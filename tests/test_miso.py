import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from miso.baselines import GMMEstimator, LMMSE, interpolate
from miso.data import PILOTS, PointDataset, decode, encode, validation_indices
from miso.metrics import evaluate
from miso.model import CSIUNet, DDPM, IndependentCSIUNet
from miso.training import random_state, restore_random_state


class MISOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(123)
        self.h = self.rng.normal(size=(80, 8, 16)) + 1j * self.rng.normal(size=(80, 8, 16))

    def test_pack_round_trip_and_no_target_leakage(self):
        np.testing.assert_allclose(decode(encode(self.h, 3), 3), self.h, rtol=1e-6, atol=1e-6)
        stats = {"h_rms": 3, "relative_mean": [0, 0, 0], "relative_std": [1, 1, 1]}
        ds = PointDataset(self.h, np.zeros((80, 3)), np.repeat([0, 1], 40), stats)
        missing = np.setdiff1d(np.arange(16), PILOTS)
        other = self.h.copy()
        other[:, :, missing] *= 30
        ds2 = PointDataset(other, np.zeros((80, 3)), ds.groups, stats)
        np.testing.assert_array_equal(ds.condition, ds2.condition)
        self.assertEqual(ds.condition.shape, (80, 17, 16))
        self.assertEqual(len(validation_indices(ds, 8)), 16)

    def test_joint_network_shape_and_position_gradient(self):
        model = CSIUNet(base_channels=8)
        x, c, xyz = torch.randn(2, 16, 16), torch.randn(2, 17, 16), torch.randn(2, 3, requires_grad=True)
        y = model(x, torch.tensor([1, 2]), c, xyz)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertGreater(float(xyz.grad.abs().sum()), 0)
        self.assertEqual(sum(p.numel() for p in CSIUNet().parameters()), 1676944)

    def test_loss_ignores_known_pilots_and_sampler_clamps_all_antennas(self):
        model = CSIUNet(base_channels=8)
        ddpm = DDPM(model, 4)
        clean = torch.randn(2, 16, 16)
        mask = torch.zeros(2, 1, 16)
        mask[:, :, PILOTS] = 1
        condition = torch.cat([clean * mask, mask], dim=1)
        xyz = torch.zeros(2, 3)
        t, noise = torch.tensor([1, 3]), torch.randn_like(clean)
        loss = ddpm.loss(clean, condition, xyz, t, noise)
        self.assertTrue(torch.isfinite(loss))
        target = ddpm.at(ddpm.root, t) * noise - ddpm.at(ddpm.sigma, t) * clean
        class KnownOnlyError(torch.nn.Module):
            def forward(self, *args):
                return target + 100 * mask
        original = ddpm.model
        ddpm.model = KnownOnlyError()
        self.assertEqual(float(ddpm.loss(clean, condition, xyz, t, noise)), 0)
        ddpm.model = original
        p1, p2 = ddpm.sample(condition, xyz, 2), ddpm.sample(condition, xyz, 2)
        torch.testing.assert_close(p1, p2)
        torch.testing.assert_close(p1[:, :, PILOTS], clean[:, :, PILOTS], atol=0, rtol=0)
        self.assertEqual(float(ddpm.root[-1]), 0)

    def test_independent_model_has_no_cross_antenna_mixing(self):
        model = IndependentCSIUNet(base_channels=8).eval()
        x, c, xyz = torch.randn(1, 16, 16), torch.randn(1, 17, 16), torch.zeros(1, 3)
        a = model(x, torch.ones(1, dtype=torch.long), c, xyz)
        other = x.clone()
        other[:, [1, 9]] += 10
        other_condition = c.clone()
        other_condition[:, [1, 9]] += 10
        b = model(other, torch.ones(1, dtype=torch.long), other_condition, xyz)
        torch.testing.assert_close(a[:, [0, 8]], b[:, [0, 8]])

    def test_oracle_and_common_phase_beamforming_invariance(self):
        perfect = evaluate(self.h, self.h, 1)
        rotated = evaluate(self.h, self.h * np.exp(1j * 0.9), 1)
        self.assertAlmostEqual(perfect["beamforming_fraction_mean"], 1)
        self.assertAlmostEqual(rotated["beamforming_fraction_mean"], 1)
        self.assertAlmostEqual(rotated["relative_antenna_phase_deg"], 0, places=10)
        self.assertGreater(rotated["weighted_phase_deg"], 50)

    def test_wrong_relative_phase_and_zero_prediction_are_not_hidden(self):
        h = np.ones((2, 8, 16), dtype=complex)
        p = h.copy()
        p[:, 1::2] *= -1
        result = evaluate(h, p, 1)
        self.assertAlmostEqual(result["beamforming_fraction_mean"], 0)
        zero = evaluate(h, np.zeros_like(h), 1)
        self.assertEqual(zero["phase_prediction_coverage"], 0)
        self.assertAlmostEqual(zero["weighted_phase_deg"], 90)

    def test_interpolation_and_lmmse_pilot_only_interface(self):
        observed = self.h[:, :, PILOTS]
        p = interpolate(observed, np.array([0,4,8,13,17,21,25,29,34,38,42,46,50,55,59,63]))
        np.testing.assert_allclose(p[:, :, PILOTS], observed)
        model = LMMSE().fit(self.h)
        np.testing.assert_allclose(model.predict(observed)[:, :, PILOTS], observed)
        with self.assertRaises(ValueError):
            model.predict(self.h)

    def test_gmm_observations_and_pilots_preserved(self):
        model = GMMEstimator().fit(self.h, self.rng.normal(size=(80, 3)), 2, components=1, position=True)
        p = model.predict(self.h[:, :, PILOTS], np.zeros((80, 3)))
        np.testing.assert_allclose(p[:, :, PILOTS], self.h[:, :, PILOTS])
        self.assertTrue(np.isfinite(p).all())
        with self.assertRaises(ValueError):
            model.predict(self.h, np.zeros((80, 3)))

    def test_lmmse_recovers_complex_linear_relationship(self):
        from miso.baselines import OBSERVED
        y = self.rng.normal(size=(80, 24)) + 1j * self.rng.normal(size=(80, 24))
        gain = self.rng.normal(size=(128, 24)) + 1j * self.rng.normal(size=(128, 24))
        gain[OBSERVED] = np.eye(24)
        full = (y @ gain.T).reshape(80, 8, 16)
        model = LMMSE().fit(full[:60])
        np.testing.assert_allclose(model.predict(full[60:, :, PILOTS]), full[60:], atol=1e-12, rtol=1e-12)

    def test_random_state_round_trip(self):
        g = torch.Generator().manual_seed(1)
        state = random_state(g)
        a = torch.randn(3, generator=g)
        restore_random_state(state, g)
        torch.testing.assert_close(a, torch.randn(3, generator=g))


if __name__ == "__main__":
    unittest.main()
