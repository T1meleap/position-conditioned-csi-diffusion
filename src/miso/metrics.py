import numpy as np
from .data import PILOTS


def evaluate(truth, prediction, scale):
    truth = np.asarray(truth, dtype=np.complex128) / scale
    prediction = np.asarray(prediction, dtype=np.complex128) / scale
    if truth.shape != prediction.shape or truth.ndim != 3 or not np.isfinite(prediction).all():
        raise ValueError("Expected matching finite [point,tx,subcarrier] channels")
    missing = np.setdiff1d(np.arange(truth.shape[-1]), PILOTS)
    h, p = truth[:, :, missing], prediction[:, :, missing]
    power = np.abs(h) ** 2
    total = power.sum()
    if total <= 0:
        raise ValueError("Zero-energy truth")
    phase = np.abs(np.angle(p * h.conj()))
    valid_prediction, valid_truth = np.abs(p) > 1e-6, np.abs(h) > 1e-6
    phase[~valid_prediction] = np.pi / 2
    point_rms = np.sqrt(np.mean(np.abs(truth) ** 2, axis=(1, 2), keepdims=True))
    strong = valid_truth & (np.abs(h) >= 0.1 * point_rms)
    hp, pp = power.sum(axis=1), (np.abs(p) ** 2).sum(axis=1)
    numerator = np.abs((h * p.conj()).sum(axis=1)) ** 2
    eta = np.divide(numerator, hp * pp, out=np.zeros_like(hp), where=(hp * pp) > 0)
    if eta.max() > 1 + 1e-10:
        raise ValueError("Beamforming oracle bound violated")
    eta = np.clip(eta, 0, 1)
    beam_valid = hp > 0
    pooled_eta = float((eta * hp).sum() / hp.sum())
    relative = np.abs(np.angle((p[:, 1:] * p[:, :1].conj()) * (h[:, 1:] * h[:, :1].conj()).conj()))
    relative[~(valid_prediction[:, 1:] & valid_prediction[:, :1])] = np.pi / 2
    relative_mask = strong[:, 1:] & strong[:, :1]
    relative_weights = np.abs(h[:, 1:]) * np.abs(h[:, :1]) * relative_mask
    def db(value):
        return float(10 * np.log10(max(float(value), 1e-30)))
    return {"complex_nmse_db": db(np.sum(np.abs(p - h) ** 2) / total),
            "magnitude_nmse_db": db(np.sum((np.abs(p) - np.abs(h)) ** 2) / total),
            "weighted_phase_deg": float(np.degrees((phase * power).sum() / total)),
            "strong_phase_deg": float(np.degrees(phase[strong].mean())) if strong.any() else None,
            "relative_antenna_phase_deg": float(np.degrees((relative * relative_weights).sum() / relative_weights.sum())) if relative_weights.sum() else None,
            "beamforming_fraction_mean": float(eta[beam_valid].mean()),
            "beamforming_fraction_power_weighted": pooled_eta,
            "beamforming_loss_db": -db(pooled_eta),
            "phase_prediction_coverage": float(valid_prediction[valid_truth].mean()),
            "beamforming_truth_coverage": float(beam_valid.mean()),
            "pilot_max_error_normalized": float(np.max(np.abs(prediction[:, :, PILOTS] - truth[:, :, PILOTS]))),
            "per_antenna_complex_nmse_db": [db(np.sum(np.abs(p[:, i] - h[:, i]) ** 2) / power[:, i].sum()) for i in range(h.shape[1])]}


SCORES = ["complex_nmse_db", "magnitude_nmse_db", "weighted_phase_deg", "strong_phase_deg",
          "relative_antenna_phase_deg", "beamforming_fraction_mean", "beamforming_fraction_power_weighted", "beamforming_loss_db"]
