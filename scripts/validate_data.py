import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from miso.data import sha256


root = Path("datasets/deepmimo_o1_drone_200_8x1")
metadata = json.loads((root / "metadata.json").read_text())
checks = []
for path in sorted((root / "maps").glob("map_*.npz")):
    with np.load(path) as data:
        eligible = np.flatnonzero(np.max(np.abs(data["h_real"][:, 0, 0] + 1j * data["h_imag"][:, 0, 0]), axis=1) > 0)
        idx = eligible[np.linspace(0, len(eligible) - 1, 256).astype(int)]
        valid = np.isfinite(data["path_power"][idx])
        power = np.where(valid, 10.0 ** (data["path_power"][idx].astype(float) / 10), 0)
        phase = np.where(valid, data["path_phase"][idx].astype(float), 0)
        delay = np.where(valid, data["path_delay"][idx].astype(float), 0)
        theta = np.deg2rad(np.nan_to_num(data["aod_el"][idx].astype(float)))
        phi = np.deg2rad(np.nan_to_num(data["aod_az"][idx].astype(float)))
        # DeepMIMO's stored theta convention is used directly; no second elevation conversion.
        array = np.exp(1j * np.pi * np.arange(8)[None, :, None] * (np.sin(theta) * np.sin(phi))[:, None, :])
        f = np.array(metadata["selected_subcarriers"]) * metadata["bandwidth_hz"] / 64
        paths = np.sqrt(power / 64)[:, :, None] * np.exp(1j * phase[:, :, None] - 2j * np.pi * delay[:, :, None] * f)
        analytic = np.einsum("nap,npk->nak", array, paths)
        actual = data["h_real"][idx, 0] + 1j * data["h_imag"][idx, 0]
        error = np.sqrt(np.sum(np.abs(analytic - actual) ** 2) / np.sum(np.abs(actual) ** 2))
        if not np.isfinite(error) or error > 1e-4:
            raise ValueError("Independent array/path reconstruction failed: " + str(error))
        checks.append({"map": path.name, "analytic_relative_rms_error": float(error), "points_checked": len(idx)})
        print(checks[-1], flush=True)
for path, digest in metadata["source_files_sha256"].items():
    assert sha256(path) == digest, path
(root / "physical_validation.json").write_text(json.dumps({"analytic_checks": checks, "old_dataset_unchanged": True}, indent=2))
