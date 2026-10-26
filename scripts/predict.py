"""Complete one or many points from pilot CSI and relative XYZ only."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miso.data import PILOTS, PointDataset
from miso.model import CSIUNet, DDPM, IndependentCSIUNet
from miso.training import sample_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        raise ValueError("Output must have .npz extension")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite prediction")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device)
    with np.load(args.observations) as data:
        observed = data["pilot_real"] + 1j * data["pilot_imag"]
        relative_xyz = data["relative_xyz"]
    if observed.ndim != 3 or observed.shape[1:] != (8, 3) or relative_xyz.shape != (len(observed), 3):
        raise ValueError("Expected pilot_real/imag [point,8,3] and relative_xyz [point,3] in meters")
    if not len(observed) or not np.isfinite(observed).all() or not np.isfinite(relative_xyz).all():
        raise ValueError("Empty or nonfinite observations")
    # Missing entries are zero placeholders and are masked out of the condition.
    placeholder = np.zeros((len(observed), 8, 16), dtype=np.complex64)
    placeholder[:, :, PILOTS] = observed
    ds = PointDataset(placeholder, relative_xyz, np.zeros(len(observed), dtype=int), saved["stats"])
    constructor = CSIUNet if saved["config"]["mode"] == "joint" else IndependentCSIUNet
    model = constructor().to(device)
    model.load_state_dict(saved["model"], strict=True)
    diffusion = DDPM(model, saved["config"]["timesteps"]).to(device)
    prediction = sample_points(diffusion, ds, args.batch_size, device, args.seed)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nonfinite generated CSI")
    norm = np.sqrt(np.sum(np.abs(prediction.astype(np.complex128)) ** 2, axis=1, keepdims=True))
    weights = np.divide(prediction.conj(), norm, out=np.zeros_like(prediction, dtype=np.complex128), where=norm > 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, h_real=prediction.real, h_imag=prediction.imag,
                        mrt_real=weights.real, mrt_imag=weights.imag, relative_xyz=relative_xyz)
    print("Saved {} points to {}".format(len(observed), args.output), flush=True)


if __name__ == "__main__":
    main()
