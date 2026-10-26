"""Post-selection map-level diagnostics; never feeds back into checkpoint selection."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miso.data import PointDataset, load_points
from miso.metrics import evaluate
from miso.model import CSIUNet, DDPM, IndependentCSIUNet
from miso.training import sample_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.run / "best.pt", map_location=device)
    model = (CSIUNet if saved["config"]["mode"] == "joint" else IndependentCSIUNet)().to(device)
    model.load_state_dict(saved["model"], strict=True)
    diffusion = DDPM(model, saved["config"]["timesteps"]).to(device)
    h, xyz, groups = load_points(saved["config"]["dataset"], "test")
    dataset = PointDataset(h, xyz, groups, saved["stats"])
    p = sample_points(diffusion, dataset, saved["config"]["evaluation_batch_size"], device, 12345)
    metrics = {str(group): evaluate(h[groups == group], p[groups == group], saved["stats"]["h_rms"]) for group in np.unique(groups)}
    output = args.run / "diagnostics"
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "test_prediction_seed12345.npz", h_real=p.real, h_imag=p.imag)
    (output / "per_map_test.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    unknown = np.setdiff1d(np.arange(16), [0, 8, 15])
    true_power = np.sum(np.abs(h[:, :, unknown].astype(np.complex128)) ** 2, axis=(1, 2))
    error = np.sum(np.abs((h - p)[:, :, unknown].astype(np.complex128)) ** 2, axis=(1, 2))
    point_nmse = 10 * np.log10(np.maximum(error / np.maximum(true_power, 1e-30), 1e-30))
    for group, ax in enumerate(axes.flat):
        selected = groups == group
        artist = ax.scatter(xyz[selected, 0], xyz[selected, 1], c=point_nmse[selected], s=5, cmap="viridis", vmin=-20, vmax=5)
        ax.set_title("Map {}: {:.2f} dB".format(group, metrics[str(group)]["complex_nmse_db"]))
        ax.set_xlabel("Relative x (m)")
        ax.set_ylabel("Relative y (m)")
    fig.subplots_adjust(right=0.90, hspace=0.35, wspace=0.20)
    fig.colorbar(artist, cax=fig.add_axes([0.92, 0.15, 0.015, 0.70]), label="Per-point complex NMSE (dB)")
    fig.savefig(output / "test_spatial_errors.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
