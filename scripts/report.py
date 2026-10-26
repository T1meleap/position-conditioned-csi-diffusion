import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["complex_nmse_db", "magnitude_nmse_db", "weighted_phase_deg", "strong_phase_deg",
        "relative_antenna_phase_deg", "beamforming_fraction_power_weighted", "beamforming_loss_db"]
LABELS = ["Complex NMSE (dB)", "Magnitude NMSE (dB)", "Phase (deg)", "Strong phase (deg)",
          "Inter-antenna phase (deg)", "MRT fraction", "MRT loss (dB)"]


def plot_curves(runs, mode, seeds, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for seed in seeds:
        data = np.genfromtxt(runs / mode / ("seed_%d" % seed) / "history.csv", delimiter=",", names=True)
        axes[0, 0].plot(data["epoch"], data["train_loss"], label=str(seed))
        axes[0, 1].plot(data["epoch"], data["validation_loss"], label=str(seed))
        valid = np.isfinite(data["val_complex_nmse_db"])
        axes[1, 0].plot(data["epoch"][valid], data["val_complex_nmse_db"][valid], label=str(seed))
        axes[1, 1].plot(data["epoch"][valid], data["val_beamforming_fraction"][valid], label=str(seed))
    for ax, title in zip(axes.flat, ["Training v-MSE", "Validation v-MSE (EMA)", "Validation complex NMSE (dB)", "Validation MRT fraction"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(title="Training seed")
    fig.tight_layout()
    fig.savefig(output / (mode + "_training_curves.png"), dpi=180)
    plt.close(fig)


def main():
    runs = ROOT / "runs" / "matrix_400_T200"
    models = [json.loads((runs / "joint" / ("seed_%d" % s) / "metrics.json").read_text()) for s in (2026, 2027, 2028)]
    independent = json.loads((runs / "independent" / "seed_2026" / "metrics.json").read_text())
    baselines = json.loads((ROOT / "runs" / "baselines_original" / "results.json").read_text())
    summary = {"joint_seeds": [2026, 2027, 2028], "test_samples": models[0]["test_samples"],
               "joint": {}, "independent_seed_2026": independent["aggregate"], "baselines": baselines,
               "checkpoint_epochs": {str(m["training_seed"]): m["best_epoch"] for m in models}}
    summary["independent_checkpoint_epoch"] = independent["best_epoch"]
    summary["runtime"] = {"python": platform.python_version(), "executable": sys.executable,
                          "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
                          "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    for key in KEYS:
        values = [m["aggregate"][key + "_mean"] for m in models]
        summary["joint"][key] = {"mean": float(np.mean(values)), "training_seed_std": float(np.std(values, ddof=1))}
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# DeepMIMO 8x1 MISO Results", "", "Three predefined joint-model seeds; mean +/- sample standard deviation across training seeds.",
             "Within each run, metrics are averaged over three diffusion sampling seeds, not prediction ensembles.",
             "All scores exclude the three observed frequencies. Independent control uses one training seed only.", "",
             "| Method | " + " | ".join(LABELS) + " |", "| --- | " + " | ".join(["---:"] * len(KEYS)) + " |"]
    lines.append("| Joint diffusion + XYZ | " + " | ".join("{:.3f} +/- {:.3f}".format(summary["joint"][k]["mean"], summary["joint"][k]["training_seed_std"]) for k in KEYS) + " |")
    lines.append("| Shared independent diffusion + XYZ | " + " | ".join("{:.3f}".format(independent["aggregate"][k + "_mean"]) for k in KEYS) + " |")
    for name, value in baselines["methods"].items():
        scores = value.get("test", value)
        label = name
        if name.startswith("GMM"):
            label += " (K={})".format(baselines["selection"][name]["components"])
        lines.append("| " + label + " | " + " | ".join("{:.3f}".format(scores[k]) for k in KEYS) + " |")
    lines += ["", "## Validation/Test Gap", "", "| Model | Full validation NMSE (dB) | Test NMSE (dB) |", "| --- | ---: | ---: |"]
    for m in models + [independent]:
        lines.append("| {} seed {} | {:.3f} | {:.3f} |".format(m["mode"], m["training_seed"],
                     m["full_validation"]["complex_nmse_db"], m["aggregate"]["complex_nmse_db_mean"]))
    lines += ["", "## Initial Interpretation", "",
              "The shared independent-antenna control gives better test CSI reconstruction than the channel-concatenated joint model.",
              "Its MRT efficiency is essentially tied with complex linear interpolation; the current results do not establish a substantial beamforming advantage.",
              "Both neural approaches retain a validation/test spatial generalization gap. The independent control has only one training seed and needs replication.",
              "The joint model should be treated as an initial reference, not a finished high-performance MISO estimator."]
    lines += ["", "MRT weights are normalized to fixed total transmit power. The fraction is energy-weighted relative to oracle CSI MRT; higher is better.",
              "Linear interpolation can retain antenna-relative phases despite a large common absolute-phase error. Better CSI NMSE does not automatically imply better MRT.",
              "The GMM grid uses K in {1,4}, common-scale covariance ridge in {0.001,0.01}, max_iter=100 and one initialization; non-converged fits are excluded.",
              "These are initial baselines, not evidence that a fully optimized GMM estimator cannot perform better.",
              "Noisy sounding, channel aging, imperfect synchronization and UAV attitude changes are not modeled.",
              "The 200 GHz carrier, 10 MHz bandwidth and five retained path slots match the previous DeepMIMO data configuration.",
              "Absolute and relative-phase errors use true-channel amplitude thresholds; zero predictions are not silently dropped."]
    audit = ROOT / "reports" / "scene_partition_audit.json"
    if audit.exists():
        scene = json.loads(audit.read_text())
        ranges = scene[0]["splits"]
        lines += ["", "## Frozen Spatial Partitions", "",
                  "Relative-x ranges (meters): " + "; ".join("{} [{:.3f}, {:.3f}]".format(k, *ranges[k]["relative_x_range_m"]) for k in ("train", "val", "test")) + ".",
                  "This is out-of-region evaluation, not random held-out points spread throughout the training region.",
                  "All selected validation/test points have a direct ray. Low-BS training maps contain only approximately 44-46% direct-ray points.",
                  "Geometry and propagation-distribution shift are present; test data is not simply a more NLOS-heavy set.",
                  "The scene audit is post-selection diagnostic information and is not an added model input."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_curves(runs, "joint", (2026, 2027, 2028), output)
    plot_curves(runs, "independent", (2026,), output)
    print((output / "RESULTS.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
