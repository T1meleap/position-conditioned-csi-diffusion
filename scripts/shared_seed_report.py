"""Report all three predeclared training seeds, not nine independent training runs."""
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miso.data import provenance, sha256
from miso.metrics import SCORES

SEEDS = (2026, 2027, 2028)
RUNS = ROOT / "runs" / "shared_independent_3seed_400_T200"
REFERENCE = ROOT / "runs" / "matrix_400_T200" / "independent" / "seed_2026"
OUTPUT = ROOT / "reports" / "shared_independent_3seed_400_T200"
TABLE_METRICS = ["complex_nmse_db", "magnitude_nmse_db", "weighted_phase_deg", "strong_phase_deg",
                 "relative_antenna_phase_deg", "beamforming_fraction_power_weighted", "beamforming_loss_db"]
LABELS = ["Complex NMSE (dB)", "Magnitude NMSE (dB)", "Phase (deg)", "Strong phase (deg)",
          "Inter-antenna phase (deg)", "MRT fraction", "MRT loss (dB)"]


def aggregate_runs(results):
    if sorted(r["training_seed"] for r in results) != list(SEEDS):
        raise ValueError("Exactly the three unique predeclared training seeds are required")
    ordered = sorted(results, key=lambda r: r["training_seed"])
    reference = ordered[0]
    common = {k: v for k, v in reference["config"].items() if k != "seed"}
    if common["mode"] != "independent":
        raise ValueError("This report is for the shared independent-antenna model only")
    summary = {}
    for result in ordered:
        if {k: v for k, v in result["config"].items() if k != "seed"} != common:
            raise ValueError("Configurations differ by more than training seed")
        if result["config"]["seed"] != result["training_seed"]:
            raise ValueError("Run seed disagrees with configuration")
        if sorted(r["sampling_seed"] for r in result["per_seed"]) != common["test_sampling_seeds"]:
            raise ValueError("Sampling seeds differ")
        for key in ("parameters", "train_samples", "validation_samples", "test_samples"):
            if result[key] != reference[key]:
                raise ValueError("Run size or architecture changed: " + key)
        for metric in SCORES:
            samples = np.array([r[metric] for r in result["per_seed"]], dtype=float)
            if not np.isfinite(samples).all():
                raise ValueError("Nonfinite score")
            if not np.isclose(samples.mean(), result["aggregate"][metric + "_mean"], rtol=1e-12, atol=1e-12):
                raise ValueError("Stored sampling-seed mean disagrees")
    for metric in SCORES:
        means = np.array([r["aggregate"][metric + "_mean"] for r in ordered], dtype=float)
        summary[metric] = {"mean": float(means.mean()), "training_seed_std": float(means.std(ddof=1)),
                           "per_training_seed": {str(s): float(v) for s, v in zip(SEEDS, means)}}
    return summary


def plot_curves(paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    columns = ["train_loss", "validation_loss", "val_complex_nmse_db", "val_magnitude_nmse_db",
               "val_weighted_phase_deg", "val_beamforming_fraction"]
    titles = ["Training v-MSE", "Validation v-MSE (EMA)", "Validation complex NMSE (dB)",
              "Validation magnitude NMSE (dB)", "Validation phase error (deg)", "Validation MRT fraction"]
    for seed, path in zip(SEEDS, paths):
        rows = np.genfromtxt(path / "history.csv", delimiter=",", names=True)
        for ax, column in zip(axes.flat, columns):
            valid = np.isfinite(rows[column])
            ax.plot(rows["epoch"][valid], rows[column][valid], label=str(seed))
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(title="Training seed")
    fig.tight_layout()
    fig.savefig(OUTPUT / "training_curves.png", dpi=180)
    plt.close(fig)


def main():
    paths = [REFERENCE] + [RUNS / ("seed_%d" % s) for s in SEEDS[1:]]
    results = [json.loads((p / "metrics.json").read_text()) for p in paths]
    summary = aggregate_runs(results)
    source = json.loads((REFERENCE / "source_hashes.json").read_text())
    if provenance(results[0]["config"]["dataset"]) != source:
        raise ValueError("Core or data provenance changed")
    for path in paths:
        if json.loads((path / "source_hashes.json").read_text()) != source:
            raise ValueError("Runs use different code/data")
        np.testing.assert_array_equal(np.load(REFERENCE / "validation_indices.npy"), np.load(path / "validation_indices.npy"))
    protected = json.loads((RUNS / "protected_artifacts.json").read_text())
    for path, expected in protected.items():
        if sha256(path) != expected:
            raise ValueError("Protected artifact changed: " + path)
    preservation = json.loads((RUNS / "preservation_check.json").read_text())
    if not preservation["protected_files_unchanged"]:
        raise ValueError("Preservation check failed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    import torch
    from miso.model import IndependentCSIUNet
    checkpoint_checks = []
    reference_stats = json.loads((Path(results[0]["config"]["dataset"]) / "normalization.json").read_text())
    for path, result in zip(paths, results):
        saved = torch.load(path / "best.pt", map_location="cpu")
        if saved["config"] != result["config"] or saved["stats"] != reference_stats or saved["epoch"] != result["best_epoch"]:
            raise ValueError("Checkpoint configuration or normalization mismatch")
        if not all(torch.isfinite(v).all() for v in saved["model"].values()):
            raise ValueError("Nonfinite checkpoint weights")
        model = IndependentCSIUNet()
        model.load_state_dict(saved["model"], strict=True)
        if sum(p.numel() for p in model.parameters()) != result["parameters"]:
            raise ValueError("Checkpoint parameter count changed")
        for score in result["per_seed"]:
            if score["pilot_max_error_normalized"] > 1e-5 or not 0 <= score["beamforming_fraction_power_weighted"] <= 1:
                raise ValueError("Pilot clamp or beamforming bound failed")
        rows = np.genfromtxt(path / "history.csv", delimiter=",", names=True)
        if int(rows["epoch"][-1]) != result["completed_epochs"]:
            raise ValueError("Training history and result disagree")
        checkpoint_checks.append({"seed": result["training_seed"], "weights_finite": True,
                                  "checkpoint_sha256": sha256(path / "best.pt")})
    payload = {"training_seeds": list(SEEDS), "run_paths": [str(p) for p in paths], "metrics": summary,
               "common_config": {k: v for k, v in results[0]["config"].items() if k != "seed"},
               "aggregation": "average three sampling-seed metrics within each training run, then mean and sample SD across three training seeds; no prediction ensemble",
               "test_samples": results[0]["test_samples"], "source_hashes": source, "preservation": preservation,
               "checkpoint_checks": checkpoint_checks,
               "per_run": [{"seed": r["training_seed"], "best_epoch": r["best_epoch"], "completed_epochs": r["completed_epochs"],
                            "full_validation": r["full_validation"], "test": r["aggregate"]} for r in results],
               "runtime": {"python": platform.python_version(), "python_executable": sys.executable,
                           "torch": torch.__version__, "cuda": torch.version.cuda}}
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Shared-Antenna Diffusion: Three Training Seeds", "",
             "DeepMIMO downlink 8-TX / 1-RX MISO; three observed frequencies per transmit antenna plus relative XYZ.",
             "Original 16000 / 8000 / 16000 spatial train/validation/test partitions are unchanged.",
             "Original seed 2026 is reused unchanged; seeds 2027 and 2028 are independently trained from random initialization.",
             "T=200, 400-epoch cap, original validation-only checkpoint selection and 30-epoch patience remain unchanged.",
             "Test metrics score only the 13 missing frequencies; the three observed pilot frequencies are excluded.",
             "Within each run, three sampling-seed metric values are averaged. Mean +/- sample SD below is across three TRAINING seeds, not nine training runs.",
             "No seed selection or prediction ensembling is used.", "", "## Per-Seed Test Results", "",
             "| Training seed | Best epoch | Completed epochs | " + " | ".join(LABELS) + " |",
             "| --- | ---: | ---: | " + " | ".join(["---:"] * len(TABLE_METRICS)) + " |"]
    for result in results:
        lines.append("| {} | {} | {} | {} |".format(result["training_seed"], result["best_epoch"], result["completed_epochs"],
                    " | ".join("{:.3f}".format(result["aggregate"][m + "_mean"]) for m in TABLE_METRICS)))
    lines += ["", "## Comparison With Frozen Baselines", "",
              "| Method | " + " | ".join(LABELS) + " |", "| --- | " + " | ".join(["---:"] * len(TABLE_METRICS)) + " |"]
    lines.append("| Shared independent diffusion + XYZ | " + " | ".join("{:.3f} +/- {:.3f}".format(summary[m]["mean"], summary[m]["training_seed_std"]) for m in TABLE_METRICS) + " |")
    baselines = json.loads((ROOT / "runs" / "baselines_original" / "results.json").read_text())
    if baselines["source_hashes"] != source or baselines["test_samples"] != results[0]["test_samples"]:
        raise ValueError("Baseline protocol mismatch")
    for name, scores in baselines["methods"].items():
        label = name + (" (K={})".format(baselines["selection"][name]["components"]) if name.startswith("GMM") else "")
        lines.append("| " + label + " | " + " | ".join("{:.3f}".format(scores[m]) for m in TABLE_METRICS) + " |")
    lines += ["", "## Validation / Test Gap", "", "| Training seed | Full validation NMSE (dB) | Full validation phase (deg) | Test NMSE (dB) | Test phase (deg) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for result in results:
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(result["training_seed"], result["full_validation"]["complex_nmse_db"],
                     result["full_validation"]["weighted_phase_deg"], result["aggregate"]["complex_nmse_db_mean"], result["aggregate"]["weighted_phase_deg_mean"]))
    lines += ["", "MRT fraction is true-channel-energy-weighted received-power efficiency relative to oracle CSI MRT at fixed total transmit power; it is not BER or prediction accuracy.",
              "The existing GMM grid and fitted baseline models are unchanged; both selected GMM models have K=1. This is not a comparison against fully optimized GMM or theoretical optimal MMSE.",
              "Training-seed variability measures optimization stability on one fixed dataset/split, not generalization to new scenes or independent spatial holdouts.",
              "No pilot measurement noise, RF calibration errors, coordinate noise or UAV dynamics are added in this extension.",
              "All protected old experiment files, frozen core/data hashes and validation indices passed preservation checks.", "",
              "![Training curves](training_curves.png)"]
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_curves(paths)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
