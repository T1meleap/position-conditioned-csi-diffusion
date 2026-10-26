"""Extend the frozen shared-antenna experiment without modifying its reference run."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miso.data import provenance, sha256
from miso.training import train

REFERENCE = ROOT / "runs" / "matrix_400_T200" / "independent" / "seed_2026"
OUTPUT = ROOT / "runs" / "shared_independent_3seed_400_T200"


def protected_files():
    paths = []
    for folder in (ROOT / "runs" / "matrix_400_T200", ROOT / "runs" / "baselines_original", ROOT / "reports"):
        paths.extend(p for p in folder.rglob("*") if p.is_file())
    metadata = json.loads((ROOT / "datasets" / "deepmimo_o1_drone_200_8x1" / "metadata.json").read_text())
    paths.extend(Path(p) for p in metadata["source_files_sha256"])
    return {str(p.resolve()): sha256(p) for p in sorted(set(paths))}


def verify_protected(manifest):
    for path, expected in manifest.items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError("Protected experiment artifact changed: " + path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("The reference used CUDA; refusing an unexpected CPU-only formal run")
    config = json.loads((REFERENCE / "config.json").read_text())
    reference = json.loads((REFERENCE / "metrics.json").read_text())
    source = json.loads((REFERENCE / "source_hashes.json").read_text())
    if config != reference["config"] or config["mode"] != "independent" or config["seed"] != 2026:
        raise ValueError("Invalid reference experiment")
    if config["epochs"] != 400 or config["timesteps"] != 200 or config["cap"] != 0:
        raise ValueError("Reference is not the full 400/T200 experiment")
    if provenance(config["dataset"]) != source:
        raise ValueError("Core code or dataset differs from the reference")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "protected_artifacts.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        verify_protected(manifest)
    else:
        manifest = protected_files()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plan = {"training_seeds": [2026, 2027, 2028], "new_training_seeds": [2027, 2028],
            "reference": str(REFERENCE), "reference_config": config, "source_hashes": source,
            "selection": "unchanged validation-only complex NMSE; no seed cherry-picking"}
    plan_path = OUTPUT / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Extension plan changed")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    results = [reference]
    for seed in (2027, 2028):
        print("START shared antenna training seed={}".format(seed), flush=True)
        result = train(config["dataset"], OUTPUT / ("seed_%d" % seed), seed=seed, epochs=config["epochs"],
                       timesteps=config["timesteps"], batch_size=config["batch_size"],
                       evaluation_batch_size=config["evaluation_batch_size"], mode=config["mode"], resume=args.resume)
        expected = dict(config, seed=seed)
        if result["config"] != expected or result["parameters"] != reference["parameters"]:
            raise ValueError("New run differs by more than training seed")
        np.testing.assert_array_equal(np.load(REFERENCE / "validation_indices.npy"),
                                      np.load(OUTPUT / ("seed_%d" % seed) / "validation_indices.npy"))
        verify_protected(manifest)
        results.append(result)
        (OUTPUT / "progress.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    if provenance(config["dataset"]) != source:
        raise ValueError("Core code or dataset changed during extension")
    verify_protected(manifest)
    (OUTPUT / "preservation_check.json").write_text(json.dumps({"protected_files": len(manifest),
        "protected_files_unchanged": True, "core_and_dataset_unchanged": True,
        "validation_indices_identical": True, "training_seeds": [2026, 2027, 2028]}, indent=2), encoding="utf-8")
    print("SHARED THREE-SEED EXTENSION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
