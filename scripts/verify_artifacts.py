import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miso.data import provenance, sha256


def main():
    dataset = ROOT / "datasets" / "deepmimo_o1_drone_200_8x1"
    metadata = json.loads((dataset / "metadata.json").read_text())
    for path, expected in metadata["source_files_sha256"].items():
        assert sha256(path) == expected, "Old source data changed: " + path
    counts = dict(train=0, val=0, test=0)
    for path in sorted((dataset / "splits").glob("*.npz")):
        with np.load(path) as data:
            sets = [data[s + "_idx"] for s in counts]
            for i, a in enumerate(sets):
                assert len(a) == len(np.unique(a))
                for b in sets[i + 1:]:
                    assert not len(np.intersect1d(a, b))
            for split in counts:
                counts[split] += len(data[split + "_idx"])
    assert counts == metadata["split_counts"]
    source = provenance(dataset)
    results = []
    for path in sorted((ROOT / "runs" / "matrix_400_T200").glob("*/seed_*/metrics.json")):
        value = json.loads(path.read_text())
        assert json.loads((path.parent / "source_hashes.json").read_text()) == source
        assert value["train_samples"] == counts["train"]
        assert value["test_samples"] == counts["test"]
        checkpoint = torch.load(path.parent / "best.pt", map_location="cpu")
        assert all(torch.isfinite(v).all() for v in checkpoint["model"].values())
        for score in value["per_seed"]:
            assert score["pilot_max_error_normalized"] < 1e-5
            assert 0 <= score["beamforming_fraction_power_weighted"] <= 1
        results.append({"run": str(path.parent), "best_epoch": checkpoint["epoch"], "weights_finite": True})
    assert len(results) == 4, "Formal training matrix is incomplete"
    baselines = json.loads((ROOT / "runs" / "baselines_original" / "results.json").read_text())
    assert baselines["source_hashes"] == source and baselines["test_samples"] == counts["test"]
    report = {"old_dataset_unchanged": True, "split_disjoint": True, "counts": counts,
              "formal_runs": results, "baseline_methods": list(baselines["methods"]), "hashes_match": True}
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    (output / "artifact_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
