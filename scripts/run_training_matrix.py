import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miso.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    jobs = [("joint", seed) for seed in (2026, 2027, 2028)] + [("independent", 2026)]
    output = ROOT / "runs" / "matrix_400_T200"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for mode, seed in jobs:
        result = train(ROOT / "datasets" / "deepmimo_o1_drone_200_8x1", output / mode / ("seed_%d" % seed),
                       seed=seed, mode=mode, resume=args.resume)
        results.append(result)
        (output / "progress.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print("MATRIX COMPLETE", flush=True)


if __name__ == "__main__":
    main()
