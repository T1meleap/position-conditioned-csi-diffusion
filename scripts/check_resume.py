"""Small actual-data interruption check; not a performance experiment."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miso.training import train


def main():
    torch.set_num_threads(4)
    dataset = ROOT / "datasets" / "deepmimo_o1_drone_200_8x1"
    full, split = ROOT / "runs" / "resume_check_full", ROOT / "runs" / "resume_check_split"
    options = dict(epochs=2, timesteps=2, batch_size=32, evaluation_batch_size=64, cap=64)
    train(dataset, full, **options)
    train(dataset, split, stop_after=1, **options)
    train(dataset, split, resume=True, **options)
    a = torch.load(full / "last_training.pt", map_location="cpu")
    b = torch.load(split / "last_training.pt", map_location="cpu")
    maximum = 0.0
    for name in ("model", "ema"):
        for key in a[name]:
            delta = float((a[name][key] - b[name][key]).abs().max())
            maximum = max(maximum, delta)
            torch.testing.assert_close(a[name][key], b[name][key], atol=0, rtol=0)
    assert a["step"] == b["step"] and a["best_epoch"] == b["best_epoch"]
    print("RESUME CHECK: identical model/EMA weights; max_difference={}".format(maximum), flush=True)


if __name__ == "__main__":
    main()
