import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from miso.training import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/deepmimo_o1_drone_200_8x1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=1024)
    parser.add_argument("--mode", choices=["joint", "independent"], default="joint")
    parser.add_argument("--cap", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    train(args.dataset, args.output, args.seed, args.epochs, args.timesteps, args.batch_size,
          args.evaluation_batch_size, args.mode, args.cap, args.resume)
