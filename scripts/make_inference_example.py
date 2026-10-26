"""Export four test-point pilot-only inputs; truth is a separate QA artifact."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from miso.data import PILOTS


def main():
    root = ROOT / "datasets" / "deepmimo_o1_drone_200_8x1"
    output = ROOT / "examples"
    output.mkdir(exist_ok=True)
    if (output / "observations.npz").exists():
        raise FileExistsError("Example observations already exist")
    with np.load(root / "splits" / "split_000.npz") as split, np.load(root / "maps" / "map_000.npz") as data:
        indices = split["test_idx"][:4]
        h = data["h_real"][indices, 0] + 1j * data["h_imag"][indices, 0]
        coordinates = data["rx_pos"][indices] - data["tx_pos"].reshape(1, 3)
    np.savez_compressed(output / "observations.npz", pilot_real=h[:, :, PILOTS].real,
                        pilot_imag=h[:, :, PILOTS].imag, relative_xyz=coordinates)
    np.savez_compressed(output / "truth_for_qa_only.npz", h_real=h.real, h_imag=h.imag)
    print("Exported four pilot-only observations; truth saved separately for verification")


if __name__ == "__main__":
    main()
