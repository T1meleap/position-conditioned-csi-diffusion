"""Read-only geometric/power diagnostics of the frozen spatial partitions."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT / "datasets" / "deepmimo_o1_drone_200_8x1"
    maps = []
    for i, path in enumerate(sorted((root / "maps").glob("*.npz"))):
        with np.load(path) as data, np.load(root / "splits" / ("split_%03d.npz" % i)) as split:
            relative = data["rx_pos"] - data["tx_pos"].reshape(1, 3)
            distance = np.linalg.norm(relative, axis=1)
            valid = np.isfinite(data["path_power"])
            direct = np.any(valid & (data["inter"] == 0), axis=1)
            h = data["h_real"][:, 0].astype(np.float64) + 1j * data["h_imag"][:, 0].astype(np.float64)
            power = np.mean(np.abs(h) ** 2, axis=(1, 2))
            summary = {"map": i, "tx_position_m": data["tx_pos"].tolist(), "splits": {}}
            for name in ("train", "val", "test"):
                idx = split[name + "_idx"]
                summary["splits"][name] = {"points": len(idx), "relative_x_range_m": [float(v) for v in (relative[idx, 0].min(), relative[idx, 0].max())],
                        "distance_range_m": [float(v) for v in (distance[idx].min(), distance[idx].max())],
                        "direct_path_fraction": float(direct[idx].mean()), "channel_rms_absolute": float(np.sqrt(power[idx].mean()))}
        maps.append(summary)
        print("Map {}: {}".format(i, json.dumps(summary["splits"])), flush=True)
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    (output / "scene_partition_audit.json").write_text(json.dumps(maps, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
