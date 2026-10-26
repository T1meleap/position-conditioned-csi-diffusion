import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PILOTS = np.array([0, 8, 15], dtype=np.int64)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_points(root, split):
    hs, coordinates, groups = [], [], []
    root = Path(root)
    for i, path in enumerate(sorted((root / "maps").glob("map_*.npz"))):
        with np.load(path) as data, np.load(root / "splits" / ("split_%03d.npz" % i)) as indices:
            selected = indices[split + "_idx"]
            if data["h_real"].ndim != 4 or data["h_real"].shape[1:] != (1, 8, 16):
                raise ValueError("Expected [point,1,8,16], not a flattened channel")
            h = data["h_real"][selected, 0] + 1j * data["h_imag"][selected, 0]
            hs.append(h.astype(np.complex64))
            coordinates.append((data["rx_pos"][selected] - data["tx_pos"].reshape(1, 3)).astype(np.float32))
            groups.append(np.full(len(selected), i, dtype=np.int32))
    if not hs:
        raise ValueError("No MISO maps")
    return np.concatenate(hs), np.concatenate(coordinates), np.concatenate(groups)


def training_stats(root, h, coordinates):
    path = Path(root) / "normalization.json"
    stats = {"h_rms": float(np.sqrt(np.mean(np.abs(h.astype(np.complex128)) ** 2))),
             "relative_mean": coordinates.mean(axis=0).astype(float).tolist(),
             "relative_std": (coordinates.std(axis=0) + 1e-12).astype(float).tolist(),
             "scope": "training points only; one shared complex scale for all antennas/frequencies"}
    if not np.isfinite(stats["h_rms"]) or stats["h_rms"] <= 0:
        raise ValueError("Invalid training RMS")
    if path.exists():
        if json.loads(path.read_text()) != stats:
            raise ValueError("Cached training normalization changed")
    else:
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def encode(h, scale):
    return np.concatenate([h.real / scale, h.imag / scale], axis=1).astype(np.float32)


def decode(value, scale):
    if value.ndim != 3 or value.shape[1] % 2:
        raise ValueError("Expected [point,2*num_tx,frequency]")
    antennas = value.shape[1] // 2
    return (value[:, :antennas] + 1j * value[:, antennas:]) * scale


class PointDataset(Dataset):
    def __init__(self, h, coordinates, groups, stats):
        self.h, self.groups, self.stats = h, groups, stats
        self.clean = encode(h, stats["h_rms"])
        self.coordinates = ((coordinates - np.array(stats["relative_mean"])) / np.array(stats["relative_std"])).astype(np.float32)
        mask = np.zeros((len(h), 1, h.shape[-1]), dtype=np.float32)
        mask[:, :, PILOTS] = 1
        self.condition = np.concatenate([self.clean * mask, mask], axis=1)

    def __len__(self):
        return len(self.h)

    def __getitem__(self, index):
        return (torch.from_numpy(self.clean[index]), torch.from_numpy(self.condition[index]),
                torch.from_numpy(self.coordinates[index]))

    def subset(self, indices):
        value = object.__new__(PointDataset)
        for name in ("h", "groups", "clean", "coordinates", "condition"):
            setattr(value, name, getattr(self, name)[indices].copy())
        value.stats = self.stats
        return value


def validation_indices(dataset, per_map=128):
    indices = []
    for group in np.unique(dataset.groups):
        pool = np.flatnonzero(dataset.groups == group)
        indices.extend(pool[np.linspace(0, len(pool) - 1, min(per_map, len(pool))).astype(int)])
    return np.asarray(indices, dtype=np.int64)


def provenance(root):
    root = Path(root).resolve()
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [root / "metadata.json", root / "normalization.json"]
    paths += sorted((root / "maps").glob("*.npz")) + sorted((root / "splits").glob("*.npz"))
    return {str(p): sha256(p) for p in paths}
