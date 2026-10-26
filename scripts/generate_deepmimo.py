"""Generate paired 1x1/1x8 channels with DeepMIMO, preserving the SISO split."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import deepmimo as dm
import numpy as np
from deepmimo.generator.channel import ChannelParameters


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Compatible SISO reference export; see docs/dataset.md")
    parser.add_argument("--scenarios", required=True, help="Local official DeepMIMO scenario directory")
    parser.add_argument("--output", default="datasets/deepmimo_o1_drone_200_8x1")
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("The source is read-only; use an independent destination")
    if output.exists():
        raise FileExistsError("Use a fresh dataset output; existing data are never overwritten")
    metadata = json.loads((source / "metadata.json").read_text())
    source_files = [source / "metadata.json", source / "pilot_normalization_stats.json"]
    source_files += sorted((source / "maps").glob("*.npz")) + sorted((source / "splits").glob("*.npz"))
    hashes = {str(p): digest(p) for p in source_files}
    with np.load(source / "maps/map_000.npz") as first:
        slots = first["path_power"].shape[1]
    dm.config.set("scenarios_folder", str(Path(args.scenarios).resolve()))
    matrices = ["rx_pos", "tx_pos", "power", "phase", "delay", "aoa_az", "aoa_el", "aod_az", "aod_el", "inter"]
    dataset = dm.load(metadata["scenario"], max_paths=slots, matrices=matrices)
    pairs = list(dataset.datasets) if type(dataset).__name__ == "MacroDataset" else [dataset]
    if len(pairs) != metadata["map_count"]:
        raise ValueError("Original TX/RX pair inventory changed")
    output.mkdir(parents=True)
    (output / "maps").mkdir()
    (output / "splits").mkdir()
    audits = []
    frequency = metadata["carrier_frequency_hz"]
    for i, pair in enumerate(pairs):
        with np.load(source / "maps" / ("map_%03d.npz" % i)) as old:
            reference = old["h_real"] + 1j * old["h_imag"]
            rx, tx = old["rx_pos"].copy(), old["tx_pos"].copy()
            np.testing.assert_array_equal(pair["rx_pos"], rx)
            np.testing.assert_array_equal(np.asarray(pair["tx_pos"]).reshape(-1, 3)[0], tx)
            arrays = {k: old[k].copy() for k in old.files if k not in ("h_real", "h_imag")}
        common = {"ue_antenna": {"shape": [1, 1], "spacing": 0.5, "rotation": [0, 0, 0]},
                  "freq_domain": 1, "num_paths": slots, "doppler": False,
                  "ofdm": {"subcarriers": metadata["num_subcarriers_generated"],
                           "selected_subcarriers": metadata["selected_subcarriers"],
                           "bandwidth": metadata["bandwidth_hz"], "rx_filter": False}}
        siso = pair.compute_channels(ChannelParameters({**common, "bs_antenna": {"shape": [1, 1], "spacing": 0.5, "rotation": [0, 0, 0]}}))[:, 0, 0, :]
        scale = np.sqrt(np.mean(np.abs(reference.astype(np.complex128)) ** 2))
        error = np.sqrt(np.mean(np.abs(siso.astype(np.complex128) - reference) ** 2)) / scale
        if not np.isfinite(error) or error > 1e-5:
            raise ValueError("SISO reconstruction failed, relative RMS error=" + str(error))
        h = pair.compute_channels(ChannelParameters({**common, "bs_antenna": {"shape": [8, 1], "spacing": 0.5, "rotation": [0, 0, 0]}})).astype(np.complex64)
        if h.shape != (len(rx), 1, 8, 16) or not np.isfinite(h).all():
            raise ValueError("Invalid MISO tensor")
        element_error = np.sqrt(np.mean(np.abs(h[:, 0, 0, :].astype(np.complex128) - siso) ** 2)) / scale
        if element_error > 1e-5:
            raise ValueError("Antenna zero must retain the SISO reference phase")
        np.savez_compressed(output / "maps" / ("map_%03d.npz" % i), **arrays,
                            h_real=h.real, h_imag=h.imag, siso_real=siso.real, siso_imag=siso.imag)
        split_path = source / "splits" / ("split_%03d.npz" % i)
        shutil.copyfile(split_path, output / "splits" / split_path.name)
        with np.load(split_path) as split:
            parts = [split[k] for k in ("train_idx", "val_idx", "test_idx")]
            if any(np.intersect1d(parts[a], parts[b]).size for a, b in ((0, 1), (0, 2), (1, 2))):
                raise ValueError("Overlapping point splits")
        audits.append({"map": i, "shape": list(h.shape), "siso_relative_rms_error": float(error),
                       "antenna_zero_relative_rms_error": float(element_error), "split_sha256": digest(split_path),
                       "actual_path_slots": slots, "max_antennas_difference": float(np.max(np.abs(h[:, 0, 1:, :] - h[:, 0, :1, :])))})
        print("GENERATED map {} shape={} siso_error={:.3g} ant0_error={:.3g}".format(i, h.shape, error, element_error), flush=True)
    if any(digest(p) != value for p, value in hashes.items()):
        raise ValueError("Original dataset was modified")
    element_indices = np.column_stack([np.zeros(8), np.arange(8), np.zeros(8)])
    info = {"created_date": "2026-10-13", "kind": "single_point_miso_frequency_completion", "source": "DeepMIMO ray tracing",
            "scenario": metadata["scenario"], "deepmimo_version": dm.__version__, "map_count": len(pairs),
            "tensor_axes": ["point", "rx_antenna", "tx_antenna", "subcarrier"], "num_rx": 1, "num_tx": 8,
            "selected_subcarriers": metadata["selected_subcarriers"], "pilots": [0, 8, 15],
            "num_subcarriers_generated": 64, "bandwidth_hz": metadata["bandwidth_hz"], "carrier_frequency_hz": frequency,
            "actual_path_slots": slots, "array_shape": [8, 1], "array_rotation_deg": [0, 0, 0],
            "array_spacing_wavelengths": 0.5, "tx_element_offsets_m": (element_indices * 0.5 * 299792458.0 / frequency).tolist(),
            "split_policy": metadata["split_policy"], "split_counts": metadata["validation"]["split_counts"],
            "pilot_observation": "per-TX CSI from noiseless orthogonal sounding; not three scalar mixed observations",
            "sounding_example": "8 orthogonal time slots at each of 3 pilot frequencies, channel constant during sounding",
            "physical_scope": "static isotropic narrowband-array ray-tracing channel; no coupling, RF calibration error or Doppler",
            "source_files_sha256": hashes, "source_unchanged": True, "map_audits": audits}
    (output / "metadata.json").write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
    print("COMPLETED " + str(output), flush=True)


if __name__ == "__main__":
    main()
