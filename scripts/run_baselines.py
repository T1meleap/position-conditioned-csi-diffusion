import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from threadpoolctl import threadpool_limits
from miso.baselines import GMMEstimator, LMMSE, interpolate
from miso.data import PILOTS, load_points, provenance, training_stats
from miso.metrics import evaluate


def main():
    root = ROOT / "datasets" / "deepmimo_o1_drone_200_8x1"
    output = ROOT / "runs" / "baselines_original"
    if output.exists():
        raise FileExistsError("Do not overwrite baseline results")
    output.mkdir(parents=True)
    ht, ct, _ = load_points(root, "train")
    hv, cv, _ = load_points(root, "val")
    stats = training_stats(root, ht, ct)
    scale = stats["h_rms"]
    source_hashes = provenance(root)
    metadata = json.loads((root / "metadata.json").read_text())
    candidates, selected, failures = [], {}, {}
    frequencies = np.array(metadata["selected_subcarriers"])
    models = {"LS RI interpolation": None}
    try:
        models["Joint raw LMMSE"] = LMMSE().fit(ht)
    except np.linalg.LinAlgError as exc:
        failures["Joint raw LMMSE"] = str(exc)
    best = float("inf")
    for ridge in (1e-6, 0.001, 0.01):
        model = LMMSE().fit(ht, ridge)
        metrics = evaluate(hv, model.predict(hv[:, :, PILOTS]), scale)
        candidates.append({"method": "Joint relative-ridge LMMSE", "config": model.info, "validation": metrics})
        if metrics["complex_nmse_db"] < best:
            best, models["Joint relative-ridge LMMSE"] = metrics["complex_nmse_db"], model
            selected["Joint relative-ridge LMMSE"] = model.info
    for position in (False, True):
        name = "GMM-MMSE XYZ" if position else "GMM-MMSE no position"
        best = float("inf")
        for components in (1, 4):
            for ridge in (0.001, 0.01):
                print("FIT {} K={} ridge={}".format(name, components, ridge), flush=True)
                model = GMMEstimator().fit(ht, ct, scale, components, ridge, position)
                metrics = evaluate(hv, model.predict(hv[:, :, PILOTS], cv), scale)
                candidates.append({"method": name, "config": model.info, "validation": metrics})
                print("VAL {} converged={} nmse={:.3f} beam={:.3f}".format(name, model.info["converged"], metrics["complex_nmse_db"], metrics["beamforming_fraction_power_weighted"]), flush=True)
                if model.info["converged"] and metrics["complex_nmse_db"] < best:
                    best, models[name] = metrics["complex_nmse_db"], model
                    selected[name] = model.info
        if name not in models:
            failures[name] = "No converged validation candidate"
    (output / "selection.json").write_text(json.dumps({"selected": selected, "candidates": candidates,
                         "failures": failures, "selection_metric": "validation complex NMSE", "test_used": False}, indent=2, allow_nan=False))
    # No held-out target is loaded until all choices have been saved.
    htest, ctest, _ = load_points(root, "test")
    results = {}
    for name, model in models.items():
        prediction = interpolate(htest[:, :, PILOTS], frequencies) if model is None else (
            model.predict(htest[:, :, PILOTS], ctest) if isinstance(model, GMMEstimator) else model.predict(htest[:, :, PILOTS]))
        results[name] = evaluate(htest, prediction, scale)
        np.savez_compressed(output / (name.replace(" ", "_") + ".npz"), real=prediction.real, imag=prediction.imag)
        if model is not None:
            # Local trusted experiment artifact, not an interchange format.
            with (output / (name.replace(" ", "_") + ".pkl")).open("wb") as stream:
                pickle.dump(model, stream)
        print("TEST {} {}".format(name, json.dumps(results[name])), flush=True)
    if provenance(root) != source_hashes:
        raise ValueError("Baseline source/data changed")
    (output / "results.json").write_text(json.dumps({"methods": results, "failures": failures,
              "source_hashes": source_hashes, "selection": selected, "test_samples": len(htest)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
