import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PointDataset, decode, load_points, provenance, training_stats, validation_indices
from .metrics import SCORES, evaluate
from .model import CSIUNet, DDPM, IndependentCSIUNet


def save_atomic(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def random_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "loader": generator.get_state()}


def restore_random_state(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
    generator.set_state(state["loader"].cpu())


@torch.no_grad()
def sample_points(diffusion, dataset, batch_size, device, seed):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    diffusion.eval()
    outputs = []
    for batch_id, (_, condition, coordinates) in enumerate(loader):
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            value = diffusion.sample(condition.to(device), coordinates.to(device), seed + batch_id)
        outputs.append(decode(value.cpu().numpy(), dataset.stats["h_rms"]))
    return np.concatenate(outputs).astype(np.complex64)


@torch.no_grad()
def validation_loss(diffusion, dataset, batch_size, device, seed):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(seed))
    generator = torch.Generator(device=device).manual_seed(seed)
    diffusion.eval()
    total = 0.0
    for clean, condition, coordinates in loader:
        clean, condition, coordinates = clean.to(device), condition.to(device), coordinates.to(device)
        t = torch.randint(diffusion.timesteps, (len(clean),), device=device, generator=generator)
        noise = torch.randn(clean.shape, device=device, generator=generator)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            loss = diffusion.loss(clean, condition, coordinates, t, noise)
        total += float(loss) * len(clean)
    return total / len(dataset)


def train(dataset_dir, output_dir, seed=2026, epochs=400, timesteps=200, batch_size=512,
          evaluation_batch_size=1024, mode="joint", cap=0, resume=False, stop_after=0):
    if epochs < 1 or timesteps < 2 or mode not in ("joint", "independent"):
        raise ValueError("Invalid training parameters")
    warmup = min(5, epochs - 1)
    config = {"dataset": str(Path(dataset_dir).resolve()), "seed": seed, "epochs": epochs, "timesteps": timesteps,
              "batch_size": batch_size, "evaluation_batch_size": evaluation_batch_size, "mode": mode, "cap": cap,
              "num_tx": 8, "coordinates": "relative_xyz", "learning_rate": 0.0002, "warmup_epochs": warmup,
              "minimum_lr_ratio": 0.1, "ema": 0.999, "weight_decay": 1e-5, "patience": 30,
              "validation_interval": 10, "validation_sampling_seed": 12345,
              "test_sampling_seeds": [12345, 23456, 34567], "objective": "unknown-bin v-MSE", "schema_version": 1}
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    exists = config_path.exists()
    if exists and json.loads(config_path.read_text()) != config:
        raise ValueError("Saved config differs; use a new directory")
    h, c, g = load_points(dataset_dir, "train")
    stats = training_stats(dataset_dir, h, c)
    train_data = PointDataset(h, c, g, stats)
    val_data = PointDataset(*load_points(dataset_dir, "val"), stats)
    if cap:
        train_data = train_data.subset(np.arange(min(cap, len(train_data))))
        val_data = val_data.subset(np.arange(min(cap, len(val_data))))
    indices = validation_indices(val_data)
    val_subset = val_data.subset(indices)
    hashes = provenance(dataset_dir)
    if exists:
        if json.loads((out / "source_hashes.json").read_text()) != hashes:
            raise ValueError("Dataset or code changed")
        np.testing.assert_array_equal(np.load(out / "validation_indices.npy"), indices)
        if (out / "metrics.json").exists():
            return json.loads((out / "metrics.json").read_text())
        if not resume:
            raise ValueError("Incomplete run exists; use --resume")
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        (out / "source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        np.save(out / "validation_indices.npy", indices)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    constructor = CSIUNet if mode == "joint" else IndependentCSIUNet
    network = constructor().to(device)
    ema = copy.deepcopy(network).eval()
    diffusion, ema_diffusion = DDPM(network, timesteps).to(device), DDPM(ema, timesteps).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0002, weight_decay=1e-5)
    def lr_factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = min(max((epoch - warmup + 1) / (epochs - warmup), 0), 1)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=loader_generator, pin_memory=device.type == "cuda")
    best, best_epoch, step, skipped, start_epoch, history, elapsed_before = float("inf"), 0, 0, 0, 1, [], 0.0
    if exists:
        state = torch.load(out / "last_training.pt", map_location=device)
        if state["config"] != config or state["stats"] != stats:
            raise ValueError("Full checkpoint settings disagree")
        network.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best, best_epoch, step, skipped = state["best"], state["best_epoch"], state["step"], state["skipped"]
        start_epoch, history, elapsed_before = state["epoch"] + 1, state["history"], state["elapsed_seconds"]
        restore_random_state(state["random_state"], loader_generator)
        print("RESUME {} seed={} epoch={}".format(mode, seed, start_epoch), flush=True)
    begin = time.perf_counter()
    def elapsed():
        return elapsed_before + time.perf_counter() - begin
    fields = ["epoch", "learning_rate", "train_loss", "validation_loss", "val_complex_nmse_db", "val_magnitude_nmse_db",
              "val_weighted_phase_deg", "val_relative_antenna_phase_deg", "val_beamforming_fraction", "step", "skipped", "elapsed_seconds"]
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
        for epoch in range(start_epoch, epochs + 1):
            if epoch - 1 - best_epoch >= 30 and (epoch - 1) % 10 == 0:
                break
            diffusion.train()
            total = 0.0
            lr = optimizer.param_groups[0]["lr"]
            for clean, condition, coordinates in loader:
                clean, condition, coordinates = clean.to(device), condition.to(device), coordinates.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    loss = diffusion.loss(clean, condition, coordinates)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    if not torch.isfinite(norm):
                        raise RuntimeError("Nonfinite gradient without AMP recovery")
                    step += 1
                    with torch.no_grad():
                        decay = 0.999 if step > 100 else 0.9
                        for ep, p in zip(ema.parameters(), network.parameters()):
                            ep.mul_(decay).add_(p, alpha=1 - decay)
                else:
                    skipped += 1
                total += float(loss.detach()) * len(clean)
            row = {"epoch": epoch, "learning_rate": lr, "train_loss": total / len(train_data),
                   "validation_loss": validation_loss(ema_diffusion, val_data, batch_size * 2, device, seed + 50000),
                   "step": step, "skipped": skipped, "elapsed_seconds": elapsed()}
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                p = sample_points(ema_diffusion, val_subset, evaluation_batch_size, device, 12345)
                score = evaluate(val_subset.h, p, stats["h_rms"])
                row.update(val_complex_nmse_db=score["complex_nmse_db"], val_magnitude_nmse_db=score["magnitude_nmse_db"],
                           val_weighted_phase_deg=score["weighted_phase_deg"], val_relative_antenna_phase_deg=score["relative_antenna_phase_deg"],
                           val_beamforming_fraction=score["beamforming_fraction_power_weighted"])
                if score["complex_nmse_db"] < best:
                    best, best_epoch = score["complex_nmse_db"], epoch
                    save_atomic(out / "best.pt", {"model": ema.state_dict(), "config": config, "stats": stats,
                                "epoch": epoch, "validation": score, "parameters": sum(p.numel() for p in network.parameters())})
                print("{} seed={} epoch={} val_nmse={:.3f} amplitude={:.3f} phase={:.2f} beam={:.3f} elapsed={:.0f}s".format(
                    mode, seed, epoch, score["complex_nmse_db"], score["magnitude_nmse_db"], score["weighted_phase_deg"],
                    score["beamforming_fraction_power_weighted"], elapsed()), flush=True)
            scheduler.step()
            history.append(row)
            writer.writerow(row)
            handle.flush()
            stopping = epoch - best_epoch >= 30 and epoch % 10 == 0
            if epoch % 10 == 0 or epoch == epochs or stopping or (stop_after and epoch >= stop_after):
                save_atomic(out / "last_training.pt", {"model": network.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "random_state": random_state(loader_generator),
                            "config": config, "stats": stats, "epoch": epoch, "history": history, "best": best,
                            "best_epoch": best_epoch, "step": step, "skipped": skipped, "elapsed_seconds": elapsed()})
            if stop_after and epoch >= stop_after and epoch < epochs:
                return {"status": "interrupted", "completed_epochs": epoch}
            if stopping:
                break
    saved = torch.load(out / "best.pt", map_location=device)
    ema.load_state_dict(saved["model"])
    full_val = evaluate(val_data.h, sample_points(ema_diffusion, val_data, evaluation_batch_size, device, 12345), stats["h_rms"])
    test_data = PointDataset(*load_points(dataset_dir, "test"), stats)
    if cap:
        test_data = test_data.subset(np.arange(min(cap, len(test_data))))
    per_seed = []
    for sampling_seed in (12345, 23456, 34567):
        p = sample_points(ema_diffusion, test_data, evaluation_batch_size, device, sampling_seed)
        score = evaluate(test_data.h, p, stats["h_rms"])
        if score["pilot_max_error_normalized"] > 1e-5:
            raise ValueError("Pilot clamp failed")
        per_seed.append({"sampling_seed": sampling_seed, **score})
    result = {"config": config, "training_seed": seed, "mode": mode, "parameters": saved["parameters"],
              "best_epoch": saved["epoch"], "completed_epochs": history[-1]["epoch"], "full_validation": full_val,
              "train_samples": len(train_data), "validation_samples": len(val_data), "test_samples": len(test_data),
              "per_seed": per_seed, "aggregate": {m + suffix: float(function([r[m] for r in per_seed]))
                  for m in SCORES for suffix, function in (("_mean", np.mean), ("_std", lambda a: np.std(a, ddof=1)))},
              "selection": {"metric": "fixed-validation-subset complex NMSE", "test_used": False}, "elapsed_seconds": elapsed()}
    if provenance(dataset_dir) != hashes:
        raise ValueError("Source or dataset changed during training")
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("DONE {} seed={} {}".format(mode, seed, json.dumps(result["aggregate"])), flush=True)
    return result
