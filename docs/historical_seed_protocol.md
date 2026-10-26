# Shared-Antenna Three-Seed Extension

Only new training seeds 2027 and 2028 are run. Seed 2026 is reused unchanged from:
`runs/matrix_400_T200/independent/seed_2026`.

New runs: `runs/shared_independent_3seed_400_T200/seed_2027` and `seed_2028`.
New report: `reports/shared_independent_3seed_400_T200/RESULTS.md`.
Original experiments and reports are preserved as snapshots, not overwritten.

## Frozen Controls

- Same DeepMIMO 8-TX / 1-RX dataset, original spatial point indices and training-only normalization.
- Same shared single-antenna U-Net, RI CSI representation, relative XYZ and three pilot frequencies.
- Same v-prediction, zero-terminal-SNR T=200 schedule, inpainting clamp and EMA.
- Same AdamW LR 0.0002, weight decay 0.00001, 5-epoch warmup/cosine schedule and batch size 512.
- Same 400-epoch cap, 30-epoch validation patience and validation-only complex-NMSE checkpoint selection.
- Same evaluation batch size 1024 and sampling seeds 12345, 23456, 34567.
- All three training seeds are reported, including any worse result; no prediction ensemble is formed.
- Existing MMSE/LS baselines are reused without fitting or tuning them again.

The statistics first average sampling-seed metrics within each run, then compute mean and sample SD
across the three training runs (ddof=1). They describe training-randomness stability on one fixed
scene and spatial partition, not three independent datasets or spatial holdouts.

## Run (PowerShell)

```powershell
Set-Location /path/to/project
$python = 'python'
& $python -u scripts\run_shared_seeds.py --resume
& $python scripts\shared_seed_report.py
```

Full-state resume is supported only for the identical configuration and frozen code/data.
`protected_artifacts.json` and `preservation_check.json` record checksums of old experiments and source data.
Each run retains `config.json`, source hashes, fixed validation indices, training history,
`best.pt`, full-state `last_training.pt`, and final per-sampling-seed metrics.
