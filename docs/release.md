# Release Notes

This repository is a publication copy of the MISO experiment only. The original SISO and RSS projects, raw data, runs and model checkpoints are not modified or uploaded.

## Included

- Exact copies of the MISO model, training, data, metrics and baseline modules.
- Training, inference, analysis and historical audit scripts.
- Unit tests, dependency files and dataset-format documentation.
- Per-run metrics, configurations, loss/validation histories and fixed validation indices as JSON.
- Baseline selection logs, original result summaries, physical audit summaries and training curves.
- Original dataset split indices exported as JSON, not channel or ray data.

Machine-specific root paths in exported JSON/Markdown are replaced with portable root labels. Exported historical hashes describe original files, not the sanitized JSON bytes. Sanitized archive manifests cannot be used to resume a historical checkpoint under a new path without rebuilding provenance. Source-module copies are byte-identical except the dataset generator's machine-specific default input paths, which are replaced by required CLI arguments. No model mathematics or trained weights are changed.

`scripts/shared_seed_report.py`, `run_shared_seeds.py`, `report.py`, `verify_artifacts.py` and `validate_data.py` preserve original audit workflows. Full historical report regeneration requires the omitted run checkpoints, protection manifests and/or SISO reference files. The archived reports are provided as evidence, not as a claim that these audit commands work from a fresh clone alone.

There are no bundled trained weights or raw data. Unit tests can run without them; full training and inference require the dataset and, for inference, a locally trained trusted checkpoint. No new formal training or full dataset generation was performed while packaging this release.
