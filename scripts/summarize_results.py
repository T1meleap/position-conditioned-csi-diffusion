"""Recompute the display row from archived per-run metrics, without checkpoints."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    'complex_nmse_db', 'magnitude_nmse_db', 'weighted_phase_deg',
    'relative_antenna_phase_deg', 'beamforming_fraction_power_weighted',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2027, 2028])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Training seeds must be unique')
    records = [json.loads((ROOT / 'results/runs/independent' / ('seed_%d' % seed) / 'metrics.json').read_text())
               for seed in args.seeds]
    row = {}
    for metric in METRICS:
        per_run = []
        for seed, record in zip(args.seeds, records):
            if record['training_seed'] != seed:
                raise ValueError('Run seed does not match its directory')
            scores = [r[metric] for r in record['per_seed']]
            value = sum(scores) / len(scores)
            if abs(value - record['aggregate'][metric + '_mean']) > 1e-10:
                raise ValueError('Saved aggregate is inconsistent')
            per_run.append(value)
        row[metric] = sum(per_run) / len(per_run)
    baseline = json.loads((ROOT / 'results/baselines/results.json').read_text())
    print('| Method | Complex NMSE (dB) | Magnitude NMSE (dB) | Phase (deg) | Inter-antenna phase (deg) | MRT efficiency |')
    print('| --- | ---: | ---: | ---: | ---: | ---: |')
    for name, values in [('Shared-antenna diffusion + XYZ', row)] + list(baseline['methods'].items()):
        cells = ['%.3f' % values[key] for key in METRICS[:-1]]
        cells.append('%.2f%%' % (100 * values[METRICS[-1]]))
        print('| ' + name + ' | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
