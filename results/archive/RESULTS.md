# DeepMIMO 8x1 MISO Results

Three predefined joint-model seeds; mean +/- sample standard deviation across training seeds.
Within each run, metrics are averaged over three diffusion sampling seeds, not prediction ensembles.
All scores exclude the three observed frequencies. Independent control uses one training seed only.

| Method | Complex NMSE (dB) | Magnitude NMSE (dB) | Phase (deg) | Strong phase (deg) | Inter-antenna phase (deg) | MRT fraction | MRT loss (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint diffusion + XYZ | -2.749 +/- 0.224 | -5.671 +/- 0.307 | 42.622 +/- 2.362 | 36.927 +/- 3.390 | 47.181 +/- 3.185 | 0.651 +/- 0.034 | 1.871 +/- 0.226 |
| Shared independent diffusion + XYZ | -7.504 | -11.847 | 15.164 | 14.299 | 7.961 | 0.964 | 0.159 |
| LS RI interpolation | 2.762 | -9.333 | 101.387 | 100.297 | 8.194 | 0.963 | 0.164 |
| Joint raw LMMSE | 15.033 | 14.455 | 75.820 | 73.159 | 32.995 | 0.743 | 1.290 |
| Joint relative-ridge LMMSE | 1.398 | -6.058 | 79.517 | 76.467 | 43.552 | 0.717 | 1.443 |
| GMM-MMSE no position (K=1) | 1.057 | -6.090 | 79.892 | 76.931 | 46.944 | 0.690 | 1.612 |
| GMM-MMSE XYZ (K=1) | 1.059 | -6.094 | 79.896 | 76.945 | 47.038 | 0.689 | 1.618 |

## Validation/Test Gap

| Model | Full validation NMSE (dB) | Test NMSE (dB) |
| --- | ---: | ---: |
| joint seed 2026 | -10.178 | -2.561 |
| joint seed 2027 | -9.663 | -2.690 |
| joint seed 2028 | -10.059 | -2.996 |
| independent seed 2026 | -20.150 | -7.504 |

## Initial Interpretation

The shared independent-antenna control gives better test CSI reconstruction than the channel-concatenated joint model.
Its MRT efficiency is essentially tied with complex linear interpolation; the current results do not establish a substantial beamforming advantage.
Both neural approaches retain a validation/test spatial generalization gap. The independent control has only one training seed and needs replication.
The joint model should be treated as an initial reference, not a finished high-performance MISO estimator.

MRT weights are normalized to fixed total transmit power. The fraction is energy-weighted relative to oracle CSI MRT; higher is better.
Linear interpolation can retain antenna-relative phases despite a large common absolute-phase error. Better CSI NMSE does not automatically imply better MRT.
The GMM grid uses K in {1,4}, common-scale covariance ridge in {0.001,0.01}, max_iter=100 and one initialization; non-converged fits are excluded.
These are initial baselines, not evidence that a fully optimized GMM estimator cannot perform better.
Noisy sounding, channel aging, imperfect synchronization and UAV attitude changes are not modeled.
The 200 GHz carrier, 10 MHz bandwidth and five retained path slots match the previous DeepMIMO data configuration.
Absolute and relative-phase errors use true-channel amplitude thresholds; zero predictions are not silently dropped.

## Frozen Spatial Partitions

Relative-x ranges (meters): train [-237.504, 69.486]; val [70.296, 122.946]; test [123.756, 202.326].
This is out-of-region evaluation, not random held-out points spread throughout the training region.
All selected validation/test points have a direct ray. Low-BS training maps contain only approximately 44-46% direct-ray points.
Geometry and propagation-distribution shift are present; test data is not simply a more NLOS-heavy set.
The scene audit is post-selection diagnostic information and is not an added model input.
