# Evaluation Protocol

The comparison table is generated from saved experiment metrics, not from a fresh training run during repository preparation. The displayed diffusion row averages the saved test metrics for training seeds 2027 and 2028, as requested for presentation. This subset was chosen after results were available; it is not a pre-registered all-seed estimate. Seed 2026 and all original three-seed statistics remain in the archive for audit. No prediction ensembling is used.

Within each training run, reported test metrics are means across fixed diffusion sampling seeds 12345, 23456 and 34567. These are sampling repeats, not additional independently trained networks. NMSE entries are arithmetic means of the saved dB metrics.

## Training and selection

- Shared per-antenna U-Net: 1,668,866 parameters; no cross-antenna feature mixing.
- T=200, 400 epochs, batch size 512, evaluation batch size 1024.
- AdamW: learning rate 0.0002, weight decay 0.00001, five warmup epochs, cosine decay with minimum ratio 0.1.
- v prediction, unknown-subcarrier v-MSE, gradient clipping, AMP and EMA.
- Checkpoint selection uses complex NMSE on a fixed 1,024-point validation subset (128 points per map).
- Full validation uses 8,000 points after checkpoint selection; final test uses 16,000 points.
- Both displayed runs select epoch 400. Test targets never select checkpoints or GMM components.

## Metrics

Only the thirteen unobserved frequencies are scored. Predictions are restored to original channel units.

- Complex NMSE: total squared complex error divided by total true complex-channel power, expressed in dB.
- Magnitude NMSE: squared amplitude error divided by true amplitude power, in dB.
- Phase: true-power-weighted circular angular error in degrees.
- Strong phase: unweighted circular error on coefficients above the implemented point-relative amplitude threshold.
- Inter-antenna phase: relative phase against antenna zero, with true-amplitude-product weighting and strong-reference masking.
- MRT efficiency: true-channel-energy-weighted received-power ratio to oracle-CSI MRT, at fixed total transmit power. It is not BER or a percentage of correct predictions.

For the convention `y = sqrt(Pt) * sum(h_m * w_m) * s + n`, the weights are `w = conj(h_est) / norm(h_est)`. A common phase rotation does not affect MRT efficiency, so large absolute phase errors need not imply poor beamforming.

## Baselines and limitations

LS uses real/imaginary linear interpolation on actual physical-bin coordinates. Raw joint LMMSE uses no internal normalization or ridge; its empirical covariance is ill-conditioned. Relative-ridge LMMSE is an explicitly separate baseline.

GMM uses full real/imaginary covariance and a training-only shared CSI scale; the position variant models CSI jointly with XYZ. The initial validation search is K in {1,4} and covariance ridge in {0.001,0.01}, one initialization and max_iter=100. Non-converged candidates cannot be selected. Both selected GMM models have K=1. This limited search is not an optimized GMM reference or theoretical optimal MMSE.

Results are from noiseless per-antenna pilots in one scenario under a fixed spatial split. They do not establish noisy-link robustness, independent-scene generalization or a statistically significant MRT advantage over LS.
