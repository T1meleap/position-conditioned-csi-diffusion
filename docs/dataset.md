# Dataset Contract

## Physical configuration

- DeepMIMO scenario: `o1_drone_200`.
- Carrier: 200 GHz; bandwidth: 10 MHz; generated OFDM bins: 64.
- Selected physical bins: `[0,4,8,13,17,21,25,29,34,38,42,46,50,55,59,63]`.
- Observed selected-axis bins: `[0,8,15]`, corresponding to physical bins `[0,34,63]`.
- TX array: `[8,1]`, half-wavelength spacing, zero rotation, array axis y.
- One RX antenna, five retained ray-path slots, static isotropic channel.
- Eight maps: two transmitter positions and four receiver heights.

## Expected directory

```text
datasets/deepmimo_o1_drone_200_8x1/
  metadata.json
  normalization.json              # Created from training data if absent
  maps/map_000.npz ... map_007.npz
  splits/split_000.npz ... split_007.npz
```

Training requires `h_real` and `h_imag` shaped `[point,1,8,16]`, `rx_pos` shaped `[point,3]` and `tx_pos` containing a three-dimensional transmitter position. Split files contain integer `train_idx`, `val_idx` and `test_idx`, with no overlaps. The original grid is 124 x 544 (67,456 points per map). All antennas of one physical point stay in the same split.

Physical validation additionally uses `path_power` (dBW), `path_phase` (radians), `path_delay` (seconds), AoA/AoD (degrees), interaction codes and grid metadata. Angle conventions must follow DeepMIMO 4.0.3; do not reinterpret its polar angle as a conventional elevation without conversion.

## Split and normalization

Original spatial x bands are 70% / 12% / 18% for train / validation / test. Selected sample counts are 16,000 / 8,000 / 16,000 across eight maps (2,000 / 1,000 / 2,000 per map). The selected counts are not the spatial area proportions. No point-level random reshuffle is used.

A shared complex RMS scale and relative-XYZ standardization are fitted only on training points. There is no per-antenna gain normalization or phase alignment. Splits are archived as JSON index arrays under `results/dataset/splits/` for traceability; convert them to NPZ arrays if using another dataset preparation tool. These indices are valid only for the original receiver order and geometry.

## Generation prerequisite and current limitation

`scripts/generate_deepmimo.py` preserves the historical experiment's SISO reference and split files. It requires:

1. The locally downloaded official DeepMIMO scenario.
2. A compatible single-antenna reference export with `metadata.json`, `pilot_normalization_stats.json`, `maps/*.npz` and `splits/*.npz`.
3. Reference metadata including scenario, map count, carrier frequency, bandwidth, generated/selected subcarriers, split policy and split counts.
4. Reference maps containing SISO `h_real/h_imag` shaped `[point,16]`, receiver/transmitter positions, path arrays and the original point order.

The script regenerates SISO channels using official DeepMIMO and checks agreement before generating the eight-antenna channels. It refuses existing output directories and verifies that the reference export is unchanged. The reference data are not included here, so a fresh clone alone cannot yet regenerate the exact dataset from the scenario name. An already-generated compatible MISO dataset can be used directly for training.

No raw scenario, ray-path data, generated CSI or sample predictions are uploaded. Check DeepMIMO's data terms before redistributing any of these materials.
