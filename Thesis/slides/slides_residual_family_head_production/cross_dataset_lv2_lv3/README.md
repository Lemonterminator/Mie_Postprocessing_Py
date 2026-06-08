# lv2 / lv3 Fixed-Point Re-evaluation

This folder contains controlled point-table re-evaluations of curated ML checkpoints.

Dataset roots are passed explicitly:

- lv2: `MLP/synthetic_data_clean_lv2`
- lv3_qc_gated: `MLP/synthetic_data_clean_lv3_qc_gated`

Source-of-truth aggregate CSVs:

- `MLP/eval/cross_dataset_lv2_lv3/all_model_dataset_metrics.csv`
- `MLP/eval/cross_dataset_lv2_lv3/headline_cdf_uncensored.csv`
- `MLP/eval/cross_dataset_lv2_lv3/post_training_gain_lv2.csv`
- `MLP/eval/cross_dataset_lv2_lv3/qc_decomposition.csv`

Interpretation:

- Read post-training gain within the same dataset root, especially lv2.
- Decompose QC as target/eval cleanup plus retrain gain.
- Treat lv3_qc_gated as sensitivity evidence, not the only final proof.
- Use cdf_uncensored for headlines and P50/Q1 slices for behavior details.
