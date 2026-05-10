# Baseline Comparison Slide Data Sources

Snapshot date: 2026-05-09

This folder freezes the data used by `Thesis/baseline_comparison_slides_zh.tex`.
The source experiment outputs remain in `MLP/`, but the slides read the copied
CSV snapshots here so that later updates are traceable.

## Primary Source Report

- `MLP/baseline/comparison_reports/gp_vs_mlp_20260509/headline_comparison.csv`
- `MLP/baseline/comparison_reports/gp_vs_mlp_20260509/headline_comparison.md`
- `MLP/baseline/comparison_reports/stage3_dp_exp_0p5_vs_HA_NS_20260509/new_mlp_5seed_aggregate.csv`
- `MLP/baseline/comparison_reports/stage3_dp_exp_0p5_vs_HA_NS_20260509/per_seed_new_mlp_eval.csv`
- `MLP/baseline/comparison_reports/stage3_dp_exp_0p5_vs_HA_NS_20260509/analysis_summary.md`

## Upstream Evaluation Runs

- Hiroyasu-Arai calibrated:
  `MLP/baseline/Hiroyasu_Arai/outputs/20260509_020253_ha_calibrated_grouped_condition_all_all_clean_diagnostic_20260509_review`
- Naber-Siebers delay:
  `MLP/baseline/Naber_Siebers/outputs/20260509_004452_ns_delay_grouped_condition_all_clean_diagnostic_20260509`
- Old Stage-3 MLP, Delta P exponent 0.25, seed 42:
  `MLP/eval/rmse_eval_clean_20260509_004118_winner_full`
- New Stage-3 MLP, Delta P exponent 0.5, five external evaluations:
  listed in `per_seed_new_mlp_eval.csv` under `run_dir` and `eval_dir`.
- Sparse heteroscedastic GP:
  `MLP/runs_mlp/gp_baseline_20260509_201444`
- GP external comparison report:
  `MLP/baseline/comparison_reports/gp_vs_mlp_20260509`

## Protocol

All external headline rows use the same external clean evaluation set:

- `n_points = 795,561`
- `n_trajectories = 33,845`
- clean diagnostic split

The GP in-pipeline bootstrap is stored separately in `gp_bootstrap_summary.json`.
It uses the full 512-point reconstructed GP test grid with `214,016` points per
seed, so it should not be mixed numerically with the MLP winner's smaller
post-evaluation subset without noting the protocol difference.

## Headline Interpretation

On the external clean diagnostic, the GP does not beat the Delta P exponent 0.5
MLP on point accuracy:

- GP mean RMSE: `6.898 mm`
- Delta P exponent 0.5 MLP five-seed mean RMSE: `6.211 mm`
- GP has `+11.1%` higher RMSE, `+13.5%` higher MAE, and `+10.2%` higher P95
  error than the new MLP mean.

The GP's main advantage is calibration closeness on this clean set:

- GP coverage: `0.670 / 0.950`
- MLP coverage: `0.774 / 0.981`

This should be written as: GP is a strong uncertainty baseline and is closer to
nominal coverage, but it does not match the MLP's accuracy under the same
external clean protocol.

## Wording Caveat

The five-seed aggregate is a Delta P exponent 0.5 Stage-3 pipeline aggregate.
Seeds 42, 17, 99, and 7 use `anchor_off`; seed 2024 uses
`raw_reliable_no_kd`. In thesis text, call this a five-seed external aggregate
for the Delta P exponent 0.5 Stage-3 pipeline. Do not call it strict
same-ablation seed stability unless seed 2024 is replaced by an `anchor_off`
evaluation.

## Refresh Steps

1. Re-run or update the comparison report under `MLP/baseline/comparison_reports/`.
2. Copy the updated `headline_comparison.csv`, `headline_comparison.md`,
   `new_mlp_5seed_aggregate.csv`, `per_seed_new_mlp_eval.csv`, and any updated
   GP bootstrap files into this folder.
3. Update this `DATA_SOURCES.md` if report paths or protocol changed.
4. Regenerate slide figures:

   ```powershell
   C:\Users\Jiang\Documents\Mie_Postprocessing_Py\.venv\Scripts\python.exe Thesis\generated\baseline_comparison_20260509\make_figures.py
   ```

5. Rebuild `Thesis/baseline_comparison_slides_zh.tex`.
