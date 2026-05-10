# Stage-3 MLP ΔP^0.5 vs H-A / N-S / GP Baselines

All rows use the same external clean evaluation protocol: 795,561 points and
33,845 trajectories.

| Model | RMSE mm | MAE mm | P95 mm | Bias mm | 1sigma cov | 2sigma cov |
|---|---:|---:|---:|---:|---:|---:|
| Hiroyasu-Arai calibrated | 9.974 | 7.686 | 20.493 | 0.625 | 0.536 | 0.886 |
| Naber-Siebers delay | 8.840 | 6.780 | 17.922 | -0.344 | 0.620 | 0.893 |
| Stage-3 MLP ΔP^0.25, seed 42 | 6.696 | 4.875 | 14.118 | -1.149 | 0.745 | 0.979 |
| Stage-3 MLP ΔP^0.5, 5-seed mean | 6.211 | 4.587 | 13.029 | -0.838 | 0.774 | 0.981 |
| Sparse heteroscedastic GP, 5-seed mean | 6.898 | 5.206 | 14.365 | -1.184 | 0.670 | 0.950 |

## Relative Changes of Stage-3 MLP ΔP^0.5

- vs Hiroyasu-Arai calibrated: RMSE -37.7%, MAE -40.3%, P95 -36.4%.
- vs Naber-Siebers delay: RMSE -29.7%, MAE -32.3%, P95 -27.3%.
- vs old Stage-3 MLP ΔP^0.25 seed 42: RMSE -7.2%, MAE -5.9%, P95 -7.7%.
- vs sparse heteroscedastic GP: RMSE -10.0%, MAE -11.9%, P95 -9.3%.

The ΔP^0.5 feature engineering improves the full-clean external evaluation
relative to both physical baselines and the previous ΔP^0.25 MLP. The
five-seed mean RMSE is 6.211 mm with a seed-to-seed standard deviation of
0.283 mm and a 95% interval of 6.023--6.463 mm. This interval stays below the
old ΔP^0.25 seed-42 RMSE of 6.696 mm.

Coverage is conservative on the clean external set. The ΔP^0.5 five-seed mean
has 1sigma coverage of 0.774 and 2sigma coverage of 0.981, above the nominal
Gaussian targets of approximately 0.683 and 0.954. This should be described as
slight over-coverage on the cleaned full diagnostic set, not as under-
calibration.

The GP is the better-calibrated uncertainty baseline on the clean external set:
its 1sigma/2sigma coverage is 0.670/0.950, close to the nominal 0.683/0.954
reference. It does not beat the ΔP^0.5 MLP on accuracy: the GP has +11.1% RMSE,
+13.5% MAE, and +10.2% P95 error relative to the MLP five-seed mean.

## Caveat for Stability Wording

The five-seed aggregate is a ΔP^0.5 pipeline aggregate, but the selected winner
is not the same ablation for every seed. Seeds 42, 17, 99, and 7 use
`anchor_off`, while seed 2024 uses `raw_reliable_no_kd`. In the thesis, this
can be reported as a five-seed external aggregate for the ΔP^0.5 Stage-3
pipeline. If the text needs to claim strict same-ablation seed stability, the
seed-2024 row should either be excluded or replaced by an `anchor_off` seed-2024
run evaluated under the same protocol.

Source files:

- `headline_comparison.md`
- `headline_comparison.csv`
- `new_mlp_5seed_aggregate.csv`
- `per_seed_new_mlp_eval.csv`
- `gp_baseline_per_seed.csv`
- `gp_bootstrap_summary.json`
