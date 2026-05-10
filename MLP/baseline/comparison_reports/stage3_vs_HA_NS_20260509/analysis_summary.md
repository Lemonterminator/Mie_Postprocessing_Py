# Stage-3 MLP vs H-A / N-S Baselines

All rows use the clean full evaluation set: 795,561 points and 33,845 trajectories.

| Model | RMSE mm | MAE mm | P95 mm | Bias mm | 1sigma cov | 2sigma cov | Mean sigma mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hiroyasu-Arai calibrated | 9.974 | 7.686 | 20.493 | 0.625 | 0.536 | 0.886 | 7.478 |
| Naber-Siebers delay | 8.840 | 6.780 | 17.922 | -0.344 | 0.620 | 0.893 | 7.632 |
| Stage-3 MLP anchor_off | 6.696 | 4.875 | 14.118 | -1.149 | 0.745 | 0.979 | 6.505 |

## Relative changes of Stage-3 MLP

- vs Hiroyasu-Arai calibrated: RMSE -32.9%, MAE -36.6%, P95 -31.1%, mean relative error -45.2%.
- vs Naber-Siebers delay: RMSE -24.3%, MAE -28.1%, P95 -21.2%, mean relative error -43.8%.

Generated files: overall_metrics.png, per_nozzle_rmse.png, time_bin_rmse.png, coverage_reliability.png.

## Condition-Level Plots

The report also includes full condition-level comparisons for the same three
methods. Each condition plot treats all points under the same
dataset/test/injection-duration condition as scattered observations, overlays
the measured mean, and shows each model's binned mean prediction with a light
1sigma uncertainty band. The legend reports condition-level 1sigma/2sigma
coverage.

- condition_plot_index.html: browser index for all condition plots.
- condition_plots/: 598 condition-level PNG files.
- pred_vs_actual_side_by_side.png: side-by-side global prediction scatter.
- residual_histogram_side_by_side.png: side-by-side residual histograms.
- per_condition_metrics.csv: condition-level RMSE and coverage table.
