# Naber--Siebers Baseline

This folder implements a fair empirical penetration-correlation baseline for
the Mie top-view CDF penetration series.

The baseline reads the same cleaned wide-series exports used by the Stage-3 MLP:

```text
MLP/synthetic_data/*/cdf/series_wide_clean/*.csv
```

Those files are generated from:

```text
Mie_scattering_top_view_results
```

by `MLP/curve_fit/fit_raw_data.py`. The baseline therefore compares models on
the same CDF penetration definition and the same delay-aligned cleaning layer.

## Models

The default model is:

```text
S(t) = K * A * sqrt(max(t - td, 0))
A = ((Pinj - Pamb) / rho_a) ** 0.25 * sqrt(dn)
```

Variants:

- `ns_no_angle`: global `K`, no fitted delay.
- `ns_delay`: global `K` plus fitted delay `td`.
- `ns_train_angle`: fitted delay plus a train-only nozzle-level cone-angle factor.
- `ns_oracle_angle`: uses each trajectory's measured cone angle; diagnostic only.

The oracle-angle variant sees information extracted from the evaluated video, so
it should not be used as the main fair comparison against the neural surrogate.

## Uncertainty

The deterministic correlation is converted into a probabilistic baseline by
adding:

- residual uncertainty calibrated on the validation split,
- optional time-binned residual scale,
- optional grouped bootstrap parameter uncertainty.

The final exported standard deviation is:

```text
sigma_total(t)^2 = sigma_residual(t)^2 + sigma_parameter(t)^2
```

## Example

Smoke test:

```bash
.venv/Scripts/python.exe MLP/baseline/Naber_Siebers/run_baseline.py \
  --variant ns_delay \
  --split-mode grouped_condition \
  --bootstrap-n 8 \
  --max-trajectories 600 \
  --tag smoke
```

Full grouped-condition run:

```bash
.venv/Scripts/python.exe MLP/baseline/Naber_Siebers/run_baseline.py \
  --variant ns_delay \
  --split-mode grouped_condition \
  --bootstrap-n 64
```

Full-clean diagnostic, useful only for comparing against already exported
whole-clean-set figures:

```bash
.venv/Scripts/python.exe MLP/baseline/Naber_Siebers/run_baseline.py \
  --variant ns_delay \
  --split-mode grouped_condition \
  --primary-split all \
  --bootstrap-n 64 \
  --tag all_clean_diagnostic
```

Leave-one-nozzle example:

```bash
.venv/Scripts/python.exe MLP/baseline/Naber_Siebers/run_baseline.py \
  --variant ns_delay \
  --split-mode leave_one_nozzle \
  --holdout-nozzle nozzle3 \
  --bootstrap-n 64
```

Regenerate the visual summary for the latest run:

```bash
.venv/Scripts/python.exe MLP/baseline/Naber_Siebers/visualize_results.py
```

For a specific run:

```bash
.venv/Scripts/python.exe MLP/baseline/Naber_Siebers/visualize_results.py \
  --baseline-run-dir MLP/baseline/Naber_Siebers/outputs/<run_name>
```

Key outputs:

- `points.csv`: pointwise truth, prediction, residual, and uncertainty.
- `per_trajectory.csv`: trajectory-level RMSE/MAE/coverage.
- `per_folder.csv`, `per_nozzle.csv`: grouped summaries.
- `time_bins.csv`: time-binned error and coverage.
- `metrics_summary.json`: thesis-table-ready metrics.
- `residual_uncertainty_by_time.csv`: calibrated residual sigma.
- `bootstrap_params.csv`: fitted parameters from bootstrap resamples.
- `visual_summary/*.png`: dashboard, uncertainty decomposition, and optional
  diagnostic comparison against the exported Stage-3 MLP evaluation.
