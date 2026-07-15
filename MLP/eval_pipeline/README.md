# eval_pipeline — layered evaluation for the penetration models (thesis Ch5)

End-to-end replacement for the ad-hoc scripts in `MLP/eval/`: given **dataset
roots** and **model checkpoints**, one command computes the full metric
protocol for every model, and a second command renders the entire figure suite
from those results. The two layers are strictly separated so figures can be
re-styled and regenerated in seconds without re-running any model.

```
Layer 1  run_eval.py      (data + weights)  ->  points + metrics artifacts
Layer 2  make_figures.py  (Layer-1 run dir) ->  full publication figure suite
```

## Quick start

```powershell
$py = ".venv\Scripts\python.exe"   # system python lacks torch

# Full thesis roster (13 checkpoints + HA/NS baselines) x {lv2, lv3_qc_gated}:
& $py MLP\eval_pipeline\run_eval.py --manifest MLP\eval_pipeline\manifests\thesis_production.json

# One ad-hoc model on one dataset:
& $py MLP\eval_pipeline\run_eval.py `
    --model "seed42=mlp:MLP/best_models/thesis_baselines/production_mlp_modeA_5seed/seed_42:42" `
    --dataset "lv3_qc_gated=MLP/synthetic_data_clean_lv3_qc_gated"

# Smoke test (first 8 conditions, seconds):
& $py MLP\eval_pipeline\run_eval.py --manifest ... --limit-conditions 8 --tag smoke

# Full roster with optimized cluster-bootstrap CIs:
& $py MLP\eval_pipeline\run_eval.py --manifest ... --bootstrap --bootstrap-workers 8

# Figures (Layer 2, reads only the run dir):
& $py MLP\eval_pipeline\make_figures.py --run MLP\eval_pipeline\runs\evalrun_<ts>_<tag>
```

## Model kinds

| kind            | checkpoint                                             | loader                                             |
|-----------------|--------------------------------------------------------|----------------------------------------------------|
| `mlp`           | run dir with `best_model_refinement.pt` + config/scaler (single / family_head / residual_fh / residual_film) | `efc.load_run_artifacts` |
| `single_svgp`   | `model.pt` (or run dir with `per_seed/seed_N/model.pt`) | `run_gp_baseline.load_gp_artifacts`                |
| `residual_svgp` | run dir with `per_seed/seed_N/model.pt`                 | `residual_multitask_svgp.load_residual_svgp_artifacts` |
| `ha` / `ns`     | fitted-params output dir of the correlation baselines   | `evaluate_ha_ns_fixed_tables.predict_baseline`     |

Dataset roots must be **explicit** (`synthetic_data_clean_lv2` /
`synthetic_data_clean_lv3_qc_gated`); the `MLP/synthetic_data` junction is
refused by design. Eval sets per root: `cdf_uncensored`, `full_clean`,
`p50_observed`, `q1_grid_all` (auto-split into observed-window /
extrapolated slices).

## Metric protocol (all in `metrics.py`, single source of truth)

* **Accuracy**: RMSE, MAE, bias, median/P90/P95/max |err|, NRMSE, relative error
* **Interval calibration**: 1σ/2σ/3σ coverage (legacy parity) + central
  50/80/90/95 % coverage + k·σ coverage curve
* **Distributional scores**: Gaussian NLL (`nll_mm`, physical space, incl.
  log 2π), CRPS (closed form; mean/median/P90/P95), Winkler interval score
  (50 %, 90 %), pinball loss (q10/q50/q90)
* **PIT calibration**: reliability curve + fixed-level ECE (thesis parity),
  PIT-histogram TVD, KS statistic/p-value, mean|z|, z-var, Spiegelhalter Z
* **Sharpness**: mean/median/IQR of predicted σ
* **Weighted variants** for `p50_observed` (by `n_points_in_p50_bin`)
* Optional seeded **cluster bootstrap 95 % CI** for RMSE/MAE/bias
  (`--bootstrap`) — resamples whole trajectories (`traj_key`, falling back to
  `condition_id`), not individual points, since points within one trajectory
  share correlated bias and case-resampling points would understate the CI
  (`--bootstrap-n`, default 2000; `--bootstrap-workers` parallelizes the
  vectorized sufficient-statistics resampling)
* σ-decile calibration audit table per eval set

Not implemented (deliberate): censored scoring rules. `full_clean` scores the
`cdf_points_all.csv` rows as the same pointwise CDF target used by the legacy
full-clean protocol, including rows marked `is_right_censored`; it is a
supplementary stress check, not a censored-likelihood evaluation. The
scaled-space GP NLL variant is also omitted because its values are not
comparable across model kinds.

## Run-directory layout (the Layer-1 ⇄ Layer-2 contract)

```
evalrun_<ts>_<tag>/
  manifest.json                     resolved models/datasets/options
  metrics_wide.csv                  one row per model x dataset x eval_set x slice
  known_number_checks.json          regression guard vs published thesis numbers
  models/<label>/<dataset>/<eval_set>/
    points.parquet|csv              canonical per-point table
                                    (meta + pen_true_mm/pen_pred_mm/pen_std_mm/resid_mm)
    metrics.json                    overall + slices (full protocol)
    per_condition.csv  per_experiment.csv  per_time_bin.csv
    per_condition_time_bin.csv  per_trajectory.csv
    per_censor_status.csv           (when is_right_censored is present)
    reliability_curve.csv  pit_histogram.csv  coverage_curve.csv
    sigma_bin_calibration.csv       (probabilistic sets only)
  figures/                          written by make_figures.py
    models/<label>/<dataset>/<eval_set>/*.png    12/10/7-figure suite per set
    comparison/<dataset>/<eval_set>/*.png        cross-model overlays/bars
  figure_manifest.json
```

Per-model figures: pred-vs-actual parity density (hexbin with log count
colorbar), residual histogram / vs-truth,
worst-conditions RMSE, error-vs-time, condition×time heatmap, PIT reliability,
PIT histogram, k·σ coverage curve, σ-bin audit, best/worst trajectory bands,
observed-vs-extrapolated. Comparison figures: RMSE/MAE/P95 bars, CRPS/NLL/ECE
bars, 1σ/2σ coverage vs nominal, reliability overlay, CRPS–sharpness scatter,
PIT small-multiples, per-seed RMSE (families with ≥3 seeds), by-fold bars
(activated when models carry `meta.holdout`, e.g. LONO rosters).

## Validation

`run_eval.py` re-derived the published numbers exactly on lv3/cdf_uncensored
(known-number checks, auto-run on every full evaluation):

* `residual_film` RMSE **4.4011** mm (published 4.401)
* `qc_retrained_residual_svgp` RMSE **3.8477** mm (published 3.848)

Full 540 k-point evaluation of one checkpoint ≈ 3–4 s on CUDA.

## Extending

* **New checkpoint**: add an entry to a manifest JSON (`label/kind/path/seed/meta`).
* **LONO roster**: one manifest entry per fold winner dir with
  `"meta": {"holdout": "NozzleX"}` — by-fold figures activate automatically.
* **New metric**: add it to `metrics.py` (`point_metrics` or
  `probabilistic_metrics`); it propagates to `metrics.json`, `metrics_wide.csv`
  and the grouped CSVs with no other changes.
* **New figure**: add a function to `figures/per_model.py` or
  `figures/comparison.py` and register it in the corresponding
  `render_*_figures` list. Figures never recompute metrics.

Legacy scripts under `MLP/eval/` are kept untouched for reproducibility of
historical artifacts; new work should use this package.
