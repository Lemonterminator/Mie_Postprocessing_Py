# Thesis Pipeline Wrappers

The `pipelines/` package provides reproducible entry points for the staged
Masters-thesis workflow. Each phase writes timestamped archives plus metadata
and manifest files, so downstream phases can record exactly which upstream run
they consumed.

## Archive Layout

- `MLP/synthetic_data_runs/{fit_run_id}/`: Phase 1 curve-fit outputs.
- `MLP/synthetic_data_runs/latest.txt`: pointer to the latest production fit.
- `MLP/audit_runs/{audit_run_id}/`: Phase 2 audit outputs.
- `MLP/audit_runs/latest.txt`: pointer to the latest audit.
- `MLP/MLP_training/*/runs/*/{run_id}/`: Phase 3 training/evaluation runs.
- `Thesis/generated/{report_run_id}/`: Phase 4 assembled thesis artifacts.
- `Thesis/generated/current/`: stable mirror used by LaTeX `\input{}` and
  `\includegraphics{}` calls.

The raw Mie scattering export is expected at `Mie_scattering_top_view_results/`
unless `--input-root` or `--raw-root` is supplied. Audit code intentionally
reuses the existing raw traversal from
`MLP.curve_fit.audit_cdf_spatial_censoring.collect_raw_inventory`, so the
expected folder and CSV structure stays aligned with the earlier implementation.

## Full Chain

Run all phases:

```bash
python pipelines/run_full.py --input-root Mie_scattering_top_view_results
```

Useful options:

```bash
python pipelines/run_full.py \
  --input-root Mie_scattering_top_view_results \
  --allow-synthetic-population \
  --train-mode single \
  --device auto \
  --promote
```

Skip phases when reusing existing archives:

```bash
python pipelines/run_full.py --skip-phase fit --skip-phase audit
python pipelines/run_full.py --skip-fit --skip-train --skip-alpha
```

Use `--dry-run` to print the chained commands without executing them. This is
the safest check on machines that do not have `Mie_scattering_top_view_results/`.

## Phase 1: Fit

```bash
python pipelines/fit/run_fit_pipeline.py \
  --input-root Mie_scattering_top_view_results \
  --n-workers 0
```

For a small smoke test on a data machine:

```bash
python pipelines/fit/run_fit_pipeline.py --nozzle-filter HS_01 --n-workers 1
```

## Phase 2: Audits

```bash
python pipelines/audit/run_audit_pipeline.py --fit-run-dir MLP/synthetic_data_runs/<fit_run_id>
```

Generated audit groups:

- B.1 `scr_naive_hold_gap.py`: spatial-censoring and naive-hold gap summary.
- D.2 `data_attrition_report.py`: raw-to-fit-to-train attrition cascade.
- B.5 `raw_coverage_heatmap.py`: CDF/regime raw coverage heatmap.
- E.1 `scr_ood_cross_audit.py`: spatial censoring crossed with OOD support.

When raw data is absent but a fit archive exists, `--allow-synthetic-population`
allows D.2 to run a structural fallback from fitted trajectories. Use that only
for pipeline wiring checks, not final thesis counts.

## Phase 3: Training

```bash
python pipelines/train/run_train_pipeline.py \
  --fit-run-dir MLP/synthetic_data_runs/<fit_run_id> \
  --audit-run-dir MLP/audit_runs/<audit_run_id> \
  --mode single
```

The wrapper forwards parent run pointers into the training scripts. Each major
training run writes `_metadata.json`, and completed stage runs write
`_thesis_metrics.json` for Phase 4 assembly.

## Phase 4: Report Assembly

```bash
python pipelines/report/run_report_pipeline.py \
  --fit-run-dir MLP/synthetic_data_runs/<fit_run_id> \
  --audit-run-dir MLP/audit_runs/<audit_run_id> \
  --promote
```

This phase collects manifest-tagged artifacts, copies them into
`Thesis/generated/{report_run_id}/`, mirrors them to `Thesis/generated/current/`,
optionally runs the E.2 alpha sensitivity sweep, and can promote canonical
figures to `Thesis/images/`.

The LaTeX sections in `Thesis/latex/sections_en/04_trajectory_surrogate_screening.tex`
and `Thesis/latex/sections_zh/04_trajectory_surrogate_screening.tex` read from
`Thesis/generated/current/` when the generated snippets and figures exist, with
fallback text/macros when they do not.

