# Best models for cross-dataset evaluation

Curated copy of the **best / comparison model from each model family and ablation**,
gathered here so the upcoming comprehensive evaluation on different datasets can load
them from one place. Only **configs + weights** were copied (`*.pt`, `*.json`); training
logs, CSVs, and plots were left in the source run dirs. Originals remain untouched under
`MLP/runs_mlp/` (this is a copy, not a move).

Copied on 2026-06-08. Integrity verified by SHA-256 against the source checkpoints.

## Layout

```
best_models/
├── thesis_baselines/
│   ├── production_mlp_modeA_5seed/      # 3-stage single-head MLP surrogate (5 seeds)
│   │   ├── bootstrap_summary.json       #   5-seed metrics + per-seed run map
│   │   ├── pipeline_config_resolved.json
│   │   ├── shared_stage1/  shared_stage2/   # shared trunk teachers (best_model_stage{1,2}.pt)
│   │   └── seed_07/ seed_17/ seed_42/ seed_99/ seed_2024/   # anchor-off Stage-3 winners
│   └── svgp_stage3_singleoutput/        # single-output Stage-3 SVGP  (per_seed/seed_42/model.pt)
├── residual_study/                      # evaluated on the PRODUCTION dataset
│   ├── family_head_baseline/            # deployed per-family-head MLP (the residual baseline)
│   ├── residual_fh/                     # residual family head, delta_L2=1e-4
│   ├── residual_film/                   # residual + last-block FiLM, film_L2=1e-4 delta_L2=1e-3
│   └── residual_svgp/                   # additive-residual multi-task SVGP, delta_L2=1e-4
└── qc_gated_retrains/                   # same 3 residual models RETRAINED on the QC-gated lv3 dataset
    ├── residual_fh/   residual_film/   residual_svgp/
```

## Manifest

Headline metrics are **CDF-uncensored RMSE / P50-observed RMSE / CDF-ECE (mm, mm, –)**.
The `thesis_baselines` and `residual_study` rows are all on the **same 542,565-point
uncensored CDF table** (anchored by the single-output SVGP = 4.193 in both the thesis and
this study). The `qc_gated_retrains` rows are on the **QC-gated lv3 table** and are *not*
directly comparable to the production-dataset rows.

| Model dir | CDF | P50 | ECE | Role | Source run (`MLP/runs_mlp/…`) |
|---|---:|---:|---:|---|---|
| `thesis_baselines/production_mlp_modeA_5seed` | 4.265 | 2.848 | – | Thesis MLP surrogate (5-seed mean; full-clean 4.735±0.037) | `full_pipeline_A_20260519_161129` (orchestrator) |
| `thesis_baselines/svgp_stage3_singleoutput` | 4.193 | 2.649 | – | Thesis' strongest single model | `gp_baseline_stage3_20260521_112229` |
| `residual_study/family_head_baseline` | 5.056 | 3.931 | 0.0708 | Deployed per-family-head MLP (residual baseline) | `distill_cdf_family_head_v3_FROZEN_anchor_off_20260529_213346` |
| `residual_study/residual_fh` | 4.631 | 3.395 | 0.0604 | Residual family head winner | `distill_cdf_residual_fh_from_prod_FROZEN_anchor_off_l2_1em04_20260530_225953` |
| `residual_study/residual_film` | 4.510 | 3.166 | 0.0404 | Residual-FiLM winner (best MLP-only) | `distill_cdf_residual_film_from_residual_fh_FROZEN_anchor_off_residual_film_last_block_film_1em04_delta_1em03_20260531_123345` |
| `residual_study/residual_svgp` | 3.987 | 2.077 | 0.0262 | Residual multi-task SVGP winner (**new best**) | `residual_multitask_svgp_prod_existing_full_svgp_l2_1em04_20260531_025634` |
| `qc_gated_retrains/residual_fh` | 4.590 | 3.366 | 0.0523 | QC-gated retrain | `…_residual_fh_…_l2_1em04_20260531_175357` |
| `qc_gated_retrains/residual_film` | 4.400 | 3.065 | 0.0366 | QC-gated retrain | `…_residual_film_…_20260531_175716` |
| `qc_gated_retrains/residual_svgp` | 3.848 | 1.829 | 0.0303 | QC-gated retrain (**overall best**) | `…_residual_multitask_svgp_…_l2_1em04_20260531_180104` |

## Loading

- **MLP models** (`family_head_baseline`, `residual_fh`, `residual_film`, production seeds):
  load `best_model_refinement.pt` (self-contained: trunk + heads), normalise inputs with
  `scaler_state.json`, architecture/contract in `train_config_used.json` + `refine_config.json`.
  Production MLP Stage-1/2 trunks are `shared_stage{1,2}/best_model_stage{1,2}.pt`.
- **SVGP models** (`svgp_stage3_singleoutput`, both `residual_svgp`): load
  `per_seed/seed_42/model.pt`, with `per_seed/seed_42/scaler_state.json`,
  `per_seed/seed_42/seed_config.json`, and run config in `gp_config_resolved.json`.

## Dependencies (note before re-pointing eval scripts)

`residual_svgp/gp_config_resolved.json` carries **absolute paths** to:
- `shared_checkpoint` → `…/gp_baseline_stage3_20260521_112229/per_seed/seed_42/model.pt`
  (the single-output SVGP base it hot-starts from — also bundled here at
  `thesis_baselines/svgp_stage3_singleoutput/per_seed/seed_42/model.pt`),
- `mlp_bootstrap` / `stage1_run` → the production MLP feature pipeline
  (`full_pipeline_A_20260519_161129`, `stage1_engineered_mse_a_plus_pressures_20260519_161130`).

The bundled `residual_svgp` `model.pt` is self-contained for inference; the absolute refs
only matter if the eval re-resolves them from config. The originals are retained in
`MLP/runs_mlp/`, so they resolve as-is.
