# Production residual-family-head warm-start summary

Date: 2026-05-30

## Experiment

- Baseline: `MLP/runs_mlp/distill_cdf_family_head_v3_FROZEN_anchor_off_20260529_213346`
- Baseline eval: `MLP/eval/point_eval_20260530_131659_latest_full_eval/per_run_metrics.csv`
- New model: `residual_family_head`
- Feature contract: `a_dp050_plus_pressures`
- Warm start: copy trunk and log-var head; copy old `mu_heads.1` to `mu_shared_head`; zero-init per-family `delta_heads`; run 10 mimic epochs; freeze trunk and shared mu during refinement.
- Sweep: `residual_delta_l2_weight in {0, 1e-4, 1e-3, 1e-2}`

## Headline

All residual runs beat the current production model on the primary CDF uncensored RMSE. The selected winner is `delta_l2 = 1e-4` by primary RMSE, with no calibration or P50 regression penalty.

| model | delta L2 | CDF RMSE | CDF MAE | CDF bias | CDF ECE | CDF CRPS | P50 RMSE | Q1 obs RMSE | Q1 extrap RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current production | - | 5.056 | 3.613 | +0.343 | 0.0708 | 2.801 | 3.931 | 3.839 | 11.295 |
| residual winner | 1e-4 | 4.631 | 3.188 | +0.180 | 0.0604 | 2.382 | 3.395 | 3.242 | 11.307 |

## Sweep

| delta L2 | CDF RMSE | CDF ECE | CDF CRPS | P50 RMSE | Q1 obs RMSE | Q1 extrap RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.668 | 0.0550 | 2.372 | 3.441 | 3.275 | 11.284 |
| 1e-4 | 4.631 | 0.0604 | 2.382 | 3.395 | 3.242 | 11.307 |
| 1e-3 | 4.653 | 0.0533 | 2.361 | 3.418 | 3.233 | 11.275 |
| 1e-2 | 4.685 | 0.0491 | 2.368 | 3.449 | 3.277 | 11.248 |

## Training log read

- All 4 runs completed 10 mimic epochs and finished successfully.
- Refinement early-stopped before 150 epochs:
  - `delta_l2=0`: epoch 104
  - `delta_l2=1e-4`: epoch 44
  - `delta_l2=1e-3`: epoch 110
  - `delta_l2=1e-2`: epoch 84
- Trainable parameter policy:
  - trunk frozen: 334,720 params fixed
  - shared mu frozen: 16,641 params fixed
  - trainable: 49,923 params (`delta_heads` + `log_var_head`)

## Verdict

The production residual-head conversion is a positive result. It improves point accuracy and calibration on the primary uncensored CDF table and substantially improves P50 and observed-window Q1. The extrapolated Q1 region is essentially flat: the winner is +0.012 mm worse than current production, while stronger shrinkage improves extrapolated Q1 at a small cost to CDF RMSE.

## Addendum: Residual SVGP Follow-Up

Date: 2026-05-31

The same deployment question was then revisited with an additive residual multi-task SVGP. The shared SVGP was hot-started from the deployed Stage-3 production checkpoint, family id stayed outside the kernel feature vector, and the per-family residuals were routed separately with a zero-residual fallback for unseen families.

Headline comparison:

| model | delta L2 | CDF RMSE | CDF ECE | CDF CRPS | P50 RMSE | Q1 obs RMSE | Q1 extrap RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| latest MLP residual family head | 1e-2 | 4.685 | 0.0491 | 2.368 | 3.449 | 3.277 | 11.248 |
| residual multitask SVGP winner | 1e-4 | 3.987 | 0.0262 | 2.040 | 2.077 | 2.134 | 10.429 |

Compared to the original single-output SVGP baseline, the residual multitask winner also improves the primary CDF uncensored RMSE from 4.193 mm to 3.987 mm.

Artifacts:

- Production sweep: `MLP/runs_mlp/residual_multitask_svgp_production_20260531_025632`
- Point-table eval: `MLP/eval/point_eval_20260531_031534_residual_svgp_prod_l2_1em04`
- Comparison CSV: `Thesis/slides/slides_residual_family_head_production/residual_svgp_vs_latest_mlp.csv`

## Next MLP-only Architecture Ablation

The next high-value MLP-only experiment should move the family residual from the output head into the representation:

`family_residual_film_adapter`

Formula:

```text
h_l' = gamma_{g,l} * h_l + beta_{g,l}
mu(x, g) = mu_shared(h_L') + delta_g(h_L')
```

The adapter is identity-initialized (`gamma=1`, `beta=0`) and the residual head is zero-initialized, so the model can be hot-started from the current production residual family head without changing the initial function. Unknown families use the identity adapter and zero residual, which preserves the same shared-model fallback property as the residual family head.

Why this is the right next ablation:

- It is genuinely MLP-specific: SVGP can route additive residual functions, but it does not have hidden representations that can be modulated layer by layer.
- It tests whether the nozzle-family difference is only an output offset or whether it changes how `a_dp050_plus_pressures` should be represented before the mean head.
- It keeps the deployed engineering story clean: hot-start existing production, freeze the trunk, train only tiny family adapters plus the delta/log-var heads, and add new families by identity-initializing a new adapter.

Suggested matrix:

| priority | mode | adapter placement | trainable during refine | regularization | reason |
|---:|---|---|---|---|---|
| 1 | `residual_film_last_block` | after final trunk block | FiLM adapter + delta heads + log-var | `film_l2 in {1e-4,1e-3,1e-2}` | smallest change; checks if family difference is late representation calibration |
| 2 | `residual_film_all_blocks` | after every trunk block | FiLM adapters + delta heads + log-var | same | lets family condition pressure interactions throughout the MLP |
| 3 | `low_rank_residual_adapter` | after final/all trunk blocks | rank-4 or rank-8 bottleneck adapters + delta heads + log-var | adapter weight decay + output shrinkage | more expressive than FiLM while still few-shot/new-family friendly |
| 4 | `family_logvar_residual` | output uncertainty only | shared mu frozen; family log-var delta | log-var delta shrinkage | targets calibration without spending mean capacity |

Recommended first run: `residual_film_last_block`, hot-started from the latest residual family-head production winner, with the trunk and shared mean frozen. If it beats the latest MLP CDF RMSE of `4.6851 mm` while keeping ECE near `0.0491`, expand to all-block FiLM.

## Addendum: Residual-FiLM MLP Production Sweep

Date: 2026-05-31

The recommended MLP-only ablation was run on the production dataset, without LONO. The source model was the latest residual family-head production checkpoint:

`MLP/runs_mlp/distill_cdf_residual_fh_from_prod_FROZEN_anchor_off_l2_1em02_20260530_230928`

The model inserted an identity-initialized FiLM adapter after the final trunk block and was refined with the trunk and shared mean frozen. Full sweep:

- `film_adapter_l2_weight in {1e-4, 1e-3, 1e-2}`
- `residual_delta_l2_weight in {1e-4, 1e-3, 1e-2}`

Headline:

| model | film L2 | delta L2 | CDF RMSE | CDF ECE | CDF CRPS | P50 RMSE | Q1 obs RMSE | Q1 extrap RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source residual FH | - | 1e-2 | 4.685 | 0.0491 | 2.368 | 3.449 | 3.277 | 11.248 |
| residual-FiLM winner | 1e-4 | 1e-3 | 4.510 | 0.0404 | 2.242 | 3.166 | 3.110 | 11.265 |

Winner rule outcome:

- Primary CDF RMSE improves by `0.1755 mm`.
- P50 RMSE improves by `0.2833 mm`.
- CDF calibration improves: ECE `0.0491 -> 0.0404`, CRPS `2.368 -> 2.242`.
- Q1 observed-window RMSE improves by `0.1673 mm`.
- Q1 extrapolated is effectively flat: `11.248 -> 11.265 mm`.

Sweep reading:

| film L2 | delta L2 | CDF RMSE | CDF ECE | P50 RMSE | Q1 obs RMSE | Q1 extrap RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 1e-4 | 4.562 | 0.0367 | 3.247 | 3.164 | 11.274 |
| 1e-4 | 1e-3 | 4.510 | 0.0404 | 3.166 | 3.110 | 11.265 |
| 1e-4 | 1e-2 | 4.554 | 0.0422 | 3.216 | 3.121 | 11.237 |
| 1e-3 | 1e-4 | 4.523 | 0.0410 | 3.183 | 3.132 | 11.238 |
| 1e-3 | 1e-3 | 4.574 | 0.0386 | 3.237 | 3.161 | 11.272 |
| 1e-3 | 1e-2 | 4.547 | 0.0426 | 3.209 | 3.153 | 11.243 |
| 1e-2 | 1e-4 | 4.525 | 0.0415 | 3.184 | 3.114 | 11.249 |
| 1e-2 | 1e-3 | 4.523 | 0.0427 | 3.174 | 3.121 | 11.216 |
| 1e-2 | 1e-2 | 4.523 | 0.0410 | 3.176 | 3.099 | 11.245 |

Verdict:

The MLP-only representation-level family adapter is a positive result. It shows that the family signal is useful not only as an output residual but also as a hidden representation modulation. The best setting uses a lightly-regularized FiLM adapter and medium residual-delta shrinkage. Stronger FiLM shrinkage is competitive but does not beat the winner on CDF RMSE.

Artifacts:

- Production sweep: `MLP/runs_mlp/residual_film_family_head_production_20260531_123024`
- Winner run: `MLP/runs_mlp/distill_cdf_residual_film_from_residual_fh_FROZEN_anchor_off_residual_film_last_block_film_1em04_delta_1em03_20260531_123345`
- Winner eval: `MLP/eval/point_eval_20260531_123655_residual_film_film_1em04_delta_1em03`
- Slides CSV: `Thesis/slides/slides_residual_family_head_production/residual_film_family_head_eval_summary.csv`
