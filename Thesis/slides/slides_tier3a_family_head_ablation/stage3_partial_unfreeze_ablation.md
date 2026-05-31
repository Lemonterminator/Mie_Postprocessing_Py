# Stage-3 partial trunk unfreeze ablation

Date: 2026-05-30

## Question

The previous family-head refinement froze the shared trunk completely. This check
tests whether a very small learning rate on the last shared trunk block can recover
some flexibility without reintroducing the trunk-drift failure mode.

## New run

- Run: `MLP/runs_mlp/distill_cdf_fh_LAST1_trunklr3e-5_anchor_0p1_20260530_215841`
- Eval: `MLP/eval/point_eval_20260530_220041_distill_cdf_fh_LAST1_trunklr3e-5_anchor_0p1_20260530_215841`
- Command-level change:
  - `--freeze-trunk`
  - `--unfreeze-trunk-last-blocks 1`
  - `--trunk-learning-rate 3e-5`
  - head learning rate remains `3e-3`
  - `lambda_anchor = 0.1`

## Headline result

| model | cdf RMSE | p50 RMSE | q1 observed RMSE | q1 extrapolated RMSE | cdf ECE | cdf CRPS |
|---|---:|---:|---:|---:|---:|---:|
| old no-head seed42 | 4.217 | 2.794 | 2.696 | 10.800 | 0.0778 | 2.300 |
| family-head frozen trunk | 5.056 | 3.931 | 3.839 | 11.295 | 0.0708 | 2.801 |
| family-head last trunk block unfrozen | 5.459 | 4.208 | 4.304 | 11.310 | 0.0513 | 2.719 |

## Interpretation

Partial unfreezing is a useful diagnostic but not a better candidate. It improves
scalar calibration error and slightly improves cdf CRPS relative to the fully
frozen family-head run, but it worsens the headline point metrics on every eval
set. The model becomes less over-dispersed and flips the cdf bias from positive
to negative, but the residual tail grows and 1-sigma coverage drops.

The result supports the earlier frozen-trunk diagnosis: even a tiny last-block
trunk update can reshape the shared representation enough to hurt the family-head
student. The next architecture ablation should prefer a residual family head
(`mu = mu_shared + delta_family`) or a small family adapter/FiLM block over direct
shared-trunk unfreezing.
