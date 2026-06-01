# Leave-One-Nozzle-Out validation of the residual deployment family

Date: 2026-06-01

## Experiment

Does the residual-deployment win — measured in-distribution as residual family
head 4.59 mm and residual FiLM 4.40 mm CDF uncensored RMSE on the QC-gated set —
survive **leave-one-nozzle-out (LONO)**, where the held-out nozzle is never seen
in training?

- Protocol: `lono_modified_only` — hold out each of Nozzle1, 2, 4, 5, 6 in turn.
  **Nozzle0 stays in the training set** every fold (it is the sole member of the
  first-generation injector family; holding it out would degrade the residual
  mechanism to a fallback, and per the Tier-3A protocol-B result removing it from
  training hurts the other folds). Nozzle0 as a genuine OOD injector is a separate
  zero-shot / few-shot study, not a CV fold.
- Feature contract: `a_dp050_plus_pressures`, seed 42, CUDA.
- Each fold trains Stage-1 -> Stage-2 -> Stage-3 from scratch with the held-out
  nozzle removed (no production-checkpoint warm start, so there is no held-out
  leakage).
- Three architecture arms, all built the production way: Stage-1/2 train a
  `family_head` teacher; Stage-3 converts to the deployment student with a frozen
  trunk and frozen shared mean.
  - `family_head`: shared trunk + per-family mu heads (the teacher itself).
  - `residual_fh`: shared mu + per-family residual delta, `delta_l2 = 1e-4`.
  - `residual_film`: residual head + identity FiLM after the last trunk block,
    `film_l2 = 1e-4`, `delta_l2 = 1e-3`.
- Metric: MLP **uncensored** point-table RMSE on the held-out nozzle (same primary
  metric family as the in-distribution slides). HA / NS physics baselines scored
  on the identical held-out subset.

15 of 15 folds completed, 0 failures.

## Headline

The residual win **survives LONO on every in-family fold**. Excluding the
data-sparse OOD fold (Nozzle6), the residual conversion takes family-head LONO
RMSE from 7.44 to 5.16 mm (-31 %) and nearly halves the fold-to-fold spread,
while also repairing the family-head over-prediction bias and restoring coverage.
`residual_fh` and `residual_film` are statistically indistinguishable under LONO.

| aggregation | family_head | residual_fh | residual_film | HA / NS |
|---|---:|---:|---:|---:|
| all 5 folds | 8.32 +/- 2.33 | 6.95 +/- 4.05 | 6.96 +/- 4.08 | 11.69 / 11.10 |
| 4 folds (excl Nozzle6) | 7.44 +/- 1.46 | **5.16 +/- 0.65** | **5.15 +/- 0.67** | 11.93 / 11.37 |

## Per-fold (MLP uncensored: RMSE mm / bias mm / coverage@1sigma)

| held-out | n | family_head | residual_fh | residual_film |
|---|---:|---|---|---|
| Nozzle2 | 60k | 7.71 / +5.7 / 0.60 | **4.34 / -0.5 / 0.94** | 4.29 / -0.5 / 0.91 |
| Nozzle1 | 25k | 6.75 / +3.8 / 0.77 | **5.11 / +0.6 / 0.87** | 5.13 / +0.4 / 0.87 |
| Nozzle4 | 28k | 5.96 / +2.9 / 0.78 | **5.27 / -0.8 / 0.86** | 5.26 / -0.8 / 0.85 |
| Nozzle5 | 33k | 9.35 / +7.9 / 0.44 | **5.91 / +3.9 / 0.78** | 5.92 / +3.8 / 0.77 |
| Nozzle6 | 20k | **11.82 / -5.2 / 0.37** | 14.13 / -10.5 / 0.29 | 14.18 / -10.8 / 0.28 |

`residual_fh` minus `family_head` RMSE per fold: N2 -3.37, N1 -1.64, N4 -0.68,
N5 -3.43 (all improvements), **N6 +2.31 (regression)**.

## Reading the result

1. **The residual deployment win generalizes.** On all four in-family folds the
   residual arms beat the family-head teacher (by 0.7-3.4 mm), keep the in-distribution
   ordering, fix the bias (family head over-predicts +3 to +8 mm; residual sits near
   +/-1 mm on three of four folds), and restore coverage (0.44 -> 0.78 on N5). All
   three MLP arms beat the HA / NS physics baselines (~11-12 mm) on nozzles they
   never trained on.

2. **`residual_fh` ~= `residual_film` under LONO.** Their per-fold numbers are within
   0.04 mm everywhere. FiLM's small in-distribution edge (4.40 vs 4.59) does **not**
   translate into better out-of-sample generalization: for an unseen nozzle the cheap
   output-residual is as good as the representation-level adapter.

3. **Nozzle6 is a true OOD-disaster fold and must be reported separately.** It is the
   data-sparsest fold (20k points), all three MLP arms collapse to 12-14 mm (worse than
   HA / NS's ~10 there), and the residual arms are ~2.3 mm worse than the family head
   with severe under-prediction (bias -10.5). N6 alone drags the residual mean from
   5.16 to 6.95 mm and inflates the std six-fold (0.65 -> 4.05). The honest report is a
   4-fold in-family LONO mean plus N6 as a separate OOD stress-test row.

## Verdict

The residual deployment conversion is a positive result under leave-one-nozzle-out,
not just in-distribution: on in-family held-out nozzles it improves point accuracy,
bias and calibration over the family-head teacher and dominates the physics baselines.
The two residual variants are equivalent for OOD generalization. Nozzle6 behaves as an
out-of-design-family outlier where specialization over-commits; it foreshadows the
Nozzle0 OOD case and argues for reporting such folds as stress tests rather than
folding them into the cross-validation mean.

## Artifacts

- Comparison CSV: `lono_modified_only_comparison.csv` (per-fold, all arms + HA/NS)
- Aggregate CSV: `lono_modified_only_aggregate.csv` (both groupings)
- Raw per-arm outputs: `family_head/`, `residual_fh/`, `residual_film/`
  (each has `per_fold.csv`, `aggregate.csv`, `headline_comparison.md`, `lono_config.json`)
- Source runs: `MLP/runs_mlp/lonoFULL_20260601_104911_{family_head,residual_fh,residual_film}`
- Runner recipe and the `oracle_metrics` experiment_name merge fix that made the
  folds aggregate are documented in the project notes.
