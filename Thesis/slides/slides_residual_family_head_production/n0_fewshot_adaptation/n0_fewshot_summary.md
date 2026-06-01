# Nozzle0 few-shot adaptation: the limit of residual deployment

Date: 2026-06-01

## Question

The LONO study showed Nozzle6 (a data-sparse out-of-design-family fold) breaks the
residual head. Nozzle0 is the other true OOD case: it is the *only* first-generation
injector (BC2022), the sole member of family 0, sharing identical orifice geometry
with the modified family but with an empirically distinct penetration signature.

Deployment question: once a new injector arrives, **how many of its conditions must
we collect before the frozen-trunk per-family residual delta head recovers in-family
accuracy?** This is the adaptation story only the residual / adapter design can tell.

## Setup

- Starting model: a residual-family-head student whose Stage-1/2/3 trunk was trained
  with **Nozzle0 held out** (`lono_5fold`, n-folds 1). The trunk has never seen N0, so
  `k=0` is genuine zero-shot OOD.
- Adaptation: freeze the entire model except `delta_heads[0]` (the N0 residual head);
  fine-tune only that head on `k` randomly-drawn N0 conditions; evaluate uncensored
  RMSE on a fixed held-out N0 test split (135 of 338 conditions, 120 292 points).
- `k in {0,1,2,5,10,20,50,all(203)}`, 5 random condition draws per k (1 for k=0/all).
- Two loss modes for the delta fine-tune:
  - `nll`: mean error weighted by predicted precision — matches production residual
    training. For an OOD nozzle the **frozen log-var head emits a hugely inflated
    sigma** (mean predicted std ~1443 mm at k=0), which throttles the mean gradient.
  - `mse`: adapt the mean directly in A-scaled space, immune to the broken frozen
    sigma — isolates what the delta head alone can do to the mean.

## Result (uncensored RMSE, mm; in-family LONO band ~5 mm for reference)

| k | NLL rmse | NLL bias | MSE rmse | MSE bias |
|---:|---:|---:|---:|---:|
| 0 (zero-shot) | 33.16 | -27.4 | 33.16 | -27.4 |
| 1 | 24.05 +/- 3.37 | -9.1 | 23.31 +/- 4.36 | -7.2 |
| 2 | 22.01 +/- 2.19 | -6.9 | 19.88 +/- 2.98 | -1.7 |
| 5 | 21.40 +/- 1.91 | -3.7 | 20.09 +/- 0.56 | +0.7 |
| 10 | 20.68 +/- 1.39 | -6.5 | 17.85 +/- 1.18 | +0.8 |
| 20 | 19.17 +/- 1.56 | -6.2 | 15.81 +/- 0.33 | +2.2 |
| 50 | 19.55 +/- 1.22 | -5.8 | 16.18 +/- 0.62 | +1.6 |
| all (203) | 19.54 | -6.6 | 15.73 | +1.5 |

## Two findings

1. **The frozen variance head throttles NLL adaptation.** Because the log-var head is
   frozen and badly mis-scaled on OOD N0 features, NLL down-weights the mean error and
   plateaus at ~19.5 mm. Adapting the mean directly (MSE) lowers the floor to ~15.7 mm
   (-4 mm) and removes the bias by k=2. So for few-shot adaptation of a new injector,
   the **variance head must be adapted too, or the mean fit must bypass it.**

2. **Even with the mean fit unthrottled, delta-only adaptation plateaus far above the
   in-family band.** MSE RMSE bottoms out at ~15.7 mm with all 203 conditions — vs the
   ~5 mm the same architecture reaches on in-family LONO folds. Bias is fixed (a delta
   head trivially removes the global offset, -27 -> ~0 by k=2), but the **residual
   spread cannot be closed**: the frozen trunk lacks the representation N0's
   first-generation geometry needs, and a per-family output offset cannot synthesize it.

## Verdict

This draws the exact boundary of the residual deployment story. Cheap per-family
delta adaptation is sufficient when the new nozzle is in-design-family (the N1/2/4/5
LONO folds recover to ~5 mm); it is **not** sufficient for a genuinely
out-of-design-family injector (N0, like the N6 LONO disaster), where it removes the
bias but leaves a large irreducible spread. Such an injector needs trunk-level
capacity (unfreeze / FiLM-all-blocks / retrain), not just a new output head.

This is the constructive counterpart to the N6 LONO result and the direct answer to
the "should we leave Nozzle0 out" question: N0 is out-of-design-family, not merely an
unseen instance — confirmed independently by holding it out (33 mm zero-shot) and by
failing to recover it with delta-only few-shot (~16 mm floor).

## Follow-ups this motivates

- Adapt the log-var head alongside delta (fixes finding 1; cheap).
- Few-shot adapt the FiLM-all-blocks variant or unfreeze the last trunk block, to test
  whether trunk-level capacity closes the residual spread (finding 2).

## Artifacts

- `fewshot_curve_nll.csv`, `fewshot_curve_mse.csv` — RMSE-vs-k mean/std per loss mode
- `fewshot_per_run_nll.csv`, `fewshot_per_run_mse.csv` — every (k, repeat) point
- `config_nll.json`, `config_mse.json` — full protocol
- Driver: `MLP/MLP_training/ood_lono/run_n0_fewshot_adaptation.py`
- Starting model: `MLP/runs_mlp/lono_residual_fh_anchor_off_20260601_145626`
  (trunk trained with N0 held out)
