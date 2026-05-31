# OOD Validation Plan — Leave-One-Nozzle-Out (LONO)

**Purpose**: rebut the #1 reviewer ask for top ML/scientific venues — "does this
generalize beyond the training data family?" Currently all splits are random
within the same nozzle/fuel family. LONO removes one entire nozzle from
training and tests on it; a model that beats HA/NS in this protocol has a
real generalization claim.

This brief is self-contained — a fresh agent / Codex run can pick it up cold.

---

## 1. Context (what just happened upstream)

The Stage-1→2→3 MLP pipeline with new ΔP^0.5 A_scale beats HA / NS on the
in-distribution clean diagnostic eval (n=795,561 points, 33,845 trajectories):

- 5-seed mean: **rmse 6.211 ± 0.283 mm, mae 4.587 ± 0.281 mm**
- HA calibrated: rmse 9.974 / mae 7.686
- NS delay:      rmse 8.840 / mae 6.780
- All splits (val/test) are random subsets of the same nozzle families.

LONO closes the generalization gap. The A_scale formula and pipeline are
fixed (committed) — only the **split assignment** changes.

A_scale formula already in place:
`A_scale = ΔP^0.5 · ρ_a^-0.25 · √d`
(`MLP/MLP_training/engineered_feature_common.py` lines 577, 1588, 1752)

The orchestrator that runs Stage-1→2→3 + ablation suite + winner selection +
5-seed bootstrap exists already:
`MLP/MLP_training/run_full_pipeline.py` (mode C = full reseeded per seed).

---

## 2. Goal

Run the same Stage-1→2→3+ablation pipeline, but with **leave-one-nozzle-out**
splits instead of random splits. Aggregate metrics across folds. Compare
LONO MLP vs HA/NS *on the held-out nozzle data only*, fold by fold.

Headline output: a CSV similar to `headline_comparison.csv` with per-fold
metrics + an aggregate row that lets the paper say:

> "Even when the test nozzle is held out from training entirely, the MLP
> retains an X% RMSE advantage over HA/NS (5-fold mean ± std)."

If MLP loses some folds: equally publishable — "we identify [nozzle Y] as
the OOD failure case; degradation is bounded by Z mm".

---

## 3. Step-by-step spec

### 3.1 Identify nozzle families

Load `MLP/runs_mlp/stage1_engineered_mse_a_only_20260509_110102/row_table.csv`
(any recent seed Stage-1 run dir). Group by `experiment_name` (or by
`dataset_key` if more granular). Count rows / `sample_group_id` per group.

Drop any nozzle with < 200 sample_group_ids (too small for held-out test).
Output a one-line printout: number of folds, rows per fold.

Expected: 3–10 nozzle families. If only 2–3, switch to leave-one-injection-
pressure-bin-out (bin `injection_pressure_bar` into terciles) — same scaffold.

### 3.2 New split assignment function

Add to `MLP/MLP_training/engineered_feature_common.py` (next to
`assign_splits_by_group`, around line 582):

```python
def assign_splits_leave_one_out(
    df_in: pd.DataFrame,
    *,
    holdout_value: str,
    holdout_column: str = "experiment_name",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """LONO split: one experiment_name → entire test split.

    Within the remaining trajectories, carve out val_ratio for validation
    by `sample_group_id` (so a trajectory's curves stay together). Train
    gets the rest.
    """
    df = df_in.copy()
    is_holdout = df[holdout_column].astype(str) == str(holdout_value)
    df_holdout = df.loc[is_holdout].copy()
    df_remaining = df.loc[~is_holdout].copy()
    if df_holdout.empty:
        raise ValueError(f"No rows match {holdout_column}={holdout_value!r}.")

    # Re-use sample_group_id-based splitting on the remaining rows for val.
    rng = np.random.default_rng(int(seed))
    groups = np.asarray(pd.Series(df_remaining["sample_group_id"]).drop_duplicates().tolist(), dtype=object)
    rng.shuffle(groups)
    n_val = int(np.floor(val_ratio * len(groups)))
    val_groups = set(groups[:n_val].tolist())

    df_remaining["sample_split"] = df_remaining["sample_group_id"].astype(str).map(
        lambda g: "val" if g in val_groups else "train"
    )
    df_holdout["sample_split"] = "test"
    return pd.concat([df_remaining, df_holdout], ignore_index=True)
```

### 3.3 LONO orchestrator

Create `MLP/MLP_training/run_lono_pipeline.py`. Mirror the structure of
`run_full_pipeline.py`. For each fold:

1. Build the canonical feature table once (shared across folds — same data,
   different split labels).
2. Re-assign splits with `assign_splits_leave_one_out(df, holdout_value=fold_name)`.
3. Train Stage-1, Stage-2, run Stage-3 ablation suite.
4. Pick winner from suite (same selection as `run_full_pipeline`).
5. Save `fold_summary.json` with the held-out nozzle's metrics.

To avoid duplicating training code: invoke the existing scripts via
`subprocess`, but pass the override row table as a CSV path. **The cleanest
hack**: write a one-shot helper that monkey-patches
`assign_splits_by_group` inside the train_stage1_mse / train_stage2_nll
process, OR (cleaner) add a `--row-table-override` CLI arg to those scripts
and a `--split-strategy` arg with values `random` (default) and `lono`.

Recommended: add CLI args. Smallest diff:

```python
# train_stage1_mse.py near parse_args:
parser.add_argument("--lono-holdout", type=str, default=None,
                    help="If set, hold out experiment_name=<value> as test; "
                         "use leave-one-nozzle-out split.")

# in main(), after build_all_stage_tables(...):
if args.lono_holdout is not None:
    representative_df = assign_splits_leave_one_out(
        stage_tables.representative,
        holdout_value=args.lono_holdout,
        val_ratio=float(config["val_ratio"]),
        seed=int(config["seed"]),
    )
else:
    representative_df = assign_splits_by_group(...)
```

Same change in `train_stage2_nll.py` and `run_stage3_ablation_suite.py`. The
`run_lono_pipeline.py` then iterates folds and passes `--lono-holdout`.

### 3.4 Computational scope and seed strategy

Per fold ≈ 13 min (one mode-C seed = Stage-1 ~1 min + Stage-2 ~6 min + Stage-3
suite ~6 min on RTX 5090).

**Main OOD protocol: 5 folds × 1 seed × pre-fixed ablation winner.**

Two parameters MUST be fixed *before* the LONO sweep — picking them
post-hoc from in-distribution test results creates hindsight bias and
invalidates the OOD claim:

- **seed = 42**: not the in-distribution best, just the project default.
- **ablation = `anchor_off`**: this was the 4/5 majority-vote winner in the
  in-distribution 5-seed mode-C run
  (`MLP/runs_mlp/full_pipeline_C_20260509_110100/bootstrap_summary.json`).
  Use it directly — do **not** re-run the full Stage-3 ablation suite per fold.
  Skip the suite; train only the `anchor_off` Stage-3 variant per fold.

This costs ~13 min/fold × 5 folds = **~65 min total**.

LONO's primary variance source is *fold-to-fold* (which nozzle is held out),
not seed-to-seed (training stochasticity). 5 folds × 1 seed gives n=5
independent OOD measurements — statistically equivalent to the
in-distribution 5-seed × 1-fold protocol.

**Sanity check (~1 h, run after main sweep)**: pick the fold whose RMSE is
closest to the across-fold median. Re-run that fold with 5 seeds. Compare:

- σ_seed (std across seeds within that fold) vs
- σ_fold (std across folds in the 5×1 main sweep).

If σ_seed ≪ σ_fold (rule of thumb: σ_seed < 0.3 · σ_fold), publish the 5×1
LONO result with a footnote citing the sanity check. If σ_seed ≈ σ_fold,
upgrade to 5 folds × 5 seeds (~5.5 h) before publication.

**Supplementary worst-case stress test (~15 min, optional)**: after the main
sweep finishes, identify the fold where MLP's *in-distribution* test RMSE on
that nozzle's rows was highest (a per-nozzle breakdown of the in-distribution
result — read from `points.csv` in the prior eval dirs and group by
`experiment_name`). Re-run *only* that fold with the in-distribution best
seed (whichever of {42,17,99,7,2024} had the lowest in-dist RMSE — likely
seed 99 from the recent run). Report this as a "stress test" supplementary
table only:

> "As an additional robustness check, we evaluate the best-performing in-
> distribution seed (99) on its in-distribution-hardest nozzle held out
> (Supplementary Table SX). Even under this adversarial selection, the MLP
> retains an X% RMSE advantage over HA/NS."

This is **not** the OOD main result — it has the hindsight-bias issue we
deliberately avoided in the main sweep. It exists only to give reviewers an
additional confidence signal: "even when we game the setup against ourselves
on a single fold, MLP still wins."

**Budget summary**:

| Stage | Compute | Output |
|---|---|---|
| Main: 5 folds × seed=42 × anchor_off | ~65 min | 5-fold OOD mean ± std |
| Sanity: median fold × 5 seeds | ~65 min | σ_seed vs σ_fold check |
| Stress test (optional): 1 fold × in-dist best seed | ~15 min | supplementary worst-case |
| **Total typical** | **~2.2 h** | full OOD evidence package |
| Worst case (5×5) | ~5.7 h | only if sanity check fails |

**Paper phrasing (main text, after sanity check passes)**:

> "OOD generalization is measured across 5 leave-one-nozzle-out folds with
> seed=42 and the majority-vote Stage-3 ablation variant (anchor_off, 4/5
> seeds in the in-distribution sweep). Both choices are fixed before the
> LONO sweep to prevent hindsight bias. A within-fold seed-replication
> study (Table SX) shows seed-induced RMSE std of σ_seed mm, an order of
> magnitude smaller than the fold-to-fold std of σ_fold mm; we therefore
> report fold-level statistics without seed averaging. Mean RMSE across
> folds = X mm (std Y), worst-fold RMSE = Z mm; even on the worst
> held-out nozzle, the MLP outperforms HA/NS by W%."

### 3.5 Per-fold evaluation

For each fold, after winner selection:

1. **In-pipeline**: winner already evaluates on its `test` split (the
   held-out nozzle). The metrics dump that's already produced
   (`winner_metrics.overall` from suite summary) is what we want.

2. **External n=795k protocol** for the held-out nozzle only: filter
   `cdf_wide_df` to rows where `experiment_name == fold_name`, then call
   `run_rmse_evaluation` (`MLP/eval/inference_rmse_on_series.py`) with that
   subset. *Add a `--filter-experiment` arg to that script*, or do the
   filtering in a wrapper.

3. **HA / NS on the same held-out subset**: pull from
   `MLP/baseline/Hiroyasu_Arai/outputs/.../per_trajectory.csv` and
   `MLP/baseline/Naber_Siebers/outputs/.../per_trajectory.csv` and filter by
   `experiment_name`. Compute the same metrics with the same
   `_finite_metrics` helper from `inference_rmse_on_series.py:48`. The CSVs
   should already exist for the prior baseline run; if they don't, re-run
   HA/NS scripts (in `MLP/baseline/Hiroyasu_Arai/` and `MLP/baseline/Naber_Siebers/`).

### 3.6 Aggregation

Write `MLP/baseline/comparison_reports/lono_<date>/`:

- `per_fold.csv`: rows = (fold_name, model ∈ {MLP_dp05, HA, NS}, metrics).
- `aggregate.csv`: 5-fold mean ± std + 95% bootstrap CI per (model, metric).
  Re-use `bootstrap_ci` from `run_full_pipeline.py:354`.
- `headline_comparison.md`: side-by-side mean ± std table for paper.
- `lono_per_fold_rmse.png`: one bar per fold, three colors for MLP/HA/NS;
  visual confirmation that MLP wins folds individually, not just on average.

### 3.7 Pitfalls

1. **Within-fold val leak**: validation must come from the *training* nozzles,
   not the held-out one. The split function above does this correctly; verify.
2. **Pre-train collapse check** (`run_pretrain_collapse_check`) is keyed off
   `representative_df`. After LONO split, the collapse diagnostic should still
   pass, but with different numbers per fold. **Allow it to fail with
   `--allow-failed-precheck`** for OOD folds where the geometry is too
   different — this is expected and not a bug.
3. **Inference scaler mismatch**: the z-score `scaler_state` is fit on the
   training rows of each fold. The held-out nozzle's features are z-scored
   *using the training-fold statistics*, not its own — this is the correct
   OOD protocol. Verify `apply_zscore` does this (it should, since it takes a
   pre-computed `zscore_params` dict).
4. **Smaller training set per fold**: if N nozzles are roughly equal-sized,
   each fold trains on ~80% of full data. Early stopping patience may need
   to drop slightly (try `--early-stopping-patience 30` if 40 is too lax).

### 3.8 Outputs the parent conversation needs back

Single CSV with rows: (model, fold, rmse, mae, bias, p95, cov_1σ, cov_2σ).
One aggregate row per model with 5-fold mean ± std and 95% CI.

Plus a one-paragraph verdict:
- MLP wins all folds vs HA/NS → strong generalization claim.
- MLP wins on average but loses one fold → identify that fold, write it as
  a known limitation; framing is still OK.
- MLP loses on multiple folds → honest write-up; the in-distribution win
  stands but the generalization claim is downgraded to "interpolation regime".

---

## 4. Files to read first

1. `MLP/MLP_training/engineered_feature_common.py` lines 533–579
   (`build_canonical_feature_table`), 582–613 (`assign_splits_by_group`),
   674–700 (`build_all_stage_tables`).
2. `MLP/MLP_training/run_full_pipeline.py` lines 192–262 (stage runners),
   268–349 (`run_one_seed`), 354–384 (`bootstrap_ci`, `aggregate_seeds`).
3. `MLP/MLP_training/train_stage1_mse.py` and `train_stage2_nll.py` —
   specifically how they obtain `representative_df` after stage_tables.
4. `MLP/MLP_training/run_stage3_ablation_suite.py` — how it forwards the
   row table to ablation variants.
5. `MLP/eval/inference_rmse_on_series.py` lines 48–63 (`_finite_metrics`),
   165–375 (`run_rmse_evaluation`) — to add `--filter-experiment` arg.
6. `MLP/baseline/comparison_reports/stage3_dp_exp_0p5_vs_HA_NS_20260509/headline_comparison.csv`
   — the in-distribution baseline numbers to compare against.
7. `MLP/runs_mlp/full_pipeline_C_20260509_110100/bootstrap_summary.json` —
   the in-distribution 5-seed result schema to mirror.
