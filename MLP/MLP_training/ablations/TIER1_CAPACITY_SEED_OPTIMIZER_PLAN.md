# Tier-1 Accuracy Ablations — Capacity, Seed Variance, Optimizer/Regularization

**Purpose**: validate that the headline in-distribution MLP RMSE (~6.21 mm 5-seed
mean on the full-clean CDF; ~7.0 mm 4-fold LONO excluding Nozzle 0) is not stuck
at a local minimum determined by un-swept defaults. Three sub-campaigns:

- **1A — Architecture sweep** (width × depth) — never reported in §5
- **1B — Seed variance baseline** — currently all LONO results are seed=42 single
  shot; we need σ_seed to interpret any other ablation difference
- **1C — Weight-decay × learning-rate sweep** — defaults `lr=4e-3, wd=2e-4`
  (Stage 1) and `lr=8e-4, wd=1e-4` (Stage 2) have never been tuned

This brief is self-contained — a fresh agent / Codex run can pick it up cold.

---

## 1. Context (what's already in place)

The Stage-1 → Stage-2 → Stage-3 MLP pipeline uses (all confirmed in
`MLP/MLP_training/efc/data_io.py`):

- `hidden_dims = [512, 512, 128]` — 3 layers, with a final 128-d bottleneck
  (`data_io.py:146` Stage-1, `:209` Stage-2)
- Stage-1: `lr=4e-3, wd=2e-4, batch=128, dropout=0.3, activation=gelu,
  epochs=300, patience=40`
- Stage-2: `lr=8e-4, wd=1e-4, batch=96, epochs=220` (warm-starts from Stage 1)
- Optimizer: **AdamW** (already; `trainers/base.py:269`)
- LONO support: existing `--lono-holdout <nozzle_name>` CLI flag on all stage
  trainers (`stage1.py:69`, mirrored in stage2/stage3)
- Production Stage-3 winner: `anchor_off` (4/5 majority vote in mode-C 5-seed
  run, per `OOD_LONO_PLAN.md:163`)

In-distribution headline (5-seed, mode-C; `MLP/runs_mlp/full_pipeline_C_20260509_110100/bootstrap_summary.json`):
- MLP RMSE 6.211 ± 0.283 mm; HA calibrated 9.974; NS delay 8.840.

LONO 5-fold seed=42 only (§5 Table 3):
- MLP 12.55 ± 12.46 mm (N0 dominates at 34.76 mm); SVGP 10.16 ± 9.57 mm.
- 4-fold excluding N0: MLP 7.00 ± 1.18 vs SVGP 5.95 ± 1.98.

Three reasons to suspect the MLP is under-tuned:
- `[512, 512, 128]` was picked early and never swept.
- `wd=2e-4` (Stage 1) is a guess that survived.
- Every LONO number is a single training trajectory.

---

## 2. Goals

| Sub | Output | Purpose in paper |
|---|---|---|
| 1A | `arch_sweep_summary.csv`: 9 hidden_dims × {val_rmse, test_rmse} on in-domain | Replace the implicit "we used [512,512,128]" with "swept across 9 configs; default is within σ_seed of best." |
| 1B | `seed_variance_summary.csv`: σ_seed on Nozzle 2 LONO fold | Footnote on §5 Tables 4/5 saying "ablation gaps below N mm are within seed variance." |
| 1C | `lr_wd_sweep_summary.csv`: best (lr, wd) on in-domain, top-3 carried to LONO | Either a strict improvement (≥ 0.5 mm) or confirm defaults are near-optimal. |

Acceptance: each campaign produces a single CSV + a 1-paragraph verdict.md that
can be lifted into §5 or appendix.

---

## 3. Sub-campaign 1A — Architecture sweep

### 3.1 Add `--hidden-dims` CLI to Stage 1, 2, 3 trainers

`MLP/MLP_training/trainers/stage1.py:53-71` already has `--learning-rate`,
`--weight-decay`. Add an adjacent argument:

```python
parser.add_argument(
    "--hidden-dims", type=str, default=None,
    help="Comma-separated hidden layer widths, e.g. '256,256,64'. "
         "Overrides DEFAULT_STAGE1_CONFIG['hidden_dims'].",
)
```

In `MLP/MLP_training/trainers/base.py:run()`, after the existing
`_set_if_not_none(overrides, "weight_decay", args.weight_decay, float)` at
line 181, add:

```python
if getattr(args, "hidden_dims", None) is not None:
    overrides["hidden_dims"] = [int(x) for x in str(args.hidden_dims).split(",") if x.strip()]
```

Mirror the same `--hidden-dims` CLI in `trainers/stage2.py`. **Stage 2 warm-
starts from Stage 1**: if `hidden_dims` differ, `model.load_state_dict(...)`
will fail. For this sweep, re-train Stage 1 from scratch for each arch config —
do NOT mix shapes.

Stage 3 (`train_stage3_distillation_plus_raw_series.py`) already inherits
`hidden_dims` from `teacher_config` (line 804). No CLI override is needed
unless you want student ≠ teacher; in this sweep we keep them equal, so no
change is needed in Stage 3.

### 3.2 The 9-point grid

| run name | hidden_dims | layers | width |
|---|---|---|---|
| arch_w128_d2 | `[128,128]` | 2 | narrow |
| arch_w128_d3 | `[128,128,128]` | 3 | narrow |
| arch_w128_d4 | `[128,128,128,128]` | 4 | narrow |
| arch_w256_d2 | `[256,256]` | 2 | mid |
| arch_w256_d3 | `[256,256,256]` | 3 | mid |
| arch_w256_d4 | `[256,256,256,256]` | 4 | mid |
| arch_w512_d2 | `[512,512]` | 2 | wide |
| arch_w512_d3 | `[512,512,512]` | 3 | wide |
| arch_w512_d4 | `[512,512,512,512]` | 4 | wide |
| **baseline** | `[512,512,128]` | 3 | bottleneck (current default, also include) |

10 runs total. All uniform-width (no bottlenecks) except the baseline reference.

### 3.3 Driver

Create `MLP/MLP_training/ablations/run_arch_sweep.py`. Mirror
`run_full_pipeline.py:run_one_seed`. For each config invoke
Stage 1 → Stage 2 → single Stage-3 `anchor_off` variant (skip the
Stage-3 ablation suite to save time).

Minimal pseudocode:

```python
ARCH_CONFIGS = [
    ("arch_w128_d2", "128,128"),
    ("arch_w128_d3", "128,128,128"),
    ...
    ("baseline_w512_512_128", "512,512,128"),
]
for name, hd_str in ARCH_CONFIGS:
    s1 = run_subprocess([
        "python", "train_stage1_mse.py", "--variant", "a_only",
        "--seed", "42", "--hidden-dims", hd_str,
    ])
    stage1_dir = latest_run_dir("stage1_engineered_mse_a_only")
    s2 = run_subprocess([
        "python", "train_stage2_nll.py", "--variant", "a_only",
        "--seed", "42", "--hidden-dims", hd_str,
        "--stage1-run-dir", stage1_dir,
    ])
    stage2_dir = latest_run_dir("stage2_nll_a_only")
    s3 = run_subprocess([
        "python", "train_stage3_distillation_plus_raw_series.py",
        "--teacher-run", stage2_dir,
        "--kd-mode", "mse_mu_plus_sigma",
        "--kd-sigma-weight", "5.0",
        "--lambda-anchor", "0.0",            # anchor_off
    ])
    record_metrics(name, hd_str, s3.run_dir)
```

### 3.4 Computational scope

Per run (1 seed, full pipeline, no ablation suite):
- Stage 1 ~1 min + Stage 2 ~6 min + Stage 3 (single variant) ~1.5 min
- **≈ 8.5 min per run on RTX 5090**

10 runs × 8.5 min ≈ **~85 min total** in-domain.

Seed = 42 fixed across the sweep. All in-domain (random split, no LONO).

### 3.5 Top-2 LONO follow-up

Take the 2 in-domain winners (lowest test RMSE) and run full 5-fold LONO with
seed=42 and `anchor_off`:

```bash
python run_lono_pipeline.py --hidden-dims <winner_dims> --seed 42 \
    --stage3-variant anchor_off
```

2 configs × 5 folds × ~8.5 min ≈ **~85 min**.

If the in-domain winner also beats the `[512,512,128]` baseline on LONO mean
RMSE, that becomes the new production architecture.

### 3.6 Outputs

`MLP/MLP_training/ablations/arch_sweep_<date>/`:
- `arch_sweep_summary.csv` — columns: `run_name, hidden_dims, params_total,
  val_rmse_mm, test_rmse_mm, train_minutes`
- `arch_sweep_lono_followup.csv` — columns: `run_name, fold_nozzle,
  test_rmse_mm, test_mae_mm, cov_1sigma, cov_2sigma`
- `verdict.md` — 1 paragraph: in-domain best, LONO best, production
  recommendation.

### 3.7 Pitfalls 1A

1. **Warm-start shape mismatch**: every config re-trains Stage 1 from
   scratch. If Sonnet tries to reuse a Stage 1 checkpoint across configs, the
   Stage 2 loader will raise on shape mismatch.
2. **Bottleneck baseline parameter count**: the `[512,512,128]` baseline has
   ~360k params; `[512,512,512]` has ~530k. If the wider uniform config wins,
   the win may be capacity, not architecture. Report `params_total` in the CSV.
3. **GELU + dropout=0.3 fixed**: do NOT sweep these together with hidden_dims
   here. That's a follow-up if 1A shows architecture-sensitive results.
4. **`a_only` variant only**: do not also sweep features in this campaign.

---

## 4. Sub-campaign 1B — Seed variance baseline

### 4.1 What to run

5 seeds × **Nozzle 2** LONO fold (the in-distribution-median fold per
`OOD_LONO_PLAN.md:175`).

Seeds: `{13, 42, 91, 137, 271}` — three primes, the project default 42, one
larger prime. Avoid round numbers to make hindsight-bias arguments
implausible.

Each run:

```bash
python run_lono_pipeline.py \
    --lono-holdout Nozzle2 \
    --seed <seed> \
    --stage3-variant anchor_off \
    --skip-ablation-suite
```

5 runs total.

### 4.2 Mode and stages

Use **mode C** (re-train all stages per seed). Mode A (share Stage 1+2) would
underestimate σ_seed by capturing only Stage-3 distillation variance, which is
not what we want — we want the full pipeline's σ_seed.

### 4.3 Outputs

`MLP/MLP_training/ablations/seed_variance_nozzle2_<date>/`:
- `per_seed.csv` — columns: `seed, rmse_mm, mae_mm, bias_mm, cov_1sigma,
  cov_2sigma, train_minutes`
- `summary.json` — `mean, std, p25, p75, 95ci_low, 95ci_high` for each
  metric across 5 seeds
- `verdict.md` — the single number σ_seed and a list of existing §5
  ablation gaps it explains away.

### 4.4 Decision rule

Apply σ_seed retroactively to existing §5 results:
- **Table 4** (Stage-2 anchor): gaps are 1.81 mm (no→mu) and 0.32 mm (mu→mu_sigma).
  - σ_seed < 0.5 mm → both gaps real
  - σ_seed > 1.5 mm → mu_sigma gap (0.32) is noise; only no_anchor → mu_anchor
    is significant
- **Table 5** (Stage-3 regime): all gaps are 0.06–0.21 mm. Almost certainly
  within seed variance. The CSV in §5 should append a column "within σ_seed?
  yes/no".

### 4.5 Budget 1B

5 runs × ~9 min (single fold, single Stage 3 variant) ≈ **~45 min**.

### 4.6 Pitfalls 1B

1. **Seed must propagate to all RNGs**: `np.random.seed`,
   `torch.manual_seed`, `torch.cuda.manual_seed_all` (all done in
   `trainers/base.py:193-196`). Also: any sklearn k-means or other utility
   used downstream must accept the same seed.
2. **CUDA non-determinism**: even with seeds fixed, cuDNN heuristics can
   produce small floating-point differences across runs. If σ_seed appears
   suspiciously small (< 0.1 mm), verify by running seed=42 twice — non-zero
   delta means cuDNN nondeterminism, which is a known and acceptable noise
   floor.
3. **Use the same `anchor_off` Stage 3 variant across all 5 seeds**: do NOT
   re-pick the ablation winner per seed. The point is to measure σ_seed of a
   fixed pipeline, not of a pipeline+selection.

---

## 5. Sub-campaign 1C — Weight-decay × learning-rate sweep

### 5.1 The grid (Stage 1 only)

| | wd=1e-5 | wd=1e-4 | wd=1e-3 |
|---|---|---|---|
| lr=1e-3 | □ | □ | □ |
| lr=4e-3 (default) | □ | □ (baseline) | □ |
| lr=1e-2 | □ | □ | □ |

9 configs, seed=42, in-domain. Stage 2 keeps its defaults (lr=8e-4,
wd=1e-4). Sweeping Stage 2 too would double the grid; we'll flag Stage 2
re-tune as a follow-up only if Stage 1 LR/WD change shifts the warm-start
notably.

### 5.2 Driver

Re-use the `run_arch_sweep.py` scaffold but iterate over `(lr, wd)`. The CLI
flags already exist (`--learning-rate`, `--weight-decay` on Stage 1). Keep
`--hidden-dims` at the default (or at 1A's winner if 1A is done first).

```python
LR_WD_GRID = [
    (1e-3, 1e-5), (1e-3, 1e-4), (1e-3, 1e-3),
    (4e-3, 1e-5), (4e-3, 1e-4), (4e-3, 1e-3),
    (1e-2, 1e-5), (1e-2, 1e-4), (1e-2, 1e-3),
]
for lr, wd in LR_WD_GRID:
    name = f"lrwd_lr{lr:.0e}_wd{wd:.0e}"
    ... # same scaffold as 1A
```

### 5.3 Top-3 LONO follow-up

The top 3 (lr, wd) configs by in-domain test RMSE go to 5-fold LONO:
3 × 5 × ~8.5 min = **~130 min**.

### 5.4 Outputs

`MLP/MLP_training/ablations/lr_wd_sweep_<date>/`:
- `lr_wd_summary.csv` — columns: `lr, wd, val_rmse_mm, test_rmse_mm,
  train_minutes`
- `lr_wd_lono_followup.csv` — same schema as 1A
- `verdict.md`

### 5.5 Pitfalls 1C

1. **lr=1e-2 may diverge** with current `grad_clip_norm=1.0`. If loss goes
   NaN in the first ~5 epochs, mark the run as failed in the CSV and drop it
   from the verdict. Do NOT lower `grad_clip_norm` — that's a separate ablation.
2. **wd=1e-3 interacts with `var_reg_weight=1e-3`**: both regularize the
   variance head. If `cov_1σ` drops sharply at wd=1e-3, log it but do not
   adjust `var_reg_weight` here.
3. **Stage-2 mu-anchor compatibility**: if Stage 1 ends in a meaningfully
   different scale due to wd, the Stage-2 mu-anchor (`lambda_mu_anchor=1e-2`)
   may need re-tuning. Out of scope for 1C; flag as follow-up.

---

## 6. Total budget

| Campaign | Runs | Wall-clock |
|---|---|---|
| 1B seed variance (run first; σ_seed informs the others) | 5 | ~45 min |
| 1A in-domain arch sweep | 10 | ~85 min |
| 1A top-2 LONO follow-up | 10 (2 × 5 folds) | ~85 min |
| 1C in-domain lr/wd sweep | 9 | ~80 min |
| 1C top-3 LONO follow-up | 15 (3 × 5 folds) | ~130 min |
| **TOTAL** | **49 runs** | **~7.5 h** |

Fits in one working day. **Recommended order: 1B → 1A → 1C** — σ_seed must be
known before interpreting any other ablation difference.

---

## 7. Files to read first

1. `MLP/MLP_training/efc/data_io.py:140-175` — Stage-1 default config (current
   `hidden_dims`, `lr`, `wd` values).
2. `MLP/MLP_training/efc/data_io.py:185-250` — Stage-2 default config.
3. `MLP/MLP_training/trainers/base.py:166-273` — the shared `run()` template;
   this is where new overrides plug in (line 181 for the
   `_set_if_not_none(...weight_decay...)` pattern to copy).
4. `MLP/MLP_training/trainers/stage1.py:53-71` — `parse_args` (where to add
   `--hidden-dims`).
5. `MLP/MLP_training/run_full_pipeline.py` — orchestrator pattern to mirror
   for `run_arch_sweep.py` and the lr/wd driver.
6. `MLP/MLP_training/ood_lono/OOD_LONO_PLAN.md` — LONO infrastructure
   (orchestrator + split function). The `--lono-holdout` flag is already
   wired up; no new code needed for LONO support here.
7. `MLP/runs_mlp/full_pipeline_C_20260509_110100/bootstrap_summary.json` — the
   in-distribution 5-seed result schema; output CSVs should be loadable
   alongside this for comparison.

---

## 8. What success looks like (paper paragraph)

After all three sub-campaigns finish, §5 of the thesis can replace its
current implicit "we used these hyperparameters" with:

> "The network architecture and training hyperparameters were validated
> against a swept alternative space comprising 9 width-depth configurations
> (Tier-1A; Appendix Table TX), 9 (lr, weight_decay) combinations (Tier-1C;
> Appendix Table TY), and a 5-seed variance baseline on the median LONO fold
> (Tier-1B; σ_seed = X mm). The production [arch] with AdamW(lr=Y,
> weight_decay=Z) is within σ_seed of the best swept alternative on the
> in-distribution test split. Stage-2 anchor ablation gaps below σ_seed are
> not interpreted as evidence."
