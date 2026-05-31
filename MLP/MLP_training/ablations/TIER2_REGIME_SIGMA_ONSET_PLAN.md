# Tier-2 Accuracy Ablations — Stage-3 Regime Boundaries, KD Confidence, Onset Auxiliary Head

**Purpose**: attack three Stage-3 design knobs that affect censoring/KD bias
regions and onset accuracy but were never properly swept. These target the
weaknesses §5 already identifies (uncertain-regime RMSE 5.88 mm, near-FOV
residual −0.66 mm, Nozzle-2 onset dependence on the μ-anchor).

Three sub-campaigns:
- **2A — Stage-3 regime boundary sweep** (`uncertain_ratio`, `teacher_ratio`)
- **2B — KD confidence σ_ref complete sweep** (only {5, 10, 20} mm currently tested)
- **2C — Onset-window auxiliary regression head** (architecture change)

This brief is self-contained — a fresh agent / Codex run can pick it up cold.

---

## 1. Context

### 1.1 Stage-3 regime assignment

Coverage-driven 0.1 ms bins in
`train_stage3_distillation_plus_raw_series.py:463-510`. Hardcoded defaults
`UNCERTAIN_RATIO = 0.7`, `TEACHER_RATIO = 0.2` (constants near the top of the
file; verify exact line via grep `UNCERTAIN_RATIO` / `TEACHER_RATIO`):

- `raw_reliable`: coverage ≥ 0.7 of peak → direct NLL on raw CDF.
- `raw_uncertain`: 0.2 ≤ coverage < 0.7 → mixed raw NLL + KD.
- `teacher_only`: coverage < 0.2 → pure KD.

CLI args already exist:
- `--uncertain-ratio FLOAT` (line ~180 in same file)
- `--teacher-ratio FLOAT`
- `--sigma-conf-ref-mm FLOAT` (line ~1215, default 10.0 at :1495)

§5 Table 5 swept the KD/raw weight inside each regime but **never the
boundary itself**.

### 1.2 KD confidence weighting

`sigma_conf_ref_mm` (default 10.0) controls how KD loss is downweighted in
high-σ_teacher regions. The exact functional form lives in
`MLP/MLP_training/stage3/kd_losses.py` — verify before describing the curve in
the paper.

§5 reports only `σ_ref ∈ {5, 10, 20}` (3 points, one decade). The parameter
should be continuously sensitive; a 3-point sweep does not establish
unimodality or rule out an interior optimum.

### 1.3 Onset window

§5 documents that the Stage-2 μ-anchor (`lambda_mu_anchor=1e-2`,
`anchor_window_ms=0.15`) carries Nozzle 2 LONO from ~24 mm RMSE down to
12.7 mm. The anchor is a *blunt instrument* — a single scalar weight on
early-time positions. An auxiliary regression head trained on the onset
window (t < 0.3 ms) might capture richer early-time structure without
requiring anchor tuning.

---

## 2. Goals

| Sub | Output | Acceptance |
|---|---|---|
| 2A | `regime_boundary_summary.csv`, 4-point sweep | Establishes whether (0.7, 0.2) defaults are a plateau or a sharp optimum. |
| 2B | `sigma_ref_summary.csv`, 6-point sweep | Either picks a strict winner (≥ 0.3 mm), or confirms σ_ref=10 is robust within σ_seed. |
| 2C | `onset_aux_head_summary.csv`, 3-config A/B | Either improves Nozzle-2 LONO onset slice by ≥ 0.5 mm, or is documented as future work. |

All three should report per-regime / per-slice metrics, not just overall RMSE.

---

## 3. Sub-campaign 2A — Stage-3 regime boundary sweep

### 3.1 The 5-point grid (4 sweeps + 1 baseline)

Anchored at default (0.7, 0.2):

| run name | uncertain_ratio | teacher_ratio | rationale |
|---|---|---|---|
| regime_default | 0.7 | 0.2 | baseline (current production) |
| regime_low_uncertain | 0.5 | 0.2 | wider uncertain band → more raw mixing in censored bins |
| regime_low_teacher | 0.7 | 0.1 | shrinks teacher-only band → more raw further out |
| regime_high_teacher | 0.7 | 0.3 | grows teacher-only band → more KD trust |
| regime_high_uncertain | 0.85 | 0.2 | narrows uncertain → mostly reliable vs teacher_only |

### 3.2 Config file

Create `MLP/MLP_training/ablations/config/stage3_regime_boundary_sweep_config.json`,
mirroring `MLP/MLP_training/ablations/config/stage3_kd_sigma_weight_sweep_config.json`:

```json
{
  "_comment": "Stage 3 regime boundary sweep: uncertain_ratio x teacher_ratio. KD mode and sigma weight fixed at production winners (mse_mu_plus_sigma, 5.0).",
  "common": {
    "teacher_run": "latest_stage2",
    "device": "auto",
    "series_split": "clean",
    "sources": ["cdf"],
    "batch_size": 128,
    "epochs": 150,
    "learning_rate": 0.003,
    "patience": 20,
    "num_workers": 0,
    "pin_memory": true,
    "precompute_dataset": true,
    "eval_fast": false,
    "eval_save_points": true,
    "eval_save_plots": false,
    "eval_batch_points": 262144,
    "eval_max_traj_plots": 0,
    "skip_post_train_eval": false,
    "synthetic_root": "MLP/synthetic_data_20260509"
  },
  "selection": { "metric": "rmse_mm", "mode": "min" },
  "winner_eval": { "enabled": false },
  "ablations": [
    {
      "name": "regime_low_uncertain",
      "enabled": true,
      "run_name_prefix": "stage3_regime_low_uncertain",
      "args": {
        "uncertain_ratio": 0.5,
        "teacher_ratio": 0.2,
        "kd_mode": "mse_mu_plus_sigma",
        "kd_sigma_weight": 5.0,
        "lambda_anchor": 0.0
      }
    },
    {
      "name": "regime_low_teacher",
      "enabled": true,
      "run_name_prefix": "stage3_regime_low_teacher",
      "args": {
        "uncertain_ratio": 0.7,
        "teacher_ratio": 0.1,
        "kd_mode": "mse_mu_plus_sigma",
        "kd_sigma_weight": 5.0,
        "lambda_anchor": 0.0
      }
    },
    {
      "name": "regime_high_teacher",
      "enabled": true,
      "run_name_prefix": "stage3_regime_high_teacher",
      "args": {
        "uncertain_ratio": 0.7,
        "teacher_ratio": 0.3,
        "kd_mode": "mse_mu_plus_sigma",
        "kd_sigma_weight": 5.0,
        "lambda_anchor": 0.0
      }
    },
    {
      "name": "regime_high_uncertain",
      "enabled": true,
      "run_name_prefix": "stage3_regime_high_uncertain",
      "args": {
        "uncertain_ratio": 0.85,
        "teacher_ratio": 0.2,
        "kd_mode": "mse_mu_plus_sigma",
        "kd_sigma_weight": 5.0,
        "lambda_anchor": 0.0
      }
    }
  ],
  "sensitivity_ablations": []
}
```

### 3.3 Run

In-domain only first:

```bash
python run_stage3_ablation_suite.py \
    --config MLP/MLP_training/ablations/config/stage3_regime_boundary_sweep_config.json \
    --teacher-run MLP/runs_mlp/stage2_nll_a_only_<latest>/
```

LONO follow-up (only the top 2 in-domain winners):

```bash
for fold in Nozzle0 Nozzle1 Nozzle2 Nozzle3 Nozzle4 Nozzle5; do
    python run_lono_pipeline.py \
        --lono-holdout $fold --seed 42 \
        --stage3-variant regime_<winner_name>
done
```

### 3.4 Output

`MLP/MLP_training/ablations/regime_boundary_<date>/`:
- `regime_boundary_summary.csv` — columns: `run_name, uncertain_ratio,
  teacher_ratio, fold, rmse_overall_mm, rmse_reliable_regime_mm,
  rmse_uncertain_regime_mm, rmse_teacher_only_regime_mm, cov_1sigma,
  cov_2sigma`. The **per-regime RMSE columns are the headline** —
  even if overall RMSE is flat, regime-slice shifts are informative.
- `regime_assignment_summary.csv` — per-fold fraction of points in each
  regime (sanity check for degenerate splits).
- `verdict.md`

### 3.5 Budget 2A

- In-domain: 4 sweeps × ~2 min (Stage 3 only, reuse Stage 1+2) ≈ 8 min
- LONO follow-up: 2 winners × 5 folds × ~2 min ≈ 20 min
- **Total: ~30 min**

### 3.6 Pitfalls 2A

1. **`teacher_ratio > uncertain_ratio` is invalid**. The validation at
   `train_stage3_distillation_plus_raw_series.py:192-195` enforces
   `0 ≤ ratio ≤ 1`, but the *semantics* require `uncertain > teacher`. All
   five configs above satisfy this; do not generate combinations that don't.
2. **Empty regimes at `uncertain_ratio=0.85`**: the `raw_reliable` band may
   collapse to a thin window. Check `regime_assignment_summary.csv` per fold;
   if any regime has < 5 % of points, mark as degenerate in the verdict
   and drop from comparisons.
3. **`teacher_only` sparseness at `teacher_ratio=0.1`**: very few bins fall
   into pure KD. The variance of `rmse_teacher_only_regime_mm` will be
   large; report it but don't over-interpret a single noisy number.
4. **Teacher reuse**: all sweeps share the same Stage-2 teacher run
   (specified via `--teacher-run`). Do NOT re-train Stage 2 per config —
   that would conflate regime effects with teacher variance.

---

## 4. Sub-campaign 2B — σ_ref complete sweep

### 4.1 The 6-point grid

σ_ref ∈ {3, 5, 7, 10, 15, 20} mm.

The points 5, 10, 20 already exist if §5 Table 6 was run with
`kd_mode=mse_mu_plus_sigma, kd_sigma_weight=5.0`. Verify by reading the
existing run logs (likely under
`MLP/runs_mlp/stage3_diag_kd_mse_mu_plus_sigma*/`). If those reuse
production form, only 3 new runs are needed: σ_ref ∈ {3, 7, 15}.
If not, run all 6.

### 4.2 Config snippet

Add to / extend `stage3_kd_sigma_weight_sweep_config.json` or create
`stage3_sigma_ref_complete_sweep_config.json`:

```json
{
  "ablations": [
    {
      "name": "sigma_ref_3",
      "enabled": true,
      "run_name_prefix": "stage3_sigma_ref_3",
      "args": {
        "sigma_conf_ref_mm": 3.0,
        "kd_mode": "mse_mu_plus_sigma",
        "kd_sigma_weight": 5.0,
        "lambda_anchor": 0.0
      }
    },
    { "name": "sigma_ref_7", "args": {"sigma_conf_ref_mm": 7.0, ...} },
    { "name": "sigma_ref_15", "args": {"sigma_conf_ref_mm": 15.0, ...} }
  ]
}
```

### 4.3 Run + output

In-domain only. Reuse the latest Stage-1+2 run via `teacher_run`.
LONO follow-up only if one σ_ref beats the default by ≥ 0.3 mm in-domain.

`MLP/MLP_training/ablations/sigma_ref_sweep_<date>/`:
- `sigma_ref_summary.csv` — columns: `sigma_ref_mm, rmse_overall_mm,
  rmse_near_fov_mm, rmse_uncertain_mm, bias_near_fov_mm, cov_1sigma,
  cov_2sigma`. **Near-FOV bias is the headline metric** for this sweep —
  that's the slice where σ_ref controls KD trust most heavily.
- `sigma_ref_curve.png` — line plot of `rmse_overall_mm` and
  `bias_near_fov_mm` vs σ_ref. Visualize unimodality.
- `verdict.md`

### 4.4 Budget 2B

- 3 new runs × ~2 min ≈ 6 min (if {5, 10, 20} already exist)
- 6 runs × ~2 min ≈ 12 min (full from scratch)
- Optional LONO follow-up only if in-domain winner exists.

### 4.5 Pitfalls 2B

1. **σ_ref=3 may over-discount KD**: if the KD weight collapses to ≈ 0,
   Stage 3 reduces to raw-only training in censored regions, which behaves
   like the legacy pre-KD baseline. Watch for `kd_loss → 0` in the training
   log; if so, label that point "KD effectively disabled" in the verdict.
2. **σ_ref=20 may over-trust noisy teacher**: should manifest as elevated
   `cov_2σ` (over-coverage) and minor RMSE inflation in `teacher_only` bins.
3. **Comparing against historical {5, 10, 20}**: those runs may use slightly
   different `epochs`, `patience`, or seed. Re-run them under identical
   config if there's any chance of mismatch — small protocol differences
   can fake a 0.2 mm trend.

---

## 5. Sub-campaign 2C — Onset-window auxiliary regression head

### 5.1 Hypothesis

The current Stage-2 μ-anchor is a global penalty that says "the mean
prediction should match the teacher's mean in the first 0.15 ms". An
auxiliary regression head trained on the onset window can carry richer
signal: instead of one scalar anchor weight, the head learns its own bias
and curvature for early-time predictions.

If 2C works, it could replace the μ-anchor (architectural simplification).
If not, the μ-anchor is the right design and the failure documents it.

### 5.2 Architecture change

Modify `PenetrationMLP` in `MLP/MLP_training/efc/models.py` (verify exact
class via grep `class PenetrationMLP`):

```python
class PenetrationMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        ...,
        onset_aux_head: bool = False,    # NEW
        onset_aux_hidden: int = 64,      # NEW
    ):
        super().__init__()
        # ... existing trunk + mu_head + log_var_head ...
        if onset_aux_head:
            self.onset_head = nn.Sequential(
                nn.Linear(hidden_dims[-1], onset_aux_hidden),
                nn.GELU(),
                nn.Linear(onset_aux_hidden, 1),
            )
        else:
            self.onset_head = None

    def forward(self, x):
        h = self.shared_trunk(x)
        mu = self.mu_head(h)
        log_var = self.log_var_head(h)
        if self.onset_head is not None:
            mu_onset = self.onset_head(h)
            return mu, log_var, mu_onset  # tuple of 3
        return mu, log_var                  # tuple of 2 (current)
```

In Stage-1 / Stage-2 loss functions (locate via grep `def stage1_loss` or
similar; likely under `MLP/MLP_training/efc/losses.py`), add the auxiliary
term:

```python
def stage_loss(outputs, target, time_normed, *, lambda_aux=0.1, onset_t_norm_max=0.06, ...):
    """t < 0.3 ms corresponds to time_normed < 0.3 / time_max_ms = 0.3 / 5 = 0.06."""
    if len(outputs) == 3:
        mu, log_var, mu_onset = outputs
    else:
        mu, log_var = outputs
        mu_onset = None
    primary = nll_or_mse(mu, log_var, target, ...)
    if mu_onset is None:
        return primary
    onset_mask = (time_normed < onset_t_norm_max)
    if onset_mask.sum() == 0:
        return primary
    aux = ((mu_onset.squeeze(-1)[onset_mask] - target[onset_mask]) ** 2).mean()
    return primary + lambda_aux * aux
```

**Important**: the auxiliary `mu_onset` is **not consumed at inference**.
It exists only as a regularizer/gradient source during training. The
production `mu` is what's evaluated.

### 5.3 The 3 configs

| run name | onset_aux | lambda_aux | Stage-2 μ-anchor | rationale |
|---|---|---|---|---|
| baseline_no_aux | False | — | `mu_anchor` (production) | current §5 winner reference |
| aux_with_anchor | True | 0.1 | `mu_anchor` | aux head AND anchor (complementary?) |
| aux_no_anchor | True | 0.1 | `no_anchor` | aux head REPLACES anchor (simpler arch wins?) |

### 5.4 Run

```bash
# baseline_no_aux is the existing production run; reuse if available.

# aux_with_anchor
python train_stage1_mse.py --variant a_only --seed 42 --onset-aux-head --lambda-aux 0.1
python train_stage2_nll.py --variant a_only --seed 42 --onset-aux-head --lambda-aux 0.1 \
    --stage2-ablation mu_anchor --stage1-run-dir <latest_stage1>
# then Stage 3 production variant

# aux_no_anchor (same as above, but Stage 2 with --stage2-ablation no_anchor)
```

LONO follow-up only on the in-domain winner (likely 5-fold on N2 first as a
fast check, then full 5-fold).

### 5.5 Output

`MLP/MLP_training/ablations/onset_aux_head_<date>/`:
- `onset_aux_summary.csv` — columns: `config, fold, rmse_overall_mm,
  rmse_onset_t_lt_0p3ms_mm, rmse_t_gt_0p3ms_mm, learned_aux_mu_bias`
- `verdict.md` — does the aux head help? does it replace the anchor?

### 5.6 Budget 2C

- 3 in-domain runs × ~10 min ≈ 30 min
- Top-1 LONO (5 folds) × ~10 min ≈ 50 min if a winner exists
- **Total: ~80 min**

### 5.7 Pitfalls 2C

1. **Onset slice has few points**: t < 0.3 ms ≈ 6 % of normalised time
   (`time_max_ms=5.0`). Some batches may have very few onset points,
   inflating the aux loss variance. Print `onset_mask.sum()` in the first
   training epoch and abort if mean per-batch onset count < 5.
2. **Aux loss must not dominate**: at `lambda_aux=0.1`, the aux term should
   sit at ~10 % of the primary loss after the first epoch. Log both
   separately in the training history (extend `epoch_history` to include
   `aux_loss_mean`).
3. **Interaction with μ-anchor**: the auxiliary head and the μ-anchor
   target the same time region. The 3-config sweep is designed to
   distinguish "complementary" from "replacement"; do not collapse to 1
   config by accident.
4. **Stage-3 KD pipeline**: the Stage-2 teacher now has an aux head. The
   Stage-3 student inherits the architecture (`hidden_dims` matched), but
   the KD loss should still only target `(mu, log_var)`, not `mu_onset`.
   Verify by checking that `teacher_outputs = teacher(x)[:2]` or similar in
   the Stage-3 trainer.

---

## 6. Total budget

| Campaign | Runs | Wall-clock |
|---|---|---|
| 2A regime boundary in-domain | 4 | 8 min |
| 2A LONO follow-up (top 2) | 10 | 20 min |
| 2B σ_ref new 3 points (assuming {5,10,20} exist) | 3 | 6 min |
| 2C onset aux head in-domain | 3 | 30 min |
| 2C LONO follow-up (conditional) | 5 | 50 min |
| **TOTAL** | **~25 runs** | **~2 h** |

---

## 7. Files to read first

1. `MLP/MLP_training/train_stage3_distillation_plus_raw_series.py:160-205` —
   Stage-3 CLI args (`--uncertain-ratio`, `--teacher-ratio`,
   `--sigma-conf-ref-mm`).
2. `MLP/MLP_training/train_stage3_distillation_plus_raw_series.py:460-525` —
   regime assignment logic; this is the code under test for 2A.
3. `MLP/MLP_training/stage3/kd_losses.py` — KD loss implementations
   (verify how `sigma_conf_ref_mm` enters before describing it in §5).
4. `MLP/MLP_training/efc/models.py` — `PenetrationMLP` class definition
   (needed for 2C).
5. `MLP/MLP_training/ablations/config/stage3_kd_sigma_weight_sweep_config.json`
   — template for 2A/2B config files.
6. `MLP/MLP_training/run_stage3_ablation_suite.py` — the orchestrator that
   reads these configs.

---

## 8. Paper paragraph hooks

If 2A, 2B, 2C all return null results, the paper paragraph is short:

> "Stage-3 regime boundaries (`uncertain_ratio`, `teacher_ratio`) and KD
> confidence scale (`sigma_conf_ref_mm`) were swept across 4 and 6
> points respectively; the production defaults are within σ_seed of all
> alternatives, validating them as defensible defaults. An auxiliary
> onset-window regression head was tested as an architectural alternative
> to the Stage-2 μ-anchor; it neither improved nor degraded Nozzle-2 LONO
> RMSE beyond seed variance."

If 2A finds a win > σ_seed:

> "The Stage-3 regime boundary was retuned from (uncertain=0.7, teacher=0.2)
> to (X, Y) based on the sweep; this improves uncertain-regime RMSE from
> 5.88 → Z mm without harming reliable-regime accuracy."

If 2C shows the aux head replaces the μ-anchor:

> "An onset-window auxiliary regression head trained on t < 0.3 ms recovers
> the same Nozzle-2 LONO behaviour as the Stage-2 μ-anchor (RMSE 12.7 mm
> vs 12.7 mm) without the global anchor penalty, simplifying the loss
> hierarchy."
