# GP Baseline Plan — Sparse Heteroscedastic GP vs Stage-2 NLL MLP

**Purpose**: a strong ML baseline against which to defend the heteroscedastic MLP
pipeline for top-tier ML/scientific venue submission. Reviewers will ask "why
not GP?" — this plan answers it head-on with a fair fight.

This brief is self-contained — a fresh conversation can pick it up cold.

---

## 1. Why this exists (and what just happened upstream)

The Stage-1 → Stage-2 → Stage-3 MLP pipeline already beats Hiroyasu-Arai (HA)
and Naber-Siebers (NS) physics baselines on penetration prediction. A
**feature-engineering ablation** (`A_scale ∝ ΔP^0.25 · ρ_a^-0.25 · √d` → `ΔP^0.5
· ρ_a^-0.25 · √d`) was just completed:

- 5-seed bootstrap CIs (mode C, run dir `MLP/runs_mlp/full_pipeline_C_20260509_110100/`):
  - rmse 10.72 → 8.92 mm (−16.8%, CIs disjoint)
  - 1σ coverage 0.497 → 0.654 (target 0.683)
  - 2σ coverage 0.911 → 0.966 (target 0.954)
  - All 6 winner metrics improved with non-overlapping 95% CIs.
- Variance-collapse diagnostic: `median_post_0p5ms_collapse_ratio` 0.378 → 0.057.
- Empirical regression on `k_quarter` (the dominant prefactor in the
  sigmoid-blended penetration model — `k_sqrt` is legacy / negligible, see
  `Thesis/images/ksqrt_ratio_distribution.png`):
  `log k_quarter ~ 0.51·log ΔP − 0.25·log ρ_a + 0.43·log d`, R²=0.345 (full table:
  `MLP/synthetic_data/fit_diagnostics/param_distributions/scaling_regression.csv`).

The exponent shift toward 0.5 was framed as: orifice-momentum physics still
dominates in the 0–5 ms window (Bernoulli velocity scale), HA's asymptotic
regime not yet established.

**Now, the GP question.** Reviewers in NeurIPS / ICML / Nature MI will ask: in
this small-data regime with a need for calibrated uncertainty, why not a
Gaussian Process? GP is the textbook strong baseline here.

---

## 2. Goal

Run a sparse heteroscedastic GP on the same data, splits, and evaluation
protocol as the MLP. Compare on the same metric set with the same 5-seed
bootstrap CI protocol. Produce a fair, defensible comparison that lets the
paper say one of:

- **(a)** "MLP pipeline ≥ GP on RMSE / calibration / inference cost — here is the
  paired-by-seed comparison."
- **(b)** "GP wins on metric X, MLP wins on metric Y — here is the trade-off
  analysis. Net deployment recommendation = MLP because of inference cost /
  Stage-3 onset CDF."
- **(c)** "GP wins overall on this metric set, but MLP retains advantages in
  inference latency and Stage-3 onset modeling. We adopt MLP for those reasons
  and use GP residuals as a Stage-2 teacher signal." (ensemble fallback)

Any of these three outcomes is a publishable framing. The goal is **the
comparison itself**, with the result being whatever the data says.

---

## 3. Fair-fight requirements (these matter — do not skip)

### 3.1 Train in `S / A_scale` space, not raw mm

The ΔP^0.5 `A_scale` collapse is the upstream win — both the MLP and the GP
should benefit. The MLP's target is `penetration / A_scale`; the GP must use
the same target. Otherwise we hand the GP a 16% RMSE handicap and the
comparison is not fair.

`A_scale` formula (already committed in `engineered_feature_common.py`):

```python
A_scale = (delta_P) ** 0.5 * (rho_air) ** -0.25 * sqrt(diameter)
```

Three call sites, all already updated to 0.5: lines 577, 1588, 1752 of
`MLP/MLP_training/engineered_feature_common.py`.

### 3.2 Same splits, same seeds

Use the **exact same train/val/test split per seed** as the mode-C MLP run.
Easiest path: load `row_table.csv` from each seed's Stage-1 run dir (it has a
`sample_split` column with values `train`/`val`/`test`). Per-seed dirs are
listed under `per_seed[*].stage1_run` in the bootstrap_summary.

5 seeds: `[42, 17, 99, 7, 2024]` (hard-coded in `full_pipeline_config.json`).

### 3.3 Heteroscedastic noise

The MLP's edge is input-dependent σ. A homoscedastic GP would cap calibration
metrics. Use one of:

- **GPyTorch + heteroscedastic likelihood** (`HeteroskedasticNoise` or a chained
  GP with a second log-variance GP).
- Or **two-stage**: train a homoscedastic SVGP for the mean, then train a second
  SVGP on `log(residual^2)` as the predicted log-variance.

The first is more principled; the second is easier to debug.

### 3.4 Sparse, not exact

Per-seed point counts: representative table has ~2,792 rows × ~700 time samples
expanded ≈ 2M points after train/val/test split (~1.4M train). Exact GP is
O(N³) — infeasible. Use **SVGP (stochastic variational GP)** with 256–512
inducing points, mini-batched ELBO, Adam optimizer. This is the standard
modern baseline.

### 3.5 Same inputs

Use the `a_only` feature variant the MLP uses:

```
features = [time_norm_0_5ms,
            tilt_angle_radian_z,
            plumes_z,
            injection_duration_us_z,
            control_backpressure_bar_z]
```

These are already z-scored in `row_table.csv`. Target = `penetration / A_scale`.

`A_scale` lives in the same row table (recomputed by the new ΔP^0.5 formula —
verify the column was generated by the new code by checking that one of the
seed-stage1 dirs is from today's run, e.g.
`stage1_engineered_mse_a_only_20260509_110102`).

---

## 4. Technical spec

### 4.1 Library

**GPyTorch on top of PyTorch + CUDA.** sklearn's `GaussianProcessRegressor`
will not scale here. The user has a working PyTorch + CUDA setup (RTX 5090 in
this environment).

### 4.2 Model

```python
# Pseudocode — fill in details with GPyTorch docs
class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        super().__init__(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                gpytorch.variational.CholeskyVariationalDistribution(num_inducing),
                learn_inducing_locations=True,
            )
        )
        # Start with ARD MaternKernel(nu=2.5) — good middle smoothness, ARD
        # lets each input dim find its own length scale (interpretable!).
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=5)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )
```

For heteroscedastic likelihood, the cleanest path is:

- Train SVGP-mean on `(features → S/A_scale)`.
- Compute residuals on training set.
- Train SVGP-logvar on `(features → log(residual^2 + ε))`.
- At inference, return mean from first GP, σ = sqrt(exp(logvar from second GP))
  in scaled space, then multiply by A_scale to get physical σ.

Or use `gpytorch.likelihoods.HeteroskedasticNoise` if comfortable.

### 4.3 Training protocol

- Inducing points: 256 (start) or 512 (if 256 underfits).
- Init inducing points: k-means on training inputs (`sklearn.cluster.KMeans`).
- Optimizer: Adam, lr=1e-2 on hyperparameters + variational params.
- Mini-batch size: 1024.
- Epochs: enough to converge ELBO — track val NLL, early-stop on plateau.
- Per-seed: train, validate on val split, evaluate on test split.

### 4.4 Evaluation

Two protocols, matching the MLP:

**(a) In-pipeline test split (n=8398 per seed)**:
- Same as MLP `winner_metrics`. Compute the same 6 metrics: rmse_mm, mae_mm,
  bias_mm, p95_abs_err_mm, coverage_1sigma, coverage_2sigma.
- Output: `gp_baseline_per_seed.csv` with one row per seed.
- Aggregate with the same `bootstrap_ci` function from `run_full_pipeline.py`
  (look for `bootstrap_ci` near line 354 — it's a 5000-resample percentile
  bootstrap with seed=20260508 for reproducibility).

**(b) External n=795k clean diagnostic**:
- Use the same loader as `MLP/eval/inference_rmse_on_series.py:run_rmse_evaluation`.
- The MLP eval on this protocol just finished today; outputs are in
  `MLP/baseline/comparison_reports/stage3_dp_exp_0p5_vs_HA_NS_20260509/`
  (check `headline_comparison.csv`).
- Wrap GP inference into the same point-collection format and call the same
  `_finite_metrics` function (lines ~48–63 of that file) to guarantee an
  apples-to-apples table.

### 4.5 Output structure

Mirror the MLP run dirs:

```
MLP/runs_mlp/gp_baseline_<timestamp>/
├── per_seed/
│   ├── seed_42/{model.pt, train_log.csv, test_metrics.json}
│   ├── seed_17/...
│   └── ...
├── bootstrap_summary.json     # same schema as the MLP one
└── gp_config_resolved.json    # kernel, inducing_count, lr, epochs, etc.

MLP/baseline/comparison_reports/gp_vs_mlp_<date>/
├── headline_comparison.csv    # extends today's table with GP rows
└── headline_comparison.md
```

The `bootstrap_summary.json` schema to copy from MLP:

```json
{
  "method": "svgp_heteroscedastic",
  "seeds": [42, 17, 99, 7, 2024],
  "per_seed": [
    {"seed": 42, "test_metrics": {"rmse_mm": ..., "mae_mm": ..., "bias_mm": ...,
                                  "p95_abs_err_mm": ..., "coverage_1sigma": ...,
                                  "coverage_2sigma": ..., "n_points": 8398}},
    ...
  ],
  "bootstrap": { "metrics": { "rmse_mm": {"mean": ..., "std": ..., "ci_lo": ..., "ci_hi": ..., "n": 5}, ... } }
}
```

This makes side-by-side comparison trivial — one `pd.concat`.

### 4.6 Wall-clock budget

Per-seed estimate: 5–15 min on the 5090 (1.4M points, 256 inducing, Adam).
Two GPs (mean + log-variance) ≈ 2× that. 5 seeds × 2 GPs × 10 min ≈ 1.5–2 h
total. Budget 3 h with buffer.

---

## 5. Pitfalls (these have specific fixes)

1. **sklearn GP**: do not. Will OOM at this scale.
2. **Training in raw mm**: hands the GP a known handicap (the A_scale collapse is
   ~16% of the variance). Always train in `S/A_scale` space.
3. **Inducing point initialization**: random init often gets stuck in poor
   variational posteriors. Use k-means on training inputs.
4. **Homoscedastic noise**: the MLP has heteroscedastic σ; matching that is
   non-trivial but mandatory for fair calibration metrics. If running out of
   time, do homoscedastic first as a sanity check, then add heteroscedastic.
5. **Mixing physical and scaled spaces**: σ in scaled space ≠ σ in mm. After
   prediction, multiply σ_scaled by A_scale to get σ in mm. Same convention as
   `engineered_feature_common.py:1099, 1111` for the MLP.
6. **Validation NLL plateau**: the GP marginal likelihood can plateau into a
   bad local optimum. If val NLL is much worse than the MLP's stage-2 val NLL
   (~0.27 across 4 of 5 seeds, 0.34 for the outlier), retrain with different
   inducing init / lr. Don't let a stuck GP unfairly lose.
7. **Time-axis kernel**: ARD over 5 dims will give the time axis its own length
   scale, which is fine. But if results look poor, try a product kernel
   `Matern(time) ⊗ Matern(conditions)` — it imposes the prior that time and
   conditions affect the function differently, which is physically true.
8. **Onset / Stage-3 comparison**: standard GP cannot replicate Stage-3's
   non-Gaussian onset CDF. Don't try to compare GP to Stage-3 directly. The
   honest comparison is **GP vs MLP-Stage-2** (mean + σ). If the user wants a
   Stage-3 analog, mention it as a limitation of GP and a strength of the
   pipeline.

---

## 6. Files to read first (in order)

1. `MLP/MLP_training/engineered_feature_common.py` lines 1–150 (config), 533–579
   (canonical feature table including A_scale formula), 721–810
   (`run_pretrain_collapse_check` — already used to validate the ΔP^0.5 win).
2. `MLP/MLP_training/run_full_pipeline.py` lines 354–400 (`bootstrap_ci`,
   `aggregate_seeds`) — copy these for GP aggregation.
3. `MLP/eval/inference_rmse_on_series.py` lines 48–100, 165–375 — the external
   eval protocol that GP must match.
4. `MLP/runs_mlp/full_pipeline_C_20260509_110100/bootstrap_summary.json` — the
   MLP comparison target.
5. `MLP/runs_mlp/stage1_engineered_mse_a_only_20260509_110102/row_table.csv` —
   one example seed's row table; columns `sample_split`, `A_scale`, the 5
   z-scored features, `penetration_at_comparison_time`, etc.
6. `MLP/baseline/comparison_reports/stage3_dp_exp_0p5_vs_HA_NS_20260509/headline_comparison.csv`
   — today's external-eval comparison; GP results should append cleanly.

---

## 7. Decision tree for the new conversation

After per-seed test metrics are in:

- If GP rmse mean is **clearly higher** than MLP (CIs disjoint, GP worse):
  framing **(a)** — celebrate, do not over-claim, write Limitations honestly.
- If GP rmse mean is **clearly lower** than MLP (CIs disjoint, GP better):
  framing **(c)** — pivot to "GP→MLP distillation" as Stage-2 enhancement, run
  one extra experiment using GP posterior mean as a soft target during MLP
  Stage-1.
- If CIs **overlap** on rmse:
  framing **(b)** — focus on calibration deltas, inference cost, and Stage-3
  capabilities. Stratified analysis (per ΔP bin, per ρ_a bin, per time window)
  almost always reveals where each model wins.

In all three cases, also check:
- **Inference latency**: time `predict(N=10000)` for both models. Expected:
  GP 100×–1000× slower at scale.
- **Calibration**: reliability diagram, ECE, CRPS for both. The MLP's
  Stage-2 NLL training should help here; GP's principled posterior should help
  too. Whoever wins, this section is publishable.

---

## 8. Hand-back to the parent conversation

When done, the GP-side outputs needed for the paper table:

- 5-seed bootstrap (mean ± std, 95% CI) on rmse_mm, mae_mm, bias_mm,
  p95_abs_err_mm, coverage_1sigma, coverage_2sigma.
- One row in `headline_comparison.csv` matching the MLP / HA / NS schema.
- Median per-prediction inference latency.
- Honest verdict + any framing recommendation (a/b/c above).
