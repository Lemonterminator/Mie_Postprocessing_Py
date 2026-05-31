# Calibration Diagnostics Plan — Reliability + CRPS + PIT for All Models

**Purpose**: complete the calibration story beyond 1σ/2σ coverage for top ML/
scientific venue submission. Reviewers will ask for reliability diagrams,
ECE, CRPS, PIT histograms — these are standard probabilistic-forecast
diagnostics. We have all the σ predictions; this is purely an evaluation
script that adds them.

This brief is self-contained — a fresh agent / Codex run can pick it up cold.

---

## 1. Context (what's already in place)

The Stage-1→2→3 pipeline produces per-point (μ, σ) predictions. The recent
mode-C run with ΔP^0.5 A_scale gave (n=795,561 clean diagnostic, 5-seed mean):

- cov_1σ = 0.774 ± 0.023  (target 0.683)
- cov_2σ = 0.981 ± 0.004  (target 0.954)

The MLP slightly **over-covers** on the clean test set (predicted σ a bit too
generous). HA / NS baselines also have σ:

- HA calibrated: cov_1σ = 0.536, cov_2σ = 0.886, mean_pred_std = 7.48 mm
- NS delay:      cov_1σ = 0.620, cov_2σ = 0.893, mean_pred_std = 7.63 mm

CSV with all baseline numbers and pointer to per-trajectory data:
`MLP/baseline/comparison_reports/stage3_vs_HA_NS_20260509/overall_metrics.csv`

The calibration story we want to tell in the paper:
- **MLP σ is approximately calibrated, HA/NS σ is not.**
- **CRPS is lower for MLP (sharper *and* still calibrated).**
- **PIT histograms** show MLP closest to uniform, HA/NS skewed.

We already have the σ predictions. We just need to compute the right summary
statistics and plots.

---

## 2. Goal

Single script that produces, for each model in {HA, NS, old-MLP, 5 new
ΔP^0.5 MLPs}:

- Reliability diagram (predicted-quantile vs observed-quantile curve).
- Expected Calibration Error (ECE).
- Mean CRPS (Continuous Ranked Probability Score).
- Sharpness = mean(σ).
- PIT histogram (Probability Integral Transform — should be uniform if
  calibrated).
- KS test against uniform on PIT, p-value reported.

All metrics on the n=795,561-point clean diagnostic protocol — same as the
existing baseline CSV.

Output:
- One reliability diagram with all 8 model curves overlaid.
- One CRPS-vs-sharpness scatter (the calibration-sharpness trade-off).
- One PIT histogram per model (small multiples 2x4 grid).
- A summary CSV: rows = models, columns = ECE, CRPS_mean, CRPS_std,
  sharpness, PIT_KS_pvalue, cov_1σ, cov_2σ.

---

## 3. Required formulas (precise)

For each predicted Gaussian N(μ_i, σ_i) and observed y_i:

### 3.1 PIT
```
pit_i = Φ((y_i - μ_i) / σ_i)
```
where Φ is the standard normal CDF (`scipy.stats.norm.cdf`).

If predictions are calibrated, `{pit_i}` is uniform on [0, 1]. Use
`scipy.stats.kstest(pit, 'uniform')` to get the KS p-value. Plot histogram
in 20 bins; flat is calibrated.

### 3.2 Reliability diagram
For nominal levels α ∈ {0.025, 0.05, 0.10, 0.20, ..., 0.95, 0.975}, compute
the empirical lower-tail fraction:
```
empirical_alpha = mean(pit_i ≤ α)
```
Plot `(α, empirical_alpha)` vs identity line `(α, α)`. A model that under-
covers lies above the line on the lower tail; over-covers, below.

### 3.3 ECE (Expected Calibration Error, regression form)
```
ECE = mean over α of |empirical_alpha(α) - α|
```
Use 19 levels uniform on (0.025, 0.975) for stable estimates.

### 3.4 CRPS (Gaussian closed form)
For Gaussian forecast and scalar observation:
```
z = (y - μ) / σ
CRPS = σ · [z · (2·Φ(z) - 1) + 2·φ(z) - 1/sqrt(π)]
```
where `φ` is standard normal pdf. **Verify the sign** with
`scipy.stats.norm.pdf` and `.cdf`. CRPS has units of the observation (mm
here). Lower is better. Report mean and std over the dataset.

### 3.5 Sharpness
```
sharpness = mean(σ_i)  # in mm
```
Lower is sharper; but sharpness alone is not good — must be paired with
calibration.

---

## 4. Data sources (per model)

For each model we need a (truth, μ, σ) array of length 795,561.

### 4.1 The 5 new ΔP^0.5 MLPs

Per-seed eval dirs were created today by
`MLP/eval/eval_new_winners_vs_baselines.py`. They are at:
```
MLP/eval/rmse_eval_clean_20260509_193855_newdp_seed42
MLP/eval/rmse_eval_clean_20260509_194004_newdp_seed17
MLP/eval/rmse_eval_clean_20260509_194112_newdp_seed99
MLP/eval/rmse_eval_clean_20260509_194219_newdp_seed7
MLP/eval/rmse_eval_clean_20260509_194328_newdp_seed2024
```

But that earlier run used `save_points=False`. **Re-run with `save_points=True`**
to get `points.csv` (columns: truth_mm, pred_mm, std_mm, traj_key, t_ms, ...).
The CLI script is `MLP/eval/inference_rmse_on_series.py`; default
`--no-save-points` is off so just don't pass `--no-save-points`.

Or call `run_rmse_evaluation` programmatically with `save_points=True,
save_plots=False, max_traj_plots=0` for speed.

### 4.2 The old MLP (single seed)

Eval dir: `MLP/eval/rmse_eval_clean_20260509_004118_winner_full/points.csv`.
Should already exist with points saved (the user's CSV at
`stage3_vs_HA_NS_20260509/overall_metrics.csv` references this dir).

### 4.3 HA calibrated

Run dir: `MLP/baseline/Hiroyasu_Arai/outputs/20260509_020253_ha_calibrated_grouped_condition_all_all_clean_diagnostic_20260509_review/`

Look for `per_trajectory.csv` or `points.csv`. If only per-trajectory exists,
the σ in there is per-trajectory; expand to point-level by repeating per its
time samples (the format should match the clean diagnostic schema). If
needed, re-run HA inference with the existing HA inference script (path:
`MLP/baseline/Hiroyasu_Arai/`).

### 4.4 NS delay

Run dir: `MLP/baseline/Naber_Siebers/outputs/20260509_004452_ns_delay_grouped_condition_all_clean_diagnostic_20260509/`
Same approach as HA.

---

## 5. Output script spec

Create `MLP/eval/calibration_diagnostics.py`:

```python
"""Calibration diagnostics — reliability, ECE, CRPS, PIT — for HA / NS / MLPs."""

# Inputs: hard-coded list of (model_name, points_csv_path) tuples
# Outputs (under MLP/baseline/comparison_reports/calibration_<date>/):
#   summary.csv, reliability_overlay.png, crps_sharpness_scatter.png,
#   pit_histograms.png, per_model_pit/<model>.csv (raw PIT values for replication)

MODELS = [
    ("Hiroyasu-Arai calibrated", "<path to HA points or per_trajectory CSV>"),
    ("Naber-Siebers delay",       "<path to NS points or per_trajectory CSV>"),
    ("Stage-3 MLP (old, ΔP^0.25)", ".../rmse_eval_clean_20260509_004118_winner_full/points.csv"),
    ("Stage-3 MLP ΔP^0.5 seed 42",   "<re-run with save_points=True>"),
    ("Stage-3 MLP ΔP^0.5 seed 17",   "..."),
    ("Stage-3 MLP ΔP^0.5 seed 99",   "..."),
    ("Stage-3 MLP ΔP^0.5 seed 7",    "..."),
    ("Stage-3 MLP ΔP^0.5 seed 2024", "..."),
]
```

Functions:

```python
def compute_pit(truth, mu, sigma):
    return scipy.stats.norm.cdf((truth - mu) / np.maximum(sigma, 1e-12))

def reliability_curve(pit, levels):
    return np.array([np.mean(pit <= a) for a in levels])

def ece(pit, levels):
    rel = reliability_curve(pit, levels)
    return float(np.mean(np.abs(rel - levels)))

def crps_gaussian(truth, mu, sigma):
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (truth - mu) / sigma_safe
    return sigma_safe * (z * (2 * scipy.stats.norm.cdf(z) - 1)
                         + 2 * scipy.stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
```

Plotting requirements:

- **Reliability**: 1 axes, 8 model curves + identity line + ±2% shaded
  band (sampling tolerance for n=795k is tiny ~0.001, so the band is
  basically invisible — fine, just show the identity).
  Use distinct line styles for HA/NS (dashed) vs MLP (solid).
  Title: "Reliability diagram (n=795,561 clean diagnostic)".
- **CRPS-sharpness scatter**: x = sharpness, y = mean CRPS, one point per
  model, labels next to each. Lower-left corner = sharper + better
  calibration. The MLP cluster should sit lower-left of HA/NS.
- **PIT histograms**: 2×4 grid, 20 bins each, x ∈ [0, 1].
  Add red horizontal line at expected uniform density (1/20 = 0.05 per bin
  if normalizing to density, or n/20 if frequency). Annotate KS p-value in
  each subplot title.

Summary CSV columns:
```
model, n_points, ece, crps_mean, crps_std, sharpness_mm,
pit_ks_pvalue, coverage_1sigma, coverage_2sigma
```

Aggregate the 5 new MLP rows into one (mean ± std) row at the bottom for
the headline comparison.

---

## 6. Implementation budget

- Re-running 5 winner evals with save_points=True: ~5 min total (I/O bound).
- Loading + diagnostics on 8 models × 795k points: ~30 sec total.
- Plot generation: ~5 sec.

Total: ~10 min including HA/NS data wrangling.

---

## 7. Pitfalls

1. **σ unit mismatch**: HA/NS σ might be in different units (e.g., mm² or
   relative). Verify by checking that 1σ coverage from raw σ matches the
   reported `coverage_1sigma=0.536` for HA. If not, fix unit conversion
   before computing PIT.
2. **σ floor / clamp**: MLP has `std_clamp_min` (default 1e-3 mm) — verify
   no points have σ at the floor; if many do, the PIT will spike at 0 and 1.
3. **Outliers in PIT**: a few |z| > 10 points can crash KS test. Clip PIT
   to (eps, 1-eps) where eps=1e-9 before passing to `scipy.stats.kstest`.
4. **Truth/pred path mismatch**: each `points.csv` may have different
   column names. Standardize: read with explicit column rename to
   {truth, pred, std}. Verify `len(points) == 795_561` after filtering.
5. **HA/NS may lack point-level σ**: if the baseline scripts only give
   per-trajectory σ, expand by repeating to all timesteps of that
   trajectory. Note this in the summary as a caveat ("HA/NS sigma is
   per-trajectory; MLP sigma is per-point").
6. **PIT bin count**: 20 is conservative; with n=795k you can use 50 and
   still have ~16k points/bin. Either is fine for visual; use 20 for the
   small-multiples plot to avoid clutter.
7. **CRPS positivity**: by construction CRPS ≥ 0. If you get negatives,
   the formula has a sign bug — check against `scoringrules` library or
   the closed-form derivation.

---

## 8. What the paper section will look like

Section 4.X "Probabilistic calibration":

> Beyond binary 1σ/2σ coverage tests, we compute the full reliability
> diagram, Expected Calibration Error (ECE), and Continuous Ranked
> Probability Score (CRPS) for all four model classes (Fig. X). The MLP
> achieves ECE = A%, vs B% (HA) and C% (NS); mean CRPS = D mm vs E mm
> (HA) and F mm (NS); and PIT histograms (Fig. Y) show near-uniformity
> for the MLP (KS p = G) but skewed distributions for HA/NS (p < 1e-50),
> indicating that the MLP's predictive σ is well-calibrated whereas the
> physics baselines' uncertainty estimates are mis-specified.

The numbers A–G come straight from `summary.csv` after this script runs.

---

## 9. Files to read first

1. `MLP/eval/inference_rmse_on_series.py` lines 48–63 (`_finite_metrics`),
   165–375 (`run_rmse_evaluation`) — to re-run with save_points=True and
   understand the points.csv schema.
2. `MLP/eval/eval_new_winners_vs_baselines.py` (just created) — driver
   for evaluating the 5 winners; modify to set `save_points=True`.
3. `MLP/baseline/comparison_reports/stage3_vs_HA_NS_20260509/overall_metrics.csv`
   — the model paths and existing summary numbers to validate against.
4. `MLP/baseline/Hiroyasu_Arai/` and `MLP/baseline/Naber_Siebers/` — to
   locate per-point σ for the physics baselines.
5. `MLP/runs_mlp/full_pipeline_C_20260509_110100/bootstrap_summary.json` —
   the 5 new winner_run_dir paths.

---

## 10. Outputs the parent conversation needs back

A single markdown table with:
- Per-model: ECE, CRPS_mean, sharpness, PIT_KS_pvalue.
- Aggregate row for the 5 new MLPs (mean ± std).
- One paragraph verdict on whether MLP-σ is meaningfully better-calibrated
  than HA/NS-σ.

Plus the three figures (reliability overlay, CRPS-sharpness, PIT histograms).
