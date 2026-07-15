"""Canonical metric library for probabilistic penetration evaluation.

Single source of truth replacing the ~9 independent copies of the point-metric
dict found across ``MLP/eval``, ``MLP/GP_training`` and ``MLP/baseline``.

Conventions
-----------
* truth / mu / sigma are 1-D float arrays in physical millimetres.
* Every deterministic metric key keeps the historical ``_mm`` suffix so the
  outputs stay join-compatible with the legacy artifacts.
* ``sigma`` is floored at ``SIGMA_FLOOR`` (1e-9 mm) exactly once, here, instead
  of the three different epsilons (1e-6/1e-9/1e-12) used by the legacy scripts.
* NLL is the *physical-space* Gaussian log score including the log(2*pi)
  constant (the ``nll_physical`` variant); the scaled-space GP variant is not
  reproduced because its values are not comparable across model kinds.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

SIGMA_FLOOR = 1e-9

#: One-sided PIT probability levels for the reliability curve / fixed-level ECE
#: (thesis parity: 19 levels over [0.025, 0.975]).
PROBABILITY_LEVELS = np.linspace(0.025, 0.975, 19)

#: PIT histogram bin edges (thesis parity: 20 equal bins).
PIT_HIST_BINS = np.linspace(0.0, 1.0, 21)

#: Central-interval nominal coverage levels reported as coverage_central_XX.
CENTRAL_COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95)

#: k*sigma grid for the empirical-vs-Gaussian coverage curve.
K_SIGMA_GRID = np.linspace(0.25, 3.0, 23)

#: Predictive-quantile levels scored with pinball loss.
PINBALL_QUANTILES = (0.10, 0.50, 0.90)

#: Central intervals scored with the Winkler / interval score.
INTERVAL_SCORE_LEVELS = (0.50, 0.90)


def _as_float(x: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def safe_sigma(sigma: np.ndarray) -> np.ndarray:
    return np.maximum(_as_float(sigma), SIGMA_FLOOR)


def zscores(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (_as_float(truth) - _as_float(mu)) / safe_sigma(sigma)


def compute_pit(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Gaussian PIT, clipped away from {0,1} for downstream log-safety."""
    pit = stats.norm.cdf(zscores(truth, mu, sigma))
    return np.clip(pit, 1e-9, 1.0 - 1e-9)


def crps_gaussian(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form Gaussian CRPS (Gneiting & Raftery 2007), per point, in mm."""
    sig = safe_sigma(sigma)
    z = (_as_float(truth) - _as_float(mu)) / sig
    return sig * (z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / math.sqrt(math.pi))


def nll_gaussian(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Per-point Gaussian negative log-likelihood in physical mm space."""
    sig = safe_sigma(sigma)
    z = (_as_float(truth) - _as_float(mu)) / sig
    return 0.5 * (math.log(2.0 * math.pi) + 2.0 * np.log(sig) + z**2)


def reliability_curve(pit: np.ndarray, levels: np.ndarray = PROBABILITY_LEVELS) -> np.ndarray:
    """Empirical lower-tail fraction P(PIT <= alpha) at each nominal alpha."""
    pit = _as_float(pit)
    return np.array([np.mean(pit <= alpha) for alpha in levels], dtype=float)


def ece_fixed_levels(pit: np.ndarray, levels: np.ndarray = PROBABILITY_LEVELS) -> float:
    """Mean |empirical - nominal| over fixed probability levels (thesis 'ece')."""
    return float(np.mean(np.abs(reliability_curve(pit, levels) - levels)))


def pit_hist_tvd(pit: np.ndarray, bins: np.ndarray = PIT_HIST_BINS) -> float:
    """Total-variation distance of the PIT histogram from uniform.

    Population-weighted calibration scalar complementing the fixed-level ECE:
    0 = perfectly uniform PIT, 1 = all mass in one bin.
    """
    counts, _ = np.histogram(_as_float(pit), bins=bins)
    frac = counts / max(counts.sum(), 1)
    uniform = 1.0 / (len(bins) - 1)
    return float(0.5 * np.sum(np.abs(frac - uniform)))


def spiegelhalter_z(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Spiegelhalter's calibration Z statistic; ~N(0,1) under perfect calibration."""
    z2 = zscores(truth, mu, sigma) ** 2
    n = z2.size
    if n == 0:
        return float("nan")
    return float(np.sum(z2 - 1.0) / math.sqrt(2.0 * n))


def interval_score(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray, level: float) -> float:
    """Mean Winkler interval score of the central ``level`` Gaussian interval (mm)."""
    alpha = 1.0 - level
    sig = safe_sigma(sigma)
    y = _as_float(truth)
    half = sig * stats.norm.ppf(1.0 - alpha / 2.0)
    lower = _as_float(mu) - half
    upper = _as_float(mu) + half
    width = upper - lower
    penalty = (2.0 / alpha) * ((lower - y) * (y < lower) + (y - upper) * (y > upper))
    return float(np.mean(width + penalty))


def pinball_loss(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray, q: float) -> float:
    """Mean pinball loss of the Gaussian predictive quantile at level ``q`` (mm)."""
    pred_q = _as_float(mu) + safe_sigma(sigma) * stats.norm.ppf(q)
    diff = _as_float(truth) - pred_q
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def central_coverage(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray, level: float) -> float:
    """Empirical coverage of the central Gaussian interval at nominal ``level``."""
    k = stats.norm.ppf(0.5 + level / 2.0)
    abs_z = np.abs(zscores(truth, mu, sigma))
    return float(np.mean(abs_z <= k))


def coverage_curve(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                   k_grid: np.ndarray = K_SIGMA_GRID) -> pd.DataFrame:
    """Empirical |z|<=k coverage vs the Gaussian nominal erf(k/sqrt(2))."""
    abs_z = np.abs(zscores(truth, mu, sigma))
    rows = []
    for k in np.asarray(k_grid, dtype=float):
        nominal = math.erf(k / math.sqrt(2.0))
        empirical = float(np.mean(abs_z <= k))
        rows.append({
            "k_sigma": float(k),
            "nominal_gaussian_coverage": nominal,
            "empirical_coverage": empirical,
            "coverage_error": empirical - nominal,
        })
    return pd.DataFrame(rows)


def point_metrics(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                  *, rel_err_floor_mm: float = 5.0) -> dict[str, float | int]:
    """Deterministic accuracy + headline uncertainty metrics.

    Superset of the legacy ``_finite_metrics`` dict — legacy keys keep their
    exact names/formulas so historical numbers remain reproducible.
    """
    truth = _as_float(truth)
    mu = _as_float(mu)
    sigma = _as_float(sigma)
    n = truth.size
    if n == 0:
        return {"n_points": 0}
    resid = mu - truth
    abs_err = np.abs(resid)
    sig = safe_sigma(sigma)
    truth_min = float(np.min(truth))
    truth_max = float(np.max(truth))
    truth_range = truth_max - truth_min
    rmse = float(np.sqrt(np.mean(resid**2)))
    rel_denom = np.maximum(np.abs(truth), rel_err_floor_mm)
    out: dict[str, float | int] = {
        "n_points": int(n),
        "rmse_mm": rmse,
        "mae_mm": float(np.mean(abs_err)),
        "bias_mm": float(np.mean(resid)),
        "median_abs_err_mm": float(np.median(abs_err)),
        "p90_abs_err_mm": float(np.quantile(abs_err, 0.90)),
        "p95_abs_err_mm": float(np.quantile(abs_err, 0.95)),
        "max_abs_err_mm": float(np.max(abs_err)),
        "coverage_1sigma": float(np.mean(abs_err <= sig)),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * sig)),
        "coverage_3sigma": float(np.mean(abs_err <= 3.0 * sig)),
        "mean_pred_std_mm": float(np.mean(sigma)),
        "median_pred_std_mm": float(np.median(sigma)),
        "p25_pred_std_mm": float(np.quantile(sigma, 0.25)),
        "p75_pred_std_mm": float(np.quantile(sigma, 0.75)),
        "mean_rel_err": float(np.mean(abs_err / rel_denom)),
        "median_rel_err": float(np.median(abs_err / rel_denom)),
        "truth_min_mm": truth_min,
        "truth_max_mm": truth_max,
        "truth_range_mm": truth_range,
        "nrmse_range": rmse / truth_range if truth_range > 0 else float("nan"),
    }
    for level in CENTRAL_COVERAGE_LEVELS:
        out[f"coverage_central_{int(round(level * 100)):02d}"] = central_coverage(truth, mu, sigma, level)
    return out


def probabilistic_metrics(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                          *, weights: np.ndarray | None = None) -> dict[str, float]:
    """Distributional scoring: NLL, CRPS, PIT calibration, interval scores.

    ``weights`` (optional, e.g. ``n_points_in_p50_bin``) produces the weighted
    variant used for the aggregated P50 eval set; weighted keys carry a
    ``_weighted`` suffix and are merged by the caller.

    Caveat: ``pit_ks_statistic``/``pit_ks_pvalue`` run the standard
    (i.i.d.-assuming) one-sample KS test on the flat per-point PIT array.
    Points within one trajectory share correlated bias, so the effective
    sample size is smaller than ``n_points`` and the reported p-value is
    optimistic (overstates significance of miscalibration) on point tables
    with many points per trajectory — read it as a descriptive diagnostic,
    not a calibrated significance test.
    """
    truth = _as_float(truth)
    mu = _as_float(mu)
    sigma = _as_float(sigma)
    if truth.size == 0:
        return {}
    pit = compute_pit(truth, mu, sigma)
    crps = crps_gaussian(truth, mu, sigma)
    nll = nll_gaussian(truth, mu, sigma)
    abs_z = np.abs(zscores(truth, mu, sigma))
    ks = stats.kstest(pit, "uniform")
    out: dict[str, float] = {
        "nll_mm": float(np.mean(nll)),
        "crps_mean_mm": float(np.mean(crps)),
        "crps_std_mm": float(np.std(crps)),
        "crps_median_mm": float(np.median(crps)),
        "crps_p90_mm": float(np.quantile(crps, 0.90)),
        "crps_p95_mm": float(np.quantile(crps, 0.95)),
        "ece": ece_fixed_levels(pit),
        "pit_hist_tvd": pit_hist_tvd(pit),
        "pit_ks_statistic": float(ks.statistic),
        "pit_ks_pvalue": float(ks.pvalue),
        "sharpness_mm": float(np.mean(sigma)),
        "mean_abs_z": float(np.mean(abs_z)),
        "z_mean": float(np.mean(zscores(truth, mu, sigma))),
        "z_var": float(np.var(zscores(truth, mu, sigma))),
        "spiegelhalter_z": spiegelhalter_z(truth, mu, sigma),
    }
    for level in INTERVAL_SCORE_LEVELS:
        out[f"interval_score_{int(round(level * 100)):02d}_mm"] = interval_score(truth, mu, sigma, level)
    for q in PINBALL_QUANTILES:
        out[f"pinball_q{int(round(q * 100)):02d}_mm"] = pinball_loss(truth, mu, sigma, q)

    if weights is not None:
        w = _as_float(weights)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if w.sum() > 0:
            wn = w / w.sum()
            rel_w = np.array([np.sum(wn * (pit <= a)) for a in PROBABILITY_LEVELS])
            out.update({
                "ece_weighted": float(np.mean(np.abs(rel_w - PROBABILITY_LEVELS))),
                "crps_mean_mm_weighted": float(np.sum(wn * crps)),
                "nll_mm_weighted": float(np.sum(wn * nll)),
                "sharpness_mm_weighted": float(np.sum(wn * sigma)),
                "coverage_1sigma_weighted": float(np.sum(wn * (abs_z <= 1.0))),
                "coverage_2sigma_weighted": float(np.sum(wn * (abs_z <= 2.0))),
                "weight_sum": float(w.sum()),
            })
    return out


def pit_histogram(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                  *, weights: np.ndarray | None = None,
                  bins: np.ndarray = PIT_HIST_BINS) -> pd.DataFrame:
    pit = compute_pit(truth, mu, sigma)
    counts, edges = np.histogram(pit, bins=bins)
    df = pd.DataFrame({
        "pit_bin_left": edges[:-1],
        "pit_bin_right": edges[1:],
        "count": counts,
        "fraction": counts / max(counts.sum(), 1),
    })
    if weights is not None:
        w = _as_float(weights)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        wcounts, _ = np.histogram(pit, bins=bins, weights=w)
        df["weighted_fraction"] = wcounts / max(wcounts.sum(), 1e-12)
    return df


def reliability_table(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                      *, weights: np.ndarray | None = None,
                      levels: np.ndarray = PROBABILITY_LEVELS) -> pd.DataFrame:
    pit = compute_pit(truth, mu, sigma)
    empirical = reliability_curve(pit, levels)
    df = pd.DataFrame({
        "probability_level": levels,
        "empirical_lower_tail_fraction": empirical,
        "abs_calibration_error": np.abs(empirical - levels),
    })
    if weights is not None:
        w = _as_float(weights)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if w.sum() > 0:
            wn = w / w.sum()
            emp_w = np.array([np.sum(wn * (pit <= a)) for a in levels])
            df["empirical_lower_tail_fraction_weighted"] = emp_w
            df["abs_calibration_error_weighted"] = np.abs(emp_w - levels)
    return df


def sigma_bin_calibration(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                          *, n_bins: int = 10) -> pd.DataFrame:
    """Per-sigma-decile coverage/error audit (legacy calibration_coverage_audit)."""
    truth = _as_float(truth)
    mu = _as_float(mu)
    sigma = _as_float(sigma)
    resid = mu - truth
    abs_err = np.abs(resid)
    sig = safe_sigma(sigma)
    df = pd.DataFrame({"sigma": sigma, "abs_err": abs_err, "abs_z": abs_err / sig, "resid": resid})
    try:
        df["bin"] = pd.qcut(df["sigma"], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["sigma"], bins=min(n_bins, 3))
    rows = []
    for interval, group in df.groupby("bin", observed=True):
        if group.empty:
            continue
        rows.append({
            "sigma_bin": str(interval),
            "n": int(len(group)),
            "mean_sigma_mm": float(group["sigma"].mean()),
            "median_sigma_mm": float(group["sigma"].median()),
            "mae_mm": float(group["abs_err"].mean()),
            "rmse_mm": float(np.sqrt(np.mean(group["resid"] ** 2))),
            "coverage_1sigma": float(np.mean(group["abs_z"] <= 1.0)),
            "coverage_2sigma": float(np.mean(group["abs_z"] <= 2.0)),
            "mean_abs_z": float(group["abs_z"].mean()),
            "p90_abs_z": float(group["abs_z"].quantile(0.90)),
        })
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, stat_fn, *, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 20260521) -> tuple[float, float]:
    """Seeded percentile bootstrap CI of ``stat_fn`` over point-level values.

    Case-resamples individual points, i.e. assumes they are i.i.d. Points
    within one trajectory/condition share correlated bias, so this
    understates uncertainty on point tables with many points per trajectory
    — prefer :func:`cluster_bootstrap_ci` when a cluster id is available.
    """
    values = _as_float(values)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats_ = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = values[rng.integers(0, values.size, values.size)]
        stats_[i] = stat_fn(sample)
    return float(np.quantile(stats_, alpha / 2.0)), float(np.quantile(stats_, 1.0 - alpha / 2.0))


def _ci_pair(values: np.ndarray, alpha: float) -> tuple[float, float]:
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def _chunk_sizes(n_boot: int, chunk_size: int) -> list[int]:
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1.")
    chunk_size = max(1, min(int(chunk_size), n_boot))
    chunks = [chunk_size] * (n_boot // chunk_size)
    remainder = n_boot % chunk_size
    if remainder:
        chunks.append(remainder)
    return chunks


def _spawn_int_seeds(seed: int, n: int) -> list[int]:
    seq = np.random.SeedSequence(seed)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in seq.spawn(n)]


def bootstrap_headline_ci(values: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05,
                          seed: int = 20260521, n_workers: int = 1,
                          chunk_size: int = 128) -> dict[str, tuple[float, float]]:
    """Vectorized point-level bootstrap CIs for RMSE/MAE/bias.

    This is a fallback for tables without a cluster key. It reuses one set of
    resamples for all three headline metrics and processes draws in chunks to
    avoid one massive ``n_boot x n_points`` allocation.
    """
    values = _as_float(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        nan = (float("nan"), float("nan"))
        return {"rmse_mm": nan, "mae_mm": nan, "bias_mm": nan}

    chunks = _chunk_sizes(n_boot, chunk_size)
    seeds = _spawn_int_seeds(seed, len(chunks))
    n_workers = max(1, int(n_workers))

    def draw(n_draws: int, draw_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(draw_seed)
        idx = rng.integers(0, values.size, size=(n_draws, values.size), dtype=np.intp)
        sample = values[idx]
        rmse = np.sqrt(np.mean(sample**2, axis=1))
        mae = np.mean(np.abs(sample), axis=1)
        bias = np.mean(sample, axis=1)
        return rmse, mae, bias

    if n_workers == 1 or len(chunks) == 1:
        blocks = [draw(n, s) for n, s in zip(chunks, seeds)]
    else:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(chunks))) as pool:
            blocks = list(pool.map(lambda args: draw(*args), zip(chunks, seeds)))

    rmse = np.concatenate([b[0] for b in blocks])
    mae = np.concatenate([b[1] for b in blocks])
    bias = np.concatenate([b[2] for b in blocks])
    return {
        "rmse_mm": _ci_pair(rmse, alpha),
        "mae_mm": _ci_pair(mae, alpha),
        "bias_mm": _ci_pair(bias, alpha),
    }


def cluster_bootstrap_headline_ci(values: np.ndarray, cluster_ids: np.ndarray, *,
                                  n_boot: int = 2000, alpha: float = 0.05,
                                  seed: int = 20260521, n_workers: int = 1,
                                  chunk_size: int = 128) -> dict[str, tuple[float, float]]:
    """Fast cluster bootstrap CIs for RMSE/MAE/bias.

    The slow generic implementation pooled every point from every sampled
    cluster for every metric. For these three headline metrics, each cluster can
    be reduced to sufficient statistics: n, sum(resid), sum(|resid|), and
    sum(resid^2). Bootstrap draws then only sum per-cluster statistics, while
    preserving the same "resample whole trajectories/conditions" semantics.
    """
    values = _as_float(values)
    clusters = np.asarray(cluster_ids).reshape(-1)
    if values.size == 0 or clusters.size != values.size:
        nan = (float("nan"), float("nan"))
        return {"rmse_mm": nan, "mae_mm": nan, "bias_mm": nan}

    codes, _ = pd.factorize(clusters, sort=False)
    mask = (codes >= 0) & np.isfinite(values)
    if not np.any(mask):
        nan = (float("nan"), float("nan"))
        return {"rmse_mm": nan, "mae_mm": nan, "bias_mm": nan}
    values = values[mask]
    codes = codes[mask]

    n_clusters = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=n_clusters).astype(float)
    sum_resid = np.bincount(codes, weights=values, minlength=n_clusters)
    sum_abs_resid = np.bincount(codes, weights=np.abs(values), minlength=n_clusters)
    sum_sq_resid = np.bincount(codes, weights=values**2, minlength=n_clusters)

    nonempty = counts > 0
    counts = counts[nonempty]
    sum_resid = sum_resid[nonempty]
    sum_abs_resid = sum_abs_resid[nonempty]
    sum_sq_resid = sum_sq_resid[nonempty]
    n_clusters = counts.size
    if n_clusters == 0:
        nan = (float("nan"), float("nan"))
        return {"rmse_mm": nan, "mae_mm": nan, "bias_mm": nan}

    chunks = _chunk_sizes(n_boot, chunk_size)
    seeds = _spawn_int_seeds(seed, len(chunks))
    n_workers = max(1, int(n_workers))

    def draw(n_draws: int, draw_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(draw_seed)
        idx = rng.integers(0, n_clusters, size=(n_draws, n_clusters), dtype=np.intp)
        boot_n = counts[idx].sum(axis=1)
        rmse = np.sqrt(sum_sq_resid[idx].sum(axis=1) / boot_n)
        mae = sum_abs_resid[idx].sum(axis=1) / boot_n
        bias = sum_resid[idx].sum(axis=1) / boot_n
        return rmse, mae, bias

    if n_workers == 1 or len(chunks) == 1:
        blocks = [draw(n, s) for n, s in zip(chunks, seeds)]
    else:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(chunks))) as pool:
            blocks = list(pool.map(lambda args: draw(*args), zip(chunks, seeds)))

    rmse = np.concatenate([b[0] for b in blocks])
    mae = np.concatenate([b[1] for b in blocks])
    bias = np.concatenate([b[2] for b in blocks])
    return {
        "rmse_mm": _ci_pair(rmse, alpha),
        "mae_mm": _ci_pair(mae, alpha),
        "bias_mm": _ci_pair(bias, alpha),
    }


def cluster_bootstrap_ci(values: np.ndarray, cluster_ids: np.ndarray, stat_fn, *,
                         n_boot: int = 2000, alpha: float = 0.05,
                         seed: int = 20260521) -> tuple[float, float]:
    """Percentile bootstrap CI resampling whole clusters (e.g. trajectories).

    Points sharing one ``traj_key``/``condition_id`` carry correlated
    residual bias, so resampling individual points (:func:`bootstrap_ci`)
    overstates the effective sample size and understates the true CI width.
    This resamples cluster ids with replacement and pools every point of
    each drawn cluster, matching the independence unit the data actually has.
    """
    values = _as_float(values)
    clusters = np.asarray(cluster_ids).reshape(-1)
    if values.size == 0 or clusters.size != values.size:
        return float("nan"), float("nan")
    unique_clusters = pd.unique(clusters)
    if unique_clusters.size == 0:
        return float("nan"), float("nan")
    by_cluster = {c: values[clusters == c] for c in unique_clusters}
    rng = np.random.default_rng(seed)
    stats_ = np.empty(n_boot, dtype=float)
    n_clusters = unique_clusters.size
    for i in range(n_boot):
        drawn = unique_clusters[rng.integers(0, n_clusters, n_clusters)]
        sample = np.concatenate([by_cluster[c] for c in drawn])
        stats_[i] = stat_fn(sample)
    return float(np.quantile(stats_, alpha / 2.0)), float(np.quantile(stats_, 1.0 - alpha / 2.0))


def grouped_point_metrics(points: pd.DataFrame, group_cols: list[str],
                          *, rel_err_floor_mm: float = 5.0) -> pd.DataFrame:
    """Apply :func:`point_metrics` per group of a canonical points table."""
    usable = [c for c in group_cols if c in points.columns]
    if not usable or points.empty:
        return pd.DataFrame()
    rows = []
    for key, group in points.groupby(usable, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = dict(zip(usable, values))
        row.update(point_metrics(
            group["pen_true_mm"].to_numpy(),
            group["pen_pred_mm"].to_numpy(),
            group["pen_std_mm"].to_numpy(),
            rel_err_floor_mm=rel_err_floor_mm,
        ))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_points(points: pd.DataFrame, *, rel_err_floor_mm: float = 5.0,
                     weight_col: str | None = None,
                     probabilistic: bool = True) -> dict[str, Any]:
    """Overall metric dict (deterministic + probabilistic) for a points table."""
    truth = points["pen_true_mm"].to_numpy()
    mu = points["pen_pred_mm"].to_numpy()
    sigma = points["pen_std_mm"].to_numpy()
    out: dict[str, Any] = point_metrics(truth, mu, sigma, rel_err_floor_mm=rel_err_floor_mm)
    if "condition_id" in points.columns:
        out["n_conditions"] = int(points["condition_id"].nunique())
    if "traj_key" in points.columns:
        out["n_trajectories"] = int(points["traj_key"].nunique())
    if probabilistic:
        weights = None
        if weight_col and weight_col in points.columns:
            weights = pd.to_numeric(points[weight_col], errors="coerce").to_numpy()
        out.update(probabilistic_metrics(truth, mu, sigma, weights=weights))
    return out
