"""Audit FOV right-censoring threshold and Hampel spike rate on cleaned traces.

Phase-1.1 sanity check before refactoring fit_raw_data. Reads existing
``series_wide_clean/*.csv`` and reports three things:

1. Per-nozzle distribution of ``last_valid_penetration_mm`` and end-slope, used
   to validate the proposed rule
   ``is_right_censored = (last_pen > 85 mm) AND (|end_slope| ~ 0)``.
2. Per-nozzle Hampel filter spike rate on the already-cleaned penetration
   series, used to validate the Hampel window/k parameters
   (window ~ 0.1 ms, k = 3.0).
3. Point-level Stage-3 supervision audit: sweep ``pen_cutoff in
   {flat 80 mm, 0.85·cap_nozzle}`` × ``slope_ratio_min in {0.3, 0.5}`` and
   count surviving raw_reliable points per nozzle / per (nozzle, folder).
   The slope ratio is local-smoothed dP/dt divided by the expected power-law
   slope ``0.25·k_quarter·(t-t0)^(-0.75)``; k_quarter and t0 are fit per
   trace on its lower band (``pen < early_fit_frac · cap_nozzle``).

Outputs go to ``--out-dir`` (default ``<data-root>/fov_spike_audit/``):
   - ``last_pen_hist_<nozzle>.png``           per-nozzle histogram of last_pen
   - ``last_pen_vs_slope_<nozzle>.png``       scatter coloured by proposed flag
   - ``hampel_spike_distribution.png``        spikes-per-trace per nozzle
   - ``audit_summary.csv``                    one row per nozzle (trace level)
   - ``threshold_sensitivity.csv``            sweep over pen / slope thresholds
   - ``pointwise_cutoff_audit.csv``           per-nozzle × (rule, ratio)
   - ``pointwise_cutoff_audit_by_cond.csv``   per-(nozzle, folder) × (rule, ratio)
   - ``pointwise_raw_reliable_frac.png``      bar chart summary
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "MLP" / "synthetic_data_20260509"
DEFAULT_SOURCE = "cdf"
DEFAULT_OUT_DIR_NAME = "fov_spike_audit"

PEN_FOV_DEFAULT_MM = 85.0
SLOPE_EPS_DEFAULT_MM_PER_MS = 2.0
HAMPEL_WINDOW_DEFAULT_MS = 0.1
HAMPEL_K_DEFAULT = 3.0

PEN_THRESHOLD_SWEEP_MM = (75.0, 80.0, 85.0, 90.0, 95.0)
SLOPE_THRESHOLD_SWEEP_MM_PER_MS = (1.0, 2.0, 5.0, 10.0)

FLAT_PEN_CUTOFF_MM_DEFAULT = 80.0
CAP_FRACTION_DEFAULT = 0.85
CAP_PERCENTILE_DEFAULT = 99.0
EARLY_FIT_FRAC_DEFAULT = 0.4
SLOPE_WINDOW_FRAMES_DEFAULT = 5
SLOPE_RATIO_THRESHOLDS_DEFAULT = (0.3, 0.5)
MIN_PEN_FOR_FIT_MM = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                   help="Root containing <Nozzle>/cdf/series_wide_clean/*.csv.")
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help="Penetration source subfolder (cdf / bw_x / bw_polar).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: <data-root>/fov_spike_audit).")
    p.add_argument("--pen-fov-mm", type=float, default=PEN_FOV_DEFAULT_MM,
                   help="Proposed FOV penetration threshold (mm).")
    p.add_argument("--slope-eps-mm-per-ms", type=float, default=SLOPE_EPS_DEFAULT_MM_PER_MS,
                   help="Proposed |end_slope| threshold (mm/ms).")
    p.add_argument("--hampel-window-ms", type=float, default=HAMPEL_WINDOW_DEFAULT_MS,
                   help="Hampel centred window half-width in ms.")
    p.add_argument("--hampel-k", type=float, default=HAMPEL_K_DEFAULT,
                   help="Hampel k * 1.4826 * MAD threshold multiplier.")
    p.add_argument("--end-slope-frames", type=int, default=4,
                   help="Number of final valid frames used for end-slope regression.")
    p.add_argument("--flat-pen-cutoff-mm", type=float, default=FLAT_PEN_CUTOFF_MM_DEFAULT,
                   help="Flat pen cutoff (mm) for the aggressive Stage-3 filter.")
    p.add_argument("--cap-fraction", type=float, default=CAP_FRACTION_DEFAULT,
                   help="Fraction-of-cap cutoff for the per-nozzle rule.")
    p.add_argument("--cap-percentile", type=float, default=CAP_PERCENTILE_DEFAULT,
                   help="Percentile of last_valid_pen_mm used as cap_nozzle.")
    p.add_argument("--early-fit-frac", type=float, default=EARLY_FIT_FRAC_DEFAULT,
                   help="Upper bound (as fraction of cap) of the clean band used to fit "
                        "k_quarter/t0; points with pen below this band are always kept.")
    p.add_argument("--slope-window-frames", type=int, default=SLOPE_WINDOW_FRAMES_DEFAULT,
                   help="Frame width for the rolling mean smoothing of local dP/dt.")
    p.add_argument("--slope-ratios", type=str,
                   default=",".join(f"{r:g}" for r in SLOPE_RATIO_THRESHOLDS_DEFAULT),
                   help="Comma-separated slope-ratio thresholds to evaluate.")
    return p.parse_args()


def infer_nozzle_label(dataset_dir_name: str) -> str:
    m = re.search(r"Nozzle(\d+)", dataset_dir_name)
    return f"Nozzle{m.group(1)}" if m else dataset_dir_name


# ─────────────────────────── per-trace measurements ───────────────────────────


def extract_endpoint_features(
    times_ms: np.ndarray,
    pens_mm: np.ndarray,
    end_slope_frames: int,
) -> dict[str, float]:
    """Return last-valid pen, last-valid time, and end-of-trace slope (mm/ms)."""
    valid = np.isfinite(times_ms) & np.isfinite(pens_mm)
    if valid.sum() < 2:
        return {
            "last_valid_pen_mm": np.nan,
            "last_valid_t_ms": np.nan,
            "end_slope_mm_per_ms": np.nan,
            "n_valid": int(valid.sum()),
        }
    t = times_ms[valid]
    p = pens_mm[valid]
    last_pen = float(p[-1])
    last_t = float(t[-1])
    tail_n = int(min(end_slope_frames, len(p)))
    if tail_n < 2:
        slope = np.nan
    else:
        t_tail = t[-tail_n:]
        p_tail = p[-tail_n:]
        slope, _ = np.polyfit(t_tail, p_tail, 1)
    return {
        "last_valid_pen_mm": last_pen,
        "last_valid_t_ms": last_t,
        "end_slope_mm_per_ms": float(slope),
        "n_valid": int(valid.sum()),
    }


def hampel_spike_mask(
    times_ms: np.ndarray,
    pens_mm: np.ndarray,
    window_ms: float,
    k: float,
) -> np.ndarray:
    """Centred Hampel filter on a single trace. Returns bool mask of spikes."""
    valid = np.isfinite(times_ms) & np.isfinite(pens_mm)
    n = len(pens_mm)
    mask = np.zeros(n, dtype=bool)
    if valid.sum() < 5:
        return mask
    idx_valid = np.flatnonzero(valid)
    t = times_ms[valid]
    p = pens_mm[valid]
    for j, i_full in enumerate(idx_valid):
        t_i = t[j]
        in_win = (t >= t_i - window_ms) & (t <= t_i + window_ms)
        if in_win.sum() < 3:
            continue
        window = p[in_win]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        scale = 1.4826 * mad
        if scale <= 0:
            continue
        if abs(p[j] - med) > k * scale:
            mask[i_full] = True
    return mask


# ───────────────────────────── per-nozzle pipeline ─────────────────────────────


def collect_traces(
    data_root: Path,
    source: str,
    end_slope_frames: int,
    hampel_window_ms: float,
    hampel_k: float,
) -> pd.DataFrame:
    """Walk all nozzle dirs, return one row per trace with audit features."""
    rows: list[dict] = []
    nozzle_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("BC")])
    for nozdir in nozzle_dirs:
        series_dir = nozdir / source / "series_wide_clean"
        if not series_dir.is_dir():
            continue
        nozzle = infer_nozzle_label(nozdir.name)
        for csv_path in sorted(series_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            time_cols = sorted([c for c in df.columns if c.startswith("time_ms_")])
            pen_cols = sorted([c for c in df.columns if c.startswith("penetration_mm_")])
            if not time_cols or not pen_cols:
                continue
            times_mat = df[time_cols].to_numpy(dtype=float)
            pens_mat = df[pen_cols].to_numpy(dtype=float)
            for i in range(len(df)):
                t_row = times_mat[i]
                p_row = pens_mat[i]
                feats = extract_endpoint_features(t_row, p_row, end_slope_frames)
                spike_mask = hampel_spike_mask(t_row, p_row, hampel_window_ms, hampel_k)
                n_finite = int((np.isfinite(t_row) & np.isfinite(p_row)).sum())
                rows.append({
                    "nozzle": nozzle,
                    "dataset_dir": nozdir.name,
                    "folder": csv_path.stem,
                    "row_in_file": i,
                    "n_valid_frames": n_finite,
                    "n_hampel_spikes": int(spike_mask.sum()),
                    "hampel_spike_rate": (int(spike_mask.sum()) / n_finite) if n_finite else np.nan,
                    **feats,
                })
    if not rows:
        raise SystemExit(f"No series_wide_clean files found under {data_root}")
    return pd.DataFrame(rows)


def censoring_rate(
    traces: pd.DataFrame,
    pen_threshold_mm: float,
    slope_threshold_mm_per_ms: float,
) -> pd.Series:
    is_cens = (
        traces["last_valid_pen_mm"].gt(pen_threshold_mm)
        & traces["end_slope_mm_per_ms"].abs().lt(slope_threshold_mm_per_ms)
    )
    return is_cens


# ───────────────────────────────── plotting ─────────────────────────────────


def plot_last_pen_hist(traces: pd.DataFrame, out_dir: Path, pen_fov_mm: float) -> None:
    nozzles = sorted(traces["nozzle"].unique())
    n = len(nozzles)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax, nozzle in zip(axes.ravel(), nozzles):
        sub = traces.loc[traces["nozzle"] == nozzle, "last_valid_pen_mm"].dropna()
        ax.hist(sub, bins=40, color="steelblue", edgecolor="white")
        ax.axvline(pen_fov_mm, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"P_FOV={pen_fov_mm:g} mm")
        ax.set_title(f"{nozzle}  (n={len(sub)})")
        ax.set_xlabel("last_valid_pen_mm")
        ax.set_ylabel("count")
        ax.legend(fontsize=8, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Per-nozzle distribution of last valid penetration", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "last_pen_hist_grid.png", dpi=140)
    plt.close(fig)


def plot_last_pen_vs_slope(
    traces: pd.DataFrame,
    out_dir: Path,
    pen_fov_mm: float,
    slope_eps: float,
) -> None:
    nozzles = sorted(traces["nozzle"].unique())
    n = len(nozzles)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows), squeeze=False)
    for ax, nozzle in zip(axes.ravel(), nozzles):
        sub = traces.loc[traces["nozzle"] == nozzle].dropna(
            subset=["last_valid_pen_mm", "end_slope_mm_per_ms"]
        )
        is_cens = censoring_rate(sub, pen_fov_mm, slope_eps)
        ax.scatter(sub.loc[~is_cens, "last_valid_pen_mm"],
                   sub.loc[~is_cens, "end_slope_mm_per_ms"].abs(),
                   s=8, c="steelblue", alpha=0.45, label="not censored")
        ax.scatter(sub.loc[is_cens, "last_valid_pen_mm"],
                   sub.loc[is_cens, "end_slope_mm_per_ms"].abs(),
                   s=10, c="crimson", alpha=0.7, label="proposed censored")
        ax.axvline(pen_fov_mm, color="crimson", linestyle="--", linewidth=0.9)
        ax.axhline(slope_eps, color="crimson", linestyle="--", linewidth=0.9)
        rate = 100.0 * is_cens.mean() if len(sub) else 0.0
        ax.set_title(f"{nozzle}  censored={rate:.1f}%")
        ax.set_xlabel("last_valid_pen_mm")
        ax.set_ylabel("|end_slope| (mm/ms)")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(
        f"Per-nozzle end-of-trace map. Proposed rule: last_pen>{pen_fov_mm:g} mm AND |slope|<{slope_eps:g} mm/ms",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "last_pen_vs_slope_grid.png", dpi=140)
    plt.close(fig)


def plot_hampel_summary(traces: pd.DataFrame, out_dir: Path, window_ms: float, k: float) -> None:
    nozzles = sorted(traces["nozzle"].unique())
    data = [traces.loc[traces["nozzle"] == n, "n_hampel_spikes"].to_numpy() for n in nozzles]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(nozzles)), 4))
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("steelblue")
        body.set_edgecolor("black")
        body.set_alpha(0.6)
    ax.set_xticks(range(1, len(nozzles) + 1))
    ax.set_xticklabels(nozzles, rotation=30, ha="right")
    ax.set_ylabel("Hampel spikes per trace")
    ax.set_title(f"Hampel spike count per trace  (window=±{window_ms:g} ms, k={k:g})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hampel_spike_distribution.png", dpi=140)
    plt.close(fig)


# ─────────────────────────── point-level audit ───────────────────────────────


def fit_quarter_root_local(t_ms: np.ndarray, pen_mm: np.ndarray) -> tuple[float, float]:
    """Fit ``pen = k · (t - t0)^0.25`` by linearising ``pen^4 = k^4·(t - t0)``.

    Returns ``(k_quarter, t0)`` or ``(nan, nan)`` on degenerate fits.
    """
    if len(t_ms) < 5:
        return float("nan"), float("nan")
    y = pen_mm ** 4
    A = np.vstack([t_ms, np.ones_like(t_ms)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope_y, intercept_y = float(coef[0]), float(coef[1])
    if slope_y <= 0:
        return float("nan"), float("nan")
    k_quarter = slope_y ** 0.25
    t0 = -intercept_y / slope_y
    return k_quarter, t0


def point_level_features(
    t_row: np.ndarray,
    p_row: np.ndarray,
    cap_mm: float,
    slope_window_frames: int,
    early_fit_frac: float,
) -> dict | None:
    valid = np.isfinite(t_row) & np.isfinite(p_row)
    if valid.sum() < 10 or not np.isfinite(cap_mm):
        return None
    t = t_row[valid]
    p = p_row[valid]
    order = np.argsort(t)
    t = t[order]
    p = p[order]

    fit_mask = (p >= MIN_PEN_FOR_FIT_MM) & (p < early_fit_frac * cap_mm)
    if fit_mask.sum() < 5:
        return None
    k_quarter, t0 = fit_quarter_root_local(t[fit_mask], p[fit_mask])
    if not np.isfinite(k_quarter):
        return None

    dpdt = np.gradient(p, t)
    w = max(1, int(slope_window_frames))
    if w > 1 and len(dpdt) >= w:
        kernel = np.ones(w) / w
        dpdt_smooth = np.convolve(dpdt, kernel, mode="same")
    else:
        dpdt_smooth = dpdt
    dt = t - t0
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.where(dt > 1e-6, 0.25 * k_quarter * np.power(np.maximum(dt, 1e-6), -0.75), np.nan)
        ratio = np.where(np.isfinite(expected) & (expected > 0), dpdt_smooth / expected, np.nan)
    return {
        "t": t,
        "pen": p,
        "local_slope": dpdt_smooth,
        "expected_slope": expected,
        "slope_ratio": ratio,
        "k_quarter": float(k_quarter),
        "t0": float(t0),
        "cap_mm": float(cap_mm),
    }


def collect_pointwise(
    data_root: Path,
    source: str,
    nozzle_caps: dict[str, float],
    slope_window_frames: int,
    early_fit_frac: float,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    nozzle_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("BC")])
    for nozdir in nozzle_dirs:
        series_dir = nozdir / source / "series_wide_clean"
        if not series_dir.is_dir():
            continue
        nozzle = infer_nozzle_label(nozdir.name)
        cap_mm = nozzle_caps.get(nozzle, float("nan"))
        for csv_path in sorted(series_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            time_cols = sorted([c for c in df.columns if c.startswith("time_ms_")])
            pen_cols = sorted([c for c in df.columns if c.startswith("penetration_mm_")])
            if not time_cols or not pen_cols:
                continue
            times_mat = df[time_cols].to_numpy(dtype=float)
            pens_mat = df[pen_cols].to_numpy(dtype=float)
            for i in range(len(df)):
                feats = point_level_features(
                    times_mat[i], pens_mat[i], cap_mm,
                    slope_window_frames, early_fit_frac,
                )
                if feats is None:
                    continue
                chunks.append(pd.DataFrame({
                    "nozzle": nozzle,
                    "folder": csv_path.stem,
                    "row_in_file": i,
                    "t_ms": feats["t"],
                    "pen_mm": feats["pen"],
                    "local_slope": feats["local_slope"],
                    "expected_slope": feats["expected_slope"],
                    "slope_ratio": feats["slope_ratio"],
                    "cap_mm": feats["cap_mm"],
                    "k_quarter": feats["k_quarter"],
                    "t0_ms": feats["t0"],
                }))
    if not chunks:
        raise SystemExit("No pointwise data collected — check series_wide_clean inputs.")
    return pd.concat(chunks, ignore_index=True)


def _summarise_kept(group: pd.DataFrame, pen_cutoff_mm: float, slope_ratio_min: float,
                    early_fit_frac: float) -> dict:
    cap_mm = float(group["cap_mm"].iloc[0])
    in_pen_band = group["pen_mm"] <= pen_cutoff_mm
    lower_safety = group["pen_mm"] < early_fit_frac * cap_mm
    slope_ok = group["slope_ratio"].ge(slope_ratio_min) | lower_safety
    raw_reliable = in_pen_band & slope_ok
    rr = group[raw_reliable]
    n_total = len(group)
    return {
        "cap_used_mm": cap_mm,
        "pen_cutoff_mm": pen_cutoff_mm,
        "slope_ratio_min": slope_ratio_min,
        "n_points_total": n_total,
        "n_after_pen_cut": int(in_pen_band.sum()),
        "n_raw_reliable": int(raw_reliable.sum()),
        "frac_raw_reliable": float(raw_reliable.mean()) if n_total else 0.0,
        "pen_min_kept_mm": float(rr["pen_mm"].min()) if len(rr) else float("nan"),
        "pen_max_kept_mm": float(rr["pen_mm"].max()) if len(rr) else float("nan"),
        "pen_p50_kept_mm": float(rr["pen_mm"].median()) if len(rr) else float("nan"),
        "pen_p99_kept_mm": float(rr["pen_mm"].quantile(0.99)) if len(rr) else float("nan"),
        "t_min_kept_ms": float(rr["t_ms"].min()) if len(rr) else float("nan"),
        "t_max_kept_ms": float(rr["t_ms"].max()) if len(rr) else float("nan"),
    }


def write_pointwise_cutoff_audit(
    points: pd.DataFrame,
    out_dir: Path,
    flat_cutoff_mm: float,
    cap_fraction: float,
    slope_ratios: tuple[float, ...],
    early_fit_frac: float,
) -> pd.DataFrame:
    rules_per_nozzle: dict[str, list[tuple[str, float]]] = {}
    for nozzle, g in points.groupby("nozzle", dropna=False):
        cap_mm = float(g["cap_mm"].iloc[0])
        rules_per_nozzle[nozzle] = [
            (f"flat_{flat_cutoff_mm:g}mm", flat_cutoff_mm),
            (f"{cap_fraction:g}_cap", cap_fraction * cap_mm),
        ]

    nozzle_rows = []
    for nozzle, g in points.groupby("nozzle", dropna=False):
        for rule_name, pen_cut in rules_per_nozzle[nozzle]:
            for sr in slope_ratios:
                row = _summarise_kept(g, pen_cut, sr, early_fit_frac)
                row.update({"nozzle": nozzle, "pen_cutoff_rule": rule_name})
                nozzle_rows.append(row)
    nozzle_df = pd.DataFrame(nozzle_rows)
    nozzle_df = nozzle_df[[
        "nozzle", "cap_used_mm", "pen_cutoff_rule", "pen_cutoff_mm", "slope_ratio_min",
        "n_points_total", "n_after_pen_cut", "n_raw_reliable", "frac_raw_reliable",
        "pen_min_kept_mm", "pen_p50_kept_mm", "pen_p99_kept_mm", "pen_max_kept_mm",
        "t_min_kept_ms", "t_max_kept_ms",
    ]].sort_values(["nozzle", "pen_cutoff_rule", "slope_ratio_min"])
    nozzle_df.to_csv(out_dir / "pointwise_cutoff_audit.csv", index=False)

    cond_rows = []
    for (nozzle, folder), g in points.groupby(["nozzle", "folder"], dropna=False):
        for rule_name, pen_cut in rules_per_nozzle[nozzle]:
            for sr in slope_ratios:
                row = _summarise_kept(g, pen_cut, sr, early_fit_frac)
                row.update({"nozzle": nozzle, "folder": folder, "pen_cutoff_rule": rule_name})
                cond_rows.append(row)
    cond_df = pd.DataFrame(cond_rows)
    cond_df = cond_df[[
        "nozzle", "folder", "cap_used_mm", "pen_cutoff_rule", "pen_cutoff_mm",
        "slope_ratio_min", "n_points_total", "n_after_pen_cut", "n_raw_reliable",
        "frac_raw_reliable", "pen_min_kept_mm", "pen_max_kept_mm",
        "t_min_kept_ms", "t_max_kept_ms",
    ]].sort_values(["nozzle", "folder", "pen_cutoff_rule", "slope_ratio_min"])
    cond_df.to_csv(out_dir / "pointwise_cutoff_audit_by_cond.csv", index=False)
    return nozzle_df


def plot_raw_reliable_frac(nozzle_df: pd.DataFrame, out_dir: Path) -> None:
    nozzles = sorted(nozzle_df["nozzle"].unique())
    combos = (
        nozzle_df[["pen_cutoff_rule", "slope_ratio_min"]]
        .drop_duplicates()
        .sort_values(["pen_cutoff_rule", "slope_ratio_min"])
        .to_records(index=False)
    )
    n_combos = len(combos)
    width = 0.8 / n_combos
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(nozzles)), 4.2))
    x = np.arange(len(nozzles))
    cmap = plt.get_cmap("tab10")
    for j, (rule, sr) in enumerate(combos):
        vals = []
        for nozzle in nozzles:
            sub = nozzle_df[
                (nozzle_df["nozzle"] == nozzle)
                & (nozzle_df["pen_cutoff_rule"] == rule)
                & (nozzle_df["slope_ratio_min"] == sr)
            ]
            vals.append(float(sub["frac_raw_reliable"].iloc[0]) if len(sub) else 0.0)
        ax.bar(x + (j - (n_combos - 1) / 2) * width, vals, width=width,
               color=cmap(j), label=f"{rule} | ratio≥{sr:g}")
    ax.set_xticks(x)
    ax.set_xticklabels(nozzles, rotation=30, ha="right")
    ax.set_ylabel("frac_raw_reliable")
    ax.set_ylim(0, 1.0)
    ax.set_title("Stage-3 raw_reliable point fraction per nozzle (pen-cut × slope-ratio)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "pointwise_raw_reliable_frac.png", dpi=140)
    plt.close(fig)


# ───────────────────────────────── summaries ─────────────────────────────────


def write_audit_summary(
    traces: pd.DataFrame,
    out_dir: Path,
    pen_fov_mm: float,
    slope_eps: float,
) -> None:
    grouped = traces.groupby("nozzle", dropna=False)
    rows = []
    for nozzle, g in grouped:
        is_cens = censoring_rate(g, pen_fov_mm, slope_eps)
        rows.append({
            "nozzle": nozzle,
            "n_traces": len(g),
            "last_pen_median_mm": float(g["last_valid_pen_mm"].median()),
            "last_pen_p90_mm": float(g["last_valid_pen_mm"].quantile(0.90)),
            "last_pen_p99_mm": float(g["last_valid_pen_mm"].quantile(0.99)),
            "frac_last_pen_gt_fov": float(g["last_valid_pen_mm"].gt(pen_fov_mm).mean()),
            "frac_slope_below_eps": float(g["end_slope_mm_per_ms"].abs().lt(slope_eps).mean()),
            "frac_proposed_censored": float(is_cens.mean()),
            "n_proposed_censored": int(is_cens.sum()),
            "median_hampel_spikes_per_trace": float(g["n_hampel_spikes"].median()),
            "mean_hampel_spike_rate": float(g["hampel_spike_rate"].mean()),
            "max_hampel_spikes_per_trace": int(g["n_hampel_spikes"].max()),
            "n_traces_with_any_spike": int(g["n_hampel_spikes"].gt(0).sum()),
        })
    df = pd.DataFrame(rows).sort_values("nozzle")
    df.to_csv(out_dir / "audit_summary.csv", index=False)


def write_threshold_sensitivity(traces: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for pen_th in PEN_THRESHOLD_SWEEP_MM:
        for slope_th in SLOPE_THRESHOLD_SWEEP_MM_PER_MS:
            is_cens = censoring_rate(traces, pen_th, slope_th)
            rows.append({
                "pen_threshold_mm": pen_th,
                "slope_threshold_mm_per_ms": slope_th,
                "n_traces": len(traces),
                "n_censored": int(is_cens.sum()),
                "frac_censored": float(is_cens.mean()),
            })
            for nozzle, g in traces.groupby("nozzle"):
                cens_g = censoring_rate(g, pen_th, slope_th)
                rows[-1][f"frac_{nozzle}"] = float(cens_g.mean())
    pd.DataFrame(rows).to_csv(out_dir / "threshold_sensitivity.csv", index=False)


# ───────────────────────────────────── main ─────────────────────────────────────


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_dir = (args.out_dir or (data_root / DEFAULT_OUT_DIR_NAME)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from {data_root}")
    print(f"Writing audit to {out_dir}")

    traces = collect_traces(
        data_root=data_root,
        source=args.source,
        end_slope_frames=args.end_slope_frames,
        hampel_window_ms=args.hampel_window_ms,
        hampel_k=args.hampel_k,
    )
    print(f"Loaded {len(traces)} traces across {traces['nozzle'].nunique()} nozzles.")

    traces.to_csv(out_dir / "per_trace_features.csv", index=False)

    plot_last_pen_hist(traces, out_dir, args.pen_fov_mm)
    plot_last_pen_vs_slope(traces, out_dir, args.pen_fov_mm, args.slope_eps_mm_per_ms)
    plot_hampel_summary(traces, out_dir, args.hampel_window_ms, args.hampel_k)
    write_audit_summary(traces, out_dir, args.pen_fov_mm, args.slope_eps_mm_per_ms)
    write_threshold_sensitivity(traces, out_dir)

    print("\nProposed rule:")
    print(f"  is_right_censored = (last_pen > {args.pen_fov_mm:g} mm)"
          f" AND (|end_slope| < {args.slope_eps_mm_per_ms:g} mm/ms)")
    summary = pd.read_csv(out_dir / "audit_summary.csv")
    print("\nPer-nozzle proposed-censored rate:")
    print(summary[["nozzle", "n_traces", "frac_proposed_censored", "median_hampel_spikes_per_trace"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    nozzle_caps = (
        traces.groupby("nozzle")["last_valid_pen_mm"]
        .quantile(args.cap_percentile / 100.0)
        .to_dict()
    )
    slope_ratios = tuple(float(s.strip()) for s in args.slope_ratios.split(",") if s.strip())
    print(f"\nPer-nozzle cap (p{args.cap_percentile:g} of last_valid_pen_mm):")
    for n, c in sorted(nozzle_caps.items()):
        print(f"  {n}: {c:.2f} mm  ->  {args.cap_fraction:g}*cap = {args.cap_fraction * c:.2f} mm")

    print("\nCollecting point-level features (this re-walks the cleaned series)…")
    points = collect_pointwise(
        data_root=data_root,
        source=args.source,
        nozzle_caps=nozzle_caps,
        slope_window_frames=args.slope_window_frames,
        early_fit_frac=args.early_fit_frac,
    )
    print(f"  collected {len(points):,} points across {points['nozzle'].nunique()} nozzles.")

    nozzle_df = write_pointwise_cutoff_audit(
        points=points,
        out_dir=out_dir,
        flat_cutoff_mm=args.flat_pen_cutoff_mm,
        cap_fraction=args.cap_fraction,
        slope_ratios=slope_ratios,
        early_fit_frac=args.early_fit_frac,
    )
    plot_raw_reliable_frac(nozzle_df, out_dir)

    print("\nStage-3 raw_reliable fraction (per nozzle × rule × slope-ratio):")
    print(
        nozzle_df[[
            "nozzle", "pen_cutoff_rule", "slope_ratio_min", "n_raw_reliable",
            "frac_raw_reliable", "pen_max_kept_mm",
        ]].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )


if __name__ == "__main__":
    main()
