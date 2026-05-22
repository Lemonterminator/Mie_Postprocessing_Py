"""Quick time-binned physics scaling screen on raw CDF penetration points.

The screen reads every ``cdf/series_wide_all/*.csv`` file under the selected
synthetic-data root, expands wide time/penetration columns to raw point samples,
bins by time, and fits

    log(S) = a log(P_ch) + b log(Delta P) + c log(d_nozzle) + const

inside each time bin.  It also reports how much scatter remains after dividing
the raw penetration by the fitted multiplicative scale and by the fixed
Hiroyasu-Arai-like scale ``DeltaP^0.25 * P_ch^-0.25 * d^0.5``.

It is intended as a fast physics sanity screen over unfit CDF points, not as a
replacement for the fitted q1 model or NN training pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "MLP" / "synthetic_data_20260509"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "fit_diagnostics" / "raw_series_physics_screening"

META_COLS = [
    "file_path",
    "file_name",
    "file_stem",
    "plume_idx",
    "diameter_mm",
    "chamber_pressure_bar",
    "injection_pressure_bar",
    "control_backpressure_bar",
    "injection_duration_us",
]
FILL_META_COLS = [
    "diameter_mm",
    "chamber_pressure_bar",
    "injection_pressure_bar",
    "control_backpressure_bar",
    "injection_duration_us",
]
HA_EXPS = {
    "chamber_pressure_bar": -0.25,
    "delta_pressure_bar": 0.25,
    "diameter_mm": 0.50,
}


def prefixed_ids(columns: list[str], prefix: str) -> set[int]:
    ids: set[int] = set()
    for col in columns:
        if not col.startswith(prefix):
            continue
        try:
            ids.add(int(col.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return ids


def fill_group_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Fill sparse per-plume metadata from rows sharing the same source cine."""
    out = df.copy()
    group_cols = [col for col in ["file_path", "file_name", "file_stem"] if col in out.columns]
    if group_cols:
        out[FILL_META_COLS] = out.groupby(group_cols, dropna=False)[FILL_META_COLS].transform(
            lambda values: values.ffill().bfill()
        )
    out[FILL_META_COLS] = out[FILL_META_COLS].ffill().bfill()
    return out


def load_raw_points(
    data_root: Path,
    *,
    source: str,
    split_dir: str,
    time_min_ms: float,
    time_max_ms: float,
    during_injection_only: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    files = sorted(data_root.glob(f"*/{source}/{split_dir}/*.csv"))
    if not files:
        raise FileNotFoundError(f"No files matched {data_root}/*/{source}/{split_dir}/*.csv")

    file_summaries: list[dict[str, object]] = []
    total_wide_rows = 0
    total_candidate_points = 0
    total_valid_points = 0

    for csv_path in files:
        header = pd.read_csv(csv_path, nrows=0)
        columns = list(header.columns)
        frame_ids = sorted(
            prefixed_ids(columns, "time_ms_").intersection(prefixed_ids(columns, "penetration_mm_"))
        )
        if not frame_ids:
            continue

        time_cols = [f"time_ms_{frame_id:03d}" for frame_id in frame_ids]
        pen_cols = [f"penetration_mm_{frame_id:03d}" for frame_id in frame_ids]
        usecols = [col for col in META_COLS + time_cols + pen_cols if col in columns]
        df = pd.read_csv(csv_path, usecols=usecols)
        df = fill_group_metadata(df)

        n_rows = len(df)
        n_frames = len(frame_ids)
        total_wide_rows += n_rows
        total_candidate_points += n_rows * n_frames

        cond = df[FILL_META_COLS].apply(pd.to_numeric, errors="coerce")
        chamber = cond["chamber_pressure_bar"].to_numpy(dtype=float)
        injection = cond["injection_pressure_bar"].to_numpy(dtype=float)
        diameter = cond["diameter_mm"].to_numpy(dtype=float)
        duration_ms = cond["injection_duration_us"].to_numpy(dtype=float) / 1000.0
        delta = injection - chamber

        t = df[time_cols].to_numpy(dtype=float)
        s = df[pen_cols].to_numpy(dtype=float)

        chamber_flat = np.repeat(chamber, n_frames)
        delta_flat = np.repeat(delta, n_frames)
        diameter_flat = np.repeat(diameter, n_frames)
        duration_flat = np.repeat(duration_ms, n_frames)
        t_flat = t.reshape(-1)
        s_flat = s.reshape(-1)

        valid = (
            np.isfinite(t_flat)
            & np.isfinite(s_flat)
            & np.isfinite(chamber_flat)
            & np.isfinite(delta_flat)
            & np.isfinite(diameter_flat)
            & (t_flat >= time_min_ms)
            & (t_flat < time_max_ms)
            & (s_flat > 0.0)
            & (chamber_flat > 0.0)
            & (delta_flat > 0.0)
            & (diameter_flat > 0.0)
        )
        if during_injection_only:
            valid &= np.isfinite(duration_flat) & (t_flat <= duration_flat)

        experiment = csv_path.parents[2].name
        valid_count = int(valid.sum())
        total_valid_points += valid_count
        file_summaries.append(
            {
                "experiment_name": experiment,
                "file": str(csv_path),
                "wide_rows": int(n_rows),
                "frames": int(n_frames),
                "valid_points": valid_count,
            }
        )
        if valid_count == 0:
            continue

        frames.append(
            pd.DataFrame(
                {
                    "experiment_name": experiment,
                    "t_ms": t_flat[valid],
                    "penetration_mm": s_flat[valid],
                    "chamber_pressure_bar": chamber_flat[valid],
                    "delta_pressure_bar": delta_flat[valid],
                    "diameter_mm": diameter_flat[valid],
                }
            )
        )

    if not frames:
        raise ValueError("No valid log-regression points were found.")

    long_df = pd.concat(frames, ignore_index=True, sort=False)
    summary = {
        "source": source,
        "split_dir": split_dir,
        "files": len(files),
        "wide_rows": int(total_wide_rows),
        "candidate_points": int(total_candidate_points),
        "valid_points": int(total_valid_points),
        "time_min_ms": float(time_min_ms),
        "time_max_ms": float(time_max_ms),
        "during_injection_only": bool(during_injection_only),
        "file_summaries": file_summaries,
    }
    return long_df, summary


def ols_loglog(sub: pd.DataFrame) -> dict[str, float]:
    """Fit one log-log OLS model and report fitted vs HA scaling residuals."""
    log_ch = np.log(sub["chamber_pressure_bar"].to_numpy(dtype=float))
    log_dp = np.log(sub["delta_pressure_bar"].to_numpy(dtype=float))
    log_d = np.log(sub["diameter_mm"].to_numpy(dtype=float))
    y = np.log(sub["penetration_mm"].to_numpy(dtype=float))

    x_no_intercept = np.column_stack([log_ch, log_dp, log_d])
    design = np.column_stack([x_no_intercept, np.ones(len(sub))])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ coef
    resid = y - y_hat

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    n, p = design.shape
    sigma2 = ss_res / max(n - p, 1)
    try:
        cov = sigma2 * np.linalg.pinv(design.T @ design)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    s_raw = sub["penetration_mm"].to_numpy(dtype=float)
    log_rel_scale_fit = x_no_intercept @ coef[:3]
    rel_scale_fit = np.exp(log_rel_scale_fit - np.mean(log_rel_scale_fit))
    s_scaled_fit = s_raw / rel_scale_fit

    log_rel_scale_ha = (
        HA_EXPS["chamber_pressure_bar"] * log_ch
        + HA_EXPS["delta_pressure_bar"] * log_dp
        + HA_EXPS["diameter_mm"] * log_d
    )
    rel_scale_ha = np.exp(log_rel_scale_ha - np.mean(log_rel_scale_ha))
    s_scaled_ha = s_raw / rel_scale_ha
    ha_intercept = float(np.mean(y - log_rel_scale_ha))
    ha_resid = y - (ha_intercept + log_rel_scale_ha)
    ha_ss_res = float(np.sum(ha_resid**2))
    ha_r2 = 1.0 - ha_ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    raw_var = float(np.var(s_raw, ddof=0))
    fit_var = float(np.var(s_scaled_fit, ddof=0))
    ha_var = float(np.var(s_scaled_ha, ddof=0))
    raw_std = float(np.std(s_raw, ddof=0))
    fit_std = float(np.std(s_scaled_fit, ddof=0))
    ha_std = float(np.std(s_scaled_ha, ddof=0))

    return {
        "coef_log_chamber_pressure": float(coef[0]),
        "coef_log_delta_pressure": float(coef[1]),
        "coef_log_diameter": float(coef[2]),
        "intercept": float(coef[3]),
        "se_log_chamber_pressure": float(se[0]),
        "se_log_delta_pressure": float(se[1]),
        "se_log_diameter": float(se[2]),
        "r2_log": float(r2),
        "log_var_collapse_ratio_fit": float(ss_res / ss_tot) if ss_tot > 1e-12 else np.nan,
        "log_var_reduction_pct_fit": float(100.0 * r2) if np.isfinite(r2) else np.nan,
        "raw_std_mm": raw_std,
        "scaled_fit_std_mm": fit_std,
        "raw_std_collapse_ratio_fit": float(fit_std / raw_std) if raw_std > 1e-12 else np.nan,
        "raw_var_collapse_ratio_fit": float(fit_var / raw_var) if raw_var > 1e-12 else np.nan,
        "raw_var_reduction_pct_fit": float(100.0 * (1.0 - fit_var / raw_var)) if raw_var > 1e-12 else np.nan,
        "ha_fixed_r2_log": float(ha_r2),
        "scaled_ha_std_mm": ha_std,
        "raw_std_collapse_ratio_ha_fixed": float(ha_std / raw_std) if raw_std > 1e-12 else np.nan,
        "raw_var_collapse_ratio_ha_fixed": float(ha_var / raw_var) if raw_var > 1e-12 else np.nan,
        "raw_var_reduction_pct_ha_fixed": float(100.0 * (1.0 - ha_var / raw_var)) if raw_var > 1e-12 else np.nan,
    }


def regress_time_bins(
    long_df: pd.DataFrame,
    *,
    bin_ms: float,
    time_min_ms: float,
    time_max_ms: float,
    min_points: int,
) -> pd.DataFrame:
    """Run the log-log screen independently in each requested time bin."""
    bins = np.arange(time_min_ms, time_max_ms + bin_ms * 0.5, bin_ms)
    rows: list[dict[str, float | int]] = []
    for left, right in zip(bins[:-1], bins[1:]):
        sub = long_df.loc[(long_df["t_ms"] >= left) & (long_df["t_ms"] < right)]
        base: dict[str, float | int] = {
            "t_left_ms": float(left),
            "t_right_ms": float(right),
            "t_center_ms": float((left + right) / 2.0),
            "n_points": int(len(sub)),
            "n_experiments": int(sub["experiment_name"].nunique()) if len(sub) else 0,
            "n_feature_conditions": int(
                sub[["chamber_pressure_bar", "delta_pressure_bar", "diameter_mm"]].drop_duplicates().shape[0]
            )
            if len(sub)
            else 0,
        }
        if len(sub) < min_points:
            rows.append(base)
            continue
        base.update(ols_loglog(sub))
        rows.append(base)
    return pd.DataFrame(rows)


def summarize_results(reg_df: pd.DataFrame, load_summary: dict[str, object]) -> dict[str, object]:
    valid = reg_df.dropna(subset=["r2_log"]).copy()
    post = valid.loc[valid["t_center_ms"] >= 0.5]

    def stats(prefix: str, df: pd.DataFrame) -> dict[str, float]:
        if df.empty:
            return {
                f"{prefix}_bins": 0,
                f"{prefix}_median_r2_log": float("nan"),
                f"{prefix}_weighted_mean_r2_log": float("nan"),
                f"{prefix}_median_raw_var_reduction_pct_fit": float("nan"),
                f"{prefix}_median_raw_var_reduction_pct_ha_fixed": float("nan"),
                f"{prefix}_median_raw_var_collapse_ratio_fit": float("nan"),
                f"{prefix}_median_raw_var_collapse_ratio_ha_fixed": float("nan"),
            }
        weights = df["n_points"].to_numpy(dtype=float)
        return {
            f"{prefix}_bins": int(len(df)),
            f"{prefix}_median_r2_log": float(np.nanmedian(df["r2_log"])),
            f"{prefix}_weighted_mean_r2_log": float(np.average(df["r2_log"], weights=weights)),
            f"{prefix}_median_raw_var_reduction_pct_fit": float(np.nanmedian(df["raw_var_reduction_pct_fit"])),
            f"{prefix}_median_raw_var_reduction_pct_ha_fixed": float(
                np.nanmedian(df["raw_var_reduction_pct_ha_fixed"])
            ),
            f"{prefix}_median_raw_var_collapse_ratio_fit": float(np.nanmedian(df["raw_var_collapse_ratio_fit"])),
            f"{prefix}_median_raw_var_collapse_ratio_ha_fixed": float(
                np.nanmedian(df["raw_var_collapse_ratio_ha_fixed"])
            ),
        }

    out: dict[str, object] = {
        key: value
        for key, value in load_summary.items()
        if key != "file_summaries"
    }
    out.update(stats("all_valid_bins", valid))
    out.update(stats("post_0p5ms", post))
    if not valid.empty:
        best = valid.loc[valid["r2_log"].idxmax()]
        out["best_r2_bin"] = {
            "t_center_ms": float(best["t_center_ms"]),
            "n_points": int(best["n_points"]),
            "r2_log": float(best["r2_log"]),
            "raw_var_reduction_pct_fit": float(best["raw_var_reduction_pct_fit"]),
            "raw_var_reduction_pct_ha_fixed": float(best["raw_var_reduction_pct_ha_fixed"]),
        }
    return out


def plot_results(reg_df: pd.DataFrame, out_dir: Path) -> None:
    valid = reg_df.dropna(subset=["r2_log"])
    if valid.empty:
        return

    t = valid["t_center_ms"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    axes = axes.ravel()

    axes[0].plot(t, valid["coef_log_chamber_pressure"], "-o", ms=3, label="P_ch")
    axes[0].axhline(-0.25, color="0.4", ls=":", lw=1.2, label="HA P_ch^-0.25 proxy")
    axes[0].set_ylabel("log-log exponent")
    axes[0].set_title("Chamber pressure exponent")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(t, valid["coef_log_delta_pressure"], "-o", ms=3, label="Delta P")
    axes[1].axhline(0.25, color="0.4", ls=":", lw=1.2, label="HA DeltaP^0.25")
    axes[1].axhline(0.50, color="0.2", ls="--", lw=1.0, label="orifice DeltaP^0.5")
    axes[1].set_title("Pressure-difference exponent")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(t, valid["coef_log_diameter"], "-o", ms=3, label="d")
    axes[2].axhline(0.50, color="0.4", ls=":", lw=1.2, label="HA d^0.5")
    axes[2].set_xlabel("time-bin center [ms]")
    axes[2].set_ylabel("log-log exponent")
    axes[2].set_title("Nozzle-diameter exponent")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    axes[3].plot(t, valid["r2_log"], "-o", ms=3, label="OLS log R2")
    axes[3].plot(t, valid["ha_fixed_r2_log"], "-o", ms=3, label="fixed HA log R2")
    axes[3].set_xlabel("time-bin center [ms]")
    axes[3].set_ylabel("R2")
    axes[3].set_ylim(bottom=min(-0.05, float(np.nanmin(valid["ha_fixed_r2_log"])) - 0.05), top=1.0)
    axes[3].set_title("Explained log variance")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "time_binned_loglog_exponents.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(t, valid["raw_var_collapse_ratio_fit"], "-o", ms=3, label="fitted log-log scale")
    ax.plot(t, valid["raw_var_collapse_ratio_ha_fixed"], "-o", ms=3, label="fixed HA scale")
    ax.axhline(1.0, color="0.3", ls="--", lw=1.0)
    ax.set_xlabel("time-bin center [ms]")
    ax.set_ylabel("raw variance ratio after scaling")
    ax.set_title("Variance collapse on raw penetration points")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "time_binned_variance_collapse.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--source", default="cdf")
    parser.add_argument("--split-dir", default="series_wide_all")
    parser.add_argument("--bin-ms", type=float, default=0.1)
    parser.add_argument("--time-min-ms", type=float, default=0.0)
    parser.add_argument("--time-max-ms", type=float, default=1.6)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument(
        "--during-injection-only",
        action="store_true",
        help="Drop points where t_ms is after injection_duration_us.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    long_df, load_summary = load_raw_points(
        args.data_root,
        source=args.source,
        split_dir=args.split_dir,
        time_min_ms=args.time_min_ms,
        time_max_ms=args.time_max_ms,
        during_injection_only=args.during_injection_only,
    )
    long_path = args.out_dir / "raw_points_long_sample.csv"
    long_df.sample(n=min(20000, len(long_df)), random_state=42).to_csv(long_path, index=False)

    counts = (
        long_df.groupby("experiment_name", dropna=False)
        .agg(
            valid_points=("penetration_mm", "size"),
            chamber_min=("chamber_pressure_bar", "min"),
            chamber_max=("chamber_pressure_bar", "max"),
            delta_p_min=("delta_pressure_bar", "min"),
            delta_p_max=("delta_pressure_bar", "max"),
            diameter_mm=("diameter_mm", "median"),
        )
        .reset_index()
    )
    counts.to_csv(args.out_dir / "raw_points_by_experiment.csv", index=False)

    reg_df = regress_time_bins(
        long_df,
        bin_ms=args.bin_ms,
        time_min_ms=args.time_min_ms,
        time_max_ms=args.time_max_ms,
        min_points=args.min_points,
    )
    reg_path = args.out_dir / "time_binned_loglog_regression.csv"
    reg_df.to_csv(reg_path, index=False)

    summary = summarize_results(reg_df, load_summary)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_results(reg_df, args.out_dir)

    display_cols = [
        "t_center_ms",
        "n_points",
        "coef_log_chamber_pressure",
        "coef_log_delta_pressure",
        "coef_log_diameter",
        "r2_log",
        "raw_var_reduction_pct_fit",
        "raw_var_reduction_pct_ha_fixed",
        "ha_fixed_r2_log",
    ]
    print(f"Loaded {load_summary['valid_points']:,} valid points from {load_summary['files']} files.")
    print(f"Saved regression table: {reg_path}")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("\nPer-bin regression:")
    print(reg_df[display_cols].to_string(index=False, float_format=lambda value: f"{value:8.4f}"))


if __name__ == "__main__":
    main()
