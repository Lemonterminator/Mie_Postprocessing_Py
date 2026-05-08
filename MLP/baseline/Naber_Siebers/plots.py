from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import erf, sqrt


def plot_pred_vs_actual(points_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    sample = points_df
    if len(sample) > 80000:
        sample = sample.sample(80000, random_state=42)
    ax.scatter(sample["pen_true_mm"], sample["pen_pred_mm"], s=2, alpha=0.18, linewidths=0)
    low = min(float(points_df["pen_true_mm"].min()), float(points_df["pen_pred_mm"].min()))
    high = max(float(points_df["pen_true_mm"].max()), float(points_df["pen_pred_mm"].max()))
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Measured penetration [mm]")
    ax.set_ylabel("Predicted penetration [mm]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_residual_hist(points_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=160)
    ax.hist(points_df["resid_mm"], bins=80, color="#4c78a8", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual [mm]")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_group_rmse(per_group: pd.DataFrame, group_col: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    ordered = per_group.sort_values("rmse_mean_mm")
    ax.barh(ordered[group_col].astype(str), ordered["rmse_mean_mm"], color="#4c78a8", alpha=0.85)
    ax.set_xlabel("Mean trajectory RMSE [mm]")
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_time_bins(time_bins: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax1 = plt.subplots(figsize=(7.2, 4.3), dpi=160)
    ax1.plot(time_bins["time_bin_center_ms"], time_bins["rmse_mm"], label="RMSE", color="#4c78a8")
    ax1.plot(time_bins["time_bin_center_ms"], time_bins["mae_mm"], label="MAE", color="#f58518")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Error [mm]")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(time_bins["time_bin_center_ms"], time_bins["coverage_1sigma"], label="1 sigma coverage", color="#54a24b")
    ax2.plot(time_bins["time_bin_center_ms"], time_bins["coverage_2sigma"], label="2 sigma coverage", color="#e45756")
    ax2.set_ylabel("Coverage")
    ax2.set_ylim(0.0, 1.05)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_trajectory(points: pd.DataFrame, out_path: Path, title: str) -> None:
    g = points.sort_values("time_ms")
    t = g["time_ms"].to_numpy(dtype=float)
    mu = g["pen_pred_mm"].to_numpy(dtype=float)
    std = g["pen_std_mm"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.plot(t, g["pen_true_mm"], "o", markersize=3, label="measured")
    ax.plot(t, mu, "-", linewidth=1.4, label="Naber--Siebers")
    ax.fill_between(t, mu - std, mu + std, alpha=0.18, label="1 sigma")
    ax.fill_between(t, mu - 2.0 * std, mu + 2.0 * std, alpha=0.10, label="2 sigma")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Penetration [mm]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_standard_plots(
    *,
    points_df: pd.DataFrame,
    per_traj: pd.DataFrame,
    per_folder: pd.DataFrame,
    per_nozzle: pd.DataFrame,
    time_bins: pd.DataFrame,
    out_dir: Path,
    title_prefix: str,
    max_trajectory_plots: int = 24,
) -> None:
    plot_pred_vs_actual(points_df, out_dir / "pred_vs_actual.png", f"{title_prefix}: prediction vs measurement")
    plot_residual_hist(points_df, out_dir / "residual_histogram.png", f"{title_prefix}: residual distribution")
    plot_group_rmse(per_folder, "folder", out_dir / "per_folder_rmse.png", f"{title_prefix}: RMSE by campaign")
    plot_group_rmse(per_nozzle, "dataset_key", out_dir / "per_nozzle_rmse.png", f"{title_prefix}: RMSE by nozzle")
    plot_time_bins(time_bins, out_dir / "time_bin_metrics.png", f"{title_prefix}: time-binned error and coverage")

    traj_dir = out_dir / "traj_plots"
    traj_dir.mkdir(exist_ok=True)
    plot_rows = per_traj.sort_values("rmse_mm", ascending=False).head(int(max_trajectory_plots))
    for _, row in plot_rows.iterrows():
        g = points_df.loc[points_df["traj_key"] == row["traj_key"]]
        safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(row["traj_key"]))[:160]
        plot_trajectory(g, traj_dir / f"{safe_name}.png", f"{row['folder']} | {row['test_name']} | plume {row['plume_idx']}")


def _sample_points(points_df: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if len(points_df) <= int(max_points):
        return points_df
    return points_df.sample(int(max_points), random_state=int(seed))


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def plot_baseline_dashboard(
    *,
    points_df: pd.DataFrame,
    per_nozzle: pd.DataFrame,
    time_bins: pd.DataFrame,
    out_path: Path,
    title: str,
    max_points: int = 80000,
) -> None:
    _require_columns(points_df, ["pen_true_mm", "pen_pred_mm", "resid_mm"], "points_df")
    _require_columns(per_nozzle, ["dataset_key", "rmse_mean_mm"], "per_nozzle")
    _require_columns(time_bins, ["time_bin_center_ms", "rmse_mm", "coverage_1sigma", "coverage_2sigma"], "time_bins")

    sample = _sample_points(points_df, max_points=max_points)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=170)
    fig.suptitle(title, fontsize=13)

    ax = axes[0, 0]
    ax.scatter(sample["pen_true_mm"], sample["pen_pred_mm"], s=3, alpha=0.16, linewidths=0, color="#3b6ea8")
    low = min(float(points_df["pen_true_mm"].min()), float(points_df["pen_pred_mm"].min()))
    high = max(float(points_df["pen_true_mm"].max()), float(points_df["pen_pred_mm"].max()))
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Measured penetration [mm]")
    ax.set_ylabel("Predicted penetration [mm]")
    ax.set_title("Prediction vs measurement")
    ax.grid(True, alpha=0.24)

    ax = axes[0, 1]
    ax.hist(points_df["resid_mm"], bins=85, color="#8f6bb3", alpha=0.86)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual [mm]")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")

    ax = axes[1, 0]
    ordered = per_nozzle.sort_values("rmse_mean_mm", ascending=True)
    ax.barh(ordered["dataset_key"].astype(str), ordered["rmse_mean_mm"], color="#4f9d69", alpha=0.9)
    ax.set_xlabel("Mean trajectory RMSE [mm]")
    ax.set_title("RMSE by nozzle")
    ax.grid(True, axis="x", alpha=0.22)

    ax1 = axes[1, 1]
    ax1.plot(time_bins["time_bin_center_ms"], time_bins["rmse_mm"], color="#c44e52", linewidth=1.7, label="RMSE")
    if "mae_mm" in time_bins.columns:
        ax1.plot(time_bins["time_bin_center_ms"], time_bins["mae_mm"], color="#dd8452", linewidth=1.4, label="MAE")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Error [mm]")
    ax1.grid(True, alpha=0.24)
    ax2 = ax1.twinx()
    ax2.plot(time_bins["time_bin_center_ms"], time_bins["coverage_1sigma"], color="#4878d0", linewidth=1.2, label="1 sigma coverage")
    ax2.plot(time_bins["time_bin_center_ms"], time_bins["coverage_2sigma"], color="#6acc64", linewidth=1.2, label="2 sigma coverage")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Coverage")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    ax1.set_title("Time-binned error and coverage")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path)
    plt.close(fig)


def build_uncertainty_decomposition(points_df: pd.DataFrame, time_bins: pd.DataFrame | None = None) -> pd.DataFrame:
    _require_columns(points_df, ["time_ms", "resid_mm", "pen_std_mm"], "points_df")
    centers = None
    width = 0.1
    if time_bins is not None and "time_bin_center_ms" in time_bins.columns and len(time_bins) > 1:
        centers = time_bins["time_bin_center_ms"].to_numpy(dtype=float)
        diffs = np.diff(np.sort(centers))
        width = float(np.nanmedian(diffs)) if np.isfinite(diffs).any() else width
    elif time_bins is not None and "time_bin_center_ms" in time_bins.columns:
        centers = time_bins["time_bin_center_ms"].to_numpy(dtype=float)

    df = points_df.copy()
    df["time_bin_center_ms"] = (np.floor(df["time_ms"].to_numpy(dtype=float) / width) * width) + 0.5 * width
    grouped = df.groupby("time_bin_center_ms", dropna=False)
    out = grouped.agg(
        n_points=("resid_mm", "size"),
        residual_rmse_mm=("resid_mm", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        total_sigma_mean_mm=("pen_std_mm", "mean"),
    ).reset_index()
    if "sigma_resid_mm" in df.columns:
        out = out.merge(
            grouped["sigma_resid_mm"].mean().rename("residual_sigma_mean_mm").reset_index(),
            on="time_bin_center_ms",
            how="left",
        )
    if "sigma_param_mm" in df.columns:
        out = out.merge(
            grouped["sigma_param_mm"].mean().rename("parameter_sigma_mean_mm").reset_index(),
            on="time_bin_center_ms",
            how="left",
        )

    if centers is not None and len(centers):
        valid = pd.DataFrame({"time_bin_center_ms": centers})
        out = valid.merge(out, on="time_bin_center_ms", how="left")
    return out.sort_values("time_bin_center_ms").reset_index(drop=True)


def plot_uncertainty_decomposition(decomp: pd.DataFrame, out_path: Path, title: str) -> None:
    _require_columns(decomp, ["time_bin_center_ms", "residual_rmse_mm", "total_sigma_mean_mm"], "decomp")
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=170)
    ax.plot(decomp["time_bin_center_ms"], decomp["residual_rmse_mm"], color="#c44e52", linewidth=1.8, label="Observed residual RMSE")
    ax.plot(decomp["time_bin_center_ms"], decomp["total_sigma_mean_mm"], color="#4878d0", linewidth=1.6, label="Total predicted sigma")
    if "residual_sigma_mean_mm" in decomp.columns:
        ax.plot(decomp["time_bin_center_ms"], decomp["residual_sigma_mean_mm"], color="#6acc64", linewidth=1.2, label="Residual sigma")
    if "parameter_sigma_mean_mm" in decomp.columns:
        ax.plot(decomp["time_bin_center_ms"], decomp["parameter_sigma_mean_mm"], color="#8f6bb3", linewidth=1.2, label="Bootstrap parameter sigma")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Scale [mm]")
    ax.set_title(title)
    ax.grid(True, alpha=0.24)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric_comparison(
    *,
    baseline_metrics: dict,
    reference_metrics: dict,
    out_path: Path,
    baseline_label: str,
    reference_label: str,
    title: str,
) -> None:
    error_metrics = [
        ("rmse_mm", "RMSE"),
        ("mae_mm", "MAE"),
        ("p95_abs_err_mm", "P95 abs. err."),
        ("mean_pred_std_mm", "Mean sigma"),
    ]
    coverage_metrics = [
        ("coverage_1sigma", "1 sigma cov."),
        ("coverage_2sigma", "2 sigma cov."),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), dpi=170)
    fig.suptitle(title, fontsize=12.5)

    ax = axes[0]
    x = np.arange(len(error_metrics))
    width = 0.36
    b_vals = [float(baseline_metrics[k]) for k, _ in error_metrics]
    r_vals = [float(reference_metrics[k]) for k, _ in error_metrics]
    ax.bar(x - width / 2, b_vals, width=width, label=baseline_label, color="#c44e52", alpha=0.9)
    ax.bar(x + width / 2, r_vals, width=width, label=reference_label, color="#4878d0", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in error_metrics], rotation=20, ha="right")
    ax.set_ylabel("mm")
    ax.set_title("Error and uncertainty scale")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(fontsize=8)

    ax = axes[1]
    x = np.arange(len(coverage_metrics))
    b_vals = [float(baseline_metrics[k]) for k, _ in coverage_metrics]
    r_vals = [float(reference_metrics[k]) for k, _ in coverage_metrics]
    ax.bar(x - width / 2, b_vals, width=width, label=baseline_label, color="#c44e52", alpha=0.9)
    ax.bar(x + width / 2, r_vals, width=width, label=reference_label, color="#4878d0", alpha=0.9)
    ax.axhline(0.6827, color="black", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.axhline(0.9545, color="black", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in coverage_metrics])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration coverage")
    ax.grid(True, axis="y", alpha=0.22)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


def build_coverage_reliability(
    datasets: list[tuple[str, pd.DataFrame]],
    *,
    z_min: float = 0.25,
    z_max: float = 3.0,
    n_steps: int = 23,
) -> pd.DataFrame:
    rows = []
    thresholds = np.linspace(float(z_min), float(z_max), int(n_steps))
    for label, df in datasets:
        _require_columns(df, ["resid_mm", "pen_std_mm"], f"{label} points")
        std = df["pen_std_mm"].to_numpy(dtype=float)
        resid_abs = np.abs(df["resid_mm"].to_numpy(dtype=float))
        ok = np.isfinite(std) & np.isfinite(resid_abs) & (std > 0.0)
        std = std[ok]
        resid_abs = resid_abs[ok]
        for z in thresholds:
            rows.append(
                {
                    "model": label,
                    "z_sigma": float(z),
                    "nominal_coverage": float(erf(float(z) / sqrt(2.0))),
                    "empirical_coverage": float(np.mean(resid_abs <= float(z) * std)) if len(std) else np.nan,
                    "n_points": int(len(std)),
                }
            )
    return pd.DataFrame(rows)


def plot_coverage_reliability(reliability: pd.DataFrame, out_path: Path, title: str) -> None:
    _require_columns(reliability, ["model", "nominal_coverage", "empirical_coverage"], "reliability")
    fig, ax = plt.subplots(figsize=(5.7, 5.2), dpi=170)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, alpha=0.65, label="Ideal")
    for label, group in reliability.groupby("model", sort=False):
        ordered = group.sort_values("nominal_coverage")
        ax.plot(ordered["nominal_coverage"], ordered["empirical_coverage"], marker="o", markersize=3, linewidth=1.4, label=str(label))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Nominal Gaussian coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.grid(True, alpha=0.24)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
