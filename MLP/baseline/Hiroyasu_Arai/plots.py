from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pred_vs_actual(points_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    sample = points_df if len(points_df) <= 80000 else points_df.sample(80000, random_state=42)
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
    ax2.plot(time_bins["time_bin_center_ms"], time_bins["coverage_1sigma"], label="1σ coverage", color="#54a24b")
    ax2.plot(time_bins["time_bin_center_ms"], time_bins["coverage_2sigma"], label="2σ coverage", color="#e45756")
    ax2.set_ylabel("Coverage")
    ax2.set_ylim(0.0, 1.05)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_trajectory(points: pd.DataFrame, out_path: Path, title: str, model_label: str = "H-A") -> None:
    g = points.sort_values("time_ms")
    t = g["time_ms"].to_numpy(dtype=float)
    mu = g["pen_pred_mm"].to_numpy(dtype=float)
    std = g["pen_std_mm"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.plot(t, g["pen_true_mm"], "o", markersize=3, label="measured")
    ax.plot(t, mu, "-", linewidth=1.4, label=model_label)
    ax.fill_between(t, mu - std, mu + std, alpha=0.18, label="1σ")
    ax.fill_between(t, mu - 2.0 * std, mu + 2.0 * std, alpha=0.10, label="2σ")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Penetration [mm]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_breakup_time_distribution(points_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if "t_b_ms" not in points_df.columns:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=155)
    vals = pd.to_numeric(points_df["t_b_ms"], errors="coerce").dropna()
    ax.hist(vals, bins=40, color="#7b6ea8", alpha=0.85)
    ax.set_xlabel("Breakup time t_b [ms]")
    ax.set_ylabel("Point count")
    ax.set_title(title)
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
        plot_trajectory(
            g,
            traj_dir / f"{safe_name}.png",
            f"{row['folder']} | {row['test_name']} | plume {row['plume_idx']}",
            model_label=title_prefix.split()[0],
        )


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
        std = df["pen_std_mm"].to_numpy(dtype=float)
        resid_abs = np.abs(df["resid_mm"].to_numpy(dtype=float))
        ok = np.isfinite(std) & np.isfinite(resid_abs) & (std > 0.0)
        std = std[ok]
        resid_abs = resid_abs[ok]
        for z in thresholds:
            rows.append({
                "model": label,
                "z_sigma": float(z),
                "nominal_coverage": float(erf(float(z) / sqrt(2.0))),
                "empirical_coverage": float(np.mean(resid_abs <= float(z) * std)) if len(std) else np.nan,
                "n_points": int(len(std)),
            })
    return pd.DataFrame(rows)
