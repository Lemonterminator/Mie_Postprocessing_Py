"""Figure export helpers for point-table evaluation directories.

The exporter is intentionally post-hoc: it reads the CSV artifacts already
written by ``inference_rmse_on_point_tables.py`` and creates publication-style
diagnostic figures inside the same evaluation directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_POINTS = 80_000
DEFAULT_TOP_CONDITIONS = 30
EVAL_SET_TITLES = {
    "cdf_uncensored": "CDF uncensored observations",
    "p50_observed": "Observed P50 aggregate",
    "q1_grid_all": "Q1 oracle grid",
}


def _finite_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = np.ones(len(out), dtype=bool)
    for col in cols:
        mask &= np.isfinite(out[col].to_numpy(dtype=float))
    return out.loc[mask].copy()


def _short_label(value: Any, *, max_len: int = 34) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "."


def _eval_title(eval_set: str) -> str:
    return EVAL_SET_TITLES.get(eval_set, eval_set.replace("_", " "))


def _condition_label_map(points: pd.DataFrame | None) -> dict[Any, str]:
    if points is None or points.empty or "condition_id" not in points.columns:
        return {}
    labels: dict[Any, str] = {}
    for condition_id, group in points.groupby("condition_id", dropna=False, sort=False):
        first = group.iloc[0]
        prefix = f"C{condition_id}"
        key = first.get("condition_key") if "condition_key" in points.columns else None
        if key is not None and not pd.isna(key):
            parts = str(key).split("|")
            exp = parts[0] if parts else str(first.get("experiment_name", ""))
            details = []
            if len(parts) > 2:
                details.append(f"d={parts[2]}")
            if len(parts) > 4:
                details.append(f"Pinj={parts[4]}")
            if len(parts) > 5:
                details.append(f"Pch={parts[5]}")
            labels[condition_id] = _short_label(" | ".join([prefix, exp, *details]), max_len=44)
            continue
        exp = first.get("experiment_name") if "experiment_name" in points.columns else None
        labels[condition_id] = _short_label(f"{prefix} | {exp}" if exp is not None and not pd.isna(exp) else prefix)
    return labels


def _save_pred_vs_actual(points: pd.DataFrame, out_path: Path, *, title: str, sample_points: int) -> str | None:
    required = ["pen_true_mm", "pen_pred_mm"]
    if any(col not in points.columns for col in required):
        return None
    df = _finite_df(points, required)
    if df.empty:
        return None
    sample = df.sample(n=min(len(df), int(sample_points)), random_state=42) if len(df) else df
    truth = sample["pen_true_mm"].to_numpy(dtype=float)
    pred = sample["pen_pred_mm"].to_numpy(dtype=float)
    lim_min = min(0.0, float(np.nanmin(truth)), float(np.nanmin(pred)))
    lim_max = max(float(np.nanmax(df["pen_true_mm"])), float(np.nanmax(df["pen_pred_mm"])))
    if not np.isfinite(lim_max) or lim_max <= lim_min:
        return None

    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=180)
    ax.scatter(truth, pred, s=2, alpha=0.12, color="#2F6FA3", linewidths=0)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#222222", linewidth=1.1, linestyle="--")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Target penetration [mm]")
    ax.set_ylabel("Predicted penetration [mm]")
    ax.set_title(f"{title}: predicted vs. target")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _save_residual_hist(points: pd.DataFrame, out_path: Path, *, title: str) -> str | None:
    if "resid_mm" not in points.columns:
        return None
    resid = pd.to_numeric(points["resid_mm"], errors="coerce").dropna().to_numpy(dtype=float)
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return None
    clipped = np.clip(resid, -30.0, 30.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax.hist(clipped, bins=90, color="#2F6FA3", alpha=0.88, edgecolor="white", linewidth=0.35)
    ax.axvline(0.0, color="#222222", linewidth=1.1)
    ax.axvline(float(np.mean(resid)), color="#C26A2E", linewidth=1.4, label="mean bias")
    ax.set_xlabel("Residual, prediction - target [mm]")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}: residual distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _save_per_condition(
    per_condition: pd.DataFrame,
    points: pd.DataFrame | None,
    out_rmse: Path,
    out_cov: Path,
    *,
    title: str,
    top_n_conditions: int,
    include_uncertainty: bool,
) -> dict[str, str]:
    if per_condition.empty or "condition_id" not in per_condition.columns or "rmse_mm" not in per_condition.columns:
        return {}
    label_map = _condition_label_map(points)
    df = per_condition.copy()
    df["rmse_mm"] = pd.to_numeric(df["rmse_mm"], errors="coerce")
    df = df.dropna(subset=["rmse_mm"]).sort_values("rmse_mm", ascending=False).head(int(top_n_conditions))
    if df.empty:
        return {}
    labels = [_short_label(label_map.get(row["condition_id"], row["condition_id"])) for _, row in df.iterrows()]
    y = np.arange(len(df))
    fig_h = max(4.4, 0.28 * len(df) + 1.8)

    fig, ax = plt.subplots(figsize=(8.8, fig_h), dpi=180)
    bars = ax.barh(y, df["rmse_mm"], color="#2F6FA3", edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE [mm]")
    ax.set_title(f"{title}: worst condition RMSE")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_rmse)
    plt.close(fig)

    outputs = {"per_condition_rmse": str(out_rmse)}
    if not include_uncertainty and out_cov.exists():
        out_cov.unlink()
    if include_uncertainty and {"coverage_1sigma", "coverage_2sigma"}.issubset(df.columns):
        cov1 = pd.to_numeric(df["coverage_1sigma"], errors="coerce").to_numpy(dtype=float)
        cov2 = pd.to_numeric(df["coverage_2sigma"], errors="coerce").to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8.8, fig_h), dpi=180)
        height = 0.36
        ax.barh(y - height / 2, cov1, height=height, label="1 sigma", color="#7A5BB5")
        ax.barh(y + height / 2, cov2, height=height, label="2 sigma", color="#3A9AA1")
        ax.axvline(0.683, color="#7A5BB5", linestyle="--", linewidth=1.0)
        ax.axvline(0.954, color="#3A9AA1", linestyle="--", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.04)
        ax.set_xlabel("Empirical coverage")
        ax.set_title(f"{title}: coverage for worst-RMSE conditions")
        ax.legend(frameon=False, ncols=2, loc="lower right")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_cov)
        plt.close(fig)
        outputs["per_condition_coverage"] = str(out_cov)
    return outputs


def _save_per_time_bin(
    per_time: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    include_uncertainty: bool,
) -> str | None:
    required = {"time_bin", "rmse_mm", "mae_mm", "bias_mm"}
    if per_time.empty or not required.issubset(per_time.columns):
        return None
    df = _finite_df(per_time, ["time_bin", "rmse_mm", "mae_mm", "bias_mm"])
    if df.empty:
        return None
    df = df.sort_values("time_bin")
    x = df["time_bin"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=180)
    ax = axes[0]
    ax.plot(x, df["rmse_mm"], marker="o", linewidth=1.8, color="#2F6FA3", label="RMSE")
    ax.plot(x, df["mae_mm"], marker="s", linewidth=1.8, color="#C26A2E", label="MAE")
    ax.plot(x, df["bias_mm"], marker="^", linewidth=1.5, color="#4F8C4A", label="Bias")
    ax.axhline(0.0, color="#222222", linewidth=0.9)
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Error [mm]")
    ax.set_title("Point error over time")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    if include_uncertainty and {"coverage_1sigma", "coverage_2sigma"}.issubset(df.columns):
        ax.plot(x, df["coverage_1sigma"], marker="o", linewidth=1.8, color="#7A5BB5", label="1 sigma")
        ax.plot(x, df["coverage_2sigma"], marker="s", linewidth=1.8, color="#3A9AA1", label="2 sigma")
        ax.axhline(0.683, color="#7A5BB5", linestyle="--", linewidth=1.0)
        ax.axhline(0.954, color="#3A9AA1", linestyle="--", linewidth=1.0)
        ax.set_ylim(0.0, 1.04)
        ax.set_ylabel("Coverage")
        ax.legend(frameon=False)
        ax.set_title("Uncertainty over time")
    elif "p95_abs_err_mm" in df.columns:
        ax.plot(x, df["p95_abs_err_mm"], marker="o", linewidth=1.8, color="#7A5BB5", label="P95 abs. error")
        ax.set_ylabel("P95 absolute error [mm]")
        ax.legend(frameon=False)
        ax.set_title("Tail error over time")
    elif "mean_pred_std_mm" in df.columns:
        ax.plot(x, df["mean_pred_std_mm"], marker="o", linewidth=1.8, color="#7A5BB5")
        ax.set_ylabel("Mean predicted std [mm]")
        ax.set_title("Prediction scale over time")
    ax.set_xlabel("Time bin")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _save_condition_time_heatmap(
    per_condition_time: pd.DataFrame,
    points: pd.DataFrame | None,
    out_path: Path,
    *,
    title: str,
    top_n_conditions: int,
) -> str | None:
    required = {"condition_id", "time_bin", "rmse_mm"}
    if per_condition_time.empty or not required.issubset(per_condition_time.columns):
        return None
    df = _finite_df(per_condition_time, ["time_bin", "rmse_mm"])
    if df.empty:
        return None
    top_conditions = (
        df.groupby("condition_id", dropna=False)["rmse_mm"]
        .mean()
        .sort_values(ascending=False)
        .head(int(top_n_conditions))
        .index
    )
    df = df.loc[df["condition_id"].isin(top_conditions)].copy()
    if df.empty:
        return None
    pivot = df.pivot_table(index="condition_id", columns="time_bin", values="rmse_mm", aggfunc="mean")
    pivot = pivot.loc[top_conditions.intersection(pivot.index)]
    if pivot.empty:
        return None
    label_map = _condition_label_map(points)
    y_labels = [_short_label(label_map.get(idx, idx), max_len=28) for idx in pivot.index]
    x_labels = [f"{float(col):.2g}" for col in pivot.columns]

    fig_h = max(4.6, 0.26 * len(pivot) + 1.8)
    fig, ax = plt.subplots(figsize=(9.4, fig_h), dpi=180)
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Time bin")
    ax.set_title(f"{title}: RMSE heatmap for worst conditions")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("RMSE [mm]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _save_reliability_curve(reliability: pd.DataFrame, out_path: Path, *, title: str) -> str | None:
    if reliability.empty or "probability_level" not in reliability.columns:
        return None
    x = pd.to_numeric(reliability["probability_level"], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=180)
    ax.plot([0, 1], [0, 1], color="#222222", linestyle="--", linewidth=1.1, label="ideal")
    plotted = False
    for col, label, color in (
        ("empirical_lower_tail_fraction_unweighted", "unweighted", "#2F6FA3"),
        ("empirical_lower_tail_fraction_weighted_by_n_points_in_p50_bin", "weighted", "#C26A2E"),
    ):
        if col not in reliability.columns:
            continue
        y = pd.to_numeric(reliability[col], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, color=color, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Nominal lower-tail probability")
    ax.set_ylabel("Empirical lower-tail fraction")
    ax.set_title(f"{title}: reliability curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _save_pit_histogram(pit_hist: pd.DataFrame, out_path: Path, *, title: str) -> str | None:
    required = {"pit_bin_left", "pit_bin_right", "fraction_unweighted"}
    if pit_hist.empty or not required.issubset(pit_hist.columns):
        return None
    left = pd.to_numeric(pit_hist["pit_bin_left"], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(pit_hist["pit_bin_right"], errors="coerce").to_numpy(dtype=float)
    width = right - left
    center = left + width / 2.0
    frac = pd.to_numeric(pit_hist["fraction_unweighted"], errors="coerce").to_numpy(dtype=float)
    if len(center) == 0:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    ax.bar(center, frac, width=width * 0.92, color="#2F6FA3", alpha=0.82, label="unweighted")
    if "weighted_fraction_by_n_points_in_p50_bin" in pit_hist.columns:
        weighted = pd.to_numeric(pit_hist["weighted_fraction_by_n_points_in_p50_bin"], errors="coerce").to_numpy(dtype=float)
        ax.step(center, weighted, where="mid", color="#C26A2E", linewidth=2.0, label="weighted")
    ax.axhline(1.0 / max(len(center), 1), color="#222222", linestyle="--", linewidth=1.0, label="uniform")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Fraction")
    ax.set_title(f"{title}: PIT histogram")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _save_probabilistic_summary(prob_summary: pd.DataFrame, out_path: Path, *, title: str) -> str | None:
    required = {"label", "ece", "crps_mean", "sharpness_mm"}
    if prob_summary.empty or not required.issubset(prob_summary.columns):
        return None
    df = prob_summary.copy()
    labels = [str(x).replace("_by_n_points_in_p50_bin", "") for x in df["label"]]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9), dpi=180)
    for ax, col, ylabel, color in (
        (axes[0], "ece", "ECE", "#2F6FA3"),
        (axes[1], "crps_mean", "CRPS [mm]", "#C26A2E"),
        (axes[2], "sharpness_mm", "Mean std [mm]", "#4F8C4A"),
    ):
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        bars = ax.bar(x, values, color=color, edgecolor="white")
        ax.bar_label(bars, fmt="%.3g", fontsize=8, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{title}: probabilistic summary", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _save_trajectory_plot(points: pd.DataFrame, traj_key: Any, out_path: Path, *, title: str) -> str | None:
    required = {"traj_key", "time_ms", "pen_true_mm", "pen_pred_mm", "pen_std_mm"}
    if not required.issubset(points.columns):
        return None
    sub = points.loc[points["traj_key"].astype(str) == str(traj_key)].copy()
    sub = _finite_df(sub, ["time_ms", "pen_true_mm", "pen_pred_mm", "pen_std_mm"]).sort_values("time_ms")
    if sub.empty:
        return None

    x = sub["time_ms"].to_numpy(dtype=float)
    truth = sub["pen_true_mm"].to_numpy(dtype=float)
    pred = sub["pen_pred_mm"].to_numpy(dtype=float)
    std = np.maximum(sub["pen_std_mm"].to_numpy(dtype=float), 0.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax.plot(x, truth, marker="o", markersize=3, linewidth=1.4, color="#222222", label="target")
    ax.plot(x, pred, marker="o", markersize=3, linewidth=1.4, color="#2F6FA3", label="prediction")
    ax.fill_between(x, pred - std, pred + std, color="#2F6FA3", alpha=0.16, label="plus/minus 1 sigma")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Penetration [mm]")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def _save_best_worst_trajectories(
    points: pd.DataFrame,
    per_trajectory: pd.DataFrame,
    out_dir: Path,
    *,
    title: str,
    max_traj_plots: int | None,
) -> dict[str, str]:
    if max_traj_plots == 0:
        return {}
    required = {"traj_key", "n_points", "rmse_mm"}
    if per_trajectory.empty or not required.issubset(per_trajectory.columns) or "traj_key" not in points.columns:
        return {}
    candidates = per_trajectory.copy()
    candidates["n_points"] = pd.to_numeric(candidates["n_points"], errors="coerce")
    candidates["rmse_mm"] = pd.to_numeric(candidates["rmse_mm"], errors="coerce")
    candidates = candidates.loc[(candidates["n_points"] >= 8) & np.isfinite(candidates["rmse_mm"])]
    if candidates.empty:
        return {}
    budget = 2 if max_traj_plots is None else max(0, int(max_traj_plots))
    outputs: dict[str, str] = {}
    selections: list[tuple[str, pd.Series]] = []
    if budget >= 1:
        selections.append(("best", candidates.sort_values("rmse_mm").iloc[0]))
    if budget >= 2 and len(candidates) > 1:
        selections.append(("worst", candidates.sort_values("rmse_mm", ascending=False).iloc[0]))
    for label, row in selections:
        traj_key = row["traj_key"]
        row_title = f"{title}: {label} trajectory, RMSE={float(row['rmse_mm']):.2f} mm"
        path = out_dir / f"trajectory_{label}.png"
        saved = _save_trajectory_plot(points, traj_key, path, title=row_title)
        if saved:
            outputs[f"trajectory_{label}"] = saved
    return outputs


def _save_q1_observed_extrapolated(points: pd.DataFrame, out_path: Path, *, title: str) -> str | None:
    if "is_observed_window" not in points.columns or "time_ms" not in points.columns:
        return None
    required = ["time_ms", "resid_mm", "pen_true_mm", "pen_pred_mm"]
    if any(col not in points.columns for col in required):
        return None
    df = _finite_df(points, required)
    if df.empty:
        return None
    observed = df["is_observed_window"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    df = df.assign(window=np.where(observed, "observed", "extrapolated"))
    rows = []
    for (window, time_ms), group in df.groupby(["window", "time_ms"], dropna=False, sort=True):
        resid = group["resid_mm"].to_numpy(dtype=float)
        rows.append(
            {
                "window": window,
                "time_ms": float(time_ms),
                "rmse_mm": float(np.sqrt(np.mean(resid * resid))),
                "mae_mm": float(np.mean(np.abs(resid))),
                "bias_mm": float(np.mean(resid)),
            }
        )
    stats = pd.DataFrame(rows)
    if stats.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), dpi=180)
    colors = {"observed": "#2F6FA3", "extrapolated": "#C26A2E"}
    for window, sub in stats.groupby("window", sort=True):
        sub = sub.sort_values("time_ms")
        axes[0].plot(sub["time_ms"], sub["rmse_mm"], marker="o", linewidth=1.8, color=colors.get(window), label=window)
        axes[1].plot(sub["time_ms"], sub["bias_mm"], marker="s", linewidth=1.8, color=colors.get(window), label=window)
    axes[0].set_ylabel("RMSE [mm]")
    axes[0].set_title("RMSE by time")
    axes[1].axhline(0.0, color="#222222", linewidth=0.9)
    axes[1].set_ylabel("Bias [mm]")
    axes[1].set_title("Bias by time")
    for ax in axes:
        ax.set_xlabel("Time [ms]")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{title}: observed vs. extrapolated grid", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _save_root_metrics(eval_dir: Path, out_dir: Path) -> dict[str, str]:
    path = eval_dir / "per_run_metrics.csv"
    if not path.exists():
        return {}
    per_run = pd.read_csv(path)
    if per_run.empty or "eval_set" not in per_run.columns:
        return {}
    outputs: dict[str, str] = {}
    labels = [str(x).replace("_", "\n") for x in per_run["eval_set"]]
    x = np.arange(len(per_run))

    if {"rmse_mm", "mae_mm"}.issubset(per_run.columns):
        width = 0.36
        fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=180)
        rmse = pd.to_numeric(per_run["rmse_mm"], errors="coerce").to_numpy(dtype=float)
        mae = pd.to_numeric(per_run["mae_mm"], errors="coerce").to_numpy(dtype=float)
        bars1 = ax.bar(x - width / 2, rmse, width=width, color="#2F6FA3", label="RMSE")
        bars2 = ax.bar(x + width / 2, mae, width=width, color="#C26A2E", label="MAE")
        ax.bar_label(bars1, fmt="%.2f", fontsize=7, padding=2)
        ax.bar_label(bars2, fmt="%.2f", fontsize=7, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Error [mm]")
        ax.set_title("Point-table evaluation errors")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        out_path = out_dir / "per_run_rmse_mae.png"
        fig.savefig(out_path)
        plt.close(fig)
        outputs["per_run_rmse_mae"] = str(out_path)

    prob_cols = [col for col in ("prob_ece", "prob_crps_mean", "prob_sharpness_mm") if col in per_run.columns]
    prob_df = per_run.loc[per_run[prob_cols].notna().any(axis=1)].copy() if prob_cols else pd.DataFrame()
    if not prob_df.empty:
        labels = [str(x).replace("_", "\n") for x in prob_df["eval_set"]]
        x = np.arange(len(prob_df))
        fig, axes = plt.subplots(1, len(prob_cols), figsize=(3.8 * len(prob_cols), 4.2), dpi=180)
        axes_arr = np.atleast_1d(axes)
        for ax, col, color in zip(axes_arr, prob_cols, ["#2F6FA3", "#C26A2E", "#4F8C4A"]):
            values = pd.to_numeric(prob_df[col], errors="coerce").to_numpy(dtype=float)
            bars = ax.bar(x, values, color=color, edgecolor="white")
            ax.bar_label(bars, fmt="%.3g", fontsize=8, padding=2)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(col.replace("prob_", "").replace("_", " "))
            ax.grid(axis="y", alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Probabilistic headline metrics", y=1.03, fontsize=12)
        fig.tight_layout()
        out_path = out_dir / "per_run_probabilistic.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        outputs["per_run_probabilistic"] = str(out_path)

    return outputs


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def export_point_eval_figures(
    eval_dir: Path | str,
    *,
    sample_points: int = DEFAULT_SAMPLE_POINTS,
    top_n_conditions: int = DEFAULT_TOP_CONDITIONS,
    max_traj_plots: int | None = None,
) -> dict[str, Any]:
    """Export diagnostic figures for a point-table evaluation directory."""
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Point eval directory not found: {eval_dir}")

    root_fig_dir = eval_dir / "figures"
    root_fig_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "eval_dir": str(eval_dir),
        "figures_dir": str(root_fig_dir),
        "sample_points": int(sample_points),
        "top_n_conditions": int(top_n_conditions),
        "eval_sets": {},
        "outputs": {},
    }
    manifest["outputs"].update(_save_root_metrics(eval_dir, root_fig_dir))

    for set_dir in sorted(path for path in eval_dir.iterdir() if path.is_dir()):
        points_path = set_dir / "points.csv"
        if not points_path.exists():
            continue
        eval_set = set_dir.name
        title = _eval_title(eval_set)
        include_uncertainty = eval_set in {"cdf_uncensored", "p50_observed"}
        fig_dir = set_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        points = pd.read_csv(points_path, low_memory=False)
        outputs: dict[str, str] = {}

        saved = _save_pred_vs_actual(points, fig_dir / "pred_vs_actual.png", title=title, sample_points=sample_points)
        if saved:
            outputs["pred_vs_actual"] = saved
        saved = _save_residual_hist(points, fig_dir / "residual_histogram.png", title=title)
        if saved:
            outputs["residual_histogram"] = saved

        per_condition = _read_csv_if_exists(set_dir / "per_condition.csv")
        if per_condition is not None:
            outputs.update(
                _save_per_condition(
                    per_condition,
                    points,
                    fig_dir / "per_condition_rmse.png",
                    fig_dir / "per_condition_coverage.png",
                    title=title,
                    top_n_conditions=top_n_conditions,
                    include_uncertainty=include_uncertainty,
                )
            )

        per_time = _read_csv_if_exists(set_dir / "per_time_bin.csv")
        if per_time is not None:
            saved = _save_per_time_bin(
                per_time,
                fig_dir / "per_time_bin_metrics.png",
                title=title,
                include_uncertainty=include_uncertainty,
            )
            if saved:
                outputs["per_time_bin_metrics"] = saved

        per_condition_time = _read_csv_if_exists(set_dir / "per_condition_time_bin.csv")
        if per_condition_time is not None:
            saved = _save_condition_time_heatmap(
                per_condition_time,
                points,
                fig_dir / "per_condition_time_rmse_heatmap.png",
                title=title,
                top_n_conditions=top_n_conditions,
            )
            if saved:
                outputs["per_condition_time_rmse_heatmap"] = saved

        reliability = _read_csv_if_exists(set_dir / "reliability_curve.csv")
        if reliability is not None:
            saved = _save_reliability_curve(reliability, fig_dir / "reliability_curve.png", title=title)
            if saved:
                outputs["reliability_curve"] = saved

        pit_hist = _read_csv_if_exists(set_dir / "pit_histogram.csv")
        if pit_hist is not None:
            saved = _save_pit_histogram(pit_hist, fig_dir / "pit_histogram.png", title=title)
            if saved:
                outputs["pit_histogram"] = saved

        prob_summary = _read_csv_if_exists(set_dir / "probabilistic_summary.csv")
        if prob_summary is not None:
            saved = _save_probabilistic_summary(prob_summary, fig_dir / "probabilistic_summary.png", title=title)
            if saved:
                outputs["probabilistic_summary"] = saved

        per_trajectory = _read_csv_if_exists(set_dir / "per_trajectory.csv")
        if per_trajectory is not None:
            outputs.update(
                _save_best_worst_trajectories(
                    points,
                    per_trajectory,
                    fig_dir,
                    title=title,
                    max_traj_plots=max_traj_plots,
                )
            )

        saved = _save_q1_observed_extrapolated(points, fig_dir / "observed_vs_extrapolated_error.png", title=title)
        if saved:
            outputs["observed_vs_extrapolated_error"] = saved

        manifest["eval_sets"][eval_set] = {
            "figures_dir": str(fig_dir),
            "n_points": int(len(points)),
            "outputs": outputs,
        }

    manifest_path = eval_dir / "figure_manifest.json"
    manifest["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--sample-points", type=int, default=DEFAULT_SAMPLE_POINTS)
    parser.add_argument("--top-n-conditions", type=int, default=DEFAULT_TOP_CONDITIONS)
    parser.add_argument("--max-traj-plots", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = export_point_eval_figures(
        args.eval_dir,
        sample_points=args.sample_points,
        top_n_conditions=args.top_n_conditions,
        max_traj_plots=args.max_traj_plots,
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
