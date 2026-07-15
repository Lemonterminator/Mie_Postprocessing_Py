"""Per-(model, dataset, eval-set) diagnostic figures.

Every function reads only Layer-1 artifacts inside one eval-set directory and
returns the written file paths; missing inputs are skipped, not errors.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from MLP.eval_pipeline.common import read_points
from MLP.eval_pipeline.datasets import coerce_bool_series
from MLP.eval_pipeline.figures.style import (
    ACCENT_HUE,
    BASELINE,
    INK_MUTED,
    INK_SECONDARY,
    SEQUENTIAL_CMAP,
    SINGLE_HUE,
    WARN_HUE,
    save_fig,
)

HEXBIN_THRESHOLD = 20_000
TIME_BIN_MS = 0.1
_CONFIDENTIAL_CONDITION_PREFIX = re.compile(r"^BC\d{8}_HZ_(?=Nozzle\d+\|)")


def sanitize_condition_label(value: object) -> str:
    """Remove the confidential archive prefix from a displayed condition key."""
    return _CONFIDENTIAL_CONDITION_PREFIX.sub("", str(value))


def _read_csv(eval_dir: Path, name: str) -> pd.DataFrame | None:
    path = eval_dir / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def _maybe_points(eval_dir: Path, max_points: int) -> pd.DataFrame | None:
    try:
        points = read_points(eval_dir)
    except FileNotFoundError:
        return None
    if len(points) > max_points:
        points = points.sample(max_points, random_state=42)
    return points


def _title(meta: dict[str, Any], what: str) -> str:
    return f"{meta['model_label']} | {meta['dataset']} | {meta['eval_set']} — {what}"


def _time_axis_ms(values: pd.Series, meta: dict[str, Any]) -> np.ndarray:
    """Return plot-ready milliseconds for mixed legacy/exact time-bin encodings.

    CDF/P50 tables carry integer time-bin ids, while q1_grid_all uses exact
    time_ms passthrough to avoid 0.1-ms floating collisions.
    """
    t = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return t
    is_integer_index = np.all(np.isclose(finite, np.round(finite), atol=1e-8))
    if is_integer_index and (meta.get("eval_set") != "q1_grid_all" or np.nanmax(finite) > 5.0):
        return t * TIME_BIN_MS
    return t


def fig_pred_vs_actual(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                       *, max_points: int, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    points = _maybe_points(eval_dir, max_points)
    if points is None:
        return []
    truth = points["pen_true_mm"].to_numpy()
    pred = points["pen_pred_mm"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    hb = ax.hexbin(
        truth, pred, gridsize=64, cmap=SEQUENTIAL_CMAP,
        mincnt=1, linewidths=0, norm=LogNorm(),
    )
    fig.colorbar(hb, ax=ax, label="points per cell (log scale)", shrink=0.85)
    lo = min(truth.min(), pred.min())
    hi = max(truth.max(), pred.max())
    pad = max((hi - lo) * 0.025, 1e-6)
    lims = [lo - pad, hi + pad]
    ax.plot(lims, lims, "--", color=INK_SECONDARY, linewidth=1.2, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    overall = meta.get("overall", {})
    ax.annotate(
        f"RMSE {overall.get('rmse_mm', float('nan')):.2f} mm\n"
        f"MAE  {overall.get('mae_mm', float('nan')):.2f} mm\n"
        f"n = {overall.get('n_points', len(points)):,}",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top",
        fontsize=9, color=INK_SECONDARY,
    )
    ax.set_xlabel("observed penetration [mm]")
    ax.set_ylabel("predicted penetration [mm]")
    ax.set_title(_title(meta, "predicted vs observed"))
    ax.legend(loc="lower right")
    return save_fig(fig, fig_dir / "pred_vs_actual", dpi=dpi, formats=formats)


def fig_residual_histogram(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                           *, max_points: int, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    points = _maybe_points(eval_dir, max_points)
    if points is None:
        return []
    resid = points["resid_mm"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.hist(resid, bins=60, color=SINGLE_HUE, alpha=0.9, edgecolor="white", linewidth=0.3)
    ax.axvline(0.0, color=INK_SECONDARY, linestyle="--", linewidth=1.2)
    overall = meta.get("overall", {})
    ax.annotate(
        f"bias {overall.get('bias_mm', float('nan')):+.2f} mm\n"
        f"P95 |err| {overall.get('p95_abs_err_mm', float('nan')):.2f} mm",
        xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=9, color=INK_SECONDARY,
    )
    ax.set_xlabel("residual (pred − true) [mm]")
    ax.set_ylabel("count")
    ax.set_title(_title(meta, "residual distribution"))
    return save_fig(fig, fig_dir / "residual_histogram", dpi=dpi, formats=formats)


def fig_residual_vs_truth(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                          *, max_points: int, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    points = _maybe_points(eval_dir, max_points)
    if points is None:
        return []
    truth = points["pen_true_mm"].to_numpy()
    resid = points["resid_mm"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if len(points) > HEXBIN_THRESHOLD:
        hb = ax.hexbin(truth, resid, gridsize=60, cmap=SEQUENTIAL_CMAP, mincnt=1, linewidths=0)
        fig.colorbar(hb, ax=ax, label="points per cell", shrink=0.85)
    else:
        ax.scatter(truth, resid, s=8, alpha=0.35, color=SINGLE_HUE, edgecolors="none")
    ax.axhline(0.0, color=INK_SECONDARY, linestyle="--", linewidth=1.2)
    ax.set_xlabel("observed penetration [mm]")
    ax.set_ylabel("residual [mm]")
    ax.set_title(_title(meta, "residual vs truth"))
    return save_fig(fig, fig_dir / "residual_vs_truth", dpi=dpi, formats=formats)


def fig_per_condition_rmse(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                           *, top_n: int, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    table = _read_csv(eval_dir, "per_condition.csv")
    if table is None or "rmse_mm" not in table.columns:
        return []
    label_col = "condition_key" if "condition_key" in table.columns else (
        "condition_id" if "condition_id" in table.columns else None)
    if label_col is None:
        return []
    worst = table.sort_values("rmse_mm", ascending=False).head(top_n).iloc[::-1].copy()
    worst["_display_label"] = worst[label_col].map(sanitize_condition_label)
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.24 * len(worst))))
    ax.barh(worst["_display_label"], worst["rmse_mm"], color=SINGLE_HUE, height=0.7)
    ax.set_xlabel("RMSE [mm]")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(_title(meta, f"worst {len(worst)} conditions by RMSE"))
    return save_fig(fig, fig_dir / "per_condition_rmse", dpi=dpi, formats=formats)


def fig_qc_per_condition_rmse_comparison(
    lv2_eval_dir: Path,
    lv3_eval_dir: Path,
    fig_dir: Path,
    *,
    model_label: str,
    top_n: int,
    dpi: int,
    formats: tuple[str, ...],
) -> list[Path]:
    """Compare the same LV2 high-RMSE conditions before and after QC gating.

    The selection is made once from LV2, then the matching LV3 values are
    plotted beside it.  This avoids the selection bias of independently taking
    the worst conditions from each population.
    """
    lv2 = _read_csv(lv2_eval_dir, "per_condition.csv")
    lv3 = _read_csv(lv3_eval_dir, "per_condition.csv")
    required = {"condition_key", "rmse_mm"}
    if lv2 is None or lv3 is None or not required <= set(lv2.columns) or not required <= set(lv3.columns):
        return []
    if lv2["condition_key"].duplicated().any() or lv3["condition_key"].duplicated().any():
        raise ValueError("QC comparison requires one unique row per condition_key in both tables.")

    paired = (lv2[["condition_key", "rmse_mm"]].rename(columns={"rmse_mm": "rmse_lv2_mm"})
              .merge(
                  lv3[["condition_key", "rmse_mm"]].rename(columns={"rmse_mm": "rmse_lv3_mm"}),
                  on="condition_key", how="inner", validate="one_to_one",
              ))
    if len(paired) != len(lv2) or len(paired) != len(lv3):
        raise ValueError("QC comparison requires identical LV2 and LV3 condition-key support.")
    paired["display_label"] = paired["condition_key"].map(sanitize_condition_label)
    if paired["display_label"].duplicated().any():
        raise ValueError("Condition-label sanitization created non-unique displayed labels.")

    selected = paired.nlargest(top_n, "rmse_lv2_mm").sort_values("rmse_lv2_mm", ascending=False)
    y = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(8.2, max(4.2, 0.30 * len(selected))))
    ax.barh(y - 0.19, selected["rmse_lv2_mm"], height=0.36, color=INK_SECONDARY, label="LV2 (pre-QC)")
    ax.barh(y + 0.19, selected["rmse_lv3_mm"], height=0.36, color=SINGLE_HUE, label="LV3 (QC-gated)")
    ax.set_yticks(y, selected["display_label"].tolist(), fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE [mm]")
    ax.set_ylabel("")
    ax.set_title(f"{model_label} | full-clean — LV2 top-{len(selected)} RMSE conditions before/after QC")
    ax.legend(loc="lower right")
    return save_fig(fig, fig_dir / "per_condition_rmse_lv2_vs_lv3_qc", dpi=dpi, formats=formats)


def fig_per_time_bin(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                     *, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    table = _read_csv(eval_dir, "per_time_bin.csv")
    if table is None or "time_bin" not in table.columns:
        return []
    table = table.sort_values("time_bin")
    t = _time_axis_ms(table["time_bin"], meta)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(t, table["rmse_mm"], color=SINGLE_HUE, label="RMSE")
    ax.plot(t, table["mae_mm"], color=ACCENT_HUE, label="MAE")
    ax.plot(t, table["bias_mm"], color=WARN_HUE, label="bias")
    ax.axhline(0.0, color=BASELINE, linewidth=1.0)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("error [mm]")
    ax.set_title(_title(meta, "error vs time"))
    ax.legend(loc="upper left", ncols=3)
    return save_fig(fig, fig_dir / "per_time_bin_metrics", dpi=dpi, formats=formats)


def fig_condition_time_heatmap(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                               *, top_n: int, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    table = _read_csv(eval_dir, "per_condition_time_bin.csv")
    if table is None or not {"condition_id", "time_bin", "rmse_mm"} <= set(table.columns):
        return []
    worst_ids = (table.groupby("condition_id")["rmse_mm"].mean()
                 .sort_values(ascending=False).head(top_n).index)
    pivot = (table[table["condition_id"].isin(worst_ids)]
             .pivot_table(index="condition_id", columns="time_bin", values="rmse_mm"))
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.22 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=SEQUENTIAL_CMAP, interpolation="nearest")
    ax.set_yticks(range(len(pivot)), [str(i) for i in pivot.index], fontsize=7)
    xt = np.arange(0, pivot.shape[1], max(1, pivot.shape[1] // 10))
    ax.set_xticks(xt, [f"{c:.1f}" for c in _time_axis_ms(pd.Series(pivot.columns[xt]), meta)])
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("condition_id")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="RMSE [mm]", shrink=0.85)
    ax.set_title(_title(meta, f"RMSE heatmap (worst {len(pivot)} conditions)"))
    return save_fig(fig, fig_dir / "condition_time_rmse_heatmap", dpi=dpi, formats=formats)


def fig_reliability(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                    *, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    table = _read_csv(eval_dir, "reliability_curve.csv")
    if table is None:
        return []
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color=BASELINE, linewidth=1.2, label="ideal")
    ax.plot(table["probability_level"], table["empirical_lower_tail_fraction"],
            color=SINGLE_HUE, marker="o", markersize=4, label="empirical")
    if "empirical_lower_tail_fraction_weighted" in table.columns:
        ax.plot(table["probability_level"], table["empirical_lower_tail_fraction_weighted"],
                color=ACCENT_HUE, linestyle="--", marker="s", markersize=3.5, label="weighted")
    ece = meta.get("overall", {}).get("ece")
    if ece is not None:
        ax.annotate(f"ECE {ece:.4f}", xy=(0.03, 0.97), xycoords="axes fraction",
                    va="top", fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("nominal lower-tail probability")
    ax.set_ylabel("empirical fraction (PIT ≤ level)")
    ax.set_title(_title(meta, "PIT reliability"))
    ax.legend(loc="lower right")
    return save_fig(fig, fig_dir / "reliability_curve", dpi=dpi, formats=formats)


def fig_pit_histogram(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                      *, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    table = _read_csv(eval_dir, "pit_histogram.csv")
    if table is None:
        return []
    centers = (table["pit_bin_left"] + table["pit_bin_right"]) / 2.0
    width = (table["pit_bin_right"] - table["pit_bin_left"]).iloc[0]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(centers, table["fraction"], width=width * 0.92, color=SINGLE_HUE)
    ax.axhline(1.0 / len(table), color=INK_SECONDARY, linestyle="--",
               linewidth=1.2, label="uniform")
    if "weighted_fraction" in table.columns:
        ax.step(np.r_[table["pit_bin_left"], table["pit_bin_right"].iloc[-1]],
                np.r_[table["weighted_fraction"], table["weighted_fraction"].iloc[-1]],
                where="post", color=ACCENT_HUE, linewidth=1.6, label="weighted")
    ax.set_xlabel("PIT")
    ax.set_ylabel("fraction")
    ax.set_title(_title(meta, "PIT histogram"))
    ax.legend(loc="upper center", ncols=2)
    return save_fig(fig, fig_dir / "pit_histogram", dpi=dpi, formats=formats)


def fig_coverage_curve(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                       *, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    table = _read_csv(eval_dir, "coverage_curve.csv")
    if table is None:
        return []
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(table["k_sigma"], table["nominal_gaussian_coverage"], "--",
            color=BASELINE, linewidth=1.4, label="Gaussian nominal")
    ax.plot(table["k_sigma"], table["empirical_coverage"],
            color=SINGLE_HUE, marker="o", markersize=4, label="empirical")
    ax.set_xlabel("k (multiple of predicted σ)")
    ax.set_ylabel("coverage P(|err| ≤ kσ)")
    ax.set_ylim(0, 1.02)
    ax.set_title(_title(meta, "coverage vs k·σ"))
    ax.legend(loc="lower right")
    return save_fig(fig, fig_dir / "coverage_curve", dpi=dpi, formats=formats)


def fig_sigma_bins(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                   *, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    table = _read_csv(eval_dir, "sigma_bin_calibration.csv")
    if table is None:
        return []
    x = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].bar(x, table["coverage_1sigma"], color=SINGLE_HUE, width=0.38, label="1σ empirical")
    axes[0].bar(x + 0.4, table["coverage_2sigma"], color=ACCENT_HUE, width=0.38, label="2σ empirical")
    axes[0].axhline(0.6827, color=SINGLE_HUE, linestyle="--", linewidth=1.0)
    axes[0].axhline(0.9545, color=ACCENT_HUE, linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("coverage")
    axes[0].set_xlabel("σ decile bin (small → large)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("coverage by predicted-σ bin")
    axes[1].plot(x, table["mean_abs_z"], color=SINGLE_HUE, marker="o", label="mean |z|")
    axes[1].axhline(0.7979, color=BASELINE, linestyle="--", linewidth=1.2,
                    label="Gaussian ideal √(2/π)")
    axes[1].set_xlabel("σ decile bin (small → large)")
    axes[1].set_ylabel("mean |z|")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_title("standardized error by σ bin")
    fig.suptitle(_title(meta, "σ-bin calibration audit"), fontsize=11)
    return save_fig(fig, fig_dir / "sigma_bin_calibration", dpi=dpi, formats=formats)


def fig_trajectories(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                     *, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    per_traj = _read_csv(eval_dir, "per_trajectory.csv")
    if per_traj is None or "traj_key" not in per_traj.columns:
        return []
    try:
        points = read_points(eval_dir)
    except FileNotFoundError:
        return []
    if "traj_key" not in points.columns:
        return []
    ranked = per_traj.dropna(subset=["rmse_mm"]).sort_values("rmse_mm")
    if ranked.empty:
        return []
    written: list[Path] = []
    for tag, row in (("best", ranked.iloc[0]), ("worst", ranked.iloc[-1])):
        traj = points[points["traj_key"] == row["traj_key"]].sort_values("time_ms")
        if traj.empty:
            continue
        t = traj["time_ms"].to_numpy()
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.fill_between(t, traj["pen_pred_mm"] - 2 * traj["pen_std_mm"],
                        traj["pen_pred_mm"] + 2 * traj["pen_std_mm"],
                        color=SINGLE_HUE, alpha=0.18, label="±2σ")
        ax.plot(t, traj["pen_pred_mm"], color=SINGLE_HUE, label="predicted")
        ax.scatter(t, traj["pen_true_mm"], s=12, color=INK_SECONDARY,
                   zorder=3, label="observed")
        ax.set_xlabel("time [ms]")
        ax.set_ylabel("penetration [mm]")
        ax.set_title(_title(meta, f"{tag} trajectory (RMSE {row['rmse_mm']:.2f} mm)\n"
                                  f"{row['traj_key']}"), fontsize=9)
        ax.legend(loc="lower right")
        written += save_fig(fig, fig_dir / f"trajectory_{tag}", dpi=dpi, formats=formats)
    return written


def fig_observed_vs_extrapolated(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                                 *, max_points: int, dpi: int,
                                 formats: tuple[str, ...]) -> list[Path]:
    try:
        points = read_points(eval_dir)
    except FileNotFoundError:
        return []
    if "is_observed_window" not in points.columns:
        return []
    observed = coerce_bool_series(points["is_observed_window"])
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for label, mask, color in (("observed window", observed, SINGLE_HUE),
                               ("extrapolated", ~observed, WARN_HUE)):
        sub = points.loc[mask]
        if sub.empty:
            continue
        binned = (sub.assign(bin=(sub["time_ms"] / 0.1).astype(int))
                  .groupby("bin")["resid_mm"]
                  .apply(lambda r: float(np.sqrt(np.mean(r**2)))))
        ax.plot(binned.index * 0.1, binned.to_numpy(), color=color, label=label)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("RMSE [mm]")
    ax.set_title(_title(meta, "observed vs extrapolated error"))
    ax.legend(loc="upper left")
    return save_fig(fig, fig_dir / "observed_vs_extrapolated_rmse", dpi=dpi, formats=formats)


def render_eval_set_figures(eval_dir: Path, fig_dir: Path, meta: dict[str, Any],
                            *, max_points: int = 80_000, top_n_conditions: int = 30,
                            dpi: int = 200, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    """Render the full per-model diagnostic suite for one eval-set directory."""
    written: list[Path] = []
    written += fig_pred_vs_actual(eval_dir, fig_dir, meta, max_points=max_points, dpi=dpi, formats=formats)
    written += fig_residual_histogram(eval_dir, fig_dir, meta, max_points=max_points, dpi=dpi, formats=formats)
    written += fig_residual_vs_truth(eval_dir, fig_dir, meta, max_points=max_points, dpi=dpi, formats=formats)
    written += fig_per_condition_rmse(eval_dir, fig_dir, meta, top_n=top_n_conditions, dpi=dpi, formats=formats)
    written += fig_per_time_bin(eval_dir, fig_dir, meta, dpi=dpi, formats=formats)
    written += fig_condition_time_heatmap(eval_dir, fig_dir, meta, top_n=top_n_conditions, dpi=dpi, formats=formats)
    written += fig_reliability(eval_dir, fig_dir, meta, dpi=dpi, formats=formats)
    written += fig_pit_histogram(eval_dir, fig_dir, meta, dpi=dpi, formats=formats)
    written += fig_coverage_curve(eval_dir, fig_dir, meta, dpi=dpi, formats=formats)
    written += fig_sigma_bins(eval_dir, fig_dir, meta, dpi=dpi, formats=formats)
    written += fig_trajectories(eval_dir, fig_dir, meta, dpi=dpi, formats=formats)
    written += fig_observed_vs_extrapolated(eval_dir, fig_dir, meta, max_points=max_points, dpi=dpi, formats=formats)
    return written
