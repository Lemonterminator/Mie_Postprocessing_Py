"""Cross-model comparison figures for one (dataset, eval-set) pair.

Inputs: the ``metrics_wide.csv`` slice for that pair (one row per model,
``slice == 'overall'``) plus each model's per-eval-set artifact directory.
Model → hue follows the *family* (fixed order from the run manifest); models
within a family are distinguished by linestyle/alpha and the legend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MLP.eval_pipeline.figures.style import (
    BASELINE,
    INK_MUTED,
    INK_SECONDARY,
    save_fig,
)

_LINESTYLES = ("-", "--", "-.", ":")


def _bar_colors(sub: pd.DataFrame, colors: dict[str, str]) -> list[str]:
    return [colors.get(f, INK_MUTED) for f in sub["model_family"]]


def _annotate_bars(ax, bars, values, fmt="{:.2f}") -> None:
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=7, color=INK_SECONDARY)


def fig_metric_bars(sub: pd.DataFrame, fig_dir: Path, colors: dict[str, str],
                    *, title: str, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    metrics = [("rmse_mm", "RMSE [mm]"), ("mae_mm", "MAE [mm]"), ("p95_abs_err_mm", "P95 |err| [mm]")]
    metrics = [(k, lab) for k, lab in metrics if k in sub.columns]
    if not metrics or sub.empty:
        return []
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(6.5, 0.55 * len(sub)), 2.6 * len(metrics)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(sub))
    for ax, (key, label) in zip(axes, metrics):
        values = pd.to_numeric(sub[key], errors="coerce").to_numpy()
        bars = ax.bar(x, values, color=_bar_colors(sub, colors), width=0.7)
        _annotate_bars(ax, bars, values)
        ax.set_ylabel(label)
    axes[-1].set_xticks(x, sub["model_label"], rotation=35, ha="right", fontsize=8)
    axes[0].set_title(title)
    return save_fig(fig, fig_dir / "metric_bars_rmse_mae_p95", dpi=dpi, formats=formats)


def fig_prob_metric_bars(sub: pd.DataFrame, fig_dir: Path, colors: dict[str, str],
                         *, title: str, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    metrics = [("crps_mean_mm", "CRPS [mm]"), ("nll_mm", "NLL"), ("ece", "ECE")]
    metrics = [(k, lab) for k, lab in metrics if k in sub.columns]
    plot_sub = sub.dropna(subset=[k for k, _ in metrics], how="all")
    if not metrics or plot_sub.empty:
        return []
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(max(6.5, 0.55 * len(plot_sub)), 2.6 * len(metrics)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(plot_sub))
    for ax, (key, label) in zip(axes, metrics):
        values = pd.to_numeric(plot_sub[key], errors="coerce").to_numpy()
        bars = ax.bar(x, values, color=_bar_colors(plot_sub, colors), width=0.7)
        _annotate_bars(ax, bars, values, fmt="{:.3f}")
        ax.set_ylabel(label)
    axes[-1].set_xticks(x, plot_sub["model_label"], rotation=35, ha="right", fontsize=8)
    axes[0].set_title(title)
    return save_fig(fig, fig_dir / "probabilistic_metric_bars", dpi=dpi, formats=formats)


def fig_coverage_comparison(sub: pd.DataFrame, fig_dir: Path, colors: dict[str, str],
                            *, title: str, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    if sub.empty or "coverage_1sigma" not in sub.columns:
        return []
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(max(7.0, 0.6 * len(sub)), 4.2))
    c1 = pd.to_numeric(sub["coverage_1sigma"], errors="coerce").to_numpy()
    c2 = pd.to_numeric(sub["coverage_2sigma"], errors="coerce").to_numpy()
    ax.bar(x - 0.2, c1, width=0.38, color=_bar_colors(sub, colors), label="1σ empirical")
    ax.bar(x + 0.2, c2, width=0.38, color=_bar_colors(sub, colors), alpha=0.55,
           label="2σ empirical")
    # Offset and mask the labels so neither nominal reference line crosses
    # through its own text (particularly visible on wide comparison panels).
    nominal_label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.92,
                         "pad": 0.15}
    ax.axhline(0.6827, color=INK_SECONDARY, linestyle="--", linewidth=1.1)
    ax.annotate("nominal 68.3%", xy=(x[-1] + 0.45, 0.6827), xytext=(4, 5),
                textcoords="offset points", fontsize=8, color=INK_SECONDARY,
                ha="left", va="bottom", bbox=nominal_label_box)
    ax.axhline(0.9545, color=INK_SECONDARY, linestyle=":", linewidth=1.1)
    ax.annotate("nominal 95.4%", xy=(x[-1] + 0.45, 0.9545), xytext=(4, 5),
                textcoords="offset points", fontsize=8, color=INK_SECONDARY,
                ha="left", va="bottom", bbox=nominal_label_box)
    ax.set_xticks(x, sub["model_label"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.6, len(sub) + 1.4)
    ax.legend(loc="lower right")
    ax.set_title(title)
    return save_fig(fig, fig_dir / "coverage_comparison", dpi=dpi, formats=formats)


def _family_styles(sub: pd.DataFrame) -> dict[str, tuple[str, float]]:
    """Per-model (linestyle, alpha) so same-family members stay distinguishable."""
    styles: dict[str, tuple[str, float]] = {}
    for family, group in sub.groupby("model_family", sort=False):
        for i, label in enumerate(group["model_label"]):
            styles[label] = (_LINESTYLES[i % len(_LINESTYLES)], max(0.45, 1.0 - 0.15 * i))
    return styles


def fig_reliability_overlay(sub: pd.DataFrame, model_dirs: dict[str, Path], fig_dir: Path,
                            colors: dict[str, str], *, title: str, dpi: int,
                            formats: tuple[str, ...]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], "--", color=BASELINE, linewidth=1.4, label="ideal")
    styles = _family_styles(sub)
    plotted = 0
    for _, row in sub.iterrows():
        eval_dir = model_dirs.get(row["model_label"])
        if eval_dir is None:
            continue
        path = eval_dir / "reliability_curve.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        ls, alpha = styles.get(row["model_label"], ("-", 1.0))
        ax.plot(table["probability_level"], table["empirical_lower_tail_fraction"],
                color=colors.get(row["model_family"], INK_MUTED), linestyle=ls,
                alpha=alpha, linewidth=1.8, label=row["model_label"])
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return []
    ax.set_xlabel("nominal lower-tail probability")
    ax.set_ylabel("empirical fraction (PIT ≤ level)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7, ncols=1 + plotted // 8)
    return save_fig(fig, fig_dir / "reliability_overlay", dpi=dpi, formats=formats)


def fig_crps_sharpness(sub: pd.DataFrame, fig_dir: Path, colors: dict[str, str],
                       *, title: str, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    needed = {"sharpness_mm", "crps_mean_mm"}
    if not needed <= set(sub.columns):
        return []
    plot_sub = sub.dropna(subset=list(needed))
    if plot_sub.empty:
        return []
    # Dense rosters make point-adjacent labels unreadable.  Keep the data
    # panel clear and use a dedicated, colour-coded legend column instead.
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    for _, row in plot_sub.iterrows():
        ax.scatter(row["sharpness_mm"], row["crps_mean_mm"], s=70,
                   color=colors.get(row["model_family"], INK_MUTED), zorder=3,
                   label=row["model_label"])
    ax.set_xlabel("sharpness — mean predicted σ [mm]")
    ax.set_ylabel("mean CRPS [mm]")
    ax.set_title(title)
    fig.subplots_adjust(right=0.66)
    ax.legend(title="model", loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=7, title_fontsize=8, markerscale=0.8, borderaxespad=0.0,
              handletextpad=0.45)
    return save_fig(fig, fig_dir / "crps_sharpness_scatter", dpi=dpi, formats=formats)


def fig_pit_grid(sub: pd.DataFrame, model_dirs: dict[str, Path], fig_dir: Path,
                 colors: dict[str, str], *, title: str, dpi: int,
                 formats: tuple[str, ...]) -> list[Path]:
    rows = [row for _, row in sub.iterrows()
            if (model_dirs.get(row["model_label"]) or Path("_")).joinpath("pit_histogram.csv").exists()]
    if not rows:
        return []
    ncols = min(4, len(rows))
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 2.4 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, rows):
        table = pd.read_csv(model_dirs[row["model_label"]] / "pit_histogram.csv")
        centers = (table["pit_bin_left"] + table["pit_bin_right"]) / 2.0
        width = (table["pit_bin_right"] - table["pit_bin_left"]).iloc[0]
        ax.bar(centers, table["fraction"], width=width * 0.92,
               color=colors.get(row["model_family"], INK_MUTED))
        ax.axhline(1.0 / len(table), color=INK_SECONDARY, linestyle="--", linewidth=0.9)
        ax.set_title(row["model_label"], fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(rows):]:
        ax.set_visible(False)
    # Reserve dedicated top/bottom bands: the panel titles must not compete
    # with the figure title, and the shared PIT label must sit below ticks.
    fig.subplots_adjust(left=0.11, right=0.99, top=0.80, bottom=0.22,
                        wspace=0.20, hspace=0.25)
    fig.suptitle(title, fontsize=11, y=0.99)
    fig.supxlabel("PIT", fontsize=9, y=0.035)
    fig.supylabel("fraction", fontsize=9)
    return save_fig(fig, fig_dir / "pit_histograms_grid", dpi=dpi, formats=formats)


def fig_per_seed_rmse(sub: pd.DataFrame, fig_dir: Path, colors: dict[str, str],
                      *, title: str, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    written: list[Path] = []
    for family, group in sub.groupby("model_family", sort=False):
        seeds = group.dropna(subset=["seed"]) if "seed" in group.columns else group.iloc[0:0]
        if len(seeds) < 3:
            continue
        seeds = seeds.sort_values("seed")
        values = pd.to_numeric(seeds["rmse_mm"], errors="coerce").to_numpy()
        mean, std = float(np.nanmean(values)), float(np.nanstd(values))
        x = np.arange(len(seeds))
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        bars = ax.bar(x, values, color=colors.get(family, INK_MUTED), width=0.65)
        _annotate_bars(ax, bars, values, fmt="{:.3f}")
        ax.axhline(mean, color=INK_SECONDARY, linestyle="--", linewidth=1.2,
                   label=f"mean {mean:.3f} ± {std:.3f} mm")
        ax.axhspan(mean - std, mean + std, color=INK_MUTED, alpha=0.12)
        ax.set_xticks(x, [f"seed {int(s)}" for s in seeds["seed"]])
        ax.set_ylabel("RMSE [mm]")
        pad = max(std * 4, 0.05)
        finite_values = values[np.isfinite(values)]
        lo, hi = (float(finite_values.min()), float(finite_values.max())) if finite_values.size else (0.0, 1.0)
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
        ax.set_title(f"{title}\n{family} per-seed RMSE")
        ax.legend(loc="upper right", fontsize=8)
        written += save_fig(fig, fig_dir / f"per_seed_rmse_{family}", dpi=dpi, formats=formats)
    return written


def fig_by_fold(sub: pd.DataFrame, fig_dir: Path, colors: dict[str, str],
                *, title: str, dpi: int, formats: tuple[str, ...]) -> list[Path]:
    """LONO-style by-fold comparison; active when models carry meta_holdout."""
    if "meta_holdout" not in sub.columns or sub["meta_holdout"].isna().all():
        return []
    written: list[Path] = []
    folds = sub.dropna(subset=["meta_holdout"]).copy()
    for metric, label, nominal in (("rmse_mm", "RMSE [mm]", None),
                                   ("coverage_1sigma", "1σ coverage", 0.6827)):
        if metric not in folds.columns:
            continue
        pivot = folds.pivot_table(index="meta_holdout", columns="model_family",
                                  values=metric, aggfunc="mean")
        x = np.arange(len(pivot))
        n_fam = len(pivot.columns)
        width = 0.8 / max(n_fam, 1)
        fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(pivot)), 4.2))
        for j, family in enumerate(pivot.columns):
            ax.bar(x + (j - (n_fam - 1) / 2) * width, pivot[family], width=width * 0.92,
                   color=colors.get(family, INK_MUTED), label=family)
        if nominal is not None:
            ax.axhline(nominal, color=INK_SECONDARY, linestyle="--", linewidth=1.1)
        ax.set_xticks(x, pivot.index, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_xlabel("held-out fold")
        ax.legend(fontsize=8)
        ax.set_title(f"{title} — by fold")
        written += save_fig(fig, fig_dir / f"by_fold_{metric}", dpi=dpi, formats=formats)
    return written


def render_comparison_figures(sub: pd.DataFrame, model_dirs: dict[str, Path], fig_dir: Path,
                              colors: dict[str, str], *, title: str, dpi: int = 200,
                              formats: tuple[str, ...] = ("png",)) -> list[Path]:
    """Render every applicable cross-model figure for one (dataset, eval set).

    Each sub-figure is independently fault-tolerant: a crash in one (e.g. a
    NaN/Inf metric from a bad checkpoint) is logged and skipped rather than
    discarding the figures already rendered earlier in this call.
    """
    import traceback

    stages = (
        ("metric_bars", lambda: fig_metric_bars(sub, fig_dir, colors, title=title, dpi=dpi, formats=formats)),
        ("prob_metric_bars", lambda: fig_prob_metric_bars(sub, fig_dir, colors, title=title, dpi=dpi, formats=formats)),
        ("coverage_comparison", lambda: fig_coverage_comparison(sub, fig_dir, colors, title=title, dpi=dpi, formats=formats)),
        ("reliability_overlay", lambda: fig_reliability_overlay(sub, model_dirs, fig_dir, colors, title=title, dpi=dpi, formats=formats)),
        ("crps_sharpness", lambda: fig_crps_sharpness(sub, fig_dir, colors, title=title, dpi=dpi, formats=formats)),
        ("pit_grid", lambda: fig_pit_grid(sub, model_dirs, fig_dir, colors, title=title, dpi=dpi, formats=formats)),
        ("per_seed_rmse", lambda: fig_per_seed_rmse(sub, fig_dir, colors, title=title, dpi=dpi, formats=formats)),
        ("by_fold", lambda: fig_by_fold(sub, fig_dir, colors, title=title, dpi=dpi, formats=formats)),
    )
    written: list[Path] = []
    for name, render in stages:
        try:
            written += render()
        except Exception:
            print(f"[FAIL] comparison figure stage {name!r} for {title}")
            traceback.print_exc()
    return written
