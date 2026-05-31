"""Post-process figures for the residual family-head production summary.

This script is intentionally tiny and reproducible: it redraws the production
summary figures and the residual SVGP follow-up figures with consistent
matplotlib styling, explicit labels, and enough padding for annotation text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SLIDES_DIR = PROJECT_ROOT / "Thesis" / "slides" / "slides_residual_family_head_production"
DEFAULT_SUMMARY_CSV = DEFAULT_SLIDES_DIR / "residual_from_prod_eval_summary.csv"
DEFAULT_VERDICT = DEFAULT_SLIDES_DIR / "verdict.json"
DEFAULT_OUTPUT = DEFAULT_SLIDES_DIR / "figs" / "winner_cdf_deltas.png"
DEFAULT_DELTA_SWEEP_OUTPUT = DEFAULT_SLIDES_DIR / "figs" / "delta_l2_sweep.png"
DEFAULT_SVGP_CONTEXT_CSV = DEFAULT_SLIDES_DIR / "residual_svgp_context_compare.csv"
DEFAULT_SVGP_CONTEXT_OUTPUT = DEFAULT_SLIDES_DIR / "figs" / "residual_svgp_context_rmse_comparison.png"
DEFAULT_SVGP_DELTA_OUTPUT = DEFAULT_SLIDES_DIR / "figs" / "residual_svgp_vs_current_best_deltas.png"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_winner(summary: pd.DataFrame, verdict: dict[str, Any] | None) -> pd.Series:
    if verdict and verdict.get("winner"):
        winner_run = str(verdict["winner"].get("run_dir", ""))
        rows = summary.loc[summary["run_dir"] == winner_run]
        if not rows.empty:
            return rows.iloc[0]
    residual = summary.loc[summary["kind"] == "residual"].copy()
    if "returncode" in residual.columns:
        residual = residual.loc[pd.to_numeric(residual["returncode"], errors="coerce").fillna(1).eq(0)]
    if residual.empty:
        raise ValueError("No residual winner row found in summary CSV.")
    residual = residual.sort_values(["cdf_rmse_mm", "delta_l2_weight"], ascending=[True, False])
    return residual.iloc[0]


def _require_model_row(summary: pd.DataFrame, model_name: str) -> pd.Series:
    rows = summary.loc[summary["model"] == model_name]
    if rows.empty:
        available = ", ".join(summary.get("model", pd.Series(dtype=str)).astype(str).tolist())
        raise ValueError(f"Could not find model={model_name!r} in {available!r}.")
    return rows.iloc[0]


def plot_winner_cdf_deltas(summary_csv: Path, verdict_path: Path, output_path: Path) -> None:
    summary = pd.read_csv(summary_csv)
    verdict = _load_json(verdict_path) if verdict_path.exists() else None

    baseline_rows = summary.loc[summary["kind"] == "baseline"]
    if baseline_rows.empty:
        raise ValueError(f"No baseline row found in {summary_csv}.")
    baseline = baseline_rows.iloc[0]
    winner = _select_winner(summary, verdict)

    metrics = [
        ("cdf_rmse_mm", "RMSE"),
        ("cdf_mae_mm", "MAE"),
        ("cdf_bias_mm", "Bias"),
        ("cdf_ece", "ECE"),
        ("cdf_crps", "CRPS"),
    ]
    deltas = [float(winner[col]) - float(baseline[col]) for col, _ in metrics]
    labels = [label for _, label in metrics]

    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    bars = ax.bar(labels, deltas, color="#1aa64a", width=0.72)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Winner minus current production on CDF uncensored", fontsize=19, pad=14)
    ax.set_ylabel("Delta [metric units]", fontsize=16)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=14)

    ymin = min(deltas) - 0.08
    ymax = 0.03
    ax.set_ylim(ymin, ymax)

    for bar, value in zip(bars, deltas):
        x = bar.get_x() + bar.get_width() / 2.0
        if value >= 0:
            y = value + 0.018
            va = "bottom"
        else:
            y = value - 0.02
            va = "top"
        ax.text(
            x,
            y,
            f"{value:+.3f}",
            ha="center",
            va=va,
            fontsize=14,
            color="black",
            clip_on=False,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_svgp_context_rmse_comparison(summary_csv: Path, output_path: Path) -> None:
    """Compare the RMSE-style tables for the SVGP context deck.

    The three rows show the old SVGP production baseline, the current best MLP
    residual-family-head model, and the residual multitask SVGP winner.
    """

    summary = pd.read_csv(summary_csv)
    model_order = [
        "current production SVGP",
        "latest MLP residual family head",
        "residual multitask SVGP winner",
    ]
    rows = {name: _require_model_row(summary, name) for name in model_order}

    metrics = [
        ("cdf_rmse_mm", "CDF unc.", "RMSE [mm]"),
        ("p50_rmse_mm", "P50 obs.", "RMSE [mm]"),
        ("q1_observed_rmse_mm", "Q1 obs.", "RMSE [mm]"),
        ("q1_extrapolated_rmse_mm", "Q1 extrap.", "RMSE [mm]"),
    ]
    values = {name: [float(rows[name][col]) for col, _, _ in metrics] for name in model_order}

    x_labels = [label for _, label, _ in metrics]
    x = range(len(x_labels))
    width = 0.24
    offsets = [-width, 0.0, width]
    colors = {
        "current production SVGP": "#6b7280",
        "latest MLP residual family head": "#f59e0b",
        "residual multitask SVGP winner": "#2563eb",
    }
    series_labels = {
        "current production SVGP": "Current production SVGP",
        "latest MLP residual family head": "Latest MLP residual head",
        "residual multitask SVGP winner": "Residual multitask SVGP",
    }

    fig, ax = plt.subplots(figsize=(13.4, 6.0))
    max_val = max(max(series) for series in values.values())
    for idx, name in enumerate(model_order):
        bars = ax.bar(
            [i + offsets[idx] for i in x],
            values[name],
            width=width,
            color=colors[name],
            label=series_labels[name],
            alpha=0.96,
        )
        for bar, val in zip(bars, values[name]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.12,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                color="#374151",
            )

    ax.set_title("RMSE context: current production SVGP vs current best MLP vs residual SVGP", fontsize=18, pad=14)
    ax.set_ylabel("RMSE [mm]", fontsize=16)
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0.0, max_val + 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_svgp_vs_current_best_deltas(summary_csv: Path, output_path: Path) -> None:
    """Plot residual multitask SVGP minus the current best MLP on all metrics."""

    summary = pd.read_csv(summary_csv)
    best = _require_model_row(summary, "latest MLP residual family head")
    winner = _require_model_row(summary, "residual multitask SVGP winner")

    metrics = [
        ("cdf_rmse_mm", "RMSE"),
        ("cdf_ece", "ECE"),
        ("cdf_crps", "CRPS"),
        ("p50_rmse_mm", "P50 RMSE"),
        ("q1_observed_rmse_mm", "Q1 obs."),
        ("q1_extrapolated_rmse_mm", "Q1 extrap."),
    ]
    deltas = [float(winner[col]) - float(best[col]) for col, _ in metrics]
    labels = [label for _, label in metrics]

    fig, ax = plt.subplots(figsize=(12.2, 6.0))
    bars = ax.bar(labels, deltas, color="#16a34a", width=0.72)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Residual multitask SVGP minus current best MLP", fontsize=19, pad=14)
    ax.set_ylabel("Delta [metric units]", fontsize=16)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    delta_span = max(deltas) - min(deltas)
    if delta_span <= 0:
        delta_span = max(abs(v) for v in deltas) or 1.0
    ax.set_ylim(min(deltas) - 0.12 * delta_span, max(deltas) + 0.12 * delta_span)

    for bar, value in zip(bars, deltas):
        x = bar.get_x() + bar.get_width() / 2.0
        if value >= 0:
            y = value + 0.03 * delta_span
            va = "bottom"
        else:
            y = value - 0.035 * delta_span
            va = "top"
        ax.text(
            x,
            y,
            f"{value:+.3f}",
            ha="center",
            va=va,
            fontsize=12,
            color="black",
            clip_on=False,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_delta_l2_sweep(summary_csv: Path, verdict_path: Path, output_path: Path) -> None:
    """Redraw the delta-L2 sweep with the legend outside the plotting area."""

    summary = pd.read_csv(summary_csv)
    verdict = _load_json(verdict_path) if verdict_path.exists() else None

    baseline_rows = summary.loc[summary["kind"] == "baseline"]
    residual_rows = summary.loc[summary["kind"] == "residual"].copy()
    if baseline_rows.empty or residual_rows.empty:
        raise ValueError(f"Expected baseline and residual rows in {summary_csv}.")
    if "returncode" in residual_rows.columns:
        residual_rows = residual_rows.loc[pd.to_numeric(residual_rows["returncode"], errors="coerce").fillna(1).eq(0)]
    if residual_rows.empty:
        raise ValueError(f"No successful residual rows found in {summary_csv}.")

    baseline = baseline_rows.iloc[0]
    residual_rows = residual_rows.sort_values("delta_l2_weight", kind="stable")
    winner = _select_winner(summary, verdict)

    x_labels = [f"{float(v):g}" for v in residual_rows["delta_l2_weight"].tolist()]
    x = range(len(x_labels))

    cdf_rmse = residual_rows["cdf_rmse_mm"].astype(float).tolist()
    p50_rmse = residual_rows["p50_rmse_mm"].astype(float).tolist()
    q1_obs_rmse = residual_rows["q1_observed_rmse_mm"].astype(float).tolist()
    cdf_ece = residual_rows["cdf_ece"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(13.8, 6.1))
    ax2 = ax.twinx()

    line1 = ax.plot(x, cdf_rmse, color="#2563eb", marker="o", linewidth=2.0, markersize=8, label="CDF RMSE")[0]
    line2 = ax.plot(x, p50_rmse, color="#059669", marker="o", linewidth=2.0, markersize=8, label="P50 RMSE")[0]
    line3 = ax.plot(x, q1_obs_rmse, color="#7c3aed", marker="o", linewidth=2.0, markersize=8, label="Q1 observed RMSE")[0]
    line4 = ax2.plot(x, cdf_ece, color="#dc2626", marker="s", linewidth=2.0, markersize=8, label="CDF ECE")[0]

    ax.axhline(float(baseline["cdf_rmse_mm"]), color="#2563eb", linestyle="--", linewidth=1.5, alpha=0.35)
    ax.axhline(float(baseline["p50_rmse_mm"]), color="#059669", linestyle="--", linewidth=1.5, alpha=0.35)
    ax2.axhline(float(baseline["cdf_ece"]), color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.35)

    ax.set_title("Residual delta shrinkage sweep", fontsize=19, pad=14)
    ax.set_xlabel("delta L2 weight", fontsize=16)
    ax.set_ylabel("RMSE [mm]", fontsize=16)
    ax2.set_ylabel("CDF ECE", fontsize=16)
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.28)

    ax.set_ylim(min(min(cdf_rmse), min(p50_rmse), min(q1_obs_rmse)) - 0.18, max(max(cdf_rmse), max(p50_rmse), max(q1_obs_rmse)) + 0.22)
    ax2.set_ylim(min(cdf_ece) - 0.0012, max(cdf_ece) + 0.0024)

    if verdict and verdict.get("winner"):
        try:
            winner_idx = int(residual_rows.index.get_loc(winner.name))  # type: ignore[arg-type]
        except Exception:
            winner_idx = None
        if winner_idx is not None:
            ax.scatter([winner_idx], [cdf_rmse[winner_idx]], s=140, facecolors="none", edgecolors="#111827", linewidths=1.6, zorder=5)

    # Use Matplotlib's automatic placement, which is the closest analogue to
    # MATLAB's "best" legend location. That keeps the legend off the right-side
    # tick labels without forcing a large blank margin.
    legend = ax.legend(
        handles=[line1, line2, line3, line4],
        labels=[line1.get_label(), line2.get_label(), line3.get_label(), line4.get_label()],
        loc="best",
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        fontsize=11,
        handlelength=1.8,
        handletextpad=0.55,
        labelspacing=0.35,
        borderaxespad=0.25,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=(legend,))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--verdict", type=Path, default=DEFAULT_VERDICT)
    parser.add_argument(
        "--figure",
        choices=("winner-cdf-deltas", "delta-l2-sweep", "svgp-context-rmse", "svgp-context-deltas"),
        default="winner-cdf-deltas",
    )
    parser.add_argument("--svgp-summary-csv", type=Path, default=DEFAULT_SVGP_CONTEXT_CSV)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        if args.figure == "winner-cdf-deltas":
            output = DEFAULT_OUTPUT
        elif args.figure == "delta-l2-sweep":
            output = DEFAULT_DELTA_SWEEP_OUTPUT
        elif args.figure == "svgp-context-rmse":
            output = DEFAULT_SVGP_CONTEXT_OUTPUT
        else:
            output = DEFAULT_SVGP_DELTA_OUTPUT
    if args.figure == "winner-cdf-deltas":
        plot_winner_cdf_deltas(args.summary_csv, args.verdict, output)
    else:
        if args.figure == "delta-l2-sweep":
            plot_delta_l2_sweep(args.summary_csv, args.verdict, output)
        elif args.figure == "svgp-context-rmse":
            plot_svgp_context_rmse_comparison(args.svgp_summary_csv, output)
        else:
            plot_svgp_vs_current_best_deltas(args.svgp_summary_csv, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
