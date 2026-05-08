from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plots import (
        build_coverage_reliability,
        build_uncertainty_decomposition,
        plot_baseline_dashboard,
        plot_coverage_reliability,
        plot_metric_comparison,
        plot_uncertainty_decomposition,
    )
else:
    from .plots import (
        build_coverage_reliability,
        build_uncertainty_decomposition,
        plot_baseline_dashboard,
        plot_coverage_reliability,
        plot_metric_comparison,
        plot_uncertainty_decomposition,
    )


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "MLP" / "baseline" / "Naber_Siebers" / "outputs"
DEFAULT_REFERENCE_EVAL = PROJECT_ROOT / "MLP" / "eval" / "rmse_eval_clean_20260429_130733_winner_full"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_run_dir(outputs_root: Path) -> Path:
    candidates = [p.parent for p in outputs_root.glob("*/metrics_summary.json")]
    if not candidates:
        raise FileNotFoundError(f"No baseline runs with metrics_summary.json found under {outputs_root}")
    return max(candidates, key=lambda p: (p / "metrics_summary.json").stat().st_mtime)


def resolve_path(path: Path | None, default: Path | None = None) -> Path | None:
    out = path if path is not None else default
    if out is None:
        return None
    out = Path(out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return out


def load_baseline_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    points_path = run_dir / "points.csv"
    per_nozzle_path = run_dir / "per_nozzle.csv"
    time_bins_path = run_dir / "time_bins.csv"
    metrics_path = run_dir / "metrics_summary.json"
    missing = [p for p in (points_path, per_nozzle_path, time_bins_path, metrics_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing baseline output files: {missing}")
    return (
        pd.read_csv(points_path, low_memory=False),
        pd.read_csv(per_nozzle_path),
        pd.read_csv(time_bins_path),
        read_json(metrics_path),
    )


def load_reference_tables(eval_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    points_path = eval_dir / "points.csv"
    metrics_path = eval_dir / "metrics_summary.json"
    missing = [p for p in (points_path, metrics_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing reference eval files: {missing}")
    return pd.read_csv(points_path, low_memory=False), read_json(metrics_path)


def write_metric_comparison_csv(
    *,
    baseline_metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
    baseline_label: str,
    reference_label: str,
    out_path: Path,
) -> None:
    rows = []
    for metric in [
        "n_points",
        "n_trajectories",
        "rmse_mm",
        "mae_mm",
        "bias_mm",
        "median_abs_err_mm",
        "p90_abs_err_mm",
        "p95_abs_err_mm",
        "mean_rel_err",
        "median_rel_err",
        "coverage_1sigma",
        "coverage_2sigma",
        "mean_pred_std_mm",
        "nrmse_range",
    ]:
        if metric not in baseline_metrics or metric not in reference_metrics:
            continue
        baseline_value = float(baseline_metrics[metric])
        reference_value = float(reference_metrics[metric])
        rows.append(
            {
                "metric": metric,
                baseline_label: baseline_value,
                reference_label: reference_value,
                "reference_minus_baseline": reference_value - baseline_value,
                "relative_change_vs_baseline": (reference_value - baseline_value) / baseline_value
                if baseline_value != 0.0
                else float("nan"),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create visual summaries for Naber--Siebers baseline outputs.")
    p.add_argument("--baseline-run-dir", type=Path, default=None, help="Baseline output run dir. Defaults to newest run.")
    p.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT, help="Root used when finding newest run.")
    p.add_argument("--output-dir", type=Path, default=None, help="Defaults to <baseline-run-dir>/visual_summary.")
    p.add_argument("--reference-eval-dir", type=Path, default=DEFAULT_REFERENCE_EVAL, help="Optional Stage-3 eval directory.")
    p.add_argument("--no-reference", action="store_true", help="Skip reference MLP comparison plots.")
    p.add_argument("--baseline-label", default="Naber--Siebers")
    p.add_argument("--reference-label", default="Stage-3 MLP")
    p.add_argument("--max-points", type=int, default=80000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = resolve_path(args.outputs_root)
    baseline_run_dir = resolve_path(args.baseline_run_dir) if args.baseline_run_dir else latest_run_dir(outputs_root)
    out_dir = resolve_path(args.output_dir) if args.output_dir else baseline_run_dir / "visual_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    points, per_nozzle, time_bins, summary = load_baseline_tables(baseline_run_dir)
    variant = summary.get("variant", "baseline")
    split_mode = summary.get("split_mode", "split")
    primary_split = summary.get("primary_eval_split", "eval")
    dashboard_title = f"Naber--Siebers {variant}: {split_mode}, {primary_split} split"
    plot_baseline_dashboard(
        points_df=points,
        per_nozzle=per_nozzle,
        time_bins=time_bins,
        out_path=out_dir / "baseline_dashboard.png",
        title=dashboard_title,
        max_points=int(args.max_points),
    )

    decomp = build_uncertainty_decomposition(points, time_bins=time_bins)
    decomp.to_csv(out_dir / "uncertainty_decomposition.csv", index=False)
    plot_uncertainty_decomposition(
        decomp,
        out_dir / "uncertainty_decomposition.png",
        "Naber--Siebers uncertainty decomposition",
    )

    written = [
        out_dir / "baseline_dashboard.png",
        out_dir / "uncertainty_decomposition.csv",
        out_dir / "uncertainty_decomposition.png",
    ]

    if not args.no_reference:
        reference_dir = resolve_path(args.reference_eval_dir)
        if reference_dir is not None and reference_dir.exists():
            ref_points, ref_summary = load_reference_tables(reference_dir)
            plot_metric_comparison(
                baseline_metrics=summary["overall"],
                reference_metrics=ref_summary["overall"],
                out_path=out_dir / "ns_vs_stage3_diagnostic_metrics.png",
                baseline_label=args.baseline_label,
                reference_label=args.reference_label,
                title="Diagnostic metrics: NS grouped-test vs existing Stage-3 clean evaluation",
            )
            write_metric_comparison_csv(
                baseline_metrics=summary["overall"],
                reference_metrics=ref_summary["overall"],
                baseline_label=args.baseline_label,
                reference_label=args.reference_label,
                out_path=out_dir / "ns_vs_stage3_diagnostic_metrics.csv",
            )
            reliability = build_coverage_reliability(
                [
                    (args.baseline_label, points),
                    (args.reference_label, ref_points),
                ]
            )
            reliability.to_csv(out_dir / "coverage_reliability_diagnostic.csv", index=False)
            plot_coverage_reliability(
                reliability,
                out_dir / "coverage_reliability_diagnostic.png",
                "Diagnostic uncertainty reliability",
            )
            written.extend(
                [
                    out_dir / "ns_vs_stage3_diagnostic_metrics.csv",
                    out_dir / "ns_vs_stage3_diagnostic_metrics.png",
                    out_dir / "coverage_reliability_diagnostic.csv",
                    out_dir / "coverage_reliability_diagnostic.png",
                ]
            )
        else:
            print(f"Reference eval dir not found; skipped comparison: {reference_dir}")

    print(f"Wrote visual summary to {out_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
