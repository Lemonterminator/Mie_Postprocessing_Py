"""Emit a Stage-3 headline CSV in the SVGP comparison-report format.

Reads a run_full_pipeline.py output directory, locates each seed's
Stage-3 winner run_dir, pulls metrics from post_train_rmse_eval.json,
and writes a CSV with:
  * one row per seed,
  * a "5-seed mean" row,
  * optionally appended HA/NS reference rows from --reference-csv.

The output CSV is consumable by run_gp_baseline.py via
``--report-source-headline``.

Usage
-----
    python MLP/MLP_training/emit_stage3_headline.py \\
        --pipeline-dir MLP/runs_mlp/full_pipeline_dp050_pressures_20260521_xxxxxx \\
        --output MLP/baseline/comparison_reports/stage3_dp050_pressures_20260521/headline_comparison.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_CSV = (
    PROJECT_ROOT / "MLP" / "baseline" / "comparison_reports"
    / "stage3_dp_exp_0p5_vs_HA_NS_20260509" / "headline_comparison.csv"
)

# Order matches existing SVGP comparison CSVs.
METRIC_COLS = [
    "n_points", "n_trajectories",
    "rmse_mm", "mae_mm", "bias_mm",
    "median_abs_err_mm", "p90_abs_err_mm", "p95_abs_err_mm",
    "coverage_1sigma", "coverage_2sigma", "mean_pred_std_mm",
    "mean_rel_err", "median_rel_err", "nrmse_range",
]

# Reference rows we want to keep from the source CSV (model name → row).
REFERENCE_KEEP_PREFIXES = (
    "Hiroyasu-Arai",
    "Naber-Siebers",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pipeline-dir", type=Path, required=True,
        help="run_full_pipeline.py output directory containing seed_*/ subdirs.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Destination headline CSV path.",
    )
    p.add_argument(
        "--model-label", type=str, default="Stage-3 MLP anchor_off (a_dp050_plus_pressures)",
        help="Per-seed model label prefix. The seed value is appended.",
    )
    p.add_argument(
        "--mean-label", type=str, default="Stage-3 MLP a_dp050_plus_pressures (5-seed mean)",
        help="Label for the aggregated mean row.",
    )
    p.add_argument(
        "--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV,
        help="Optional source CSV providing HA/NS reference rows.",
    )
    p.add_argument(
        "--no-reference", action="store_true",
        help="Skip appending HA/NS reference rows.",
    )
    return p.parse_args()


def _load_winner_metrics(seed_dir: Path) -> tuple[int, dict[str, Any]] | None:
    summary_path = seed_dir / "seed_summary.json"
    if not summary_path.exists():
        print(f"[warn] {summary_path} missing; skip {seed_dir.name}")
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = int(summary.get("seed"))
    winner_run_dir = summary.get("winner_run_dir")
    if not winner_run_dir:
        print(f"[warn] {summary_path} has no winner_run_dir; skip seed {seed}.")
        return None
    eval_pointer = Path(winner_run_dir) / "post_train_rmse_eval.json"
    if not eval_pointer.exists():
        print(f"[warn] {eval_pointer} missing; skip seed {seed}.")
        return None
    payload = json.loads(eval_pointer.read_text(encoding="utf-8"))
    overall = dict(payload.get("overall", {}))
    if not overall:
        print(f"[warn] {eval_pointer} has empty 'overall'; skip seed {seed}.")
        return None
    return seed, overall


def _make_row(*, model: str, overall: dict[str, Any], seed: Any, std: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"model": model}
    for col in METRIC_COLS:
        row[col] = overall.get(col)
    row["seed"] = seed
    row["std"] = std
    return row


def _load_reference_rows(reference_csv: Path) -> list[dict[str, Any]]:
    if not reference_csv.exists():
        print(f"[warn] reference CSV not found: {reference_csv}")
        return []
    ref = pd.read_csv(reference_csv)
    keep = ref[ref["model"].astype(str).str.startswith(REFERENCE_KEEP_PREFIXES)]
    return keep.to_dict(orient="records")


def main() -> None:
    args = parse_args()
    pipeline_dir = args.pipeline_dir.expanduser().resolve()
    if not pipeline_dir.exists():
        raise FileNotFoundError(f"pipeline-dir not found: {pipeline_dir}")

    seed_dirs = sorted(p for p in pipeline_dir.iterdir() if p.is_dir() and p.name.startswith("seed_"))
    if not seed_dirs:
        raise RuntimeError(f"No seed_* subdirs found under {pipeline_dir}.")

    seed_rows: list[dict[str, Any]] = []
    overalls_by_seed: dict[int, dict[str, Any]] = {}
    for seed_dir in seed_dirs:
        loaded = _load_winner_metrics(seed_dir)
        if loaded is None:
            continue
        seed, overall = loaded
        overalls_by_seed[seed] = overall
        seed_rows.append(
            _make_row(
                model=f"{args.model_label} (seed {seed})",
                overall=overall,
                seed=seed,
                std=np.nan,
            )
        )

    if not seed_rows:
        raise RuntimeError("No usable seed metrics collected.")

    # Mean row across seeds.
    mean_row: dict[str, Any] = {"model": args.mean_label}
    for col in METRIC_COLS:
        vals = [overall.get(col) for overall in overalls_by_seed.values()
                if isinstance(overall.get(col), (int, float))]
        mean_row[col] = float(np.mean(vals)) if vals else None
    mean_row["seed"] = f"mean({len(overalls_by_seed)})"
    rmse_vals = [o.get("rmse_mm") for o in overalls_by_seed.values()
                 if isinstance(o.get("rmse_mm"), (int, float))]
    mae_vals = [o.get("mae_mm") for o in overalls_by_seed.values()
                if isinstance(o.get("mae_mm"), (int, float))]
    cov1_vals = [o.get("coverage_1sigma") for o in overalls_by_seed.values()
                 if isinstance(o.get("coverage_1sigma"), (int, float))]
    parts: list[str] = []
    if len(rmse_vals) > 1:
        parts.append(f"rmse±{float(np.std(rmse_vals, ddof=1)):.2f}")
    if len(mae_vals) > 1:
        parts.append(f"mae±{float(np.std(mae_vals, ddof=1)):.2f}")
    if len(cov1_vals) > 1:
        parts.append(f"cov1±{float(np.std(cov1_vals, ddof=1)):.3f}")
    mean_row["std"] = " | ".join(parts) if parts else 0.0

    rows: list[dict[str, Any]] = []
    if not args.no_reference:
        rows.extend(_load_reference_rows(args.reference_csv))
    rows.extend(seed_rows)
    rows.append(mean_row)

    output_csv = args.output.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    column_order = ["model"] + METRIC_COLS + ["seed", "std"]
    # Preserve any extra columns from reference CSV after the canonical ones.
    extra = [c for c in out.columns if c not in column_order]
    out = out[column_order + extra]
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out)} rows to {output_csv}")

    # Also emit a markdown copy for quick inspection.
    md_path = output_csv.with_suffix(".md")
    try:
        md_text = out.to_markdown(index=False)
    except ImportError:
        md_text = "```\n" + out.to_string(index=False) + "\n```\n"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"Wrote markdown to {md_path}")


if __name__ == "__main__":
    main()
