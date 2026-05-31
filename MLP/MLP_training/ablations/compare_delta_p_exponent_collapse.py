"""Experiment: compare A_scale variance collapse for delta_P exponent 0.25 vs 0.5.

Loads the representative row_table.csv from the most recent stage1 run, recomputes
A_scale with both delta_P exponents while keeping rho_air^-0.25 and diameter^0.5
fixed, runs the pretrain collapse check on each, and writes a side-by-side
comparison summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # MLP_training/

from engineered_feature_common import run_pretrain_collapse_check


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROW_TABLE = (
    PROJECT_ROOT
    / "MLP"
    / "runs_mlp"
    / "stage1_engineered_mse_a_only_20260508_235519"
    / "row_table.csv"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "MLP"
    / "synthetic_data"
    / "fit_diagnostics"
    / "delta_p_exponent_collapse"
)


def make_a_scale(df: pd.DataFrame, dp_exp: float) -> pd.DataFrame:
    out = df.copy()
    a = (
        np.power(out["delta_pressure_bar_phys"].astype(float), dp_exp)
        * np.power(out["ambient_density_kg_m3"].astype(float), -0.25)
        * np.sqrt(out["diameter_mm"].astype(float))
    )
    out["A_scale"] = a.astype(float)
    out["log_A"] = np.log(a.astype(float))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-table", type=str, default=str(DEFAULT_ROW_TABLE))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_table_path = Path(args.row_table).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    if not row_table_path.exists():
        raise FileNotFoundError(f"Source row_table.csv not found: {row_table_path}")

    base_df = pd.read_csv(row_table_path)
    out_root.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict] = {}
    metrics_frames: dict[str, pd.DataFrame] = {}
    cases = [("dp_exp_0p25", 0.25), ("dp_exp_0p50", 0.50)]

    for label, dp_exp in cases:
        df = make_a_scale(base_df, dp_exp)
        out_dir = out_root / label
        report = run_pretrain_collapse_check(df, output_dir=out_dir)
        reports[label] = report
        metrics_frames[label] = pd.read_csv(out_dir / "pretrain_collapse_metrics.csv")
        print(f"\n=== {label} (A = delta_P^{dp_exp} * rho_a^-0.25 * sqrt(d)) ===")
        print(json.dumps(report, indent=2, default=str))

    headline_keys = [
        "median_post_0p5ms_collapse_ratio",
        "mean_sparse_r2_physical",
        "mean_sparse_r2_scaled",
        "sparse_r2_ratio_scaled_over_physical",
        "all_post_0p5ms_collapse_lt_one",
        "passed",
    ]
    rows = []
    for key in headline_keys:
        row = {"metric": key}
        for label, _ in cases:
            row[label] = reports[label][key]
        rows.append(row)
    comp = pd.DataFrame(rows)
    comp_path = out_root / "comparison_summary.csv"
    comp.to_csv(comp_path, index=False)

    per_time_rows = []
    for label, _ in cases:
        df_metrics = metrics_frames[label].copy()
        df_metrics["case"] = label
        per_time_rows.append(df_metrics)
    per_time = pd.concat(per_time_rows, ignore_index=True)
    per_time.to_csv(out_root / "per_time_metrics.csv", index=False)

    print(f"\nSaved headline comparison to: {comp_path}")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
