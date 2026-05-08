"""Export thesis-facing training hyperparameter and run-manifest tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "MLP" / "runs_mlp"
DEFAULT_STAGE1 = DEFAULT_RUNS_ROOT / "stage1_engineered_mse_a_only_20260429_113223"
DEFAULT_STAGE2 = DEFAULT_RUNS_ROOT / "stage2_engineered_nll_a_only_20260429_113440"
DEFAULT_STAGE3 = DEFAULT_RUNS_ROOT / "distill_cdf_onset_v2_ablate_anchor_off_20260429_130554"
DEFAULT_METRICS = PROJECT_ROOT / "MLP" / "eval" / "rmse_eval_clean_20260429_130733_winner_full" / "metrics_summary.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "MLP" / "runs_mlp" / "training_manifest_20260429"
DEFAULT_THESIS_TABLE_DIR = PROJECT_ROOT / "Thesis" / "generated"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _split_counts(run_dir: Path) -> dict[str, int]:
    row_path = run_dir / "row_table.csv"
    if not row_path.exists():
        return {}
    df = pd.read_csv(row_path, usecols=lambda c: c == "sample_split")
    return {str(k): int(v) for k, v in df["sample_split"].value_counts(dropna=False).sort_index().items()}


def _row_count(run_dir: Path) -> int | None:
    row_path = run_dir / "row_table.csv"
    if not row_path.exists():
        return None
    return int(sum(1 for _ in row_path.open("r", encoding="utf-8")) - 1)


def build_manifest(stage1: Path, stage2: Path, stage3: Path, metrics_path: Path | None) -> pd.DataFrame:
    s1 = _load_json(stage1 / "train_config_used.json")
    s2 = _load_json(stage2 / "train_config_used.json")
    s3 = _load_json(stage3 / "refine_config.json")
    metrics = _load_json(metrics_path) if metrics_path and metrics_path.exists() else {}
    overall = metrics.get("overall", {})

    rows = [
        {
            "stage": "Stage-1",
            "run_dir": stage1.name,
            "objective": "representative MSE",
            "rows": _row_count(stage1),
            "split_counts": json.dumps(_split_counts(stage1), ensure_ascii=False),
            "lr": s1.get("learning_rate"),
            "weight_decay": s1.get("weight_decay"),
            "batch_size": s1.get("batch_size"),
            "n_points": s1.get("n_points"),
            "lambda_var": s1.get("var_reg_weight"),
            "c0": s1.get("log_var_prior"),
            "lambda_d1": s1.get("d1_positive_weight"),
            "lambda_d2": s1.get("d2_concave_weight"),
            "tc_ms": s1.get("d2_start_ms"),
            "tau_ms": s1.get("d2_transition_ms"),
            "lambda_onset": None,
            "lambda_anchor": None,
            "sigma_ref_mm": None,
            "raw_weights": None,
            "kd_weights": None,
            "rmse_mm": None,
            "mae_mm": None,
            "coverage_1sigma": None,
            "coverage_2sigma": None,
        },
        {
            "stage": "Stage-2",
            "run_dir": stage2.name,
            "objective": "filtered heteroscedastic NLL",
            "rows": _row_count(stage2),
            "split_counts": json.dumps(_split_counts(stage2), ensure_ascii=False),
            "lr": s2.get("learning_rate"),
            "weight_decay": s2.get("weight_decay"),
            "batch_size": s2.get("batch_size"),
            "n_points": s2.get("n_points"),
            "lambda_var": None,
            "c0": None,
            "lambda_d1": s2.get("d1_positive_weight"),
            "lambda_d2": s2.get("d2_concave_weight"),
            "tc_ms": s2.get("d2_start_ms"),
            "tau_ms": s2.get("d2_transition_ms"),
            "lambda_onset": None,
            "lambda_anchor": None,
            "sigma_ref_mm": None,
            "raw_weights": None,
            "kd_weights": None,
            "rmse_mm": None,
            "mae_mm": None,
            "coverage_1sigma": None,
            "coverage_2sigma": None,
        },
        {
            "stage": "Stage-3",
            "run_dir": stage3.name,
            "objective": "raw CDF + KD refinement",
            "rows": int(overall["n_trajectories"]) if "n_trajectories" in overall else None,
            "split_counts": None,
            "lr": s3.get("learning_rate"),
            "weight_decay": s3.get("weight_decay"),
            "batch_size": s3.get("batch_size"),
            "n_points": s3.get("n_points"),
            "lambda_var": None,
            "c0": None,
            "lambda_d1": s3.get("d1_positive_weight"),
            "lambda_d2": s3.get("d2_concave_weight"),
            "tc_ms": s3.get("d2_start_ms"),
            "tau_ms": s3.get("d2_transition_ms"),
            "lambda_onset": s3.get("lambda_onset"),
            "lambda_anchor": s3.get("lambda_anchor"),
            "sigma_ref_mm": s3.get("sigma_conf_ref_mm"),
            "raw_weights": json.dumps(s3.get("raw_weights"), ensure_ascii=False),
            "kd_weights": json.dumps(s3.get("kd_weights"), ensure_ascii=False),
            "rmse_mm": overall.get("rmse_mm"),
            "mae_mm": overall.get("mae_mm"),
            "coverage_1sigma": overall.get("coverage_1sigma"),
            "coverage_2sigma": overall.get("coverage_2sigma"),
        },
    ]
    return pd.DataFrame(rows)


def write_latex_table(df: pd.DataFrame, out_path: Path) -> None:
    view = df.loc[
        :,
        [
            "stage",
            "objective",
            "lambda_var",
            "c0",
            "lambda_d1",
            "lambda_d2",
            "tc_ms",
            "tau_ms",
            "lambda_onset",
            "lambda_anchor",
            "sigma_ref_mm",
        ],
    ].copy()
    out_path.write_text(
        view.to_latex(index=False, escape=True, na_rep="--", float_format=lambda x: f"{x:.6g}"),
        encoding="utf-8",
    )


def run_export(
    *,
    stage1: Path = DEFAULT_STAGE1,
    stage2: Path = DEFAULT_STAGE2,
    stage3: Path = DEFAULT_STAGE3,
    metrics: Path | None = DEFAULT_METRICS,
    out_dir: Path = DEFAULT_OUT_DIR,
    thesis_table_dir: Path | None = DEFAULT_THESIS_TABLE_DIR,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_manifest(stage1, stage2, stage3, metrics)
    csv_path = out_dir / "training_run_manifest.csv"
    tex_path = out_dir / "training_hyperparameters_table.tex"
    df.to_csv(csv_path, index=False)
    write_latex_table(df, tex_path)

    outputs = {
        "csv": str(csv_path),
        "latex": str(tex_path),
    }
    if thesis_table_dir is not None:
        thesis_table_dir.mkdir(parents=True, exist_ok=True)
        thesis_tex = thesis_table_dir / "training_hyperparameters_table.tex"
        write_latex_table(df, thesis_tex)
        outputs["thesis_latex"] = str(thesis_tex)

    (out_dir / "training_run_manifest_summary.json").write_text(
        json.dumps(outputs, indent=2),
        encoding="utf-8",
    )
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-run", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--stage2-run", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--stage3-run", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--metrics-summary", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--thesis-table-dir", type=Path, default=DEFAULT_THESIS_TABLE_DIR)
    parser.add_argument("--no-thesis-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = run_export(
        stage1=args.stage1_run,
        stage2=args.stage2_run,
        stage3=args.stage3_run,
        metrics=args.metrics_summary,
        out_dir=args.out_dir,
        thesis_table_dir=None if args.no_thesis_copy else args.thesis_table_dir,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
