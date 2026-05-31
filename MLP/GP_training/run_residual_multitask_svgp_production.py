"""Production driver for hot-started additive residual multi-task SVGP.

Search terms: RESIDUAL_SVGP, production sweep, hot-start shared base.

The default sweep freezes the existing full Stage-3 SVGP as shared_mu, trains
only per-family delta GPs plus a final shared log-variance GP, evaluates the
same fixed point tables used by MLP production, and writes a compact package
under Thesis/slides for later comparison.
"""

from __future__ import annotations

import argparse
import json
import shutil
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLP_ROOT = PROJECT_ROOT / "MLP"
RUNS_ROOT = MLP_ROOT / "runs_mlp"
EVAL_ROOT = MLP_ROOT / "eval"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT = MLP_ROOT / "GP_training" / "run_residual_multitask_svgp.py"
EVAL_SCRIPT = MLP_ROOT / "GP_training" / "eval_residual_multitask_svgp_on_point_tables.py"
DEFAULT_SHARED_RUN = MLP_ROOT / "runs_mlp" / "gp_baseline_stage3_20260521_112229"
DEFAULT_SHARED_CHECKPOINT = DEFAULT_SHARED_RUN / "per_seed" / "seed_42" / "model.pt"
DEFAULT_MLP_BOOTSTRAP = MLP_ROOT / "runs_mlp" / "full_pipeline_A_20260519_161129" / "bootstrap_summary.json"
DEFAULT_SLIDES_DIR = PROJECT_ROOT / "Thesis" / "slides" / "slides_residual_multitask_svgp"

BASELINE = {
    # Existing single-output Stage-3 SVGP fixed-table baseline.  This remains
    # separate from the latest MLP comparison that we do after the sweep.
    "kind": "baseline",
    "run_dir": str(DEFAULT_SHARED_RUN),
    "eval_csv": str(MLP_ROOT / "baseline" / "comparison_reports" / "stage3_fixed_table_eval_20260521" / "main_table.md"),
    "shared_base": "existing_full_svgp",
    "delta_l2_weight": np.nan,
    "cdf_rmse_mm": 4.192763,
    "cdf_mae_mm": 2.893807,
    "cdf_bias_mm": -0.040461,
    "cdf_ece": np.nan,
    "cdf_crps": np.nan,
    "p50_rmse_mm": 2.648853,
    "q1_all_rmse_mm": 9.836484,
    "q1_observed_rmse_mm": 2.749035,
    "q1_extrapolated_rmse_mm": 10.590671,
    "returncode": np.nan,
    "train_log": "",
    "eval_log": "",
}


def python_exe() -> str:
    return str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))


def path_arg(path: Path) -> str:
    return str(path)


def weight_tag(value: float) -> str:
    if float(value) == 0.0:
        return "0"
    return f"{float(value):.0e}".replace("-", "m").replace("+", "")


def run_subprocess_streaming(cmd: list[str], log_path: Path) -> tuple[int, str]:
    print(f"\n[run] {' '.join(shlex.quote(part) for part in cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            captured.append(line)
        proc.wait()
    return int(proc.returncode), "".join(captured)


def latest_run_dir(runs_root: Path, *, prefix: str, shared_base: str, l2_weight: float) -> Path:
    tag = weight_tag(l2_weight)
    pattern = f"{prefix}_{shared_base}_l2_{tag}_*"
    candidates = [path for path in runs_root.glob(pattern) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories match {runs_root / pattern}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_train_cmd(args: argparse.Namespace, *, l2_weight: float, run_prefix: str) -> list[str]:
    cmd = [
        python_exe(),
        "-u",
        path_arg(TRAIN_SCRIPT),
        "--mode", "stage3",
        "--seed", str(args.seed),
        "--device", args.device,
        "--mlp-bootstrap", path_arg(args.mlp_bootstrap),
        "--runs-root", path_arg(args.runs_root),
        "--shared-base", args.shared_base,
        "--shared-checkpoint", path_arg(args.shared_checkpoint),
        "--delta-l2-weight", str(float(l2_weight)),
        "--num-inducing", str(args.num_inducing),
        "--epochs", str(args.epochs),
        "--delta-epochs", str(args.delta_epochs),
        "--var-epochs", str(args.var_epochs),
        "--batch-points", str(args.batch_points),
        "--eval-batch-points", str(args.eval_batch_points),
        "--kmeans-samples", str(args.kmeans_samples),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--log-interval", str(args.log_interval),
        "--run-name-prefix", run_prefix,
    ]
    if args.synthetic_root is not None:
        cmd.extend(["--synthetic-root", path_arg(args.synthetic_root)])
    if args.max_curves_per_split is not None:
        cmd.extend(["--max-curves-per-split", str(args.max_curves_per_split)])
    if args.max_real_cdf_rows is not None:
        cmd.extend(["--max-real-cdf-rows", str(args.max_real_cdf_rows)])
    if args.mean_only:
        cmd.append("--mean-only")
    return cmd


def build_eval_cmd(args: argparse.Namespace, *, run_dir: Path, tag: str) -> list[str]:
    cmd = [
        python_exe(),
        "-u",
        path_arg(EVAL_SCRIPT),
        "--residual-gp-run", path_arg(run_dir),
        "--seed", str(args.seed),
        "--device", args.device,
        "--output-root", path_arg(args.eval_root),
        "--tag", tag,
        "--batch-points", str(args.eval_batch_points),
        "--eval-set", "cdf_uncensored",
        "--eval-set", "p50_observed",
        "--eval-set", "q1_grid_all",
    ]
    if not args.save_points:
        cmd.append("--no-save-points")
    if args.synthetic_root is not None:
        cmd.extend(["--synthetic-root", path_arg(args.synthetic_root)])
    return cmd


def metric_row(
    *,
    run_dir: Path,
    eval_dir: Path,
    l2_weight: float,
    returncode: int,
    train_log: Path,
    eval_log: Path,
    shared_base: str,
) -> dict[str, Any]:
    csv_path = eval_dir / "per_run_metrics.csv"
    df = pd.read_csv(csv_path)

    def value(eval_set: str, col: str) -> float:
        rows = df.loc[df["eval_set"] == eval_set]
        if rows.empty or col not in rows.columns:
            return float("nan")
        return float(rows.iloc[0][col])

    return {
        "kind": "residual",
        "run_dir": str(run_dir),
        "eval_csv": str(csv_path),
        "shared_base": shared_base,
        "delta_l2_weight": float(l2_weight),
        "cdf_rmse_mm": value("cdf_uncensored", "rmse_mm"),
        "cdf_mae_mm": value("cdf_uncensored", "mae_mm"),
        "cdf_bias_mm": value("cdf_uncensored", "bias_mm"),
        "cdf_ece": value("cdf_uncensored", "prob_ece"),
        "cdf_crps": value("cdf_uncensored", "prob_crps_mean"),
        "p50_rmse_mm": value("p50_observed", "rmse_mm"),
        "q1_all_rmse_mm": value("q1_grid_all", "rmse_mm"),
        "q1_observed_rmse_mm": value("q1_grid_observed_window", "rmse_mm"),
        "q1_extrapolated_rmse_mm": value("q1_grid_extrapolated", "rmse_mm"),
        "returncode": int(returncode),
        "train_log": str(train_log),
        "eval_log": str(eval_log),
    }


def choose_winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Apply the planned SVGP winner rule to completed residual rows."""

    residuals = [
        row for row in rows
        if row.get("kind") == "residual"
        and int(row.get("returncode", 1)) == 0
        and np.isfinite(float(row.get("cdf_rmse_mm", np.nan)))
    ]
    if not residuals:
        return None
    eligible = [
        row for row in residuals
        if float(row["p50_rmse_mm"]) <= float(BASELINE["p50_rmse_mm"]) + 0.25
    ]
    if not eligible:
        eligible = residuals
    best_rmse = min(float(row["cdf_rmse_mm"]) for row in eligible)
    tied = [row for row in eligible if abs(float(row["cdf_rmse_mm"]) - best_rmse) <= 0.03]
    return max(tied, key=lambda row: float(row.get("delta_l2_weight", 0.0)))


def write_slides_package(
    *,
    slides_dir: Path,
    summary_csv: Path,
    verdict: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Copy sweep results to Thesis/slides/slides_residual_multitask_svgp."""

    slides_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_csv, slides_dir / "residual_multitask_svgp_eval_summary.csv")
    (slides_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    table = pd.DataFrame(rows).copy()
    keep_cols = [
        "kind",
        "shared_base",
        "delta_l2_weight",
        "cdf_rmse_mm",
        "cdf_mae_mm",
        "cdf_bias_mm",
        "cdf_ece",
        "cdf_crps",
        "p50_rmse_mm",
        "q1_observed_rmse_mm",
        "q1_extrapolated_rmse_mm",
    ]
    keep_cols = [col for col in keep_cols if col in table.columns]
    if keep_cols:
        table = table.loc[:, keep_cols]
    winner = verdict.get("winner") or {}
    lines = [
        "# Additive Residual Multi-Task SVGP",
        "",
        f"- Baseline CDF uncensored RMSE: {verdict['baseline_cdf_rmse_mm']:.6f} mm",
        f"- Baseline P50 observed RMSE: {verdict['baseline_p50_rmse_mm']:.6f} mm",
        f"- Passed primary goal: {bool(verdict.get('passed_primary_goal'))}",
        "",
    ]
    if winner:
        lines.extend([
            "## Winner",
            "",
            f"- Run: {winner.get('run_dir', '')}",
            f"- Shared base: {winner.get('shared_base', '')}",
            f"- Delta L2: {winner.get('delta_l2_weight', '')}",
            f"- CDF uncensored RMSE: {float(winner.get('cdf_rmse_mm', float('nan'))):.6f} mm",
            f"- P50 observed RMSE: {float(winner.get('p50_rmse_mm', float('nan'))):.6f} mm",
            "",
        ])
    else:
        lines.extend(["## Winner", "", "- No eligible residual run selected.", ""])
    lines.extend([
        "## Sweep Table",
        "",
        "```csv",
        table.to_csv(index=False).strip() if keep_cols else "",
        "```",
        "",
        f"Summary CSV: `{slides_dir / 'residual_multitask_svgp_eval_summary.csv'}`",
    ])
    (slides_dir / "residual_multitask_svgp_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    parser.add_argument("--mlp-bootstrap", type=Path, default=DEFAULT_MLP_BOOTSTRAP)
    parser.add_argument("--synthetic-root", type=Path, default=None,
                        help="Synthetic-data root for residual SVGP real-CDF training and point-table evaluation.")
    parser.add_argument("--shared-base", choices=("existing_full_svgp", "modified_only_shared"), default="existing_full_svgp")
    parser.add_argument("--shared-checkpoint", type=Path, default=DEFAULT_SHARED_CHECKPOINT)
    parser.add_argument("--delta-l2-weights", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--num-inducing", type=int, default=256)
    parser.add_argument("--kmeans-samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--delta-epochs", type=int, default=60)
    parser.add_argument("--var-epochs", type=int, default=80)
    parser.add_argument("--batch-points", type=int, default=1024)
    parser.add_argument("--eval-batch-points", type=int, default=65536)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--max-curves-per-split", type=int, default=None)
    parser.add_argument("--max-real-cdf-rows", type=int, default=None)
    parser.add_argument("--mean-only", action="store_true")
    parser.add_argument("--save-points", action="store_true")
    parser.add_argument("--slides-dir", type=Path, default=DEFAULT_SLIDES_DIR)
    parser.add_argument("--no-slides-package", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.runs_root / f"residual_multitask_svgp_production_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = "residual_multitask_svgp_prod"
    config = vars(args).copy()
    config["delta_l2_weights"] = [float(v) for v in args.delta_l2_weights]
    (sweep_dir / "experiment_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    rows: list[dict[str, Any]] = [dict(BASELINE)]
    for l2_weight in args.delta_l2_weights:
        tag = f"l2_{weight_tag(float(l2_weight))}"
        train_log = sweep_dir / f"train_{tag}.log"
        eval_log = sweep_dir / f"eval_{tag}.log"
        run_dir: Path | None = None
        train_rc = 0
        if args.skip_train:
            try:
                run_dir = latest_run_dir(
                    args.runs_root,
                    prefix=run_prefix,
                    shared_base=args.shared_base,
                    l2_weight=float(l2_weight),
                )
            except FileNotFoundError:
                if not args.skip_eval:
                    raise
        else:
            train_rc, _ = run_subprocess_streaming(
                build_train_cmd(args, l2_weight=float(l2_weight), run_prefix=run_prefix),
                train_log,
            )
        if train_rc == 0:
            if run_dir is None and not args.skip_eval:
                run_dir = latest_run_dir(
                    args.runs_root,
                    prefix=run_prefix,
                    shared_base=args.shared_base,
                    l2_weight=float(l2_weight),
                )
        else:
            rows.append({
                "kind": "residual",
                "run_dir": "",
                "eval_csv": "",
                "shared_base": args.shared_base,
                "delta_l2_weight": float(l2_weight),
                "cdf_rmse_mm": np.nan,
                "cdf_mae_mm": np.nan,
                "cdf_bias_mm": np.nan,
                "cdf_ece": np.nan,
                "cdf_crps": np.nan,
                "p50_rmse_mm": np.nan,
                "q1_all_rmse_mm": np.nan,
                "q1_observed_rmse_mm": np.nan,
                "q1_extrapolated_rmse_mm": np.nan,
                "returncode": int(train_rc),
                "train_log": str(train_log),
                "eval_log": str(eval_log),
            })
            continue

        if args.skip_eval:
            rows.append({
                "kind": "residual",
                "run_dir": str(run_dir or ""),
                "eval_csv": "",
                "shared_base": args.shared_base,
                "delta_l2_weight": float(l2_weight),
                "cdf_rmse_mm": np.nan,
                "cdf_mae_mm": np.nan,
                "cdf_bias_mm": np.nan,
                "cdf_ece": np.nan,
                "cdf_crps": np.nan,
                "p50_rmse_mm": np.nan,
                "q1_all_rmse_mm": np.nan,
                "q1_observed_rmse_mm": np.nan,
                "q1_extrapolated_rmse_mm": np.nan,
                "returncode": 0,
                "status": "eval_skipped" if run_dir is not None else "train_and_eval_skipped",
                "train_log": str(train_log),
                "eval_log": str(eval_log),
            })
            continue

        if run_dir is None:
            raise RuntimeError(f"No residual run available for l2={l2_weight}.")

        eval_rc = 0
        eval_dir = None
        eval_rc, _ = run_subprocess_streaming(
            build_eval_cmd(args, run_dir=run_dir, tag=f"residual_svgp_prod_{tag}"),
            eval_log,
        )
        candidates = [path for path in args.eval_root.glob(f"point_eval_*_residual_svgp_prod_{tag}") if path.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"Could not find eval output for tag residual_svgp_prod_{tag}.")
        eval_dir = max(candidates, key=lambda path: path.stat().st_mtime)
        rows.append(metric_row(
            run_dir=run_dir,
            eval_dir=eval_dir,
            l2_weight=float(l2_weight),
            returncode=eval_rc,
            train_log=train_log,
            eval_log=eval_log,
            shared_base=args.shared_base,
        ))

    summary_df = pd.DataFrame(rows)
    summary_csv = sweep_dir / "residual_multitask_svgp_eval_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    winner = choose_winner(rows)
    verdict = {
        "baseline_cdf_rmse_mm": float(BASELINE["cdf_rmse_mm"]),
        "baseline_p50_rmse_mm": float(BASELINE["p50_rmse_mm"]),
        "baseline_q1_extrapolated_rmse_mm": float(BASELINE["q1_extrapolated_rmse_mm"]),
        "winner": winner,
        "passed_primary_goal": bool(winner and float(winner["cdf_rmse_mm"]) < float(BASELINE["cdf_rmse_mm"])),
        "summary_csv": str(summary_csv),
    }
    (sweep_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    if not args.no_slides_package:
        write_slides_package(
            slides_dir=args.slides_dir,
            summary_csv=summary_csv,
            verdict=verdict,
            rows=rows,
        )
    print(f"\nWrote summary: {summary_csv}")
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
