"""Production residual-FiLM-family-head ablation.

This is the MLP-only follow-up to the residual-family-head experiment.  It
hot-starts from an existing residual family-head production checkpoint, inserts
identity-initialized family FiLM adapters into the MLP trunk, refines on the
same Stage-3 production data, and evaluates on the canonical point tables.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MLP_ROOT = PROJECT_ROOT / "MLP"
RUNS_ROOT = MLP_ROOT / "runs_mlp"
EVAL_ROOT = MLP_ROOT / "eval"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN_SCRIPT = MLP_ROOT / "MLP_training" / "train_stage3_distillation_plus_raw_series.py"
POINT_EVAL_SCRIPT = MLP_ROOT / "eval" / "inference_rmse_on_point_tables.py"

SOURCE_RUN = RUNS_ROOT / "distill_cdf_residual_fh_from_prod_FROZEN_anchor_off_l2_1em04_20260530_225953"
SOURCE_EVAL = EVAL_ROOT / "point_eval_20260530_230219_residual_fh_prod_l2_1em04" / "per_run_metrics.csv"
RUN_PREFIX_BASE = "distill_cdf_residual_film_from_residual_fh_FROZEN_anchor_off"


def python_exe() -> str:
    return str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))


def path_for_child(path: Path | str) -> str:
    text = str(path)
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        drive = text[5].upper()
        rest = text[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return text


def slug_float(value: float) -> str:
    text = f"{float(value):.0e}" if value != 0 else "0"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def run_subprocess_streaming(cmd: list[str], log_path: Path, *, dry_run: bool) -> tuple[int, str]:
    print(f"\n[run] {' '.join(shlex.quote(str(p)) for p in cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return 0, ""
    captured: list[str] = []
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            [str(p) for p in cmd],
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
            f.write(line)
            captured.append(line)
        proc.wait()
    return int(proc.returncode), "".join(captured)


def find_latest_run(prefix: str, started_at: float) -> Path | None:
    candidates = [
        path
        for path in RUNS_ROOT.glob(f"{prefix}_*")
        if path.is_dir() and path.stat().st_mtime >= started_at - 2.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_metrics(eval_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(eval_csv)
    by_set = {str(row["eval_set"]): row for _, row in df.iterrows()}

    def get(eval_set: str, key: str) -> float:
        if eval_set not in by_set or key not in by_set[eval_set]:
            return float("nan")
        return float(by_set[eval_set][key])

    return {
        "cdf_rmse_mm": get("cdf_uncensored", "rmse_mm"),
        "cdf_mae_mm": get("cdf_uncensored", "mae_mm"),
        "cdf_bias_mm": get("cdf_uncensored", "bias_mm"),
        "cdf_ece": get("cdf_uncensored", "prob_ece"),
        "cdf_crps": get("cdf_uncensored", "prob_crps_mean"),
        "p50_rmse_mm": get("p50_observed", "rmse_mm"),
        "q1_all_rmse_mm": get("q1_grid_all", "rmse_mm"),
        "q1_observed_rmse_mm": get("q1_grid_observed_window", "rmse_mm"),
        "q1_extrapolated_rmse_mm": get("q1_grid_extrapolated", "rmse_mm"),
    }


def build_train_cmd(
    args: argparse.Namespace,
    *,
    film_weight: float,
    delta_weight: float,
    run_prefix: str,
) -> list[str]:
    cmd = [
        python_exe(),
        "-u",
        path_for_child(TRAIN_SCRIPT),
        path_for_child(args.source_run),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--patience", str(args.patience),
        "--num-workers", "0",
        "--pin-memory",
        "--precompute-dataset",
        "--run-name-prefix", run_prefix,
        "--ablation-name", f"residual_film_{args.architecture_mode}_film_{slug_float(film_weight)}_delta_{slug_float(delta_weight)}",
        "--lambda-anchor", "0.0",
        "--kd-mode", "mse_mu_plus_sigma",
        "--kd-sigma-weight", "5.0",
        "--student-architecture-mode", args.architecture_mode,
        "--residual-shared-family-id", "1",
        "--residual-delta-l2-weight", str(delta_weight),
        "--film-adapter-l2-weight", str(film_weight),
        "--residual-mimic-epochs", str(args.mimic_epochs),
        "--residual-mimic-delta-l2-weight", str(delta_weight),
        "--freeze-trunk",
        "--freeze-residual-shared-mu",
        "--skip-post-train-eval",
    ]
    if args.synthetic_root is not None:
        cmd.extend(["--synthetic-root", path_for_child(args.synthetic_root)])
    return cmd


def build_eval_cmd(args: argparse.Namespace, *, run_dir: Path, tag: str) -> list[str]:
    cmd = [
        python_exe(),
        "-u",
        path_for_child(POINT_EVAL_SCRIPT),
        "--refinement-run", path_for_child(run_dir),
        "--eval-set", "cdf_uncensored",
        "--eval-set", "p50_observed",
        "--eval-set", "q1_grid_all",
        "--device", args.device,
        "--output-root", path_for_child(args.eval_root),
        "--tag", tag,
        "--batch-points", str(args.eval_batch_points),
        "--no-save-plots",
    ]
    if args.synthetic_root is not None:
        cmd.extend(["--synthetic-root", path_for_child(args.synthetic_root)])
    return cmd


def select_winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline = next(row for row in rows if row.get("kind") == "source_baseline")
    baseline_p50 = float(baseline["p50_rmse_mm"])
    baseline_ece = float(baseline["cdf_ece"])
    candidates = [
        row for row in rows
        if row.get("kind") == "residual_film"
        and float(row.get("cdf_rmse_mm", float("inf"))) < float("inf")
        and float(row.get("p50_rmse_mm", float("inf"))) <= baseline_p50 + 0.25
        and float(row.get("cdf_ece", float("inf"))) <= baseline_ece + 0.01
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            round(float(row["cdf_rmse_mm"]) / 0.05),
            float(row["cdf_rmse_mm"]),
            -float(row["film_adapter_l2_weight"]),
            -float(row["residual_delta_l2_weight"]),
        )
    )
    return candidates[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    p.add_argument("--source-eval", type=Path, default=SOURCE_EVAL)
    p.add_argument("--synthetic-root", type=Path, default=None,
                   help="Synthetic-data root for Stage-3 refinement and point-table evaluation.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    p.add_argument(
        "--architecture-mode",
        choices=("residual_film_last_block", "residual_film_all_blocks"),
        default="residual_film_last_block",
    )
    p.add_argument("--film-l2-weights", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    p.add_argument("--residual-delta-l2-weights", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    p.add_argument("--mimic-epochs", type=int, default=0)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.003)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--eval-batch-points", type=int, default=262144)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (RUNS_ROOT / f"residual_film_family_head_production_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    source_metrics = load_metrics(args.source_eval)
    rows.append({
        "kind": "source_baseline",
        "run_dir": str(args.source_run),
        "eval_csv": str(args.source_eval),
        "architecture_mode": "source",
        "film_adapter_l2_weight": float("nan"),
        "residual_delta_l2_weight": float("nan"),
        **source_metrics,
    })

    for film_weight in args.film_l2_weights:
        for delta_weight in args.residual_delta_l2_weights:
            film_slug = slug_float(float(film_weight))
            delta_slug = slug_float(float(delta_weight))
            run_prefix = f"{RUN_PREFIX_BASE}_{args.architecture_mode}_film_{film_slug}_delta_{delta_slug}"
            started = time.time()
            train_log = output_dir / f"train_film_{film_slug}_delta_{delta_slug}.log"
            train_cmd = build_train_cmd(
                args,
                film_weight=float(film_weight),
                delta_weight=float(delta_weight),
                run_prefix=run_prefix,
            )
            rc, _ = run_subprocess_streaming(train_cmd, train_log, dry_run=bool(args.dry_run))
            if rc != 0:
                rows.append({
                    "kind": "residual_film",
                    "architecture_mode": args.architecture_mode,
                    "film_adapter_l2_weight": float(film_weight),
                    "residual_delta_l2_weight": float(delta_weight),
                    "returncode": int(rc),
                    "train_log": str(train_log),
                })
                continue
            if args.dry_run:
                continue

            run_dir = find_latest_run(run_prefix, started)
            if run_dir is None:
                raise RuntimeError(f"Could not find residual-FiLM run for prefix {run_prefix!r}.")

            eval_log = output_dir / f"eval_film_{film_slug}_delta_{delta_slug}.log"
            eval_tag = f"residual_film_film_{film_slug}_delta_{delta_slug}"
            eval_cmd = build_eval_cmd(args, run_dir=run_dir, tag=eval_tag)
            rc, _ = run_subprocess_streaming(eval_cmd, eval_log, dry_run=False)
            if rc != 0:
                rows.append({
                    "kind": "residual_film",
                    "architecture_mode": args.architecture_mode,
                    "run_dir": str(run_dir),
                    "film_adapter_l2_weight": float(film_weight),
                    "residual_delta_l2_weight": float(delta_weight),
                    "returncode": int(rc),
                    "train_log": str(train_log),
                    "eval_log": str(eval_log),
                })
                continue

            eval_dirs = [
                path for path in args.eval_root.glob(f"point_eval_*_{eval_tag}")
                if path.is_dir() and (path / "per_run_metrics.csv").exists()
            ]
            if not eval_dirs:
                raise RuntimeError(f"Could not find point-eval output for tag {eval_tag!r}.")
            eval_csv = max(eval_dirs, key=lambda path: path.stat().st_mtime) / "per_run_metrics.csv"
            rows.append({
                "kind": "residual_film",
                "architecture_mode": args.architecture_mode,
                "run_dir": str(run_dir),
                "eval_csv": str(eval_csv),
                "film_adapter_l2_weight": float(film_weight),
                "residual_delta_l2_weight": float(delta_weight),
                "returncode": 0,
                "train_log": str(train_log),
                "eval_log": str(eval_log),
                **load_metrics(eval_csv),
            })

    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / "residual_film_family_head_eval_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    winner = select_winner(rows)
    source = rows[0]
    verdict = {
        "source_cdf_rmse_mm": float(source["cdf_rmse_mm"]),
        "source_p50_rmse_mm": float(source["p50_rmse_mm"]),
        "source_cdf_ece": float(source["cdf_ece"]),
        "winner": winner,
        "passed_primary_goal": bool(winner and float(winner["cdf_rmse_mm"]) < float(source["cdf_rmse_mm"])),
        "summary_csv": str(summary_path),
    }
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(verdict, indent=2, default=str))


if __name__ == "__main__":
    main()
