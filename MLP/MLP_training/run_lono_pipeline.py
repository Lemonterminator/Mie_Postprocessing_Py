"""Leave-One-Nozzle-Out (LONO) validation pipeline.

For each unique experiment_name (nozzle), train Stage-1 + Stage-2 +
Stage-3 anchor_off variant with that nozzle held out. Evaluate the held-out
nozzle's data against HA / NS baselines on the same (filtered) point set.

Wall clock: ~13 min/fold on RTX 5090. 5 folds ≈ 65 min total.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from engineered_feature_common import (
    DEFAULT_STAGE1_CONFIG,
    build_all_stage_tables,
    build_dataset_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLP_ROOT = PROJECT_ROOT / "MLP"
TRAINING_ROOT = MLP_ROOT / "MLP_training"
RUNS_ROOT = MLP_ROOT / "runs_mlp"
EVAL_SCRIPT = MLP_ROOT / "eval" / "inference_rmse_on_series.py"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

STAGE1_SCRIPT = TRAINING_ROOT / "train_stage1_mse.py"
STAGE2_SCRIPT = TRAINING_ROOT / "train_stage2_nll.py"
STAGE3_SUITE_SCRIPT = TRAINING_ROOT / "run_stage3_ablation_suite.py"
STAGE3_SUITE_CONFIG = TRAINING_ROOT / "stage3_ablation_suite_config.json"

HA_POINTS_CSV = (
    PROJECT_ROOT
    / "MLP" / "baseline" / "Hiroyasu_Arai" / "outputs"
    / "20260509_020253_ha_calibrated_grouped_condition_all_all_clean_diagnostic_20260509_review"
    / "points.csv"
)
NS_POINTS_CSV = (
    PROJECT_ROOT
    / "MLP" / "baseline" / "Naber_Siebers" / "outputs"
    / "20260509_004452_ns_delay_grouped_condition_all_clean_diagnostic_20260509"
    / "points.csv"
)

SAVED_RUN_DIR_RE = re.compile(r"Saved run_dir:\s*(.+)$")
SUITE_SUMMARY_RE = re.compile(r"Suite summary saved to:\s*(.+)$")
EVAL_OUTPUT_RE = re.compile(r"Wrote RMSE evaluation to\s+(.+)$", re.MULTILINE)


def python_exe() -> str:
    return str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))


def run_subprocess_streaming(cmd: list[str], log_path: Path) -> tuple[int, str]:
    """Run cmd, stream stdout/stderr to console + log, return (rc, captured_text)."""
    print(f"\n[run] {' '.join(shlex.quote(p) for p in cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            captured.append(line)
        proc.wait()
        rc = int(proc.returncode)
    return rc, "".join(captured)


def parse_saved_run_dir(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        m = SAVED_RUN_DIR_RE.search(line.strip())
        if m:
            return Path(m.group(1).strip())
    return None


def parse_suite_summary_path(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        m = SUITE_SUMMARY_RE.search(line.strip())
        if m:
            return Path(m.group(1).strip())
    return None


def identify_nozzle_families(*, data_dir: str, registry, min_groups: int = 50) -> list[str]:
    """Build representative table once and group by experiment_name."""
    print("Building canonical feature table to enumerate nozzle families ...")
    stage_tables = build_all_stage_tables(
        data_dir, registry,
        comparison_time_s=float(DEFAULT_STAGE1_CONFIG["comparison_time_s"]),
        max_curves=None,
        output_dir=None,  # do NOT write the precheck plot here
    )
    rep = stage_tables.representative
    holdout_col = "experiment_name" if "experiment_name" in rep.columns else "source_dataset_name"
    if holdout_col not in rep.columns:
        raise KeyError("representative table missing experiment_name/source_dataset_name; required for LONO.")
    counts = rep.groupby(holdout_col)["sample_group_id"].nunique().sort_values(ascending=False)
    print("\nPer-nozzle sample_group_id counts:")
    print(counts.to_string())
    valid = counts[counts >= min_groups]
    print(f"\nKeeping {len(valid)} nozzles with >= {min_groups} sample_group_ids")
    return [str(name) for name in valid.index]


def find_anchor_off_run_dir(suite_summary_path: Path) -> Path:
    """Pull the anchor_off variant's run_dir from the suite summary JSON."""
    summary = json.loads(suite_summary_path.read_text(encoding="utf-8"))
    for entry in summary.get("results", []):
        if str(entry.get("name")) == "anchor_off":
            run_dir = entry.get("run_dir")
            if run_dir:
                return Path(run_dir)
    # Fallback: selection.best
    best = (summary.get("selection") or {}).get("best") or {}
    if str(best.get("name")) == "anchor_off" and best.get("run_dir"):
        return Path(best["run_dir"])
    raise RuntimeError(f"Could not locate anchor_off run_dir in {suite_summary_path}")


def metrics_from_points(df: pd.DataFrame) -> dict[str, float]:
    truth = df["pen_true_mm"].to_numpy(dtype=float)
    pred = df["pen_pred_mm"].to_numpy(dtype=float)
    std = df["pen_std_mm"].to_numpy(dtype=float)
    resid = pred - truth
    abs_err = np.abs(resid)
    std_safe = np.maximum(std, 1e-12)
    return {
        "n_points": int(len(df)),
        "rmse_mm": float(np.sqrt(np.mean(resid ** 2))) if len(df) else float("nan"),
        "mae_mm": float(np.mean(abs_err)) if len(df) else float("nan"),
        "bias_mm": float(np.mean(resid)) if len(df) else float("nan"),
        "p95_abs_err_mm": float(np.quantile(abs_err, 0.95)) if len(df) else float("nan"),
        "coverage_1sigma": float(np.mean(abs_err <= std_safe)) if len(df) else float("nan"),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * std_safe)) if len(df) else float("nan"),
        "mean_pred_std_mm": float(np.mean(std)) if len(df) else float("nan"),
    }


def filter_points(points_csv: Path, holdout: str, experiment_col: str) -> pd.DataFrame:
    df = pd.read_csv(points_csv, low_memory=False)
    if experiment_col not in df.columns:
        raise KeyError(f"{points_csv} missing column {experiment_col!r}")
    return df.loc[df[experiment_col].astype(str) == str(holdout)].copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=str, default=DEFAULT_STAGE1_CONFIG["data_dir"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Limit to N largest nozzles (default: 5, matching the planned main LONO sweep).")
    p.add_argument("--min-groups", type=int, default=50)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--skip-folds", type=str, nargs="*", default=None,
                   help="Skip these experiment_name values (e.g. already-running fold).")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_one_fold(
    *,
    holdout: str,
    fold_dir: Path,
    seed: int,
    device: str,
    dry_run: bool,
) -> dict[str, Any]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    # ── Stage 1 ──
    s1_log = fold_dir / "stage1.log"
    s1_cmd = [
        python_exe(), str(STAGE1_SCRIPT),
        "--variant", "a_only",
        "--device", device,
        "--seed", str(seed),
        "--lono-holdout", holdout,
        "--allow-failed-precheck",
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s1_cmd)}")
        stage1_run = Path("DRY_RUN/stage1")
    else:
        rc, out = run_subprocess_streaming(s1_cmd, s1_log)
        if rc != 0:
            raise RuntimeError(f"Stage-1 failed (rc={rc}). See {s1_log}.")
        stage1_run = parse_saved_run_dir(out)
        if stage1_run is None:
            raise RuntimeError("Could not parse Stage-1 run_dir.")
    timing["stage1_s"] = time.time() - t0

    # ── Stage 2 ──
    s2_log = fold_dir / "stage2.log"
    s2_cmd = [
        python_exe(), str(STAGE2_SCRIPT),
        str(stage1_run),
        "--device", device,
        "--seed", str(seed),
        "--lono-holdout", holdout,
        "--allow-failed-precheck",
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s2_cmd)}")
        stage2_run = Path("DRY_RUN/stage2")
    else:
        rc, out = run_subprocess_streaming(s2_cmd, s2_log)
        if rc != 0:
            raise RuntimeError(f"Stage-2 failed (rc={rc}). See {s2_log}.")
        stage2_run = parse_saved_run_dir(out)
        if stage2_run is None:
            raise RuntimeError("Could not parse Stage-2 run_dir.")
    timing["stage2_s"] = time.time() - t0

    # ── Stage 3 (anchor_off only) ──
    s3_log = fold_dir / "stage3.log"
    s3_cmd = [
        python_exe(), str(STAGE3_SUITE_SCRIPT),
        "--config", str(STAGE3_SUITE_CONFIG),
        "--teacher-run", str(stage2_run),
        "--device", device,
        "--seed", str(seed),
        "--only", "anchor_off",
        "--lono-holdout", holdout,
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s3_cmd)}")
        suite_summary = Path("DRY_RUN/suite_summary.json")
        anchor_run = Path("DRY_RUN/anchor_off")
    else:
        rc, out = run_subprocess_streaming(s3_cmd, s3_log)
        if rc != 0:
            raise RuntimeError(f"Stage-3 suite failed (rc={rc}). See {s3_log}.")
        suite_summary = parse_suite_summary_path(out)
        if suite_summary is None:
            raise RuntimeError("Could not parse suite_summary path.")
        anchor_run = find_anchor_off_run_dir(suite_summary)
    timing["stage3_s"] = time.time() - t0

    record: dict[str, Any] = {
        "holdout": holdout,
        "stage1_run": str(stage1_run),
        "stage2_run": str(stage2_run),
        "stage3_suite_summary": str(suite_summary),
        "winner_run": str(anchor_run),
        "timing": timing,
    }

    # ── External eval (n=795k, save_points=True) ──
    eval_log = fold_dir / "eval.log"
    eval_cmd = [
        python_exe(), str(EVAL_SCRIPT),
        "--refinement-run", str(anchor_run),
        "--split", "clean",
        "--filter-experiment", holdout,
        "--tag", f"lono_{holdout}",
        "--batch-points", "262144",
        "--no-save-plots",
        "--max-traj-plots", "0",
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(eval_cmd)}")
        eval_dir = Path(f"DRY_RUN/eval_{holdout}")
    else:
        rc, out = run_subprocess_streaming(eval_cmd, eval_log)
        if rc != 0:
            raise RuntimeError(f"External eval failed (rc={rc}). See {eval_log}.")
        match = EVAL_OUTPUT_RE.search(out)
        if match:
            eval_dir = Path(match.group(1).strip())
        else:
            # Fallback: glob the eval root for the latest dir matching tag.
            eval_root = MLP_ROOT / "eval"
            candidates = sorted(eval_root.glob(f"rmse_eval_clean_*_lono_{holdout}"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise RuntimeError(f"Could not locate eval dir for fold {holdout}.")
            eval_dir = candidates[0]
    record["eval_dir"] = str(eval_dir)
    record["timing"]["eval_s"] = time.time() - t0

    # ── Per-fold MLP metrics on held-out subset ──
    if not dry_run:
        mlp_points_csv = eval_dir / "points.csv"
        if not mlp_points_csv.exists():
            raise FileNotFoundError(f"Missing {mlp_points_csv}")
        mlp_pts = filter_points(mlp_points_csv, holdout, "folder")
        record["mlp_metrics"] = metrics_from_points(mlp_pts)
        # HA / NS on the same held-out subset
        if HA_POINTS_CSV.exists():
            record["ha_metrics"] = metrics_from_points(
                filter_points(HA_POINTS_CSV, holdout, "experiment_name"))
        if NS_POINTS_CSV.exists():
            record["ns_metrics"] = metrics_from_points(
                filter_points(NS_POINTS_CSV, holdout, "experiment_name"))
    else:
        record["mlp_metrics"] = {}
        record["ha_metrics"] = {}
        record["ns_metrics"] = {}

    (fold_dir / "fold_summary.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def aggregate(records: list[dict[str, Any]], output_dir: Path) -> None:
    metric_keys = ["rmse_mm", "mae_mm", "bias_mm", "p95_abs_err_mm",
                   "coverage_1sigma", "coverage_2sigma", "mean_pred_std_mm"]
    rows = []
    for rec in records:
        for model_name, key in [("MLP_dp05", "mlp_metrics"),
                                ("HA", "ha_metrics"),
                                ("NS", "ns_metrics")]:
            m = rec.get(key, {})
            if not m:
                continue
            row = {"holdout": rec["holdout"], "model": model_name}
            for k in metric_keys + ["n_points"]:
                row[k] = m.get(k)
            rows.append(row)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(output_dir / "per_fold.csv", index=False)

    # Aggregate: per-model fold mean ± std
    agg_rows = []
    for model_name in ["MLP_dp05", "HA", "NS"]:
        sub = per_fold.loc[per_fold["model"] == model_name]
        if sub.empty:
            continue
        agg = {"model": model_name, "n_folds": len(sub)}
        for k in metric_keys:
            vals = pd.to_numeric(sub[k], errors="coerce").dropna().to_numpy()
            agg[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            agg[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(agg)
    pd.DataFrame(agg_rows).to_csv(output_dir / "aggregate.csv", index=False)

    # Markdown headline
    md = ["# LONO 5-fold result (mean ± std across folds)\n"]
    md.append("| model | rmse_mm | mae_mm | bias_mm | p95_mm | cov_1σ | cov_2σ |")
    md.append("|---|---|---|---|---|---|---|")
    for agg in agg_rows:
        md.append(
            f"| {agg['model']} | "
            f"{agg['rmse_mm_mean']:.3f} ± {agg['rmse_mm_std']:.3f} | "
            f"{agg['mae_mm_mean']:.3f} ± {agg['mae_mm_std']:.3f} | "
            f"{agg['bias_mm_mean']:.3f} ± {agg['bias_mm_std']:.3f} | "
            f"{agg['p95_abs_err_mm_mean']:.3f} ± {agg['p95_abs_err_mm_std']:.3f} | "
            f"{agg['coverage_1sigma_mean']:.3f} ± {agg['coverage_1sigma_std']:.3f} | "
            f"{agg['coverage_2sigma_mean']:.3f} ± {agg['coverage_2sigma_std']:.3f} |"
        )
    md.append("\n## Per fold (held-out nozzle)\n")
    md.append("| holdout | model | rmse_mm | mae_mm | bias_mm | p95_mm | cov_1σ | cov_2σ |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in per_fold.iterrows():
        md.append(
            f"| {r['holdout']} | {r['model']} | "
            f"{r['rmse_mm']:.3f} | {r['mae_mm']:.3f} | {r['bias_mm']:.3f} | "
            f"{r['p95_abs_err_mm']:.3f} | {r['coverage_1sigma']:.3f} | "
            f"{r['coverage_2sigma']:.3f} |"
        )
    (output_dir / "headline_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n=== Aggregate written ===\n")
    print((output_dir / "headline_comparison.md").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else RUNS_ROOT / f"lono_pipeline_{timestamp}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"LONO output directory: {output_dir}")

    registry = build_dataset_registry()
    nozzles = identify_nozzle_families(
        data_dir=args.data_dir, registry=registry, min_groups=args.min_groups,
    )
    if args.skip_folds:
        nozzles = [n for n in nozzles if n not in set(args.skip_folds)]
    if args.n_folds is not None and args.n_folds < len(nozzles):
        nozzles = nozzles[: args.n_folds]
    print(f"\nWill run {len(nozzles)} folds: {nozzles}\n")

    (output_dir / "lono_config.json").write_text(json.dumps({
        "seed": args.seed,
        "device": args.device,
        "min_groups": args.min_groups,
        "n_folds": len(nozzles),
        "nozzles": nozzles,
        "skip_folds": args.skip_folds,
    }, indent=2), encoding="utf-8")

    records: list[dict[str, Any]] = []
    for i, holdout in enumerate(nozzles, start=1):
        print(f"\n========== Fold {i}/{len(nozzles)}: holdout={holdout} ==========")
        fold_dir = output_dir / f"fold_{i:02d}_{holdout}"
        try:
            rec = run_one_fold(
                holdout=holdout, fold_dir=fold_dir,
                seed=args.seed, device=args.device, dry_run=args.dry_run,
            )
            records.append(rec)
        except Exception as e:
            print(f"!! Fold {holdout} FAILED: {e}")
            (fold_dir / "fold_FAILED.json").write_text(
                json.dumps({"error": str(e)}, indent=2), encoding="utf-8"
            )

    (output_dir / "all_folds.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    if records and not args.dry_run:
        aggregate(records, output_dir)
    print(f"\nLONO pipeline complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
