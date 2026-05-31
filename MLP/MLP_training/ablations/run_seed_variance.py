"""Tier-1B seed variance driver.

Runs 5 full pipelines (Stage 1 -> Stage 2 -> Stage 3 anchor_off) holding out
Nozzle 2 each time, with five different seeds. Produces sigma_seed: the
single number used to decide which existing Section-5 ablation gaps are
above noise.

Spec lives in MLP/MLP_training/ablations/TIER1_CAPACITY_SEED_OPTIMIZER_PLAN.md
section 4.

Wall clock: ~9 min/seed on RTX 5090. 5 seeds ~= 45 min.

Usage
-----
    python MLP/MLP_training/ablations/run_seed_variance.py
    python MLP/MLP_training/ablations/run_seed_variance.py --device cpu --dry-run

Outputs (under MLP/runs_mlp/seed_variance_nozzle2_<timestamp>/):
    per_seed.csv         one row per seed with rmse_mm/mae_mm/bias_mm/coverage
    summary.json         mean/std/p25/p75/ci across the 5 seeds
    verdict.md           sigma_seed headline + which Section-5 gaps survive

Decision rule (from plan section 4.4):
    sigma_seed < 0.5 mm -> both Stage-2 anchor gaps real
    sigma_seed > 1.5 mm -> mu_sigma gap (0.32 mm) is noise
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MLP_ROOT = PROJECT_ROOT / "MLP"
TRAINING_ROOT = MLP_ROOT / "MLP_training"
CONFIG_ROOT = TRAINING_ROOT / "config"
RUNS_ROOT = MLP_ROOT / "runs_mlp"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

STAGE1_SCRIPT = TRAINING_ROOT / "train_stage1_mse.py"
STAGE2_SCRIPT = TRAINING_ROOT / "train_stage2_nll.py"
STAGE3_SUITE_SCRIPT = TRAINING_ROOT / "run_stage3_ablation_suite.py"
STAGE3_SUITE_CONFIG = CONFIG_ROOT / "stage3_ablation_suite_config.json"
EVAL_SCRIPT = MLP_ROOT / "eval" / "inference_rmse_on_series.py"

SAVED_RUN_DIR_RE = re.compile(r"Saved run_dir:\s*(.+)$")
SUITE_SUMMARY_RE = re.compile(r"Suite summary saved to:\s*(.+)$")
EVAL_OUTPUT_RE = re.compile(r"Wrote RMSE evaluation to\s+(.+)$", re.MULTILINE)

DEFAULT_SEEDS = [13, 42, 91, 137, 271]
DEFAULT_HOLDOUT = "Nozzle2"
STAGE3_ONLY = "anchor_off"


def python_exe() -> str:
    return str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))


def run_subprocess_streaming(cmd: list[str], log_path: Path) -> tuple[int, str]:
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
    return int(proc.returncode), "".join(captured)


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


def find_named_run_dir(suite_summary_path: Path, name: str) -> Path:
    summary = json.loads(suite_summary_path.read_text(encoding="utf-8"))
    for entry in summary.get("results", []):
        if str(entry.get("name")) == name:
            run_dir = entry.get("run_dir")
            if run_dir:
                return Path(run_dir)
    best = (summary.get("selection") or {}).get("best") or {}
    if str(best.get("name")) == name and best.get("run_dir"):
        return Path(best["run_dir"])
    raise RuntimeError(f"Could not locate {name!r} run_dir in {suite_summary_path}")


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


def run_one_seed_fold(
    *,
    seed: int,
    holdout: str,
    seed_dir: Path,
    device: str,
    dry_run: bool,
) -> dict[str, Any]:
    seed_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    # Stage 1
    s1_log = seed_dir / "stage1.log"
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
        stage1_run = Path(f"DRY_RUN/stage1_seed{seed}")
    else:
        rc, out = run_subprocess_streaming(s1_cmd, s1_log)
        if rc != 0:
            raise RuntimeError(f"Stage-1 failed (rc={rc}). See {s1_log}.")
        stage1_run = parse_saved_run_dir(out)
        if stage1_run is None:
            raise RuntimeError("Could not parse Stage-1 run_dir.")
    timing["stage1_s"] = time.time() - t0

    # Stage 2
    s2_log = seed_dir / "stage2.log"
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
        stage2_run = Path(f"DRY_RUN/stage2_seed{seed}")
    else:
        rc, out = run_subprocess_streaming(s2_cmd, s2_log)
        if rc != 0:
            raise RuntimeError(f"Stage-2 failed (rc={rc}). See {s2_log}.")
        stage2_run = parse_saved_run_dir(out)
        if stage2_run is None:
            raise RuntimeError("Could not parse Stage-2 run_dir.")
    timing["stage2_s"] = time.time() - t0

    # Stage 3 (anchor_off only)
    s3_log = seed_dir / "stage3.log"
    s3_cmd = [
        python_exe(), str(STAGE3_SUITE_SCRIPT),
        "--config", str(STAGE3_SUITE_CONFIG),
        "--teacher-run", str(stage2_run),
        "--device", device,
        "--seed", str(seed),
        "--only", STAGE3_ONLY,
        "--lono-holdout", holdout,
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s3_cmd)}")
        anchor_run = Path(f"DRY_RUN/{STAGE3_ONLY}_seed{seed}")
        suite_summary = Path(f"DRY_RUN/suite_seed{seed}.json")
    else:
        rc, out = run_subprocess_streaming(s3_cmd, s3_log)
        if rc != 0:
            raise RuntimeError(f"Stage-3 suite failed (rc={rc}). See {s3_log}.")
        suite_summary = parse_suite_summary_path(out)
        if suite_summary is None:
            raise RuntimeError("Could not parse suite_summary path.")
        anchor_run = find_named_run_dir(suite_summary, STAGE3_ONLY)
    timing["stage3_s"] = time.time() - t0

    # External eval on the held-out nozzle subset
    eval_log = seed_dir / "eval.log"
    eval_cmd = [
        python_exe(), str(EVAL_SCRIPT),
        "--refinement-run", str(anchor_run),
        "--split", "clean",
        "--filter-experiment", holdout,
        "--tag", f"seed_var_seed{seed}_{holdout}",
        "--batch-points", "262144",
        "--no-save-plots",
        "--max-traj-plots", "0",
    ]
    t0 = time.time()
    metrics: dict[str, float] = {}
    if dry_run:
        print(f"[dry-run] {' '.join(eval_cmd)}")
        eval_dir = Path(f"DRY_RUN/eval_seed{seed}")
    else:
        rc, out = run_subprocess_streaming(eval_cmd, eval_log)
        if rc != 0:
            raise RuntimeError(f"External eval failed (rc={rc}). See {eval_log}.")
        match = EVAL_OUTPUT_RE.search(out)
        if match:
            eval_dir = Path(match.group(1).strip())
        else:
            candidates = sorted(
                (MLP_ROOT / "eval").glob(f"rmse_eval_clean_*_seed_var_seed{seed}_{holdout}"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if not candidates:
                raise RuntimeError(f"Could not locate eval dir for seed {seed}.")
            eval_dir = candidates[0]
        points_csv = eval_dir / "points.csv"
        if not points_csv.exists():
            raise FileNotFoundError(f"Missing {points_csv}")
        df = pd.read_csv(points_csv, low_memory=False)
        df = df.loc[df["folder"].astype(str) == str(holdout)].copy()
        metrics = metrics_from_points(df)
    timing["eval_s"] = time.time() - t0

    record = {
        "seed": seed,
        "holdout": holdout,
        "stage1_run": str(stage1_run),
        "stage2_run": str(stage2_run),
        "stage3_suite_summary": str(suite_summary),
        "winner_run": str(anchor_run),
        "eval_dir": str(eval_dir),
        "metrics": metrics,
        "timing": timing,
    }
    (seed_dir / "seed_summary.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def bootstrap_summary(values: list[float], *, n_resample: int = 5000) -> dict[str, float]:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p25": float("nan"),
                "p75": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    rng = np.random.default_rng(20260528)
    if arr.size > 1:
        boot_means = rng.choice(arr, size=(n_resample, arr.size), replace=True).mean(axis=1)
        ci_lo = float(np.quantile(boot_means, 0.025))
        ci_hi = float(np.quantile(boot_means, 0.975))
        std = float(arr.std(ddof=1))
    else:
        ci_lo = ci_hi = float(arr[0])
        std = 0.0
    return {
        "mean": float(arr.mean()),
        "std": std,
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "n": int(arr.size),
    }


def write_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for rec in records:
        m = rec.get("metrics", {}) or {}
        rows.append({
            "seed": rec["seed"],
            "rmse_mm": m.get("rmse_mm"),
            "mae_mm": m.get("mae_mm"),
            "bias_mm": m.get("bias_mm"),
            "p95_abs_err_mm": m.get("p95_abs_err_mm"),
            "coverage_1sigma": m.get("coverage_1sigma"),
            "coverage_2sigma": m.get("coverage_2sigma"),
            "n_points": m.get("n_points"),
            "stage1_minutes": (rec.get("timing", {}).get("stage1_s") or 0.0) / 60.0,
            "stage2_minutes": (rec.get("timing", {}).get("stage2_s") or 0.0) / 60.0,
            "stage3_minutes": (rec.get("timing", {}).get("stage3_s") or 0.0) / 60.0,
        })
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output_dir / "per_seed.csv", index=False)

    metric_keys = ["rmse_mm", "mae_mm", "bias_mm", "p95_abs_err_mm",
                   "coverage_1sigma", "coverage_2sigma"]
    summary = {
        "holdout": DEFAULT_HOLDOUT,
        "seeds": [int(r["seed"]) for r in records],
        "metrics": {k: bootstrap_summary(per_seed[k].tolist()) for k in metric_keys},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    rmse_stats = summary["metrics"]["rmse_mm"]
    sigma_seed = rmse_stats.get("std", float("nan"))
    rule = []
    if not np.isnan(sigma_seed):
        if sigma_seed < 0.5:
            rule.append("sigma_seed < 0.5 mm: both Stage-2 anchor gaps (1.81, 0.32) are real.")
        elif sigma_seed > 1.5:
            rule.append("sigma_seed > 1.5 mm: mu_sigma gap (0.32) is noise; only no_anchor->mu_anchor is significant.")
        else:
            rule.append(f"0.5 <= sigma_seed = {sigma_seed:.2f} mm <= 1.5: no_anchor->mu_anchor (1.81) real; mu_sigma (0.32) borderline.")

    md_lines = [
        f"# Tier-1B seed variance verdict ({DEFAULT_HOLDOUT})",
        "",
        f"- Seeds:           {summary['seeds']}",
        f"- N folds:         1 ({DEFAULT_HOLDOUT})",
        f"- RMSE mean:       {rmse_stats.get('mean', float('nan')):.4f} mm",
        f"- **sigma_seed:**  {sigma_seed:.4f} mm  (std over {rmse_stats.get('n', 0)} seeds)",
        f"- 95% CI mean:     [{rmse_stats.get('ci_lo', float('nan')):.4f}, {rmse_stats.get('ci_hi', float('nan')):.4f}] mm",
        "",
        "## Decision rule applied",
        "",
    ] + [f"- {line}" for line in rule] + [
        "",
        "## Section-5 gaps interpreted against this sigma_seed",
        "",
        "| Table | Comparison | Gap (mm) | Survives sigma_seed? |",
        "|---|---|---|---|",
        f"| Table 4 | no_anchor -> mu_anchor | 1.81 | {'yes' if sigma_seed < 1.81 else 'no'} |",
        f"| Table 4 | mu_anchor -> mu_sigma_anchor | 0.32 | {'yes' if sigma_seed < 0.32 else 'no'} |",
        "| Table 5 | regime gaps 0.06-0.21 | <=0.21 | likely no (within noise) |",
    ]
    (output_dir / "verdict.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("\n=== Tier-1B verdict ===")
    print((output_dir / "verdict.md").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                   help="Seeds to evaluate (default: 13 42 91 137 271).")
    p.add_argument("--holdout", type=str, default=DEFAULT_HOLDOUT,
                   help="LONO held-out nozzle (default: Nozzle2 - median fold).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else RUNS_ROOT / f"seed_variance_{args.holdout.lower()}_{timestamp}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tier-1B seed variance output dir: {output_dir}")

    (output_dir / "config.json").write_text(json.dumps({
        "seeds": args.seeds,
        "holdout": args.holdout,
        "device": args.device,
        "stage3_only": STAGE3_ONLY,
        "stage3_suite_config": str(STAGE3_SUITE_CONFIG),
    }, indent=2), encoding="utf-8")

    records: list[dict[str, Any]] = []
    for i, seed in enumerate(args.seeds, start=1):
        print(f"\n========== Seed {i}/{len(args.seeds)}: {seed} ==========")
        seed_dir = output_dir / f"seed_{seed:03d}"
        try:
            rec = run_one_seed_fold(
                seed=int(seed), holdout=args.holdout, seed_dir=seed_dir,
                device=args.device, dry_run=args.dry_run,
            )
            records.append(rec)
        except Exception as e:
            print(f"!! Seed {seed} FAILED: {e}")
            (seed_dir / "seed_FAILED.json").write_text(
                json.dumps({"error": str(e)}, indent=2), encoding="utf-8"
            )

    (output_dir / "all_seeds.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    if records and not args.dry_run:
        write_outputs(records, output_dir)
    print(f"\nTier-1B done. Output: {output_dir}")


if __name__ == "__main__":
    main()
