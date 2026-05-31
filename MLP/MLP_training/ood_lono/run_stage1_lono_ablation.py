"""Stage-1 feature-variant ablation over LONO folds.

Runs train_stage1_mse.py for each (variant, holdout) combination,
collects test_summary.csv metrics, and prints a comparison table.

Usage
-----
    python MLP/MLP_training/ood_lono/run_stage1_lono_ablation.py
    python MLP/MLP_training/ood_lono/run_stage1_lono_ablation.py --device cpu
    python MLP/MLP_training/ood_lono/run_stage1_lono_ablation.py --n-folds 3
    python MLP/MLP_training/ood_lono/run_stage1_lono_ablation.py --dry-run
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

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

if __package__ in {None, ""}:
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here))           # ood_lono/ — for sibling imports
    sys.path.insert(0, str(_here.parent))    # MLP_training/ — for engineered_feature_common

from engineered_feature_common import (
    DEFAULT_STAGE1_CONFIG,
    build_all_stage_tables,
    build_dataset_registry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MLP_ROOT = PROJECT_ROOT / "MLP"
TRAINING_ROOT = MLP_ROOT / "MLP_training"
RUNS_ROOT = MLP_ROOT / "runs_mlp"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
STAGE1_SCRIPT = TRAINING_ROOT / "train_stage1_mse.py"

SAVED_RUN_DIR_RE = re.compile(r"Saved run_dir:\s*(.+)$")

VARIANTS = [
    "legacy_9_no_scale",
    "a_dp025",
    "a_dp025_plus_pressures",
    "a_dp025_plus_diameter",
    "a_dp025_plus_pressures_diameter",
    "a_dp050",
    "a_dp050_plus_pressures",
    "a_dp050_plus_diameter",
    "a_dp050_plus_pressures_diameter",
]


def python_exe() -> str:
    return str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))


def get_nozzle_folds(n_folds: int, min_groups: int) -> list[str]:
    print("Building representative table to enumerate nozzle families ...")
    registry = build_dataset_registry(None)
    stage_tables = build_all_stage_tables(
        DEFAULT_STAGE1_CONFIG["data_dir"],
        registry,
        comparison_time_s=float(DEFAULT_STAGE1_CONFIG["comparison_time_s"]),
        max_curves=None,
        output_dir=None,
    )
    rep = stage_tables.representative
    col = "experiment_name" if "experiment_name" in rep.columns else "source_dataset_name"
    counts = rep.groupby(col)["sample_group_id"].nunique().sort_values(ascending=False)
    print("\nPer-nozzle sample_group_id counts:")
    print(counts.to_string())
    valid = counts[counts >= min_groups]
    selected = [str(n) for n in valid.index[:n_folds]]
    print(f"\nSelected {len(selected)} folds: {selected}\n")
    return selected


def run_stage1(
    *,
    variant: str,
    holdout: str,
    seed: int,
    device: str,
    log_path: Path,
    dry_run: bool,
) -> Path | None:
    cmd = [
        python_exe(), str(STAGE1_SCRIPT),
        "--variant", variant,
        "--device", device,
        "--seed", str(seed),
        "--lono-holdout", holdout,
        "--allow-failed-precheck",
    ]
    print(f"\n[run] {' '.join(shlex.quote(p) for p in cmd)}", flush=True)
    if dry_run:
        return None
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
    if rc != 0:
        print(f"[warn] Stage-1 failed (rc={rc}) for variant={variant}, holdout={holdout}. See {log_path}.")
        return None
    stdout = "".join(captured)
    for line in reversed(stdout.splitlines()):
        m = SAVED_RUN_DIR_RE.search(line.strip())
        if m:
            return Path(m.group(1).strip())
    print(f"[warn] Could not parse run_dir for variant={variant}, holdout={holdout}")
    return None


def read_test_summary(run_dir: Path) -> dict[str, float]:
    p = run_dir / "test_summary.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if df.empty:
        return {}
    row = df.iloc[-1].to_dict()
    return {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-groups", type=int, default=50)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (RUNS_ROOT / f"stage1_lono_ablation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    folds = get_nozzle_folds(args.n_folds, args.min_groups)

    results: list[dict] = []
    flag_path = output_dir / "results.json"

    # Resume from prior partial run if possible
    if flag_path.exists() and not args.dry_run:
        prior = json.loads(flag_path.read_text(encoding="utf-8"))
        done_keys = {(r["variant"], r["holdout"]) for r in prior}
        results = prior
        print(f"[resume] Found {len(results)} completed runs.")
    else:
        done_keys: set[tuple[str, str]] = set()

    total = len(VARIANTS) * len(folds)
    done = len(done_keys)
    print(f"\nTotal runs: {total}. Already done: {done}. Remaining: {total - done}\n")

    for variant in VARIANTS:
        for holdout in folds:
            if (variant, holdout) in done_keys:
                print(f"[skip] variant={variant}, holdout={holdout}")
                continue
            log_path = output_dir / f"{variant}__{holdout}.log"
            t0 = time.time()
            run_dir = run_stage1(
                variant=variant,
                holdout=holdout,
                seed=args.seed,
                device=args.device,
                log_path=log_path,
                dry_run=args.dry_run,
            )
            elapsed = time.time() - t0
            if args.dry_run:
                continue
            metrics = read_test_summary(run_dir) if run_dir else {}
            record = {
                "variant": variant,
                "holdout": holdout,
                "run_dir": str(run_dir) if run_dir else None,
                "elapsed_s": round(elapsed, 1),
                **metrics,
            }
            results.append(record)
            done_keys.add((variant, holdout))
            flag_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"  -> physical_mae={metrics.get('physical_mae', float('nan')):.4f}  "
                  f"mse_scaled={metrics.get('mse_scaled', float('nan')):.5f}  "
                  f"elapsed={elapsed:.0f}s")

    if args.dry_run:
        print("\n[dry-run] No runs executed.")
        return

    if not results:
        print("No results collected.")
        return

    df = pd.DataFrame(results)

    # Per-variant aggregate (mean across folds)
    agg = (
        df.groupby("variant")[["physical_mae", "mse_scaled"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["mae_mean", "mae_std", "mse_mean", "mse_std"]
    agg = agg.sort_values("mae_mean")

    print("\n" + "=" * 80)
    print("STAGE-1 LONO ABLATION RESULTS")
    print("=" * 80)
    print(f"\nMetric: physical_mae (mm) — mean and std across {len(folds)} LONO folds\n")
    print(agg.to_string())

    # Per-fold breakdown table
    pivot = df.pivot_table(index="variant", columns="holdout", values="physical_mae", aggfunc="mean")
    pivot["MEAN"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("MEAN")
    print("\n\nPer-fold physical_mae (mm):")
    print(pivot.round(3).to_string())

    # Save summary
    summary_path = output_dir / "ablation_summary.csv"
    agg.to_csv(summary_path)
    pivot_path = output_dir / "ablation_pivot.csv"
    pivot.round(4).to_csv(pivot_path)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Pivot saved to:   {pivot_path}")
    print(f"Results JSON:     {flag_path}")


if __name__ == "__main__":
    main()
