"""Phase 1: Curve-fitting pipeline wrapper.

Invokes ``MLP.curve_fit.workflows.raw_fit`` with a versioned ``OUTPUT_ROOT``.
The fit workflow also generates diagnostics, CDF right-censoring point
tables, and p50-q1 oracle baselines. Writes a timestamped archive to
``MLP/synthetic_data_runs/{run_id}/`` and updates
``MLP/synthetic_data_runs/latest.txt``.

Usage
-----
    python pipelines/fit/run_fit_pipeline.py [options]

Options
-------
    --output-root PATH    Override default archive root (MLP/synthetic_data_runs/)
    --input-root PATH     Path to raw Mie scattering results (default: inferred)
    --n-workers N         Worker count for the fit workflow (0 = all CPUs)
    --mirror-canonical    Also mirror outputs to MLP/synthetic_data/ (default: on)
    --no-mirror           Turn off canonical mirror
    --dry-run             Print what would run without executing
    --nozzle-filter NAME  Restrict to one nozzle name (for smoke-testing)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.common.archive_layout import (
    SYNTHETIC_DATA_RUNS,
    SYNTHETIC_DATA_CANONICAL,
    append_manifest,
    make_run_dir,
    update_latest,
)
from pipelines.common.run_metadata import make_run_id, write_metadata, finalize_metadata

FIT_MODULE = "MLP.curve_fit.workflows.raw_fit"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--input-root", type=Path, default=None)
    p.add_argument("--n-workers", type=int, default=0)
    p.add_argument("--mirror-canonical", action="store_true", default=True)
    p.add_argument("--no-mirror", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--nozzle-filter", type=str, default=None,
                   help="Restrict to one nozzle name (for smoke-testing).")
    return p.parse_args()


def _run(cmd: list[str], env: dict, dry_run: bool, label: str) -> None:
    print(f"\n[fit pipeline] {label}")
    if dry_run:
        print(f"  DRY-RUN: {' '.join(str(c) for c in cmd)}")
        return
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} exited with code {result.returncode}")


def _run_fit(
    run_dir: Path,
    *,
    input_root: Path | None,
    n_workers: int,
    dry_run: bool,
    nozzle_filter: str | None,
) -> None:
    env = os.environ.copy()
    env["FIT_OUTPUT_ROOT"] = str(run_dir)
    if input_root:
        env["FIT_INPUT_ROOT"] = str(input_root)
    env["FIT_N_WORKERS"] = str(n_workers)

    cmd = [sys.executable, "-m", FIT_MODULE]
    if nozzle_filter:
        # raw_fit reads FIT_NOZZLE_FILTER at config import time in worker subprocesses.
        env["FIT_NOZZLE_FILTER"] = nozzle_filter

    _run(cmd, env=env, dry_run=dry_run, label=FIT_MODULE)


def _mirror_canonical(run_dir: Path, dry_run: bool) -> None:
    """Copy run_dir contents to MLP/synthetic_data/ (canonical consumer path)."""
    dest = SYNTHETIC_DATA_CANONICAL
    print(f"\n[fit pipeline] Mirroring {run_dir} -> {dest}")
    if dry_run:
        print("  DRY-RUN: skipping copy")
        return
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(run_dir, dest, ignore=shutil.ignore_patterns("_metadata.json", "_manifest.csv", "latest.txt"))
    print("  Mirror complete.")


def _record_outputs(run_dir: Path) -> None:
    """Walk run_dir and record significant outputs in _manifest.csv."""
    role_map = {
        "fit_report.csv": ("fit_report", "Per-folder fit statistics"),
        "spatial_censoring_audit/plume_spatial_censoring_audit.csv": ("scr_audit_detail", "Per-trajectory SCR audit"),
        "spatial_censoring_audit/spatial_censoring_summary_overall.csv": ("scr_audit_summary", "SCR overall summary"),
        "fit_survival_report/filter_survival_by_source.csv": ("filter_survival", "Filter survival by penetration source"),
        "cdf_right_censoring_points/cdf_points_uncensored.csv": ("cdf_uncensored_points", "Point-level CDF uncensored samples"),
        "cdf_right_censoring_points/cdf_condition_censoring_summary.csv": ("cdf_censoring_condition_summary", "CDF condition-level censoring summary"),
        "p50_q1_oracle/p50_q1_fit_metrics.csv": ("p50_q1_oracle_fit_metrics", "Per-condition p50-q1 oracle fit metrics"),
        "p50_q1_oracle/p50_q1_predictions.csv": ("p50_q1_oracle_predictions", "Per-condition p50-q1 extrapolation predictions"),
    }
    for rel, (role, desc) in role_map.items():
        if (run_dir / rel).exists():
            append_manifest(run_dir, role=role, filename=rel, description=desc)


def run(args: argparse.Namespace) -> Path:
    mirror = args.mirror_canonical and not args.no_mirror

    run_id = make_run_id("fit")
    archive_root = args.output_root or SYNTHETIC_DATA_RUNS
    if args.dry_run:
        run_dir = archive_root / run_id
    else:
        archive_root.mkdir(parents=True, exist_ok=True)
        run_dir = make_run_dir(archive_root, run_id)
    print(f"\n[fit pipeline] Run directory: {run_dir}")

    config = {
        "input_root": str(args.input_root) if args.input_root else "default",
        "n_workers": args.n_workers,
        "mirror_canonical": mirror,
        "nozzle_filter": args.nozzle_filter,
    }
    t0 = time.monotonic()
    if not args.dry_run:
        write_metadata(run_dir, phase="fit", config=config)

    _run_fit(
        run_dir,
        input_root=args.input_root,
        n_workers=args.n_workers,
        dry_run=args.dry_run,
        nozzle_filter=args.nozzle_filter,
    )
    if mirror:
        _mirror_canonical(run_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[fit pipeline] DRY-RUN: not writing metadata, manifests, or latest.txt")
    else:
        _record_outputs(run_dir)
        finalize_metadata(run_dir, started_wall=t0)
        update_latest(archive_root, run_dir)
    print(f"\n[fit pipeline] Done. Run ID: {run_id}")
    return run_dir


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
