"""Phase 1: Curve-fitting pipeline wrapper.

Invokes fit_raw_data.py (with versioned OUTPUT_ROOT), then the audit and
summary scripts. Writes a timestamped archive to MLP/synthetic_data_runs/{run_id}/
and updates MLP/synthetic_data_runs/latest.txt.

Usage
-----
    python pipelines/fit/run_fit_pipeline.py [options]

Options
-------
    --output-root PATH    Override default archive root (MLP/synthetic_data_runs/)
    --input-root PATH     Path to raw Mie scattering results (default: inferred)
    --ablation-dual-fit   Enable four-param + q1 dual-fit (for B.3 audit). Writes
                          to a separate {run_id}_dualfit/ archive, not promoted to latest.
    --n-workers N         Worker count for fit_raw_data (0 = all CPUs)
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

FIT_SCRIPT = REPO_ROOT / "MLP" / "curve_fit" / "fit_raw_data.py"
AUDIT_SCRIPT = REPO_ROOT / "MLP" / "curve_fit" / "audit_cdf_spatial_censoring.py"
SURVIVAL_SCRIPT = REPO_ROOT / "MLP" / "curve_fit" / "summarize_filter_survival.py"
DATASET_SCRIPT = REPO_ROOT / "MLP" / "curve_fit" / "summarize_dataset.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--input-root", type=Path, default=None)
    p.add_argument("--ablation-dual-fit", action="store_true", default=False,
                   help="Run a side-archive with four-param dual-fit for B.3 audit.")
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
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} exited with code {result.returncode}")


def _run_fit(
    run_dir: Path,
    *,
    input_root: Path | None,
    ablation: bool,
    n_workers: int,
    dry_run: bool,
    nozzle_filter: str | None,
) -> None:
    env = os.environ.copy()
    env["FIT_OUTPUT_ROOT"] = str(run_dir)
    if input_root:
        env["FIT_INPUT_ROOT"] = str(input_root)
    if ablation:
        env["FIT_ABLATION_QUARTER_ONLY"] = "1"
    env["FIT_N_WORKERS"] = str(n_workers)

    cmd = [sys.executable, str(FIT_SCRIPT), "--no-chain"]
    if nozzle_filter:
        # patch: restrict names list via env var (fit_raw_data reads FIT_NOZZLE_FILTER)
        env["FIT_NOZZLE_FILTER"] = nozzle_filter

    _run(cmd, env=env, dry_run=dry_run, label="fit_raw_data.py")


def _run_audit(run_dir: Path, *, dry_run: bool) -> None:
    out_dir = run_dir / "spatial_censoring_audit"
    cmd = [
        sys.executable, str(AUDIT_SCRIPT),
        "--synthetic-root", str(run_dir),
        "--out-dir", str(out_dir),
    ]
    _run(cmd, env=os.environ.copy(), dry_run=dry_run, label="audit_cdf_spatial_censoring.py")


def _run_survival(run_dir: Path, *, dry_run: bool) -> None:
    report_csv = run_dir / "fit_report.csv"
    if not dry_run and not report_csv.exists():
        print(f"  SKIP summarize_filter_survival: fit_report.csv not found at {report_csv}")
        return
    cmd = [
        sys.executable, str(SURVIVAL_SCRIPT),
        "--fit-report", str(report_csv),
        "--out-dir", str(run_dir / "fit_survival_report"),
    ]
    _run(cmd, env=os.environ.copy(), dry_run=dry_run, label="summarize_filter_survival.py")


def _run_dataset_summary(dry_run: bool) -> None:
    # summarize_dataset.py has no argparse; writes to MLP/curve_fit/ by default.
    cmd = [sys.executable, str(DATASET_SCRIPT)]
    _run(cmd, env=os.environ.copy(), dry_run=dry_run, label="summarize_dataset.py")


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
        "ablation_dual_fit": args.ablation_dual_fit,
        "mirror_canonical": mirror,
        "nozzle_filter": args.nozzle_filter,
    }
    t0 = time.monotonic()
    if not args.dry_run:
        write_metadata(run_dir, phase="fit", config=config)

    _run_fit(
        run_dir,
        input_root=args.input_root,
        ablation=False,
        n_workers=args.n_workers,
        dry_run=args.dry_run,
        nozzle_filter=args.nozzle_filter,
    )
    _run_audit(run_dir, dry_run=args.dry_run)
    _run_survival(run_dir, dry_run=args.dry_run)
    _run_dataset_summary(dry_run=args.dry_run)

    if mirror:
        _mirror_canonical(run_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[fit pipeline] DRY-RUN: not writing metadata, manifests, or latest.txt")
    else:
        _record_outputs(run_dir)
        finalize_metadata(run_dir, started_wall=t0)
        update_latest(archive_root, run_dir)
    print(f"\n[fit pipeline] Done. Run ID: {run_id}")

    # Optional dual-fit side-run (for B.3 comparison audit)
    if args.ablation_dual_fit:
        dual_run_id = run_id + "_dualfit"
        dual_dir = archive_root / dual_run_id if args.dry_run else make_run_dir(archive_root, dual_run_id)
        t1 = time.monotonic()
        if not args.dry_run:
            write_metadata(
                dual_dir,
                phase="fit_dualfit",
                config={**config, "ablation_dual_fit": True},
                parent_run_ids={"production_fit": run_id},
            )
        _run_fit(
            dual_dir,
            input_root=args.input_root,
            ablation=True,
            n_workers=args.n_workers,
            dry_run=args.dry_run,
            nozzle_filter=args.nozzle_filter,
        )
        if not args.dry_run:
            _record_outputs(dual_dir)
            finalize_metadata(dual_dir, started_wall=t1)
        print(f"[fit pipeline] Dual-fit archive: {dual_run_id}")

    return run_dir


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
