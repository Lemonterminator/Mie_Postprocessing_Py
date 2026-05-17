"""Phase 2: evidence-chain audit pipeline.

Runs the individual audit scripts into ``MLP/audit_runs/{run_id}``, records a
manifest, and stores parent pointers to the fit archive used as input.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.common.archive_layout import AUDIT_RUNS, SYNTHETIC_DATA_RUNS, make_run_dir, resolve_latest, update_latest
from pipelines.common.run_metadata import finalize_metadata, make_run_id, write_metadata

SCRIPT_DIR = REPO_ROOT / "pipelines" / "audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-run-id", default=None)
    parser.add_argument("--fit-run-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=AUDIT_RUNS)
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "Mie_scattering_top_view_results")
    parser.add_argument("--dualfit-run-dir", type=Path, default=None)
    parser.add_argument("--cdf-audit-csv", type=Path, default=None)
    parser.add_argument("--support-csv", type=Path, default=None)
    parser.add_argument("--stage2-manifest", type=Path, default=None)
    parser.add_argument("--allow-synthetic-population", action="store_true")
    parser.add_argument("--skip", choices=("b1", "d2", "b3", "b5", "e1"), action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def resolve_fit_run(args: argparse.Namespace) -> Path:
    if args.fit_run_dir is not None:
        return args.fit_run_dir.resolve()
    if args.fit_run_id:
        return (SYNTHETIC_DATA_RUNS / args.fit_run_id).resolve()
    try:
        return resolve_latest(SYNTHETIC_DATA_RUNS).resolve()
    except FileNotFoundError:
        if args.dry_run:
            return SYNTHETIC_DATA_RUNS / "LATEST_FIT_RUN"
        raise


def _run(cmd: list[str], *, dry_run: bool, continue_on_error: bool) -> None:
    print("[audit]", " ".join(cmd), flush=True)
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        message = f"Audit command failed with code {completed.returncode}: {' '.join(cmd)}"
        if continue_on_error:
            print("WARNING:", message)
            return
        raise RuntimeError(message)


def _dualfit_available(path: Path | None) -> bool:
    if path is not None:
        return path.exists()
    return any(p.is_dir() for p in SYNTHETIC_DATA_RUNS.glob("*_dualfit"))


def run(args: argparse.Namespace) -> Path:
    fit_run_dir = resolve_fit_run(args)
    run_id = make_run_id("audit")
    run_dir = args.output_root / run_id if args.dry_run else make_run_dir(args.output_root, run_id)
    config = {
        "fit_run_dir": str(fit_run_dir),
        "raw_root": str(args.raw_root),
        "dualfit_run_dir": None if args.dualfit_run_dir is None else str(args.dualfit_run_dir),
        "cdf_audit_csv": None if args.cdf_audit_csv is None else str(args.cdf_audit_csv),
        "support_csv": None if args.support_csv is None else str(args.support_csv),
        "skip": list(args.skip),
        "dry_run": bool(args.dry_run),
    }
    started = time.monotonic()
    if not args.dry_run:
        write_metadata(run_dir, phase="audit", config=config, parent_run_ids={"fit": fit_run_dir.name})

    commands: list[tuple[str, list[str]]] = [
        (
            "b1",
            [
                sys.executable,
                str(SCRIPT_DIR / "scr_naive_hold_gap.py"),
                "--fit-run-dir",
                str(fit_run_dir),
                "--out-dir",
                str(run_dir / "b1_scr_gap"),
            ],
        ),
        (
            "d2",
            [
                sys.executable,
                str(SCRIPT_DIR / "data_attrition_report.py"),
                "--fit-run-dir",
                str(fit_run_dir),
                "--raw-root",
                str(args.raw_root),
                "--out-dir",
                str(run_dir / "d2_data_attrition"),
            ],
        ),
        (
            "b5",
            [
                sys.executable,
                str(SCRIPT_DIR / "raw_coverage_heatmap.py"),
                "--fit-run-dir",
                str(fit_run_dir),
                "--out-dir",
                str(run_dir / "b5_raw_coverage"),
            ],
        ),
        (
            "b3",
            [
                sys.executable,
                str(SCRIPT_DIR / "q1_vs_two_regime_comparison.py"),
                "--out-dir",
                str(run_dir / "b3_q1_vs_two_regime"),
            ],
        ),
        (
            "e1",
            [
                sys.executable,
                str(SCRIPT_DIR / "scr_ood_cross_audit.py"),
                "--fit-run-dir",
                str(fit_run_dir),
                "--out-dir",
                str(run_dir / "e1_scr_ood"),
            ],
        ),
    ]

    if args.allow_synthetic_population:
        commands[1][1].append("--allow-synthetic-population")
    if args.stage2_manifest is not None:
        commands[1][1].extend(["--stage2-manifest", str(args.stage2_manifest)])
    if args.cdf_audit_csv is not None:
        commands[2][1].extend(["--input-csv", str(args.cdf_audit_csv)])
    if args.dualfit_run_dir is not None:
        commands[3][1].extend(["--dualfit-run-dir", str(args.dualfit_run_dir)])
    if args.support_csv is not None:
        commands[4][1].extend(["--support-csv", str(args.support_csv)])

    skipped = set(args.skip)
    if "b3" not in skipped and not _dualfit_available(args.dualfit_run_dir):
        skipped.add("b3")
        print("[audit] skip b3: no *_dualfit archive found; run Phase 1 with --ablation-dual-fit to enable it.")
    for key, cmd in commands:
        if key in skipped:
            print(f"[audit] skip {key}")
            continue
        _run(cmd, dry_run=args.dry_run, continue_on_error=args.continue_on_error)

    if not args.dry_run:
        finalize_metadata(run_dir, started_wall=started)
        update_latest(args.output_root, run_dir)
        (run_dir / "audit_pipeline_summary.json").write_text(
            json.dumps({"run_id": run_id, "fit_run_dir": str(fit_run_dir), "skipped": sorted(skipped)}, indent=2),
            encoding="utf-8",
        )
    else:
        print("[audit] DRY-RUN: not writing metadata, summary, or latest.txt")
    print(f"[audit] Done: {run_dir}")
    return run_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
