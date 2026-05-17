"""Top-level four-phase pipeline entry point.

Default chain:
  Phase 1 fit -> Phase 2 audits -> Phase 3 MLP training -> Phase 4 thesis assembly

Use ``--skip-phase fit`` etc. for partial reruns.  The phase wrappers keep
``latest.txt`` pointers, so skipped phases reuse the latest completed archive.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.common.archive_layout import AUDIT_RUNS, SYNTHETIC_DATA_RUNS, resolve_latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-phase", choices=("fit", "audit", "train", "report"), action="append", default=[])
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--nozzle-filter", default=None)
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument("--include-dualfit-audit", action="store_true")
    parser.add_argument("--allow-synthetic-population", action="store_true")
    parser.add_argument("--train-mode", choices=("single", "A", "B", "C"), default="single")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--skip-alpha", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _skips(args: argparse.Namespace) -> set[str]:
    out = set(args.skip_phase)
    for flag, phase in (
        (args.skip_fit, "fit"),
        (args.skip_audit, "audit"),
        (args.skip_train, "train"),
        (args.skip_report, "report"),
    ):
        if flag:
            out.add(phase)
    return out


def _run(cmd: list[str], *, dry_run: bool, continue_on_error: bool) -> None:
    print("[full]", " ".join(cmd), flush=True)
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        message = f"Command failed with code {completed.returncode}: {' '.join(cmd)}"
        if continue_on_error:
            print("WARNING:", message)
            return
        raise RuntimeError(message)


def run(args: argparse.Namespace) -> None:
    skips = _skips(args)

    if "fit" not in skips:
        fit_cmd = [sys.executable, str(REPO_ROOT / "pipelines" / "fit" / "run_fit_pipeline.py"), "--n-workers", str(args.n_workers)]
        if args.input_root is not None:
            fit_cmd.extend(["--input-root", str(args.input_root)])
        if args.nozzle_filter:
            fit_cmd.extend(["--nozzle-filter", args.nozzle_filter])
        if args.include_dualfit_audit:
            fit_cmd.append("--ablation-dual-fit")
        if args.dry_run:
            fit_cmd.append("--dry-run")
        _run(fit_cmd, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
    else:
        print("[full] skip fit")

    fit_run = None
    dualfit_run = None
    if not args.dry_run:
        fit_run = resolve_latest(SYNTHETIC_DATA_RUNS)
        maybe_dual = fit_run.parent / f"{fit_run.name}_dualfit"
        if maybe_dual.exists():
            dualfit_run = maybe_dual

    if "audit" not in skips:
        audit_cmd = [sys.executable, str(REPO_ROOT / "pipelines" / "audit" / "run_audit_pipeline.py")]
        if fit_run is not None:
            audit_cmd.extend(["--fit-run-dir", str(fit_run)])
        if dualfit_run is not None:
            audit_cmd.extend(["--dualfit-run-dir", str(dualfit_run)])
        if args.allow_synthetic_population:
            audit_cmd.append("--allow-synthetic-population")
        if args.continue_on_error:
            audit_cmd.append("--continue-on-error")
        if args.dry_run:
            audit_cmd.append("--dry-run")
        _run(audit_cmd, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
    else:
        print("[full] skip audit")

    audit_run = None
    if not args.dry_run:
        try:
            audit_run = resolve_latest(AUDIT_RUNS)
        except FileNotFoundError:
            audit_run = None

    if "train" not in skips:
        train_cmd = [
            sys.executable,
            str(REPO_ROOT / "pipelines" / "train" / "run_train_pipeline.py"),
            "--mode",
            args.train_mode,
        ]
        if fit_run is not None:
            train_cmd.extend(["--fit-run-dir", str(fit_run)])
        if audit_run is not None:
            train_cmd.extend(["--audit-run-dir", str(audit_run)])
        if args.device:
            train_cmd.extend(["--device", args.device])
        if args.seeds:
            train_cmd.append("--seeds")
            train_cmd.extend(str(seed) for seed in args.seeds)
        if args.dry_run:
            train_cmd.append("--dry-run")
        _run(train_cmd, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
    else:
        print("[full] skip train")

    if "report" not in skips:
        report_cmd = [sys.executable, str(REPO_ROOT / "pipelines" / "report" / "run_report_pipeline.py")]
        if fit_run is not None:
            report_cmd.extend(["--fit-run-dir", str(fit_run)])
        if audit_run is not None:
            report_cmd.extend(["--audit-run-dir", str(audit_run)])
        if args.skip_alpha:
            report_cmd.append("--skip-alpha")
        if args.promote:
            report_cmd.append("--promote")
        if args.dry_run:
            report_cmd.append("--dry-run")
        _run(report_cmd, dry_run=args.dry_run, continue_on_error=args.continue_on_error)
    else:
        print("[full] skip report")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
