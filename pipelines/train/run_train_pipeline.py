"""Phase 3: thin wrapper around the existing MLP full training pipeline."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.common.archive_layout import AUDIT_RUNS, SYNTHETIC_DATA_RUNS, resolve_latest

TRAIN_SCRIPT = REPO_ROOT / "MLP" / "MLP_training" / "run_full_pipeline.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-run-id", default=None)
    parser.add_argument("--fit-run-dir", type=Path, default=None)
    parser.add_argument("--audit-run-id", default=None)
    parser.add_argument("--audit-run-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("single", "A", "B", "C"), default="single")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--include-sensitivity", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def _resolve(root: Path, run_id: str | None, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    if run_id:
        return (root / run_id).resolve()
    try:
        return resolve_latest(root).resolve()
    except FileNotFoundError:
        return None


def run(args: argparse.Namespace) -> None:
    fit_run = _resolve(SYNTHETIC_DATA_RUNS, args.fit_run_id, args.fit_run_dir)
    audit_run = _resolve(AUDIT_RUNS, args.audit_run_id, args.audit_run_dir)
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--mode", args.mode]
    if fit_run is not None:
        cmd.extend(["--fit-run-dir", str(fit_run), "--data-dir", str(fit_run)])
    if audit_run is not None:
        cmd.extend(["--audit-run-dir", str(audit_run)])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.seeds:
        cmd.append("--seeds")
        cmd.extend(str(seed) for seed in args.seeds)
    if args.include_sensitivity:
        cmd.append("--include-sensitivity")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.output_root is not None:
        cmd.extend(["--output-root", str(args.output_root)])
    print("[train]", " ".join(cmd), flush=True)
    if args.dry_run:
        return
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Training pipeline failed with code {completed.returncode}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
