"""Phase 4: assemble generated thesis artifacts and optionally promote figures."""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_GENERATED = REPO_ROOT / "Thesis" / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--fit-run-dir", type=Path, default=None)
    parser.add_argument("--audit-run-dir", type=Path, default=None)
    parser.add_argument("--train-root", type=Path, default=REPO_ROOT / "MLP" / "runs_mlp")
    parser.add_argument("--frames-npz", type=Path, default=None)
    parser.add_argument("--skip-alpha", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("[report]", " ".join(cmd), flush=True)
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Report command failed with code {completed.returncode}")


def _append_provenance(generated_dir: Path, rows: list[dict[str, str]]) -> None:
    path = generated_dir / "_provenance.csv"
    fields = ["phase", "source_run_id", "role", "source_file", "generated_file", "description"]
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)
    shutil.copy2(path, generated_dir / "_manifest.csv")


def _copy_alpha_outputs(alpha_dir: Path, generated_dir: Path) -> None:
    mapping = {
        "alpha_sensitivity_csv": alpha_dir / "alpha_sensitivity_screening_rate.csv",
        "alpha_sensitivity_figure": alpha_dir / "alpha_sensitivity_curves.png",
        "alpha_sensitivity_tex": alpha_dir / "alpha_sensitivity_summary.tex",
    }
    rows = []
    for role, src in mapping.items():
        if not src.exists():
            continue
        dest = generated_dir / f"{role}{src.suffix}"
        shutil.copy2(src, dest)
        current_dir = generated_dir.parent / "current"
        if current_dir.exists():
            shutil.copy2(dest, current_dir / dest.name)
        rows.append(
            {
                "phase": "report",
                "source_run_id": generated_dir.name,
                "role": role,
                "source_file": str(src),
                "generated_file": dest.name,
                "description": "E.2 alpha sensitivity artifact",
            }
        )
    if rows:
        _append_provenance(generated_dir, rows)


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or f"thesis_{datetime.now(tz=timezone.utc):%Y%m%d_%H%M%S}"
    generated_dir = THESIS_GENERATED / run_id
    assemble_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "assemble_thesis_artifacts.py"),
        "--run-id",
        run_id,
    ]
    if args.fit_run_dir is not None:
        assemble_cmd.extend(["--fit-run-dir", str(args.fit_run_dir)])
    if args.audit_run_dir is not None:
        assemble_cmd.extend(["--audit-run-dir", str(args.audit_run_dir)])
    if args.train_root is not None:
        assemble_cmd.extend(["--train-root", str(args.train_root)])
    _run(assemble_cmd, dry_run=args.dry_run)

    if not args.skip_alpha:
        alpha_dir = generated_dir / "e2_alpha_sensitivity"
        alpha_cmd = [sys.executable, str(SCRIPT_DIR / "alpha_sensitivity_sweep.py"), "--out-dir", str(alpha_dir)]
        if args.frames_npz is not None:
            alpha_cmd.extend(["--frames-npz", str(args.frames_npz)])
        _run(alpha_cmd, dry_run=args.dry_run)
        if not args.dry_run:
            _copy_alpha_outputs(alpha_dir, generated_dir)

    if args.promote:
        promote_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "promote_to_thesis.py"),
            "--generated-run-dir",
            str(generated_dir),
        ]
        _run(promote_cmd, dry_run=args.dry_run)
    print(f"[report] Done: {generated_dir}")
    return generated_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
