"""Run ablation 1: remove KD from raw-reliable bins.

Hypothesis: deterministic/raw-supported Nozzle0 bins should not be pulled toward
the teacher if raw CDF supervision is already available.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = Path("MLP") / "training" / "train_stage3_distillation_plus_raw_series.py"
DEFAULT_TEACHER_RUN = Path("MLP") / "runs_mlp" / "stage2_engineered_nll_a_only_20260410_020607"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def path_for_child(path: Path) -> str:
    text = str(path)
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        drive = text[5].upper()
        rest = text[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return text


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Ablation 1: set raw_reliable KD weight to 0 while keeping raw supervision."
    )
    parser.add_argument("--teacher-run", type=Path, default=DEFAULT_TEACHER_RUN)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--series-split", choices=("clean", "all"), default="clean")
    parser.add_argument("--sources", nargs="+", default=["cdf"], help="Raw source(s) passed to stage3.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def main() -> None:
    args, extra = parse_args()
    python_exe = DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)
    cmd = [
        str(python_exe),
        str(TRAIN_SCRIPT),
        path_for_child(args.teacher_run),
        "--device", args.device,
        "--series-split", args.series_split,
        "--sources", *args.sources,
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--patience", str(args.patience),
        "--run-name-prefix", "distill_cdf_onset_v2_ablate_raw_reliable_no_kd",
        "--ablation-name", "raw_reliable_no_kd",
        "--raw-reliable-kd-weight", "0.0",
    ]
    if args.no_train:
        cmd.append("--no-train")
    cmd.extend(extra)

    print("Running:", flush=True)
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
