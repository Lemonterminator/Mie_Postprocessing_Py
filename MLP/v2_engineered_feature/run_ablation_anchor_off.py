"""Run ablation 3: disable or reduce the early-time anchor loss.

Hypothesis: the 0-0.1 ms student underprediction is driven by the anchor/onset
regularization rather than by the teacher alone. By default this sets
lambda_anchor=0 while keeping the onset loss unchanged.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = Path("MLP") / "v2_engineered_feature" / "train_stage3_distillation_plus_raw_series.py"
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
        description="Ablation 3: disable or reduce the early-time anchor term."
    )
    parser.add_argument("--teacher-run", type=Path, default=DEFAULT_TEACHER_RUN)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--series-split", choices=("clean", "all"), default="clean")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lambda-anchor", type=float, default=0.0)
    parser.add_argument("--anchor-window-ms", type=float, default=None)
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
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning-rate", str(args.learning_rate),
        "--patience", str(args.patience),
        "--run-name-prefix", "distill_cdf_onset_v2_ablate_anchor_off",
        "--ablation-name", "anchor_off",
        "--lambda-anchor", str(args.lambda_anchor),
    ]
    if args.anchor_window_ms is not None:
        cmd.extend(["--anchor-window-ms", str(args.anchor_window_ms)])
    if args.no_train:
        cmd.append("--no-train")
    cmd.extend(extra)

    print("Running:", flush=True)
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
