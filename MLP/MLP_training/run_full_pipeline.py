"""End-to-end MLP training pipeline orchestrator.

Drives Stage-1 MSE -> Stage-2 NLL -> Stage-3 distillation ablation suite, picks
the winner, and (in modes A/B/C) sweeps the configured seed list to produce a
bootstrap-CI table over the winner metrics.

Modes (set via ``--mode``)
--------------------------
single  Default. One seed (config.default_seed). Full Stage 1 -> 2 -> 3 +
        ablations -> winner. Produces the production model.
A       Stage 1+2 trained once and shared; Stage 3 ablation suite re-run per
        seed. Isolates the distillation step's variability.
B       Stage 1 trained once and shared; Stage 2+3 re-run per seed.
C       Full pipeline (Stage 1+2+3) re-trained per seed. Most rigorous.

All seeds are hard-coded in ``full_pipeline_config.json`` so collaborators can
audit the choice. Override the list at the CLI only for ad-hoc debugging.

Usage
-----
    python MLP/MLP_training/run_full_pipeline.py            # single, seed 42
    python MLP/MLP_training/run_full_pipeline.py --mode A
    python MLP/MLP_training/run_full_pipeline.py --mode C --device cuda
    python MLP/MLP_training/run_full_pipeline.py --dry-run
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLP_ROOT = PROJECT_ROOT / "MLP"
TRAINING_ROOT = MLP_ROOT / "MLP_training"
RUNS_ROOT = MLP_ROOT / "runs_mlp"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_CONFIG = TRAINING_ROOT / "full_pipeline_config.json"

STAGE1_SCRIPT = TRAINING_ROOT / "train_stage1_mse.py"
STAGE2_SCRIPT = TRAINING_ROOT / "train_stage2_nll.py"
STAGE3_SUITE_SCRIPT = TRAINING_ROOT / "run_stage3_ablation_suite.py"

SAVED_RUN_DIR_PATTERN = re.compile(r"Saved run_dir:\s*(.+)$")
SUITE_SUMMARY_PATTERN = re.compile(r"Suite summary saved to:\s*(.+)$")

WINNER_METRIC_KEYS = (
    "rmse_mm",
    "mae_mm",
    "bias_mm",
    "p95_abs_err_mm",
    "coverage_1sigma",
    "coverage_2sigma",
    "n_points",
)


# ----------------------------- helpers -----------------------------


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if "seeds" not in config or not isinstance(config["seeds"], list):
        raise ValueError(f"Config {path} missing 'seeds' list.")
    if "default_seed" not in config:
        raise ValueError(f"Config {path} missing 'default_seed'.")
    if config["default_seed"] not in config["seeds"]:
        raise ValueError("default_seed must appear in the seeds list.")
    if "modes" not in config or not isinstance(config["modes"], dict):
        raise ValueError(f"Config {path} missing 'modes' block.")
    return config


def python_exe() -> str:
    return str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))


def resolve_device(requested: str) -> str:
    """Resolve the orchestrator-level ``auto`` device before calling stages.

    Stage-3 accepts ``auto`` directly, but Stage-1/2 call ``torch.device`` on
    the received value. Resolve once here so all child commands see a concrete
    PyTorch device string.
    """
    requested_norm = str(requested).strip().lower()
    if requested_norm == "auto":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Cannot resolve --device auto because torch is not importable.") from exc
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_norm == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Cannot use --device cuda because torch is not importable.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is false. "
                "Use --device cpu, or fix the CUDA/PyTorch installation."
            )
    if requested_norm not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested!r}")
    return requested_norm


def fmt_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_subprocess(
    cmd: list[str],
    *,
    log_path: Path,
    dry_run: bool,
) -> tuple[int, str]:
    """Run cmd, tee stdout/stderr to a log file and to console, return rc + tail.

    Returns (returncode, full_stdout). On dry-run returns (0, "") without
    invoking subprocess.
    """
    print(f"[run] {fmt_cmd(cmd)}", flush=True)
    if dry_run:
        return 0, ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            captured_lines.append(line)
        proc.wait()
        rc = int(proc.returncode)
    return rc, "".join(captured_lines)


def parse_saved_run_dir(stdout: str) -> Path | None:
    """Parse the 'Saved run_dir: <path>' line printed by Stage 1/2 scripts."""
    for line in reversed(stdout.splitlines()):
        m = SAVED_RUN_DIR_PATTERN.search(line.strip())
        if m:
            return Path(m.group(1).strip())
    return None


def parse_suite_summary_path(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        m = SUITE_SUMMARY_PATTERN.search(line.strip())
        if m:
            return Path(m.group(1).strip())
    return None


def write_flag(seed_dir: Path, name: str, payload: dict[str, Any]) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    flag = seed_dir / f"{name}.flag.json"
    flag.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_flag(seed_dir: Path, name: str) -> dict[str, Any] | None:
    flag = seed_dir / f"{name}.flag.json"
    if not flag.exists():
        return None
    try:
        return json.loads(flag.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ----------------------------- stage runners -----------------------------


def run_stage1(*, seed: int, variant: str, device: str, log_path: Path, dry_run: bool) -> Path:
    cmd = [
        python_exe(),
        str(STAGE1_SCRIPT),
        "--variant", variant,
        "--device", device,
        "--seed", str(int(seed)),
    ]
    rc, stdout = run_subprocess(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"Stage 1 failed (rc={rc}). See {log_path}.")
    if dry_run:
        return Path("DRY_RUN/stage1_run")
    run_dir = parse_saved_run_dir(stdout)
    if run_dir is None:
        raise RuntimeError("Could not parse Stage-1 run_dir from stdout.")
    return run_dir


def run_stage2(
    *,
    stage1_run_dir: Path,
    seed: int,
    device: str,
    log_path: Path,
    dry_run: bool,
) -> Path:
    cmd = [
        python_exe(),
        str(STAGE2_SCRIPT),
        str(stage1_run_dir),
        "--device", device,
        "--seed", str(int(seed)),
    ]
    rc, stdout = run_subprocess(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"Stage 2 failed (rc={rc}). See {log_path}.")
    if dry_run:
        return Path("DRY_RUN/stage2_run")
    run_dir = parse_saved_run_dir(stdout)
    if run_dir is None:
        raise RuntimeError("Could not parse Stage-2 run_dir from stdout.")
    return run_dir


def run_stage3_suite(
    *,
    stage2_run_dir: Path,
    seed: int,
    device: str,
    suite_config_path: Path,
    include_sensitivity: bool,
    log_path: Path,
    dry_run: bool,
) -> Path | None:
    cmd = [
        python_exe(),
        str(STAGE3_SUITE_SCRIPT),
        "--config", str(suite_config_path),
        "--teacher-run", str(stage2_run_dir),
        "--device", device,
        "--seed", str(int(seed)),
    ]
    if include_sensitivity:
        cmd.append("--include-sensitivity")
    rc, stdout = run_subprocess(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        raise RuntimeError(f"Stage 3 suite failed (rc={rc}). See {log_path}.")
    if dry_run:
        return Path("DRY_RUN/stage3_suite_summary.json")
    return parse_suite_summary_path(stdout)


# ----------------------------- per-seed driver -----------------------------


def run_one_seed(
    *,
    seed: int,
    seed_dir: Path,
    variant: str,
    device: str,
    suite_config_path: Path,
    include_sensitivity: bool,
    rerun_stage1: bool,
    rerun_stage2: bool,
    shared_stage1_run: Path | None,
    shared_stage2_run: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Train (or reuse) Stage 1, 2 and run Stage 3 ablation suite for one seed."""
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1
    if rerun_stage1 or shared_stage1_run is None:
        cached = read_flag(seed_dir, "stage1")
        if cached and Path(cached["run_dir"]).exists() and not dry_run:
            print(f"[seed {seed}] Stage 1 already done -> {cached['run_dir']}")
            stage1_run = Path(cached["run_dir"])
        else:
            stage1_run = run_stage1(
                seed=seed, variant=variant, device=device,
                log_path=seed_dir / "stage1.log", dry_run=dry_run,
            )
            if not dry_run:
                write_flag(seed_dir, "stage1", {"run_dir": str(stage1_run), "seed": seed})
    else:
        stage1_run = shared_stage1_run
        print(f"[seed {seed}] Reusing shared Stage 1 -> {stage1_run}")

    # Stage 2
    if rerun_stage2 or shared_stage2_run is None:
        cached = read_flag(seed_dir, "stage2")
        if cached and Path(cached["run_dir"]).exists() and not dry_run:
            print(f"[seed {seed}] Stage 2 already done -> {cached['run_dir']}")
            stage2_run = Path(cached["run_dir"])
        else:
            stage2_run = run_stage2(
                stage1_run_dir=stage1_run, seed=seed, device=device,
                log_path=seed_dir / "stage2.log", dry_run=dry_run,
            )
            if not dry_run:
                write_flag(seed_dir, "stage2", {"run_dir": str(stage2_run), "seed": seed})
    else:
        stage2_run = shared_stage2_run
        print(f"[seed {seed}] Reusing shared Stage 2 -> {stage2_run}")

    # Stage 3 ablation suite (always re-run per seed; that's the whole point)
    cached = read_flag(seed_dir, "stage3")
    if cached and Path(cached["summary"]).exists() and not dry_run:
        print(f"[seed {seed}] Stage 3 already done -> {cached['summary']}")
        suite_summary = Path(cached["summary"])
    else:
        suite_summary = run_stage3_suite(
            stage2_run_dir=stage2_run, seed=seed, device=device,
            suite_config_path=suite_config_path,
            include_sensitivity=include_sensitivity,
            log_path=seed_dir / "stage3.log", dry_run=dry_run,
        )
        if not dry_run and suite_summary is not None:
            write_flag(seed_dir, "stage3", {"summary": str(suite_summary), "seed": seed})

    record = {
        "seed": seed,
        "stage1_run": str(stage1_run),
        "stage2_run": str(stage2_run),
        "stage3_suite_summary": str(suite_summary) if suite_summary else None,
    }
    if not dry_run and suite_summary and suite_summary.exists():
        suite = json.loads(suite_summary.read_text(encoding="utf-8"))
        best = suite.get("selection", {}).get("best") or {}
        overall = (best.get("post_eval") or {}).get("overall") or {}
        record["winner_name"] = best.get("name")
        record["winner_run_dir"] = best.get("run_dir")
        record["winner_metrics"] = {k: overall.get(k) for k in WINNER_METRIC_KEYS}
    (seed_dir / "seed_summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


# ----------------------------- bootstrap CI -----------------------------


def bootstrap_ci(values: list[float], *, n_resample: int = 5000, alpha: float = 0.05) -> dict[str, float]:
    """Percentile bootstrap CI for the mean. Returns mean, std, lo, hi."""
    import numpy as np
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and (v != v))], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    if arr.size == 1:
        return {"mean": float(arr[0]), "std": 0.0, "ci_lo": float(arr[0]), "ci_hi": float(arr[0]), "n": 1}
    rng = np.random.default_rng(20260508)  # fixed for reproducibility
    means = rng.choice(arr, size=(n_resample, arr.size), replace=True).mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "ci_lo": lo, "ci_hi": hi, "n": int(arr.size)}


def aggregate_seeds(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    metric_table: dict[str, list[float]] = {k: [] for k in WINNER_METRIC_KEYS}
    for rec in records:
        m = rec.get("winner_metrics") or {}
        for k in WINNER_METRIC_KEYS:
            v = m.get(k)
            if isinstance(v, (int, float)):
                metric_table[k].append(float(v))
    return {
        "n_seeds": len(records),
        "seeds": [rec["seed"] for rec in records],
        "winner_names": [rec.get("winner_name") for rec in records],
        "metrics": {k: bootstrap_ci(vals) for k, vals in metric_table.items()},
    }


# ----------------------------- main -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full MLP pipeline + multi-seed bootstrap CI orchestrator.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--mode", choices=("single", "A", "B", "C"), default="single",
                   help="single (default, one seed); A: Stage3 reseeded; B: Stage2+3 reseeded; C: full reseeded.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Override the config seed list (debug only).")
    p.add_argument("--variant", default=None, help="Override config.variant (a_only or a_plus_log_a).")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    p.add_argument("--include-sensitivity", action="store_true",
                   help="Include sensitivity_ablations from the Stage-3 suite config.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Where to put full_pipeline_<timestamp>/. Defaults to MLP/runs_mlp.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_config(config_path)

    mode_spec = config["modes"][args.mode]
    variant = args.variant or config.get("variant", "a_only")
    requested_device = args.device or config.get("device", "auto")
    device = resolve_device(requested_device)
    suite_cfg_raw = config.get("stage3_ablation_config")
    suite_config_path = Path(suite_cfg_raw)
    if not suite_config_path.is_absolute():
        suite_config_path = PROJECT_ROOT / suite_config_path
    include_sensitivity = bool(args.include_sensitivity or config.get("include_sensitivity", False))

    if mode_spec.get("single_seed_only", False):
        seeds: list[int] = [int(config["default_seed"])]
    else:
        seeds = [int(s) for s in (args.seeds or config["seeds"])]

    output_root = (args.output_root or RUNS_ROOT).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = output_root / f"full_pipeline_{args.mode}_{timestamp}"
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    # Persist the resolved config so the run is self-describing.
    (pipeline_dir / "pipeline_config_resolved.json").write_text(
        json.dumps({
            "mode": args.mode,
            "mode_spec": mode_spec,
            "seeds": seeds,
            "variant": variant,
            "requested_device": requested_device,
            "device": device,
            "suite_config": str(suite_config_path),
            "include_sensitivity": include_sensitivity,
            "config_source": str(config_path),
            "dry_run": bool(args.dry_run),
        }, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Pipeline:        {pipeline_dir}")
    print(f"Mode:            {args.mode}  ({mode_spec.get('description', '')})")
    print(f"Seeds:           {seeds}")
    print(f"Variant:         {variant}")
    print(f"Device:          {device}  (requested: {requested_device})")
    print(f"Suite config:    {suite_config_path}")
    print(f"Sensitivity:     {include_sensitivity}")
    print(f"Dry-run:         {args.dry_run}")
    print()

    rerun_s1 = bool(mode_spec["rerun_stage1_per_seed"])
    rerun_s2 = bool(mode_spec["rerun_stage2_per_seed"])

    if (not rerun_s2) and rerun_s1:
        raise ValueError(
            "Invalid mode spec: cannot share Stage 2 without sharing Stage 1 "
            "(Stage 2 warm-starts from a specific Stage-1 run)."
        )

    # Shared upstream stages: A shares S1 and S2; B shares only S1.
    shared_s1: Path | None = None
    shared_s2: Path | None = None
    if not rerun_s1:
        shared_dir = pipeline_dir / "shared_stage1"
        shared_s1 = run_stage1(
            seed=int(config["default_seed"]), variant=variant, device=device,
            log_path=shared_dir / "stage1.log", dry_run=args.dry_run,
        )
        write_flag(shared_dir, "stage1_shared", {"run_dir": str(shared_s1)})
    if not rerun_s2:
        assert shared_s1 is not None, "shared Stage-1 must be set before shared Stage-2"
        shared_dir = pipeline_dir / "shared_stage2"
        shared_s2 = run_stage2(
            stage1_run_dir=shared_s1,
            seed=int(config["default_seed"]), device=device,
            log_path=shared_dir / "stage2.log", dry_run=args.dry_run,
        )
        write_flag(shared_dir, "stage2_shared", {"run_dir": str(shared_s2)})

    seed_records: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        print()
        print(f"==== seed {seed} ({index}/{len(seeds)}) ====")
        seed_dir = pipeline_dir / f"seed_{seed}"
        record = run_one_seed(
            seed=seed,
            seed_dir=seed_dir,
            variant=variant,
            device=device,
            suite_config_path=suite_config_path,
            include_sensitivity=include_sensitivity,
            rerun_stage1=rerun_s1,
            rerun_stage2=rerun_s2,
            shared_stage1_run=shared_s1,
            shared_stage2_run=shared_s2,
            dry_run=args.dry_run,
        )
        seed_records.append(record)

    summary = {
        "mode": args.mode,
        "seeds": seeds,
        "shared_stage1_run": str(shared_s1) if shared_s1 else None,
        "shared_stage2_run": str(shared_s2) if shared_s2 else None,
        "per_seed": seed_records,
        "bootstrap": aggregate_seeds(seed_records) if not args.dry_run else None,
    }
    summary_path = pipeline_dir / "bootstrap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Summary written: {summary_path}")
    if not args.dry_run and summary["bootstrap"]:
        print()
        print("Winner-metric bootstrap (mean ± std, 95% CI):")
        for metric, stats in summary["bootstrap"]["metrics"].items():
            if stats["n"] == 0:
                print(f"  {metric:>16s}: no data")
                continue
            print(
                f"  {metric:>16s}: {stats['mean']:.4f} ± {stats['std']:.4f}  "
                f"[{stats['ci_lo']:.4f}, {stats['ci_hi']:.4f}]  (n={stats['n']})"
            )


if __name__ == "__main__":
    main()
