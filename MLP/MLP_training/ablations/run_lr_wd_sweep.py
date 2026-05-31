"""Tier-1C learning-rate x weight-decay sweep driver.

Sweeps Stage-1 (learning_rate, weight_decay) on a 3x3 grid (9 configs) at
seed=42, in-domain. Stage 2 keeps its defaults to avoid doubling the grid.
Top-3 in-domain winners get a 5-fold LONO follow-up.

Spec lives in MLP/MLP_training/ablations/TIER1_CAPACITY_SEED_OPTIMIZER_PLAN.md
section 5.

Wall clock: ~8.5 min per in-domain run on RTX 5090. 9 configs ~ 80 min;
top-3 LONO follow-up (3 winners x 5 folds) ~ 130 min.

Usage
-----
    python MLP/MLP_training/ablations/run_lr_wd_sweep.py
    python MLP/MLP_training/ablations/run_lr_wd_sweep.py --dry-run
    python MLP/MLP_training/ablations/run_lr_wd_sweep.py --lono-followup

Outputs (under MLP/runs_mlp/lr_wd_sweep_<timestamp>/):
    lr_wd_summary.csv               one row per (lr, wd) (in-domain)
    lr_wd_lono_followup.csv         one row per (winner, fold) (only if --lono-followup)
    verdict.md                       winner, baseline comparison, recommendation

Failure handling
----------------
lr=1e-2 may diverge with grad_clip_norm=1.0; if Stage 1 raises NaN early the
record is marked failed and dropped from the verdict aggregation. Do NOT
relax grad_clip_norm here - that is a separate ablation.
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

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MLP_ROOT = PROJECT_ROOT / "MLP"
TRAINING_ROOT = MLP_ROOT / "MLP_training"
CONFIG_ROOT = TRAINING_ROOT / "config"
RUNS_ROOT = MLP_ROOT / "runs_mlp"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

STAGE1_SCRIPT = TRAINING_ROOT / "train_stage1_mse.py"
STAGE2_SCRIPT = TRAINING_ROOT / "train_stage2_nll.py"
STAGE3_SUITE_SCRIPT = TRAINING_ROOT / "run_stage3_ablation_suite.py"
STAGE3_SUITE_CONFIG = CONFIG_ROOT / "stage3_ablation_suite_config.json"
EVAL_SCRIPT = MLP_ROOT / "eval" / "inference_rmse_on_series.py"
LONO_PIPELINE = TRAINING_ROOT / "ood_lono" / "run_lono_pipeline.py"

SAVED_RUN_DIR_RE = re.compile(r"Saved run_dir:\s*(.+)$")
SUITE_SUMMARY_RE = re.compile(r"Suite summary saved to:\s*(.+)$")
EVAL_OUTPUT_RE = re.compile(r"Wrote RMSE evaluation to\s+(.+)$", re.MULTILINE)

STAGE3_ONLY = "anchor_off"
DEFAULT_SEED = 42

LR_WD_GRID: list[tuple[float, float]] = [
    (1e-3, 1e-5), (1e-3, 1e-4), (1e-3, 1e-3),
    (4e-3, 1e-5), (4e-3, 1e-4), (4e-3, 1e-3),
    (1e-2, 1e-5), (1e-2, 1e-4), (1e-2, 1e-3),
]
# Production defaults that we expect to appear as a near-optimum.
BASELINE_LR, BASELINE_WD = 4e-3, 2e-4


def python_exe() -> str:
    return str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))


def run_subprocess_streaming(cmd: list[str], log_path: Path) -> tuple[int, str]:
    print(f"\n[run] {' '.join(shlex.quote(p) for p in cmd)}", flush=True)
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
    return int(proc.returncode), "".join(captured)


def parse_saved_run_dir(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        m = SAVED_RUN_DIR_RE.search(line.strip())
        if m:
            return Path(m.group(1).strip())
    return None


def parse_suite_summary_path(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        m = SUITE_SUMMARY_RE.search(line.strip())
        if m:
            return Path(m.group(1).strip())
    return None


def find_named_run_dir(suite_summary_path: Path, name: str) -> Path:
    summary = json.loads(suite_summary_path.read_text(encoding="utf-8"))
    for entry in summary.get("results", []):
        if str(entry.get("name")) == name:
            run_dir = entry.get("run_dir")
            if run_dir:
                return Path(run_dir)
    best = (summary.get("selection") or {}).get("best") or {}
    if str(best.get("name")) == name and best.get("run_dir"):
        return Path(best["run_dir"])
    raise RuntimeError(f"Could not locate {name!r} run_dir in {suite_summary_path}")


def read_eval_metrics(eval_dir: Path) -> dict[str, float]:
    summary_json = eval_dir / "summary.json"
    if summary_json.exists():
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        overall = payload.get("overall") or payload
        return {
            "rmse_mm": overall.get("rmse_mm"),
            "mae_mm": overall.get("mae_mm"),
            "bias_mm": overall.get("bias_mm"),
            "coverage_1sigma": overall.get("coverage_1sigma"),
            "coverage_2sigma": overall.get("coverage_2sigma"),
        }
    pts = eval_dir / "points.csv"
    if not pts.exists():
        return {}
    df = pd.read_csv(pts, low_memory=False)
    resid = df["pen_pred_mm"].to_numpy(float) - df["pen_true_mm"].to_numpy(float)
    abs_err = np.abs(resid)
    std_safe = np.maximum(df["pen_std_mm"].to_numpy(float), 1e-12)
    return {
        "rmse_mm": float(np.sqrt(np.mean(resid ** 2))),
        "mae_mm": float(np.mean(abs_err)),
        "bias_mm": float(np.mean(resid)),
        "coverage_1sigma": float(np.mean(abs_err <= std_safe)),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * std_safe)),
    }


def fmt_lr_wd(lr: float, wd: float) -> str:
    return f"lr{lr:.0e}_wd{wd:.0e}".replace("e-0", "e-").replace("+", "")


def run_one_config(
    *,
    lr: float,
    wd: float,
    config_dir: Path,
    seed: int,
    device: str,
    dry_run: bool,
) -> dict[str, Any]:
    config_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    # Stage 1 with sweep (lr, wd)
    s1_log = config_dir / "stage1.log"
    s1_cmd = [
        python_exe(), str(STAGE1_SCRIPT),
        "--variant", "a_only",
        "--device", device,
        "--seed", str(seed),
        "--learning-rate", str(lr),
        "--weight-decay", str(wd),
        "--allow-failed-precheck",
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s1_cmd)}")
        stage1_run = Path(f"DRY_RUN/stage1_{fmt_lr_wd(lr, wd)}")
    else:
        rc, out = run_subprocess_streaming(s1_cmd, s1_log)
        if rc != 0:
            raise RuntimeError(f"Stage-1 failed for (lr={lr},wd={wd}) (rc={rc}). See {s1_log}.")
        if "nan" in out.lower() and "loss" in out.lower():
            print(f"[warn] possible NaN loss in Stage 1 for lr={lr} wd={wd}; check log.")
        stage1_run = parse_saved_run_dir(out)
        if stage1_run is None:
            raise RuntimeError("Could not parse Stage-1 run_dir.")
    timing["stage1_s"] = time.time() - t0

    # Stage 2 with default lr/wd
    s2_log = config_dir / "stage2.log"
    s2_cmd = [
        python_exe(), str(STAGE2_SCRIPT),
        str(stage1_run),
        "--device", device,
        "--seed", str(seed),
        "--allow-failed-precheck",
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s2_cmd)}")
        stage2_run = Path(f"DRY_RUN/stage2_{fmt_lr_wd(lr, wd)}")
    else:
        rc, out = run_subprocess_streaming(s2_cmd, s2_log)
        if rc != 0:
            raise RuntimeError(f"Stage-2 failed for (lr={lr},wd={wd}) (rc={rc}). See {s2_log}.")
        stage2_run = parse_saved_run_dir(out)
        if stage2_run is None:
            raise RuntimeError("Could not parse Stage-2 run_dir.")
    timing["stage2_s"] = time.time() - t0

    # Stage 3 (anchor_off only)
    s3_log = config_dir / "stage3.log"
    s3_cmd = [
        python_exe(), str(STAGE3_SUITE_SCRIPT),
        "--config", str(STAGE3_SUITE_CONFIG),
        "--teacher-run", str(stage2_run),
        "--device", device,
        "--seed", str(seed),
        "--only", STAGE3_ONLY,
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s3_cmd)}")
        anchor_run = Path(f"DRY_RUN/{STAGE3_ONLY}_{fmt_lr_wd(lr, wd)}")
        suite_summary = Path(f"DRY_RUN/suite_{fmt_lr_wd(lr, wd)}.json")
    else:
        rc, out = run_subprocess_streaming(s3_cmd, s3_log)
        if rc != 0:
            raise RuntimeError(f"Stage-3 suite failed for (lr={lr},wd={wd}) (rc={rc}). See {s3_log}.")
        suite_summary = parse_suite_summary_path(out)
        if suite_summary is None:
            raise RuntimeError("Could not parse suite_summary path.")
        anchor_run = find_named_run_dir(suite_summary, STAGE3_ONLY)
    timing["stage3_s"] = time.time() - t0

    # In-domain eval
    eval_log = config_dir / "eval.log"
    tag = f"lrwd_{fmt_lr_wd(lr, wd)}"
    eval_cmd = [
        python_exe(), str(EVAL_SCRIPT),
        "--refinement-run", str(anchor_run),
        "--split", "clean",
        "--tag", tag,
        "--batch-points", "262144",
        "--no-save-plots",
        "--max-traj-plots", "0",
    ]
    t0 = time.time()
    metrics: dict[str, float] = {}
    eval_dir = Path("")
    if dry_run:
        print(f"[dry-run] {' '.join(eval_cmd)}")
        eval_dir = Path(f"DRY_RUN/eval_{tag}")
    else:
        rc, out = run_subprocess_streaming(eval_cmd, eval_log)
        if rc != 0:
            raise RuntimeError(f"External eval failed for (lr={lr},wd={wd}) (rc={rc}). See {eval_log}.")
        match = EVAL_OUTPUT_RE.search(out)
        if match:
            eval_dir = Path(match.group(1).strip())
        else:
            candidates = sorted(
                (MLP_ROOT / "eval").glob(f"rmse_eval_clean_*_{tag}"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if not candidates:
                raise RuntimeError(f"Could not locate eval dir for {tag}.")
            eval_dir = candidates[0]
        metrics = read_eval_metrics(eval_dir)
    timing["eval_s"] = time.time() - t0

    record = {
        "lr": lr, "wd": wd,
        "config_label": fmt_lr_wd(lr, wd),
        "seed": seed,
        "stage1_run": str(stage1_run),
        "stage2_run": str(stage2_run),
        "stage3_suite_summary": str(suite_summary),
        "winner_run": str(anchor_run),
        "eval_dir": str(eval_dir),
        "metrics": metrics,
        "timing": timing,
    }
    (config_dir / "config_summary.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def write_in_domain_outputs(records: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    rows = []
    for rec in records:
        m = rec.get("metrics") or {}
        rows.append({
            "config_label": rec["config_label"],
            "lr": rec["lr"], "wd": rec["wd"],
            "test_rmse_mm": m.get("rmse_mm"),
            "test_mae_mm": m.get("mae_mm"),
            "test_bias_mm": m.get("bias_mm"),
            "coverage_1sigma": m.get("coverage_1sigma"),
            "coverage_2sigma": m.get("coverage_2sigma"),
            "train_minutes": sum(rec.get("timing", {}).values()) / 60.0,
        })
    df = pd.DataFrame(rows).sort_values("test_rmse_mm", na_position="last")
    df.to_csv(output_dir / "lr_wd_summary.csv", index=False)
    return df


def run_lono_followup(
    *,
    winner_records: list[dict[str, Any]],
    output_dir: Path,
    device: str,
    seed: int,
    dry_run: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for winner in winner_records:
        lr, wd = float(winner["lr"]), float(winner["wd"])
        label = fmt_lr_wd(lr, wd)
        lono_dir = output_dir / f"lono_{label}"
        lono_dir.mkdir(parents=True, exist_ok=True)
        lono_log = lono_dir / "lono.log"
        cmd = [
            python_exe(), str(LONO_PIPELINE),
            "--seed", str(seed),
            "--device", device,
            "--n-folds", "5",
            "--output-dir", str(lono_dir),
            "--stage3-only", STAGE3_ONLY,
            "--learning-rate", str(lr),
            "--weight-decay", str(wd),
        ]
        if dry_run:
            print(f"[dry-run] {' '.join(cmd)}  (intended lr={lr}, wd={wd})")
            continue
        rc, _ = run_subprocess_streaming(cmd, lono_log)
        if rc != 0:
            print(f"!! LONO follow-up failed for {label} (rc={rc}). Skipping.")
            continue
        per_fold_csv = lono_dir / "per_fold.csv"
        if not per_fold_csv.exists():
            print(f"!! Missing {per_fold_csv}; skipping.")
            continue
        per_fold = pd.read_csv(per_fold_csv)
        for _, r in per_fold.loc[per_fold["model"].isin(["MLP_series", "MLP_uncensored"])].iterrows():
            rows.append({
                "config_label": label,
                "lr": lr, "wd": wd,
                "model_eval": r["model"],
                "fold_nozzle": r["holdout"],
                "test_rmse_mm": r.get("rmse_mm"),
                "test_mae_mm": r.get("mae_mm"),
                "coverage_1sigma": r.get("coverage_1sigma"),
                "coverage_2sigma": r.get("coverage_2sigma"),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "lr_wd_lono_followup.csv", index=False)
    return df


def write_verdict(in_domain: pd.DataFrame, lono: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Tier-1C lr x wd sweep verdict", ""]
    if in_domain.empty:
        lines.append("No in-domain results recorded.")
    else:
        best = in_domain.iloc[0]
        lines.append(f"- In-domain best: **{best['config_label']}** "
                     f"(lr={best['lr']:.0e}, wd={best['wd']:.0e}, RMSE={best['test_rmse_mm']:.4f} mm).")
        lines.append(f"- Production baseline: lr={BASELINE_LR:.0e}, wd={BASELINE_WD:.0e}.")
        lines.append("")
        lines.append("Top in-domain configs:")
        lines.append("")
        lines.append("| rank | config | lr | wd | RMSE (mm) |")
        lines.append("|---|---|---|---|---|")
        for i, (_, r) in enumerate(in_domain.head(5).iterrows(), start=1):
            lines.append(
                f"| {i} | {r['config_label']} | {r['lr']:.0e} | {r['wd']:.0e} | {r['test_rmse_mm']:.4f} |"
            )
    lines.append("")
    if not lono.empty:
        lines.append("## LONO follow-up (5-fold mean per config)")
        agg = (
            lono.loc[lono["model_eval"] == "MLP_series"]
                .groupby("config_label")["test_rmse_mm"].agg(["mean", "std", "count"])
                .reset_index().sort_values("mean")
        )
        lines.append("")
        lines.append("| config | LONO mean RMSE | std | n_folds |")
        lines.append("|---|---|---|---|")
        for _, r in agg.iterrows():
            lines.append(f"| {r['config_label']} | {r['mean']:.4f} | {r['std']:.4f} | {int(r['count'])} |")
    else:
        lines.append("(No LONO follow-up recorded. Re-run with --lono-followup.)")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("If the in-domain top configuration is within sigma_seed (Tier-1B) of the")
    lines.append("production defaults, report the sweep as 'defaults near-optimal'.")
    lines.append("Otherwise promote the winner if its LONO mean RMSE also improves by")
    lines.append("at least 0.5 mm.")
    (output_dir / "verdict.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== Tier-1C verdict ===")
    print((output_dir / "verdict.md").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--lono-followup", action="store_true",
                   help="After in-domain sweep, run 5-fold LONO on the top-K winners.")
    p.add_argument("--top-k-lono", type=int, default=3,
                   help="Number of in-domain winners to push to LONO (default: 3).")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else RUNS_ROOT / f"lr_wd_sweep_{timestamp}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tier-1C lr/wd sweep output dir: {output_dir}")

    (output_dir / "config.json").write_text(json.dumps({
        "seed": args.seed,
        "device": args.device,
        "grid": [{"lr": lr, "wd": wd} for (lr, wd) in LR_WD_GRID],
        "baseline_lr": BASELINE_LR, "baseline_wd": BASELINE_WD,
        "stage3_only": STAGE3_ONLY,
        "stage3_suite_config": str(STAGE3_SUITE_CONFIG),
    }, indent=2), encoding="utf-8")

    records: list[dict[str, Any]] = []
    for i, (lr, wd) in enumerate(LR_WD_GRID, start=1):
        label = fmt_lr_wd(lr, wd)
        print(f"\n========== Config {i}/{len(LR_WD_GRID)}: {label} ==========")
        config_dir = output_dir / label
        try:
            rec = run_one_config(
                lr=lr, wd=wd, config_dir=config_dir,
                seed=args.seed, device=args.device, dry_run=args.dry_run,
            )
            records.append(rec)
        except Exception as e:
            print(f"!! Config {label} FAILED: {e}")
            (config_dir / "config_FAILED.json").write_text(
                json.dumps({"error": str(e), "lr": lr, "wd": wd}, indent=2),
                encoding="utf-8",
            )

    (output_dir / "all_configs.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    in_domain_df = pd.DataFrame()
    lono_df = pd.DataFrame()
    if records and not args.dry_run:
        in_domain_df = write_in_domain_outputs(records, output_dir)
        if args.lono_followup and not in_domain_df.empty:
            winners = in_domain_df.head(args.top_k_lono).to_dict(orient="records")
            lono_df = run_lono_followup(
                winner_records=winners, output_dir=output_dir,
                device=args.device, seed=args.seed, dry_run=args.dry_run,
            )
        write_verdict(in_domain_df, lono_df, output_dir)

    print(f"\nTier-1C done. Output: {output_dir}")


if __name__ == "__main__":
    main()
