"""Tier-2C onset-window auxiliary regression head driver.

Three configurations comparing whether an auxiliary onset regression head can
replace (or complement) the Stage-2 mu-anchor:

    baseline_no_aux    aux head OFF, Stage-2 mu_anchor ON (current production)
    aux_with_anchor    aux head ON,  Stage-2 mu_anchor ON (complementary)
    aux_no_anchor      aux head ON,  Stage-2 no_anchor   (aux REPLACES anchor)

Spec lives in MLP/MLP_training/ablations/TIER2_REGIME_SIGMA_ONSET_PLAN.md
section 5.

PREREQUISITE: The model class PenetrationMLP and the Stage-1 / Stage-2 loss
must accept the auxiliary onset head; the trainers must accept new CLI flags
`--onset-aux-head` (bool) and `--lambda-aux FLOAT`. Plan section 5.2 shows
how to wire it. This driver only orchestrates.

Wall clock: ~10 min/config (Stage 1+2+3) on RTX 5090; 3 configs ~ 30 min
in-domain. Top-1 LONO follow-up (Nozzle 2 fast check then full 5-fold) up
to ~50 min more.

Usage
-----
    python MLP/MLP_training/ablations/run_onset_aux_head.py
    python MLP/MLP_training/ablations/run_onset_aux_head.py --lono-followup
    python MLP/MLP_training/ablations/run_onset_aux_head.py --dry-run

Outputs (under MLP/runs_mlp/onset_aux_head_<timestamp>/):
    onset_aux_summary.csv           one row per config (in-domain + onset slice)
    onset_aux_lono_followup.csv     one row per (winner, fold) (only if --lono-followup)
    verdict.md                       does aux help? does it replace the anchor?
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
ONSET_T_MS = 0.3   # onset slice cutoff used for the per-slice CSV columns

# (name, stage1_args, stage2_args)
ONSET_CONFIGS: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    (
        "baseline_no_aux",
        # No aux head; Stage-2 mu_anchor stays ON (the production default).
        {},
        {"stage2_ablation": "mu_anchor"},
    ),
    (
        "aux_with_anchor",
        {"onset_aux_head": True, "lambda_aux": 0.1},
        {"onset_aux_head": True, "lambda_aux": 0.1, "stage2_ablation": "mu_anchor"},
    ),
    (
        "aux_no_anchor",
        {"onset_aux_head": True, "lambda_aux": 0.1},
        {"onset_aux_head": True, "lambda_aux": 0.1, "stage2_ablation": "no_anchor"},
    ),
]


def python_exe() -> str:
    return str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))


def cli_flag(key: str) -> str:
    return f"--{key.replace('_', '-')}"


def append_cli(cmd: list[str], args: dict[str, Any]) -> None:
    for k, v in args.items():
        if v is None or v is False:
            continue
        flag = cli_flag(k)
        if v is True:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])


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


def metrics_from_points(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    resid = df["pen_pred_mm"].to_numpy(float) - df["pen_true_mm"].to_numpy(float)
    abs_err = np.abs(resid)
    std_safe = np.maximum(df["pen_std_mm"].to_numpy(float), 1e-12)
    return {
        "n_points": int(len(df)),
        "rmse_mm": float(np.sqrt(np.mean(resid ** 2))),
        "mae_mm": float(np.mean(abs_err)),
        "bias_mm": float(np.mean(resid)),
        "coverage_1sigma": float(np.mean(abs_err <= std_safe)),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * std_safe)),
    }


def slice_metrics(points_csv: Path) -> dict[str, dict[str, float]]:
    """Split points by time_ms < 0.3 vs >= 0.3 (onset cutoff)."""
    if not points_csv.exists():
        return {}
    df = pd.read_csv(points_csv, low_memory=False)
    # Find the time column (eval scripts may name it 'time_ms', 't_ms', etc.).
    time_col = next((c for c in ("time_ms", "t_ms", "time") if c in df.columns), None)
    if time_col is None:
        # No time column: just return overall metrics.
        return {"overall": metrics_from_points(df)}
    onset = df.loc[df[time_col] < ONSET_T_MS]
    late = df.loc[df[time_col] >= ONSET_T_MS]
    return {
        "overall": metrics_from_points(df),
        f"onset_t_lt_{ONSET_T_MS}ms": metrics_from_points(onset),
        f"t_ge_{ONSET_T_MS}ms": metrics_from_points(late),
    }


def run_one_config(
    *,
    name: str,
    stage1_args: dict[str, Any],
    stage2_args: dict[str, Any],
    config_dir: Path,
    seed: int,
    device: str,
    dry_run: bool,
) -> dict[str, Any]:
    config_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    # Stage 1
    s1_log = config_dir / "stage1.log"
    s1_cmd = [
        python_exe(), str(STAGE1_SCRIPT),
        "--variant", "a_only",
        "--device", device,
        "--seed", str(seed),
        "--allow-failed-precheck",
    ]
    append_cli(s1_cmd, stage1_args)
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s1_cmd)}")
        stage1_run = Path(f"DRY_RUN/stage1_{name}")
    else:
        rc, out = run_subprocess_streaming(s1_cmd, s1_log)
        if rc != 0:
            raise RuntimeError(f"Stage-1 failed for {name} (rc={rc}). See {s1_log}.")
        stage1_run = parse_saved_run_dir(out)
        if stage1_run is None:
            raise RuntimeError("Could not parse Stage-1 run_dir.")
    timing["stage1_s"] = time.time() - t0

    # Stage 2
    s2_log = config_dir / "stage2.log"
    s2_cmd = [
        python_exe(), str(STAGE2_SCRIPT),
        str(stage1_run),
        "--device", device,
        "--seed", str(seed),
        "--allow-failed-precheck",
    ]
    append_cli(s2_cmd, stage2_args)
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s2_cmd)}")
        stage2_run = Path(f"DRY_RUN/stage2_{name}")
    else:
        rc, out = run_subprocess_streaming(s2_cmd, s2_log)
        if rc != 0:
            raise RuntimeError(f"Stage-2 failed for {name} (rc={rc}). See {s2_log}.")
        stage2_run = parse_saved_run_dir(out)
        if stage2_run is None:
            raise RuntimeError("Could not parse Stage-2 run_dir.")
    timing["stage2_s"] = time.time() - t0

    # Stage 3 (anchor_off only - KD operates on (mu, log_var), aux head ignored downstream)
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
        anchor_run = Path(f"DRY_RUN/{STAGE3_ONLY}_{name}")
        suite_summary = Path(f"DRY_RUN/suite_{name}.json")
    else:
        rc, out = run_subprocess_streaming(s3_cmd, s3_log)
        if rc != 0:
            raise RuntimeError(f"Stage-3 suite failed for {name} (rc={rc}). See {s3_log}.")
        suite_summary = parse_suite_summary_path(out)
        if suite_summary is None:
            raise RuntimeError("Could not parse suite_summary path.")
        anchor_run = find_named_run_dir(suite_summary, STAGE3_ONLY)
    timing["stage3_s"] = time.time() - t0

    # External eval: in-domain, full clean split. We split onset vs late from
    # the resulting points.csv (eval script already saves time_ms per point).
    eval_log = config_dir / "eval.log"
    eval_cmd = [
        python_exe(), str(EVAL_SCRIPT),
        "--refinement-run", str(anchor_run),
        "--split", "clean",
        "--tag", f"onset_{name}",
        "--batch-points", "262144",
        "--no-save-plots",
        "--max-traj-plots", "0",
    ]
    t0 = time.time()
    slices: dict[str, dict[str, float]] = {}
    eval_dir = Path("")
    if dry_run:
        print(f"[dry-run] {' '.join(eval_cmd)}")
        eval_dir = Path(f"DRY_RUN/eval_{name}")
    else:
        rc, out = run_subprocess_streaming(eval_cmd, eval_log)
        if rc != 0:
            raise RuntimeError(f"External eval failed for {name} (rc={rc}). See {eval_log}.")
        match = EVAL_OUTPUT_RE.search(out)
        if match:
            eval_dir = Path(match.group(1).strip())
        else:
            candidates = sorted(
                (MLP_ROOT / "eval").glob(f"rmse_eval_clean_*_onset_{name}"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if not candidates:
                raise RuntimeError(f"Could not locate eval dir for {name}.")
            eval_dir = candidates[0]
        slices = slice_metrics(eval_dir / "points.csv")
    timing["eval_s"] = time.time() - t0

    record = {
        "name": name,
        "stage1_args": stage1_args,
        "stage2_args": stage2_args,
        "seed": seed,
        "stage1_run": str(stage1_run),
        "stage2_run": str(stage2_run),
        "stage3_suite_summary": str(suite_summary),
        "winner_run": str(anchor_run),
        "eval_dir": str(eval_dir),
        "slices": slices,
        "timing": timing,
    }
    (config_dir / "config_summary.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def write_in_domain_outputs(records: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    rows = []
    for rec in records:
        slices = rec.get("slices", {}) or {}
        overall = slices.get("overall", {}) or {}
        onset = slices.get(f"onset_t_lt_{ONSET_T_MS}ms", {}) or {}
        late = slices.get(f"t_ge_{ONSET_T_MS}ms", {}) or {}
        rows.append({
            "name": rec["name"],
            "rmse_overall_mm": overall.get("rmse_mm"),
            f"rmse_onset_t_lt_{ONSET_T_MS}ms_mm": onset.get("rmse_mm"),
            f"rmse_t_ge_{ONSET_T_MS}ms_mm": late.get("rmse_mm"),
            "bias_overall_mm": overall.get("bias_mm"),
            "bias_onset_mm": onset.get("bias_mm"),
            "coverage_1sigma": overall.get("coverage_1sigma"),
            "n_points_onset": onset.get("n_points"),
            "train_minutes": sum(rec.get("timing", {}).values()) / 60.0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "onset_aux_summary.csv", index=False)
    return df


def run_lono_followup(
    *,
    winner_name: str,
    output_dir: Path,
    device: str,
    seed: int,
    dry_run: bool,
    full_5fold: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lono_dir = output_dir / f"lono_{winner_name}"
    lono_dir.mkdir(parents=True, exist_ok=True)
    lono_log = lono_dir / "lono.log"
    cmd = [
        python_exe(), str(LONO_PIPELINE),
        "--seed", str(seed),
        "--device", device,
        "--output-dir", str(lono_dir),
        "--stage3-only", STAGE3_ONLY,
    ]
    if full_5fold:
        cmd.extend(["--n-folds", "5"])
    else:
        # Fast Nozzle-2 only check: keep only N2 by skipping the others.
        cmd.extend(["--n-folds", "1"])
    # Forward the aux-head flags. baseline_no_aux is the only winner_name
    # where we should NOT set --onset-aux-head; the other two configs need it.
    if winner_name != "baseline_no_aux":
        cmd.extend(["--onset-aux-head", "--lambda-aux", "0.1"])
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return pd.DataFrame()
    rc, _ = run_subprocess_streaming(cmd, lono_log)
    if rc != 0:
        print(f"!! LONO follow-up failed for {winner_name} (rc={rc}).")
        return pd.DataFrame()
    per_fold_csv = lono_dir / "per_fold.csv"
    if not per_fold_csv.exists():
        return pd.DataFrame()
    per_fold = pd.read_csv(per_fold_csv)
    for _, r in per_fold.loc[per_fold["model"].isin(["MLP_series", "MLP_uncensored"])].iterrows():
        rows.append({
            "config_name": winner_name,
            "model_eval": r["model"],
            "fold_nozzle": r["holdout"],
            "test_rmse_mm": r.get("rmse_mm"),
            "test_mae_mm": r.get("mae_mm"),
            "coverage_1sigma": r.get("coverage_1sigma"),
            "coverage_2sigma": r.get("coverage_2sigma"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "onset_aux_lono_followup.csv", index=False)
    return df


def write_verdict(in_domain: pd.DataFrame, lono: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Tier-2C onset auxiliary head verdict", ""]
    if in_domain.empty:
        lines.append("No in-domain results recorded.")
    else:
        lines.append("In-domain comparison:")
        lines.append("")
        lines.append("| config | overall RMSE | onset RMSE (t<0.3 ms) | late RMSE | bias_onset |")
        lines.append("|---|---|---|---|---|")
        for _, r in in_domain.iterrows():
            lines.append(
                f"| {r['name']} | {r['rmse_overall_mm']:.4f} | "
                f"{r[f'rmse_onset_t_lt_{ONSET_T_MS}ms_mm']:.4f} | "
                f"{r[f'rmse_t_ge_{ONSET_T_MS}ms_mm']:.4f} | {r['bias_onset_mm']:+.4f} |"
            )
        baseline = in_domain.loc[in_domain["name"] == "baseline_no_aux"]
        if not baseline.empty:
            b = baseline.iloc[0]
            for _, r in in_domain.iterrows():
                if r["name"] == "baseline_no_aux":
                    continue
                delta = r[f"rmse_onset_t_lt_{ONSET_T_MS}ms_mm"] - b[f"rmse_onset_t_lt_{ONSET_T_MS}ms_mm"]
                lines.append(f"- Delta(onset RMSE) {r['name']} vs baseline: {delta:+.4f} mm.")
    lines.append("")
    if not lono.empty:
        lines.append("## LONO follow-up")
        lines.append("")
        lines.append("| config | fold | model_eval | RMSE |")
        lines.append("|---|---|---|---|")
        for _, r in lono.iterrows():
            lines.append(f"| {r['config_name']} | {r['fold_nozzle']} | {r['model_eval']} | {r['test_rmse_mm']:.4f} |")
    else:
        lines.append("(No LONO follow-up recorded. Re-run with --lono-followup.)")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- If `aux_no_anchor` matches or beats `baseline_no_aux` on Nozzle-2 LONO,")
    lines.append("  the auxiliary head can replace the mu-anchor (architectural simplification).")
    lines.append("- If `aux_with_anchor` beats both, anchor and aux are complementary - keep both.")
    lines.append("- If neither aux config beats baseline by more than sigma_seed (Tier-1B), the")
    lines.append("  current mu-anchor is the right design; document as future work.")
    (output_dir / "verdict.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== Tier-2C verdict ===")
    print((output_dir / "verdict.md").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--lono-followup", action="store_true",
                   help="Run LONO follow-up on the best aux config (only if it beats baseline in-domain).")
    p.add_argument("--lono-full-5fold", action="store_true",
                   help="Use full 5-fold LONO instead of Nozzle-2 fast check.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else RUNS_ROOT / f"onset_aux_head_{timestamp}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tier-2C onset-aux-head output dir: {output_dir}")

    (output_dir / "config.json").write_text(json.dumps({
        "seed": args.seed,
        "device": args.device,
        "configs": [
            {"name": n, "stage1_args": s1, "stage2_args": s2}
            for (n, s1, s2) in ONSET_CONFIGS
        ],
        "stage3_only": STAGE3_ONLY,
        "stage3_suite_config": str(STAGE3_SUITE_CONFIG),
        "onset_t_ms": ONSET_T_MS,
    }, indent=2), encoding="utf-8")

    records: list[dict[str, Any]] = []
    for i, (name, s1_args, s2_args) in enumerate(ONSET_CONFIGS, start=1):
        print(f"\n========== Config {i}/{len(ONSET_CONFIGS)}: {name} ==========")
        config_dir = output_dir / name
        try:
            rec = run_one_config(
                name=name, stage1_args=s1_args, stage2_args=s2_args,
                config_dir=config_dir, seed=args.seed, device=args.device,
                dry_run=args.dry_run,
            )
            records.append(rec)
        except Exception as e:
            print(f"!! Config {name} FAILED: {e}")
            (config_dir / "config_FAILED.json").write_text(
                json.dumps({"error": str(e)}, indent=2), encoding="utf-8"
            )

    (output_dir / "all_configs.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    in_domain_df = pd.DataFrame()
    lono_df = pd.DataFrame()
    if records and not args.dry_run:
        in_domain_df = write_in_domain_outputs(records, output_dir)

        # Pick winner: lowest onset RMSE among aux variants if either beats baseline.
        baseline = in_domain_df.loc[in_domain_df["name"] == "baseline_no_aux"]
        aux_only = in_domain_df.loc[in_domain_df["name"] != "baseline_no_aux"]
        winner = None
        if args.lono_followup and not aux_only.empty:
            onset_col = f"rmse_onset_t_lt_{ONSET_T_MS}ms_mm"
            if not baseline.empty:
                baseline_onset = float(baseline.iloc[0][onset_col])
                best = aux_only.sort_values(onset_col).iloc[0]
                if float(best[onset_col]) < baseline_onset:
                    winner = str(best["name"])
            else:
                winner = str(aux_only.sort_values(onset_col).iloc[0]["name"])
        if winner is not None:
            lono_df = run_lono_followup(
                winner_name=winner, output_dir=output_dir,
                device=args.device, seed=args.seed,
                dry_run=args.dry_run, full_5fold=args.lono_full_5fold,
            )
        elif args.lono_followup:
            print("[skip-lono] No aux config beat baseline in-domain; skipping LONO follow-up.")

        write_verdict(in_domain_df, lono_df, output_dir)

    print(f"\nTier-2C done. Output: {output_dir}")


if __name__ == "__main__":
    main()
