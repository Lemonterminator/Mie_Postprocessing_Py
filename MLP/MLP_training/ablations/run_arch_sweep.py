"""Tier-1A architecture sweep driver.

Sweeps hidden_dims across 9 width x depth configurations plus the production
[512, 512, 128] bottleneck baseline. For each configuration the full Stage 1
-> Stage 2 -> Stage 3 (anchor_off only) pipeline is retrained from scratch,
all at seed=42, in-domain (random group split, no LONO holdout).

Spec lives in MLP/MLP_training/ablations/TIER1_CAPACITY_SEED_OPTIMIZER_PLAN.md
section 3.

PREREQUISITE: Stage 1 and Stage 2 trainers must accept a `--hidden-dims` CLI
flag (comma-separated widths, e.g. "256,256,64"). The plan section 3.1 shows
how to wire it through trainers/base.py::run() + trainers/stage{1,2}.py.
This driver only orchestrates; it does not patch the trainers itself.

Stage 1 re-train per config is mandatory: Stage 2 warm-starts from Stage 1,
so hidden_dims must match exactly between the two stages.

Wall clock: ~8.5 min per in-domain run on RTX 5090. 10 configs ~ 85 min;
top-2 LONO follow-up (2 winners x 5 folds) ~ another 85 min.

Usage
-----
    python MLP/MLP_training/ablations/run_arch_sweep.py
    python MLP/MLP_training/ablations/run_arch_sweep.py --dry-run
    python MLP/MLP_training/ablations/run_arch_sweep.py --lono-followup

Outputs (under MLP/runs_mlp/arch_sweep_<timestamp>/):
    arch_sweep_summary.csv          one row per config (in-domain)
    arch_sweep_lono_followup.csv    one row per (winner, fold) (only if --lono-followup)
    verdict.md                       in-domain best, LONO best, recommendation
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

# (run_name, hidden_dims_csv) — 9 uniform configs + 1 bottleneck baseline.
ARCH_CONFIGS: list[tuple[str, str]] = [
    ("arch_w128_d2",            "128,128"),
    ("arch_w128_d3",            "128,128,128"),
    ("arch_w128_d4",            "128,128,128,128"),
    ("arch_w256_d2",            "256,256"),
    ("arch_w256_d3",            "256,256,256"),
    ("arch_w256_d4",            "256,256,256,256"),
    ("arch_w512_d2",            "512,512"),
    ("arch_w512_d3",            "512,512,512"),
    ("arch_w512_d4",            "512,512,512,512"),
    ("baseline_w512_512_128",   "512,512,128"),  # current production bottleneck
]


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


def param_count(hidden_dims_csv: str, input_dim: int = 8, output_dim: int = 2) -> int:
    """Rough parameter count for trunk + heads (excluding LayerNorm/biases differences).

    Conservative: counts weight matrices + biases for Linear layers in a uniform
    MLP. The actual PenetrationMLP may add LayerNorm; this is an upper-bound
    approximation good enough for the params_total column in the CSV.
    """
    widths = [int(x) for x in hidden_dims_csv.split(",") if x.strip()]
    if not widths:
        return 0
    total = 0
    prev = input_dim
    for w in widths:
        total += prev * w + w
        prev = w
    total += prev * output_dim + output_dim
    return total


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


def run_one_arch(
    *,
    run_name: str,
    hidden_dims_csv: str,
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
        "--hidden-dims", hidden_dims_csv,
        "--allow-failed-precheck",
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s1_cmd)}")
        stage1_run = Path(f"DRY_RUN/stage1_{run_name}")
    else:
        rc, out = run_subprocess_streaming(s1_cmd, s1_log)
        if rc != 0:
            raise RuntimeError(f"Stage-1 failed for {run_name} (rc={rc}). See {s1_log}.")
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
        "--hidden-dims", hidden_dims_csv,
        "--allow-failed-precheck",
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s2_cmd)}")
        stage2_run = Path(f"DRY_RUN/stage2_{run_name}")
    else:
        rc, out = run_subprocess_streaming(s2_cmd, s2_log)
        if rc != 0:
            raise RuntimeError(f"Stage-2 failed for {run_name} (rc={rc}). See {s2_log}.")
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
        anchor_run = Path(f"DRY_RUN/{STAGE3_ONLY}_{run_name}")
        suite_summary = Path(f"DRY_RUN/suite_{run_name}.json")
    else:
        rc, out = run_subprocess_streaming(s3_cmd, s3_log)
        if rc != 0:
            raise RuntimeError(f"Stage-3 suite failed for {run_name} (rc={rc}). See {s3_log}.")
        suite_summary = parse_suite_summary_path(out)
        if suite_summary is None:
            raise RuntimeError("Could not parse suite_summary path.")
        anchor_run = find_named_run_dir(suite_summary, STAGE3_ONLY)
    timing["stage3_s"] = time.time() - t0

    # In-domain eval (no fold restriction): use clean split, no filter
    eval_log = config_dir / "eval.log"
    eval_cmd = [
        python_exe(), str(EVAL_SCRIPT),
        "--refinement-run", str(anchor_run),
        "--split", "clean",
        "--tag", f"arch_{run_name}",
        "--batch-points", "262144",
        "--no-save-plots",
        "--max-traj-plots", "0",
    ]
    t0 = time.time()
    metrics: dict[str, float] = {}
    eval_dir = Path("")
    if dry_run:
        print(f"[dry-run] {' '.join(eval_cmd)}")
        eval_dir = Path(f"DRY_RUN/eval_{run_name}")
    else:
        rc, out = run_subprocess_streaming(eval_cmd, eval_log)
        if rc != 0:
            raise RuntimeError(f"External eval failed for {run_name} (rc={rc}). See {eval_log}.")
        match = EVAL_OUTPUT_RE.search(out)
        if match:
            eval_dir = Path(match.group(1).strip())
        else:
            candidates = sorted(
                (MLP_ROOT / "eval").glob(f"rmse_eval_clean_*_arch_{run_name}"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if not candidates:
                raise RuntimeError(f"Could not locate eval dir for {run_name}.")
            eval_dir = candidates[0]
        metrics = read_eval_metrics(eval_dir)
    timing["eval_s"] = time.time() - t0

    record = {
        "run_name": run_name,
        "hidden_dims": hidden_dims_csv,
        "params_total_approx": param_count(hidden_dims_csv),
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
            "run_name": rec["run_name"],
            "hidden_dims": rec["hidden_dims"],
            "params_total": rec["params_total_approx"],
            "test_rmse_mm": m.get("rmse_mm"),
            "test_mae_mm": m.get("mae_mm"),
            "test_bias_mm": m.get("bias_mm"),
            "coverage_1sigma": m.get("coverage_1sigma"),
            "coverage_2sigma": m.get("coverage_2sigma"),
            "train_minutes": sum(rec.get("timing", {}).values()) / 60.0,
        })
    df = pd.DataFrame(rows).sort_values("test_rmse_mm", na_position="last")
    df.to_csv(output_dir / "arch_sweep_summary.csv", index=False)
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
        run_name = winner["run_name"]
        hidden_dims_csv = winner["hidden_dims"]
        lono_dir = output_dir / f"lono_{run_name}"
        lono_dir.mkdir(parents=True, exist_ok=True)
        lono_log = lono_dir / "lono.log"
        cmd = [
            python_exe(), str(LONO_PIPELINE),
            "--seed", str(seed),
            "--device", device,
            "--n-folds", "5",
            "--output-dir", str(lono_dir),
            "--stage3-only", STAGE3_ONLY,
            "--hidden-dims", hidden_dims_csv,
        ]
        if dry_run:
            print(f"[dry-run] {' '.join(cmd)}  (intended hidden_dims={hidden_dims_csv})")
            continue
        rc, _ = run_subprocess_streaming(cmd, lono_log)
        if rc != 0:
            print(f"!! LONO follow-up failed for {run_name} (rc={rc}). Skipping.")
            continue
        per_fold_csv = lono_dir / "per_fold.csv"
        if not per_fold_csv.exists():
            print(f"!! Missing {per_fold_csv}; skipping.")
            continue
        per_fold = pd.read_csv(per_fold_csv)
        for _, r in per_fold.loc[per_fold["model"].isin(["MLP_series", "MLP_uncensored"])].iterrows():
            rows.append({
                "run_name": run_name,
                "hidden_dims": hidden_dims_csv,
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
    df.to_csv(output_dir / "arch_sweep_lono_followup.csv", index=False)
    return df


def write_verdict(in_domain: pd.DataFrame, lono: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Tier-1A architecture sweep verdict", ""]
    if in_domain.empty:
        lines.append("No in-domain results recorded.")
    else:
        best = in_domain.iloc[0]
        baseline = in_domain.loc[in_domain["run_name"] == "baseline_w512_512_128"]
        lines.append(f"- In-domain best: **{best['run_name']}** "
                     f"(hidden_dims={best['hidden_dims']}, RMSE={best['test_rmse_mm']:.4f} mm, "
                     f"params={int(best['params_total'])}).")
        if not baseline.empty:
            b = baseline.iloc[0]
            lines.append(f"- Production baseline: {b['hidden_dims']} -> RMSE={b['test_rmse_mm']:.4f} mm.")
            lines.append(f"- Delta vs baseline: {best['test_rmse_mm'] - b['test_rmse_mm']:+.4f} mm.")
    lines.append("")
    if not lono.empty:
        lines.append("## LONO follow-up (5-fold mean per architecture)")
        agg = (
            lono.loc[lono["model_eval"] == "MLP_series"]
                .groupby("run_name")["test_rmse_mm"].agg(["mean", "std", "count"])
                .reset_index().sort_values("mean")
        )
        lines.append("")
        lines.append("| run_name | LONO mean RMSE | std | n_folds |")
        lines.append("|---|---|---|---|")
        for _, r in agg.iterrows():
            lines.append(f"| {r['run_name']} | {r['mean']:.4f} | {r['std']:.4f} | {int(r['count'])} |")
    else:
        lines.append("(No LONO follow-up recorded. Re-run with --lono-followup.)")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("If the in-domain winner also beats `baseline_w512_512_128` on LONO mean RMSE")
    lines.append("by more than sigma_seed (see Tier-1B verdict), promote it as the new production")
    lines.append("architecture. Otherwise keep the baseline and report this sweep as a defensible")
    lines.append("non-result.")
    (output_dir / "verdict.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== Tier-1A verdict ===")
    print((output_dir / "verdict.md").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--lono-followup", action="store_true",
                   help="After the in-domain sweep, run 5-fold LONO on the top-2 winners.")
    p.add_argument("--top-k-lono", type=int, default=2,
                   help="Number of in-domain winners to push to LONO (default: 2).")
    p.add_argument("--only", type=str, nargs="*", default=None,
                   help="Restrict to a subset of arch run names.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else RUNS_ROOT / f"arch_sweep_{timestamp}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tier-1A architecture sweep output dir: {output_dir}")

    configs = [
        (name, dims) for (name, dims) in ARCH_CONFIGS
        if args.only is None or name in set(args.only)
    ]
    (output_dir / "config.json").write_text(json.dumps({
        "seed": args.seed,
        "device": args.device,
        "configs": [{"name": n, "hidden_dims": d, "params_total_approx": param_count(d)}
                    for (n, d) in configs],
        "stage3_only": STAGE3_ONLY,
        "stage3_suite_config": str(STAGE3_SUITE_CONFIG),
    }, indent=2), encoding="utf-8")

    records: list[dict[str, Any]] = []
    for i, (name, dims) in enumerate(configs, start=1):
        print(f"\n========== Arch {i}/{len(configs)}: {name} ({dims}) ==========")
        config_dir = output_dir / name
        try:
            rec = run_one_arch(
                run_name=name, hidden_dims_csv=dims, config_dir=config_dir,
                seed=args.seed, device=args.device, dry_run=args.dry_run,
            )
            records.append(rec)
        except Exception as e:
            print(f"!! Arch {name} FAILED: {e}")
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

        if args.lono_followup and not in_domain_df.empty:
            winners = in_domain_df.head(args.top_k_lono).to_dict(orient="records")
            winner_records = [
                {"run_name": w["run_name"], "hidden_dims": w["hidden_dims"]}
                for w in winners
            ]
            lono_df = run_lono_followup(
                winner_records=winner_records, output_dir=output_dir,
                device=args.device, seed=args.seed, dry_run=args.dry_run,
            )

        write_verdict(in_domain_df, lono_df, output_dir)

    print(f"\nTier-1A done. Output: {output_dir}")


if __name__ == "__main__":
    main()
