"""Tier-3A family-aware architecture sweep driver.

Three architectures x two LONO protocols, targeting the documented Nozzle-0
cross-family failure (MLP LONO RMSE 34.76 mm vs SVGP 26.99 mm on N0):

    A_single_baseline   PenetrationMLP, train on N0..N5            (current production)
    B_modified_only     PenetrationMLP, train on N1..N5            (honest narrowing)
    C_family_head       FamilyAwarePenetrationMLP, train on N0..N5 (shared trunk + per-family heads)

LONO protocols:
    lono_5fold              hold out each of {N0..N5} (current Section-5 protocol)
    lono_modified_only      hold out each of {N1..N5} only (N0 always in training)

Spec lives in MLP/MLP_training/ablations/TIER3_FAMILY_SHAPE_PLAN.md section 3.

PREREQUISITE: The code changes described in plan section 3.1-3.4 must be in
place before this driver does anything useful:
  - FamilyAwarePenetrationMLP class in MLP/MLP_training/efc/models.py
  - family_id channel added in engineered_feature_common.py
  - --family-aware CLI flag on Stage 1, 2, 3 trainers and the LONO pipeline
  - --protocol {lono_5fold, lono_modified_only} on the LONO pipeline
  - WeightedRandomSampler (or equivalent) for per-family batch balancing
  - Inference-time fallback when family-0 head has 0 training samples
    (per plan section 3.7 pitfall 3, route through family-1 head and report
     "N0 fold equivalent to baseline")

Wall clock: A is 5 folds x ~13 min ~ 65 min (reuse if available).
            B is 5 folds x ~10 min ~ 50 min.
            C x 2 protocols x 5 folds x ~13 min ~ 130 min.
Total: ~3-4 h.

PREREQUISITE 2: Tier-1B must have produced sigma_seed - the verdict here is
not interpretable without it.

Usage
-----
    python MLP/MLP_training/ablations/run_family_head_sweep.py
    python MLP/MLP_training/ablations/run_family_head_sweep.py --skip A
    python MLP/MLP_training/ablations/run_family_head_sweep.py --dry-run

Outputs (under MLP/runs_mlp/family_head_sweep_<timestamp>/):
    family_head_lono.csv            one row per (architecture, protocol, fold)
    family_head_summary.md          cross-tabulated comparison A/B/C
    verdict.md                      paper paragraph hooks
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
RUNS_ROOT = MLP_ROOT / "runs_mlp"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
LONO_PIPELINE = TRAINING_ROOT / "ood_lono" / "run_lono_pipeline.py"

STAGE3_ONLY = "anchor_off"
DEFAULT_SEED = 42

# (run_label, architecture_mode, protocol, extra_lono_args)
# A: single arch, full 5-fold (the current Section-5 protocol). Default if no
#    recent A run is available, otherwise pass --skip A to reuse the prior one.
# B: single arch, but only modified-family folds (N0 always in training).
#    Requires --protocol lono_modified_only on the LONO pipeline.
# C1: family head, full 5-fold (honest LONO; N0 fold cannot benefit).
# C2: family head, modified-only (where the family head can actually shine).
SWEEP_CONFIGS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("A_single_baseline",       "single",      "lono_5fold",          {}),
    ("B_modified_only",         "single",      "lono_modified_only",  {}),
    ("C_family_head_5fold",     "family_head", "lono_5fold",          {}),
    ("C_family_head_modified",  "family_head", "lono_modified_only",  {}),
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


def run_one_sweep(
    *,
    run_label: str,
    architecture_mode: str,
    protocol: str,
    extra_lono_args: dict[str, Any],
    sweep_dir: Path,
    seed: int,
    device: str,
    dry_run: bool,
    data_dir: Path | None = None,
) -> Path | None:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    log_path = sweep_dir / "lono.log"
    cmd = [
        python_exe(), str(LONO_PIPELINE),
        "--seed", str(seed),
        "--device", device,
        "--n-folds", "5",
        "--output-dir", str(sweep_dir),
        "--stage3-only", STAGE3_ONLY,
        "--architecture-mode", architecture_mode,
        "--protocol", protocol,
    ]
    if data_dir is not None:
        cmd.extend(["--data-dir", str(data_dir)])
    for k, v in extra_lono_args.items():
        if v is None or v is False:
            continue
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return None
    rc, _ = run_subprocess_streaming(cmd, log_path)
    if rc != 0:
        print(f"!! Sweep {run_label} failed (rc={rc}). See {log_path}.")
        return None
    per_fold = sweep_dir / "per_fold.csv"
    if not per_fold.exists():
        print(f"!! Sweep {run_label} produced no per_fold.csv.")
        return None
    print(f"[{run_label}] done in {(time.time() - t0)/60.0:.1f} min.")
    return per_fold


def aggregate_lono_results(
    sweep_records: list[tuple[str, str, str, Path | None]],
    output_dir: Path,
) -> pd.DataFrame:
    """Combine per_fold.csv from each sweep into one master CSV."""
    rows: list[dict[str, Any]] = []
    for run_label, architecture_mode, protocol, per_fold_path in sweep_records:
        if per_fold_path is None or not per_fold_path.exists():
            continue
        df = pd.read_csv(per_fold_path)
        df = df.loc[df["model"].isin(["MLP_series", "MLP_uncensored"])].copy()
        for _, r in df.iterrows():
            rows.append({
                "run_label": run_label,
                "architecture": architecture_mode,
                "protocol": protocol,
                "model_eval": r["model"],
                "fold_nozzle": r["holdout"],
                "rmse_mm": r.get("rmse_mm"),
                "mae_mm": r.get("mae_mm"),
                "bias_mm": r.get("bias_mm"),
                "coverage_1sigma": r.get("coverage_1sigma"),
                "coverage_2sigma": r.get("coverage_2sigma"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "family_head_lono.csv", index=False)
    return df


def write_summary_and_verdict(df: pd.DataFrame, output_dir: Path) -> None:
    """Cross-tabulate A vs B vs C and write summary + verdict."""
    if df.empty:
        (output_dir / "family_head_summary.md").write_text(
            "# Tier-3A family head sweep\n\nNo results recorded.\n", encoding="utf-8"
        )
        return

    series = df.loc[df["model_eval"] == "MLP_series"]
    agg = (
        series.groupby(["run_label", "fold_nozzle"], as_index=False)["rmse_mm"]
              .mean()
              .pivot(index="fold_nozzle", columns="run_label", values="rmse_mm")
    )

    overall = (
        series.groupby("run_label")["rmse_mm"].agg(["mean", "std", "count"]).reset_index()
              .rename(columns={"mean": "mean_rmse_mm", "std": "std_rmse_mm", "count": "n_folds"})
    )

    lines = ["# Tier-3A family-aware head LONO summary", ""]
    lines.append("## Per-fold RMSE (mm), MLP_series")
    lines.append("")
    headers = ["fold"] + list(agg.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for fold, row in agg.iterrows():
        vals = []
        for c in agg.columns:
            v = row[c]
            vals.append(f"{v:.4f}" if isinstance(v, (int, float)) and not np.isnan(v) else "-")
        lines.append(f"| {fold} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Aggregate RMSE (mm)")
    lines.append("")
    lines.append("| run_label | n_folds | mean_rmse | std_rmse |")
    lines.append("|---|---|---|---|")
    for _, r in overall.iterrows():
        std = r["std_rmse_mm"]
        lines.append(
            f"| {r['run_label']} | {int(r['n_folds'])} | {r['mean_rmse_mm']:.4f} | "
            f"{('-' if pd.isna(std) else f'{std:.4f}')} |"
        )
    (output_dir / "family_head_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # Verdict
    verdict_lines = ["# Tier-3A verdict", ""]
    a_label = "A_single_baseline"
    c1 = "C_family_head_5fold"
    c2 = "C_family_head_modified"
    a = overall.loc[overall["run_label"] == a_label]
    cc1 = overall.loc[overall["run_label"] == c1]
    cc2 = overall.loc[overall["run_label"] == c2]
    if not a.empty and not cc1.empty:
        d = float(cc1.iloc[0]["mean_rmse_mm"]) - float(a.iloc[0]["mean_rmse_mm"])
        verdict_lines.append(f"- C vs A on lono_5fold: delta mean RMSE = {d:+.4f} mm.")
    if not a.empty and not cc2.empty:
        # On lono_modified_only, A's baseline number isn't apples-to-apples
        # (different fold set). Just report C's modified-only number.
        verdict_lines.append(
            f"- C on lono_modified_only: mean RMSE = {float(cc2.iloc[0]['mean_rmse_mm']):.4f} mm."
        )
    verdict_lines.append("")
    verdict_lines.append("Reminder: on lono_5fold, the N0 fold cannot improve under C because the")
    verdict_lines.append("family-0 head sees zero training data when N0 is held out. Report C's")
    verdict_lines.append("N0 fold as expected null. The interesting comparisons are:")
    verdict_lines.append("- C vs A on N1..N5 folds (does sharing trunk help these?)")
    verdict_lines.append("- C vs A on lono_modified_only (family head's main shot at improving)")
    verdict_lines.append("- B vs A on modified folds (does narrowing scope alone improve?)")
    verdict_lines.append("")
    verdict_lines.append("Apply sigma_seed (Tier-1B) before declaring any of these improvements real.")
    (output_dir / "verdict.md").write_text("\n".join(verdict_lines), encoding="utf-8")
    print("\n=== Tier-3A summary ===")
    print((output_dir / "family_head_summary.md").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Synthetic data root. Defaults to MLP/synthetic_data next to this repo.")
    p.add_argument("--skip", type=str, nargs="*", default=None,
                   help="Skip these run_labels (e.g. 'A_single_baseline' if reusing Section-5 run).")
    p.add_argument("--only", type=str, nargs="*", default=None,
                   help="Restrict to a subset of run_labels.")
    p.add_argument("--reuse-a", type=Path, default=None,
                   help="If set, copy A_single_baseline's per_fold.csv from this path instead of re-running.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Resolve data_dir: prefer CLI, then MLP/synthetic_data next to the repo root.
    data_dir: Path | None = None
    if args.data_dir is not None:
        data_dir = Path(args.data_dir).resolve()
    else:
        candidate = MLP_ROOT / "synthetic_data"
        if candidate.exists():
            data_dir = candidate
    if data_dir is not None:
        print(f"Data dir: {data_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else RUNS_ROOT / f"family_head_sweep_{timestamp}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tier-3A family head sweep output dir: {output_dir}")

    (output_dir / "config.json").write_text(json.dumps({
        "seed": args.seed,
        "device": args.device,
        "configs": [
            {"run_label": rl, "architecture_mode": am, "protocol": pr, "extra": ex}
            for (rl, am, pr, ex) in SWEEP_CONFIGS
        ],
        "stage3_only": STAGE3_ONLY,
    }, indent=2), encoding="utf-8")

    skip = set(args.skip or [])
    only = set(args.only or []) if args.only else None
    sweep_records: list[tuple[str, str, str, Path | None]] = []

    for run_label, architecture_mode, protocol, extra in SWEEP_CONFIGS:
        if only is not None and run_label not in only:
            continue
        if run_label in skip:
            print(f"\n========== Skipping {run_label} (per --skip) ==========")
            continue
        if run_label == "A_single_baseline" and args.reuse_a is not None and args.reuse_a.exists():
            print(f"\n========== Reusing A from {args.reuse_a} ==========")
            sweep_records.append((run_label, architecture_mode, protocol, args.reuse_a))
            continue
        print(f"\n========== Running {run_label} (arch={architecture_mode}, protocol={protocol}) ==========")
        sweep_dir = output_dir / run_label
        try:
            per_fold = run_one_sweep(
                run_label=run_label, architecture_mode=architecture_mode,
                protocol=protocol, extra_lono_args=extra, sweep_dir=sweep_dir,
                seed=args.seed, device=args.device, dry_run=args.dry_run,
                data_dir=data_dir,
            )
            sweep_records.append((run_label, architecture_mode, protocol, per_fold))
        except Exception as e:
            print(f"!! {run_label} FAILED: {e}")
            (sweep_dir / "sweep_FAILED.json").write_text(
                json.dumps({"error": str(e)}, indent=2), encoding="utf-8"
            )
            sweep_records.append((run_label, architecture_mode, protocol, None))

    (output_dir / "all_sweeps.json").write_text(json.dumps([
        {
            "run_label": rl, "architecture_mode": am, "protocol": pr,
            "per_fold_csv": str(pf) if pf else None,
        }
        for (rl, am, pr, pf) in sweep_records
    ], indent=2), encoding="utf-8")

    if not args.dry_run:
        df = aggregate_lono_results(sweep_records, output_dir)
        write_summary_and_verdict(df, output_dir)

    print(f"\nTier-3A done. Output: {output_dir}")


if __name__ == "__main__":
    main()
