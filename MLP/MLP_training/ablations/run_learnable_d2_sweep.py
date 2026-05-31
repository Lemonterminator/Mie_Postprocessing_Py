"""Tier-3B learnable per-nozzle d2 concavity weight sweep.

Three variants targeting the Nozzle-3 step-like wake-catch-up morphology that
violates the smooth-concave prior baked into Stage-1 via d2_concave_weight:

    baseline                 lambda_d2 = 5e-4 globally (current default)
    no_d2_penalty            lambda_d2 = 0      globally
    per_nozzle_learned       lambda_d2[nozzle_id] learnable, 6-way, init ~5e-4

Each variant gets in-domain training + 5-fold LONO. The headline diagnostic
for `per_nozzle_learned` is whether lambda_d2[N3] -> 0 (floor 1e-5) while the
other nozzles stay near 5e-4 - that would directly confirm the hypothesis.

Spec lives in MLP/MLP_training/ablations/TIER3_FAMILY_SHAPE_PLAN.md section 4.

PREREQUISITE: The code changes described in plan section 4.3-4.4 must be in
place before this driver does anything useful:
  - log_lambda_d2 parameter in PenetrationMLP (efc/models.py)
  - per-sample d2-weight indexing in the Stage-1 loss
  - --learnable-d2, --n-families-for-d2, --d2-concave-weight CLI flags on
    Stage-1 trainer (and propagated to the LONO pipeline)
  - nozzle_id (6-way) channel added in engineered_feature_common.py, kept
    separate from the 3A 2-way family_id channel
  - learned_lambda_d2_values.csv export from each Stage-1 run (final
    softplus(W) + floor per nozzle)

Wall clock: ~10 min per in-domain run x 3 variants ~ 30 min;
            3 variants x 5-fold LONO x ~10 min ~ 150 min.
Total: ~3 h.

PREREQUISITE 2: Tier-1B must have produced sigma_seed - the verdict here is
not interpretable without it.

LONO + per-nozzle lambda_d2 caveat (plan section 4.8 pitfall 5): when
fold=N3, no N3 training data exists so lambda_d2[3] stays at init and the
gain on the N3-held-out fold is **zero by construction**. To actually test
the N3 hypothesis, also run an in-distribution evaluation where N3 is in
training and look at N3's per-trajectory RMSE on the in-distribution test
split - that is what --in-domain-eval enables here.

Usage
-----
    python MLP/MLP_training/ablations/run_learnable_d2_sweep.py
    python MLP/MLP_training/ablations/run_learnable_d2_sweep.py --only per_nozzle_learned
    python MLP/MLP_training/ablations/run_learnable_d2_sweep.py --dry-run

Outputs (under MLP/runs_mlp/learnable_d2_sweep_<timestamp>/):
    learnable_d2_lono.csv               variant x fold per-slice RMSE
    learned_lambda_d2_values.csv        final lambda_d2[nozzle] per fold (per_nozzle_learned only)
    in_domain_per_nozzle.csv            in-domain test RMSE sliced by source nozzle
    verdict.md                           does lambda_d2[N3] -> 0? does N3 improve in-domain?
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

# (variant_name, stage1_args)
# Stage 2/3 inherit their own defaults; learnable-d2 lives entirely in Stage 1.
D2_VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("baseline",            {}),
    ("no_d2_penalty",       {"d2_concave_weight": 0.0}),
    ("per_nozzle_learned",  {"learnable_d2": True, "n_families_for_d2": 6}),
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


def in_domain_per_nozzle_metrics(points_csv: Path) -> pd.DataFrame:
    """Per-nozzle RMSE/MAE/bias from a single eval points.csv."""
    if not points_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(points_csv, low_memory=False)
    if "folder" not in df.columns:
        return pd.DataFrame()
    rows = []
    for nozzle, sub in df.groupby("folder", sort=True):
        resid = sub["pen_pred_mm"].to_numpy(float) - sub["pen_true_mm"].to_numpy(float)
        abs_err = np.abs(resid)
        rows.append({
            "nozzle": str(nozzle),
            "n_points": int(len(sub)),
            "rmse_mm": float(np.sqrt(np.mean(resid ** 2))),
            "mae_mm": float(np.mean(abs_err)),
            "bias_mm": float(np.mean(resid)),
        })
    return pd.DataFrame(rows)


def run_one_in_domain(
    *,
    variant_name: str,
    stage1_args: dict[str, Any],
    variant_dir: Path,
    seed: int,
    device: str,
    dry_run: bool,
) -> dict[str, Any]:
    variant_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    s1_log = variant_dir / "stage1.log"
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
        stage1_run = Path(f"DRY_RUN/stage1_{variant_name}")
    else:
        rc, out = run_subprocess_streaming(s1_cmd, s1_log)
        if rc != 0:
            raise RuntimeError(f"Stage-1 failed for {variant_name} (rc={rc}). See {s1_log}.")
        stage1_run = parse_saved_run_dir(out)
        if stage1_run is None:
            raise RuntimeError("Could not parse Stage-1 run_dir.")
    timing["stage1_s"] = time.time() - t0

    s2_log = variant_dir / "stage2.log"
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
        stage2_run = Path(f"DRY_RUN/stage2_{variant_name}")
    else:
        rc, out = run_subprocess_streaming(s2_cmd, s2_log)
        if rc != 0:
            raise RuntimeError(f"Stage-2 failed for {variant_name} (rc={rc}). See {s2_log}.")
        stage2_run = parse_saved_run_dir(out)
        if stage2_run is None:
            raise RuntimeError("Could not parse Stage-2 run_dir.")
    timing["stage2_s"] = time.time() - t0

    s3_log = variant_dir / "stage3.log"
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
        anchor_run = Path(f"DRY_RUN/{STAGE3_ONLY}_{variant_name}")
        suite_summary = Path(f"DRY_RUN/suite_{variant_name}.json")
    else:
        rc, out = run_subprocess_streaming(s3_cmd, s3_log)
        if rc != 0:
            raise RuntimeError(f"Stage-3 suite failed for {variant_name} (rc={rc}). See {s3_log}.")
        suite_summary = parse_suite_summary_path(out)
        if suite_summary is None:
            raise RuntimeError("Could not parse suite_summary path.")
        anchor_run = find_named_run_dir(suite_summary, STAGE3_ONLY)
    timing["stage3_s"] = time.time() - t0

    eval_log = variant_dir / "eval.log"
    eval_cmd = [
        python_exe(), str(EVAL_SCRIPT),
        "--refinement-run", str(anchor_run),
        "--split", "clean",
        "--tag", f"d2_{variant_name}",
        "--batch-points", "262144",
        "--no-save-plots",
        "--max-traj-plots", "0",
    ]
    t0 = time.time()
    per_nozzle = pd.DataFrame()
    eval_dir = Path("")
    if dry_run:
        print(f"[dry-run] {' '.join(eval_cmd)}")
        eval_dir = Path(f"DRY_RUN/eval_{variant_name}")
    else:
        rc, out = run_subprocess_streaming(eval_cmd, eval_log)
        if rc != 0:
            raise RuntimeError(f"External eval failed for {variant_name} (rc={rc}). See {eval_log}.")
        match = EVAL_OUTPUT_RE.search(out)
        if match:
            eval_dir = Path(match.group(1).strip())
        else:
            candidates = sorted(
                (MLP_ROOT / "eval").glob(f"rmse_eval_clean_*_d2_{variant_name}"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if not candidates:
                raise RuntimeError(f"Could not locate eval dir for {variant_name}.")
            eval_dir = candidates[0]
        per_nozzle = in_domain_per_nozzle_metrics(eval_dir / "points.csv")
        per_nozzle["variant"] = variant_name
    timing["eval_s"] = time.time() - t0

    # If this is the learnable variant, expect the Stage-1 run to have emitted
    # learned_lambda_d2_values.csv. Lift it into the variant dir.
    learned_csv = Path(stage1_run) / "learned_lambda_d2_values.csv"
    if learned_csv.exists() and not dry_run:
        copy_to = variant_dir / "learned_lambda_d2_values.csv"
        copy_to.write_text(learned_csv.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "variant": variant_name,
        "stage1_args": stage1_args,
        "stage1_run": str(stage1_run),
        "stage2_run": str(stage2_run),
        "stage3_suite_summary": str(suite_summary),
        "winner_run": str(anchor_run),
        "eval_dir": str(eval_dir),
        "per_nozzle": per_nozzle.to_dict(orient="records") if not per_nozzle.empty else [],
        "timing": timing,
    }


def run_one_lono(
    *,
    variant_name: str,
    stage1_args: dict[str, Any],
    lono_dir: Path,
    seed: int,
    device: str,
    dry_run: bool,
) -> Path | None:
    lono_dir.mkdir(parents=True, exist_ok=True)
    lono_log = lono_dir / "lono.log"
    cmd = [
        python_exe(), str(LONO_PIPELINE),
        "--seed", str(seed),
        "--device", device,
        "--n-folds", "5",
        "--output-dir", str(lono_dir),
        "--stage3-only", STAGE3_ONLY,
    ]
    append_cli(cmd, stage1_args)
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return None
    rc, _ = run_subprocess_streaming(cmd, lono_log)
    if rc != 0:
        print(f"!! LONO sweep failed for {variant_name} (rc={rc}).")
        return None
    per_fold = lono_dir / "per_fold.csv"
    return per_fold if per_fold.exists() else None


def aggregate_lono(records: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        per_fold_path = rec.get("lono_per_fold_csv")
        if per_fold_path is None or not Path(per_fold_path).exists():
            continue
        df = pd.read_csv(per_fold_path)
        df = df.loc[df["model"].isin(["MLP_series", "MLP_uncensored"])].copy()
        for _, r in df.iterrows():
            rows.append({
                "variant": rec["variant"],
                "model_eval": r["model"],
                "fold_nozzle": r["holdout"],
                "rmse_mm": r.get("rmse_mm"),
                "mae_mm": r.get("mae_mm"),
                "bias_mm": r.get("bias_mm"),
                "coverage_1sigma": r.get("coverage_1sigma"),
                "coverage_2sigma": r.get("coverage_2sigma"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "learnable_d2_lono.csv", index=False)
    return df


def aggregate_in_domain(records: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        for r in rec.get("per_nozzle", []) or []:
            rows.append({**r})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "in_domain_per_nozzle.csv", index=False)
    return df


def gather_learned_lambdas(records: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("variant") != "per_nozzle_learned":
            continue
        learned_csv = Path(rec.get("stage1_run", "")) / "learned_lambda_d2_values.csv"
        if not learned_csv.exists():
            # Driver also copies it into the variant_dir; check there.
            alt = output_dir / "per_nozzle_learned" / "learned_lambda_d2_values.csv"
            if alt.exists():
                learned_csv = alt
            else:
                continue
        df = pd.read_csv(learned_csv)
        df["context"] = "in_domain"
        rows.extend(df.to_dict(orient="records"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "learned_lambda_d2_values.csv", index=False)
    return df


def write_verdict(
    lono_df: pd.DataFrame,
    in_domain_df: pd.DataFrame,
    learned_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = ["# Tier-3B learnable lambda_d2 verdict", ""]
    if in_domain_df.empty:
        lines.append("No in-domain results recorded.")
    else:
        lines.append("## In-domain per-nozzle RMSE (mm)")
        lines.append("")
        pivot = in_domain_df.pivot_table(
            index="nozzle", columns="variant", values="rmse_mm", aggfunc="first"
        ).sort_index()
        cols = list(pivot.columns)
        lines.append("| nozzle | " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
        for nozzle, row in pivot.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                vals.append(f"{v:.4f}" if isinstance(v, (int, float)) and not np.isnan(v) else "-")
            lines.append(f"| {nozzle} | " + " | ".join(vals) + " |")
        # N3 highlight
        if "Nozzle3" in pivot.index and "baseline" in pivot.columns:
            base = float(pivot.loc["Nozzle3", "baseline"])
            for c in cols:
                if c == "baseline":
                    continue
                v = pivot.loc["Nozzle3", c]
                if isinstance(v, (int, float)) and not np.isnan(v):
                    lines.append(f"- Delta(N3 in-domain RMSE) {c} vs baseline: {float(v) - base:+.4f} mm.")
    lines.append("")
    if not learned_df.empty:
        lines.append("## Final learned lambda_d2[nozzle] (per_nozzle_learned)")
        lines.append("")
        lines.append(learned_df.to_markdown(index=False))
        lines.append("")
        lines.append("Hypothesis check: lambda_d2[N3] -> floor (1e-5) while others stay near 5e-4")
        lines.append("would directly confirm the step-morphology mis-specification story.")
    if not lono_df.empty:
        lines.append("")
        lines.append("## LONO mean RMSE (mm) by variant")
        lines.append("")
        agg = (
            lono_df.loc[lono_df["model_eval"] == "MLP_series"]
                   .groupby("variant")["rmse_mm"].agg(["mean", "std", "count"]).reset_index()
                   .sort_values("mean")
        )
        lines.append("| variant | n_folds | mean | std |")
        lines.append("|---|---|---|---|")
        for _, r in agg.iterrows():
            std = r["std"]
            lines.append(
                f"| {r['variant']} | {int(r['count'])} | {r['mean']:.4f} | "
                f"{('-' if pd.isna(std) else f'{std:.4f}')} |"
            )
    lines.append("")
    lines.append("## Decision rules (plan section 4.8 + acceptance bar in section 2)")
    lines.append("- LONO + per-nozzle lambda is biased downwards on its target fold (no")
    lines.append("  gradient flows into lambda_d2[held-out]). Read the LONO numbers as a")
    lines.append("  *control*, NOT as the test of the N3 hypothesis.")
    lines.append("- The hypothesis test is the in-domain N3 row: per_nozzle_learned should")
    lines.append("  improve over baseline by >=1 mm, while the LONO 5-fold mean must not")
    lines.append("  degrade by >0.3 mm.")
    lines.append("- Apply sigma_seed (Tier-1B) before declaring any of these real.")
    (output_dir / "verdict.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== Tier-3B verdict ===")
    print((output_dir / "verdict.md").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--only", type=str, nargs="*", default=None,
                   help="Restrict to a subset of variant names.")
    p.add_argument("--skip-in-domain", action="store_true",
                   help="Skip the in-domain training/eval pass (use cached records).")
    p.add_argument("--skip-lono", action="store_true",
                   help="Skip the 5-fold LONO pass.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else RUNS_ROOT / f"learnable_d2_sweep_{timestamp}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tier-3B learnable d2 sweep output dir: {output_dir}")

    variants = [
        (n, args_) for (n, args_) in D2_VARIANTS
        if args.only is None or n in set(args.only)
    ]
    (output_dir / "config.json").write_text(json.dumps({
        "seed": args.seed,
        "device": args.device,
        "variants": [{"name": n, "stage1_args": a} for (n, a) in variants],
        "stage3_only": STAGE3_ONLY,
    }, indent=2), encoding="utf-8")

    records: list[dict[str, Any]] = []
    for i, (name, s1_args) in enumerate(variants, start=1):
        print(f"\n========== Variant {i}/{len(variants)}: {name} ==========")
        variant_dir = output_dir / name
        rec: dict[str, Any] = {"variant": name, "stage1_args": s1_args}

        if not args.skip_in_domain:
            try:
                in_dom = run_one_in_domain(
                    variant_name=name, stage1_args=s1_args, variant_dir=variant_dir,
                    seed=args.seed, device=args.device, dry_run=args.dry_run,
                )
                rec.update(in_dom)
            except Exception as e:
                print(f"!! In-domain {name} FAILED: {e}")
                (variant_dir / "in_domain_FAILED.json").write_text(
                    json.dumps({"error": str(e)}, indent=2), encoding="utf-8"
                )

        if not args.skip_lono:
            lono_dir = variant_dir / "lono"
            try:
                per_fold = run_one_lono(
                    variant_name=name, stage1_args=s1_args, lono_dir=lono_dir,
                    seed=args.seed, device=args.device, dry_run=args.dry_run,
                )
                rec["lono_per_fold_csv"] = str(per_fold) if per_fold else None
            except Exception as e:
                print(f"!! LONO {name} FAILED: {e}")
                (lono_dir / "lono_FAILED.json").write_text(
                    json.dumps({"error": str(e)}, indent=2), encoding="utf-8"
                )
                rec["lono_per_fold_csv"] = None

        (variant_dir / "variant_summary.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8"
        )
        records.append(rec)

    (output_dir / "all_variants.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    if not args.dry_run:
        lono_df = aggregate_lono(records, output_dir)
        in_domain_df = aggregate_in_domain(records, output_dir)
        learned_df = gather_learned_lambdas(records, output_dir)
        write_verdict(lono_df, in_domain_df, learned_df, output_dir)

    print(f"\nTier-3B done. Output: {output_dir}")


if __name__ == "__main__":
    main()
