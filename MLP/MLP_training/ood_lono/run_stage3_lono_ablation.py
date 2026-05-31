"""Stage-3 ablation LONO pipeline.

For each LONO fold:
  1. Run Stage-1 (a_dp050_plus_pressures) once.
  2. Run Stage-2 once (default --stage2-ablation no_anchor).
  3. Run the Stage-3 ablation suite via run_stage3_ablation_suite.py
     with --only filtered to the selected ablations (default = the 4
     main entries in stage3_ablation_suite_config.json).
  4. For each ablation's run_dir, evaluate on cdf_points_uncensored.csv
     filtered to the held-out nozzle.

Reports per-ablation uncensored RMSE / MAE / bias / coverage_{1,2}σ /
mean_pred_std aggregated across LONO folds (mean ± std).

Usage
-----
    python MLP/MLP_training/ood_lono/run_stage3_lono_ablation.py
    python MLP/MLP_training/ood_lono/run_stage3_lono_ablation.py --device cpu
    python MLP/MLP_training/ood_lono/run_stage3_lono_ablation.py --n-folds 2
    python MLP/MLP_training/ood_lono/run_stage3_lono_ablation.py --dry-run
    python MLP/MLP_training/ood_lono/run_stage3_lono_ablation.py \\
        --stage3-ablations baseline anchor_off
"""
from __future__ import annotations

import argparse
import json
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

if __package__ in {None, ""}:
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here))           # ood_lono/ — for run_lono_pipeline
    sys.path.insert(0, str(_here.parent))    # MLP_training/ — for engineered_feature_common

from run_lono_pipeline import (
    RUNS_ROOT,
    STAGE1_SCRIPT,
    STAGE2_SCRIPT,
    STAGE3_SUITE_CONFIG,
    STAGE3_SUITE_SCRIPT,
    eval_on_uncensored_points,
    identify_nozzle_families,
    parse_saved_run_dir,
    parse_suite_summary_path,
    python_exe,
    run_subprocess_streaming,
)
from engineered_feature_common import DEFAULT_STAGE1_CONFIG, build_dataset_registry

STAGE1_VARIANT = "a_dp050_plus_pressures"
DEFAULT_STAGE2_ABLATION = "no_anchor"
DEFAULT_STAGE3_ABLATIONS = [
    "baseline",
    "raw_reliable_no_kd",
    "raw_uncertain_raw05_kd025",
    "anchor_off",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=str, default=DEFAULT_STAGE1_CONFIG["data_dir"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-groups", type=int, default=50)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--skip-folds", type=str, nargs="*", default=None)
    p.add_argument(
        "--stage2-ablation",
        choices=("no_anchor", "mu_anchor", "mu_sigma_anchor"),
        default=DEFAULT_STAGE2_ABLATION,
        help="Stage-2 teacher ablation used as KD source. Default: no_anchor.",
    )
    p.add_argument(
        "--stage3-ablations",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Subset of Stage-3 ablation names to run (must match entries in "
            "stage3_ablation_suite_config.json). Default: 4 main ablations."
        ),
    )
    p.add_argument(
        "--stage3-config",
        type=Path,
        default=STAGE3_SUITE_CONFIG,
        help="Stage-3 ablation suite config path. Default: stage3_ablation_suite_config.json.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _run_stage1(*, holdout: str, fold_dir: Path, seed: int, device: str, dry_run: bool) -> Path:
    log = fold_dir / "stage1.log"
    cmd = [
        python_exe(), str(STAGE1_SCRIPT),
        "--variant", STAGE1_VARIANT,
        "--device", device,
        "--seed", str(seed),
        "--lono-holdout", holdout,
        "--allow-failed-precheck",
    ]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return Path("DRY_RUN/stage1")
    rc, out = run_subprocess_streaming(cmd, log)
    if rc != 0:
        raise RuntimeError(f"Stage-1 failed (rc={rc}). See {log}.")
    run_dir = parse_saved_run_dir(out)
    if run_dir is None:
        raise RuntimeError("Could not parse Stage-1 run_dir.")
    return run_dir


def _run_stage2(
    *, stage1_run: Path, stage2_ablation: str, holdout: str,
    fold_dir: Path, seed: int, device: str, dry_run: bool,
) -> Path:
    log = fold_dir / "stage2.log"
    cmd = [
        python_exe(), str(STAGE2_SCRIPT),
        str(stage1_run),
        "--stage2-ablation", stage2_ablation,
        "--device", device,
        "--seed", str(seed),
        "--lono-holdout", holdout,
        "--allow-failed-precheck",
    ]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return Path(f"DRY_RUN/stage2_{stage2_ablation}")
    rc, out = run_subprocess_streaming(cmd, log)
    if rc != 0:
        raise RuntimeError(f"Stage-2 ({stage2_ablation}) failed (rc={rc}). See {log}.")
    run_dir = parse_saved_run_dir(out)
    if run_dir is None:
        raise RuntimeError(f"Could not parse Stage-2 ({stage2_ablation}) run_dir.")
    return run_dir


def _run_stage3_suite(
    *, stage2_run: Path, ablations: list[str], holdout: str,
    fold_dir: Path, config: Path, seed: int, device: str, dry_run: bool,
) -> dict[str, Path]:
    """Return {ablation_name: run_dir} from the suite's results."""
    log = fold_dir / "stage3.log"
    cmd = [
        python_exe(), str(STAGE3_SUITE_SCRIPT),
        "--config", str(config),
        "--teacher-run", str(stage2_run),
        "--device", device,
        "--seed", str(seed),
        "--only", *ablations,
        "--lono-holdout", holdout,
    ]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return {name: Path(f"DRY_RUN/stage3_{name}") for name in ablations}

    rc, out = run_subprocess_streaming(cmd, log)
    if rc != 0:
        raise RuntimeError(f"Stage-3 suite failed (rc={rc}). See {log}.")
    suite_summary = parse_suite_summary_path(out)
    if suite_summary is None:
        raise RuntimeError("Could not parse suite_summary path from Stage-3 suite output.")
    summary = json.loads(suite_summary.read_text(encoding="utf-8"))
    by_name: dict[str, Path] = {}
    for entry in summary.get("results", []):
        name = str(entry.get("name"))
        run_dir = entry.get("run_dir")
        if name in ablations and run_dir:
            by_name[name] = Path(run_dir)
    missing = sorted(set(ablations) - set(by_name))
    if missing:
        raise RuntimeError(
            f"Stage-3 suite summary missing run_dir for ablations: {missing} "
            f"(suite_summary={suite_summary})."
        )
    return by_name


def run_one_fold(
    *,
    holdout: str,
    fold_dir: Path,
    seed: int,
    device: str,
    dry_run: bool,
    stage2_ablation: str,
    stage3_ablations: list[str],
    stage3_config: Path,
) -> dict[str, Any]:
    fold_dir.mkdir(parents=True, exist_ok=True)

    timing: dict[str, float] = {}
    t0 = time.time()
    stage1_run = _run_stage1(
        holdout=holdout, fold_dir=fold_dir, seed=seed, device=device, dry_run=dry_run,
    )
    timing["stage1_s"] = time.time() - t0

    t0 = time.time()
    stage2_run = _run_stage2(
        stage1_run=stage1_run, stage2_ablation=stage2_ablation, holdout=holdout,
        fold_dir=fold_dir, seed=seed, device=device, dry_run=dry_run,
    )
    timing["stage2_s"] = time.time() - t0

    t0 = time.time()
    stage3_runs = _run_stage3_suite(
        stage2_run=stage2_run, ablations=stage3_ablations, holdout=holdout,
        fold_dir=fold_dir, config=stage3_config,
        seed=seed, device=device, dry_run=dry_run,
    )
    timing["stage3_suite_s"] = time.time() - t0

    ablation_records: dict[str, Any] = {}
    for ablation in stage3_ablations:
        run_dir = stage3_runs[ablation]
        print(f"\n--- [{holdout}] Stage-3 ablation {ablation} eval ---", flush=True)
        t0 = time.time()
        if dry_run:
            metrics: dict[str, float] = {}
        else:
            metrics = eval_on_uncensored_points(run_dir, holdout, device=device)
        eval_s = time.time() - t0
        rec: dict[str, Any] = {
            "stage3_run": str(run_dir),
            "uncensored_metrics": metrics,
            "eval_s": eval_s,
        }
        (fold_dir / f"{ablation}_summary.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8"
        )
        ablation_records[ablation] = rec

    record: dict[str, Any] = {
        "holdout": holdout,
        "stage1_run": str(stage1_run),
        "stage2_run": str(stage2_run),
        "stage2_ablation": stage2_ablation,
        "timing": timing,
        "ablations": ablation_records,
    }
    (fold_dir / "fold_summary.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def aggregate_and_report(
    records: list[dict[str, Any]],
    output_dir: Path,
    ablations: list[str],
) -> None:
    metric_keys = [
        "rmse_mm", "mae_mm", "bias_mm",
        "coverage_1sigma", "coverage_2sigma", "mean_pred_std_mm",
    ]
    rows: list[dict[str, Any]] = []
    for rec in records:
        holdout = rec["holdout"]
        for ablation, abl_rec in rec.get("ablations", {}).items():
            m = abl_rec.get("uncensored_metrics", {})
            if not m:
                continue
            row: dict[str, Any] = {"holdout": holdout, "ablation": ablation}
            for k in metric_keys + ["n_points"]:
                row[k] = m.get(k)
            rows.append(row)

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(output_dir / "per_fold.csv", index=False)

    agg_rows: list[dict[str, Any]] = []
    for ablation in ablations:
        sub = per_fold[per_fold["ablation"] == ablation]
        if sub.empty:
            continue
        agg: dict[str, Any] = {"ablation": ablation, "n_folds": int(len(sub))}
        for k in metric_keys:
            vals = pd.to_numeric(sub[k], errors="coerce").dropna().to_numpy()
            agg[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            agg[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(agg)
    pd.DataFrame(agg_rows).to_csv(output_dir / "aggregate.csv", index=False)

    def _fmt(v: Any) -> str:
        return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "—"

    md = ["# Stage-3 Ablation LONO — Uncensored-point Metrics (mean ± std across folds)\n"]
    md.append("| ablation | rmse_mm | mae_mm | bias_mm | cov_1σ | cov_2σ | mean_σ_mm |")
    md.append("|---|---|---|---|---|---|---|")
    for agg in agg_rows:
        md.append(
            f"| {agg['ablation']} | "
            f"{_fmt(agg['rmse_mm_mean'])} ± {_fmt(agg['rmse_mm_std'])} | "
            f"{_fmt(agg['mae_mm_mean'])} ± {_fmt(agg['mae_mm_std'])} | "
            f"{_fmt(agg['bias_mm_mean'])} ± {_fmt(agg['bias_mm_std'])} | "
            f"{_fmt(agg['coverage_1sigma_mean'])} ± {_fmt(agg['coverage_1sigma_std'])} | "
            f"{_fmt(agg['coverage_2sigma_mean'])} ± {_fmt(agg['coverage_2sigma_std'])} | "
            f"{_fmt(agg['mean_pred_std_mm_mean'])} ± {_fmt(agg['mean_pred_std_mm_std'])} |"
        )

    md.append("\n## Per-fold breakdown\n")
    md.append("| holdout | ablation | rmse_mm | mae_mm | bias_mm | cov_1σ | cov_2σ |")
    md.append("|---|---|---|---|---|---|---|")
    for _, r in per_fold.sort_values(["holdout", "ablation"]).iterrows():
        md.append(
            f"| {r['holdout']} | {r['ablation']} | "
            f"{_fmt(r['rmse_mm'])} | {_fmt(r['mae_mm'])} | {_fmt(r['bias_mm'])} | "
            f"{_fmt(r.get('coverage_1sigma', float('nan')))} | "
            f"{_fmt(r.get('coverage_2sigma', float('nan')))} |"
        )

    (output_dir / "headline_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n=== Aggregate ===\n")
    print((output_dir / "headline_comparison.md").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    stage3_ablations = args.stage3_ablations or DEFAULT_STAGE3_ABLATIONS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else RUNS_ROOT / f"stage3_lono_ablation_{timestamp}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Stage-3 ablation LONO output: {output_dir}")

    registry = build_dataset_registry()
    nozzles = identify_nozzle_families(
        data_dir=args.data_dir, registry=registry, min_groups=args.min_groups,
    )
    if args.skip_folds:
        nozzles = [n for n in nozzles if n not in set(args.skip_folds)]
    if args.n_folds is not None and args.n_folds < len(nozzles):
        nozzles = nozzles[: args.n_folds]
    print(f"\n{len(nozzles)} folds × {len(stage3_ablations)} ablations")
    print(f"Folds:            {nozzles}")
    print(f"Stage-2 ablation: {args.stage2_ablation}")
    print(f"Stage-3 ablations:{stage3_ablations}")
    print(f"Stage-3 config:   {args.stage3_config}\n")

    (output_dir / "run_config.json").write_text(
        json.dumps({
            "seed": args.seed,
            "device": args.device,
            "stage1_variant": STAGE1_VARIANT,
            "stage2_ablation": args.stage2_ablation,
            "stage3_ablations": stage3_ablations,
            "stage3_config": str(args.stage3_config),
            "nozzles": nozzles,
        }, indent=2),
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    for i, holdout in enumerate(nozzles, start=1):
        print(f"\n========== Fold {i}/{len(nozzles)}: holdout={holdout} ==========")
        fold_dir = output_dir / f"fold_{i:02d}_{holdout}"
        try:
            rec = run_one_fold(
                holdout=holdout,
                fold_dir=fold_dir,
                seed=args.seed,
                device=args.device,
                dry_run=args.dry_run,
                stage2_ablation=args.stage2_ablation,
                stage3_ablations=stage3_ablations,
                stage3_config=args.stage3_config,
            )
            records.append(rec)
        except Exception as exc:
            print(f"!! Fold {holdout} FAILED: {exc}")
            (fold_dir / "fold_FAILED.json").write_text(
                json.dumps({"error": str(exc)}, indent=2), encoding="utf-8"
            )

    (output_dir / "all_folds.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    if records and not args.dry_run:
        aggregate_and_report(records, output_dir, stage3_ablations)
    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
