"""Recover RMSE eval for fold 1-3 of the C_family_head_5fold sweep.

These folds completed Stage-3 training but their automatic post-training RMSE
eval crashed because the student `train_config_used.json` inherited
`architecture_mode=family_head` from the teacher, while the student is a plain
PenetrationMLP. The student configs have already been patched by hand; this
script re-runs the external eval and downstream metrics, writes a proper
fold_summary.json into each fold dir, and finally regenerates per_fold.csv +
all_folds.json by re-aggregating across all 5 folds.

Run AFTER folds 4-5 have completed so the aggregation includes everyone.

Usage:
    python MLP/MLP_training/ablations/recover_failed_c_folds.py [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here.parent / "ood_lono"))
    sys.path.insert(0, str(_here.parent))

import time

from run_lono_pipeline import (
    EVAL_OUTPUT_RE,
    EVAL_SCRIPT,
    HA_POINTS_CSV,
    MLP_ROOT,
    NS_POINTS_CSV,
    aggregate,
    eval_on_uncensored_points,
    filter_points,
    metrics_from_points,
    oracle_metrics_for_holdout,
    python_exe,
    run_subprocess_streaming,
)


SWEEP_ROOT = (
    MLP_ROOT
    / "runs_mlp"
    / "family_head_sweep_20260528_212646"
    / "C_family_head_5fold"
)


FAILED_FOLDS: dict[str, dict[str, Path]] = {
    "BC20220627_HZ_Nozzle0": {
        "fold_dir": SWEEP_ROOT / "fold_01_BC20220627_HZ_Nozzle0",
        "stage1_run": MLP_ROOT / "runs_mlp" / "stage1_engineered_mse_a_only_20260528_212654",
        "stage2_run": MLP_ROOT / "runs_mlp" / "stage2_engineered_nll_no_anchor_a_only_20260528_212723",
        "anchor_run": MLP_ROOT / "runs_mlp" / "distill_cdf_onset_v2_ablate_anchor_off_20260528_212945",
    },
    "BC20241017_HZ_Nozzle2": {
        "fold_dir": SWEEP_ROOT / "fold_02_BC20241017_HZ_Nozzle2",
        "stage1_run": MLP_ROOT / "runs_mlp" / "stage1_engineered_mse_a_only_20260528_213031",
        "stage2_run": MLP_ROOT / "runs_mlp" / "stage2_engineered_nll_no_anchor_a_only_20260528_213135",
        "anchor_run": MLP_ROOT / "runs_mlp" / "distill_cdf_onset_v2_ablate_anchor_off_20260528_213608",
    },
    "BC20241003_HZ_Nozzle1": {
        "fold_dir": SWEEP_ROOT / "fold_03_BC20241003_HZ_Nozzle1",
        "stage1_run": MLP_ROOT / "runs_mlp" / "stage1_engineered_mse_a_only_20260528_213806",
        "stage2_run": MLP_ROOT / "runs_mlp" / "stage2_engineered_nll_no_anchor_a_only_20260528_213852",
        "anchor_run": MLP_ROOT / "runs_mlp" / "distill_cdf_onset_v2_ablate_anchor_off_20260528_214340",
    },
}


def recover_one(holdout: str, paths: dict[str, Path], device: str) -> dict:
    fold_dir = paths["fold_dir"]
    anchor_run = paths["anchor_run"]
    print(f"\n========== Recovering fold {holdout} ==========")

    # Sanity check: student config must have architecture_mode = single after the patch.
    cfg_path = anchor_run / "train_config_used.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if str(cfg.get("architecture_mode", "single")).lower() != "single":
        raise RuntimeError(
            f"{cfg_path} still has architecture_mode={cfg.get('architecture_mode')!r}; "
            "patch it to 'single' before running this recovery script."
        )

    record: dict = {
        "holdout": holdout,
        "stage1_run": str(paths["stage1_run"]),
        "stage2_run": str(paths["stage2_run"]),
        "stage3_suite_summary": None,
        "winner_run": str(anchor_run),
        "timing": {},
    }

    eval_log = fold_dir / "eval.log"
    eval_cmd = [
        python_exe(), str(EVAL_SCRIPT),
        "--refinement-run", str(anchor_run),
        "--split", "clean",
        "--filter-experiment", holdout,
        "--tag", f"lono_{holdout}",
        "--batch-points", "262144",
        "--no-save-plots",
        "--max-traj-plots", "0",
    ]
    t0 = time.time()
    rc, out = run_subprocess_streaming(eval_cmd, eval_log)
    if rc != 0:
        raise RuntimeError(f"External eval failed (rc={rc}). See {eval_log}.")
    match = EVAL_OUTPUT_RE.search(out)
    if match:
        eval_dir = Path(match.group(1).strip())
    else:
        eval_root = MLP_ROOT / "eval"
        candidates = sorted(
            eval_root.glob(f"rmse_eval_clean_*_lono_{holdout}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(f"Could not locate eval dir for fold {holdout}.")
        eval_dir = candidates[0]
    record["eval_dir"] = str(eval_dir)
    record["timing"]["eval_s"] = time.time() - t0

    mlp_points_csv = eval_dir / "points.csv"
    if not mlp_points_csv.exists():
        raise FileNotFoundError(f"Missing {mlp_points_csv}")
    record["mlp_metrics"] = metrics_from_points(
        filter_points(mlp_points_csv, holdout, "folder")
    )
    if HA_POINTS_CSV.exists():
        record["ha_metrics"] = metrics_from_points(
            filter_points(HA_POINTS_CSV, holdout, "experiment_name")
        )
    if NS_POINTS_CSV.exists():
        record["ns_metrics"] = metrics_from_points(
            filter_points(NS_POINTS_CSV, holdout, "experiment_name")
        )

    print(f"[fold {holdout}] Running eval on uncensored points ...", flush=True)
    record["mlp_uncensored_metrics"] = eval_on_uncensored_points(
        anchor_run, holdout, device=device
    )
    record["q1_oracle_metrics"] = oracle_metrics_for_holdout(holdout)

    (fold_dir / "fold_summary.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    failed_marker = fold_dir / "fold_FAILED.json"
    if failed_marker.exists():
        failed_marker.unlink()
        print(f"[fold {holdout}] Removed stale fold_FAILED.json marker.")

    print(
        f"[fold {holdout}] RMSE_mm = "
        f"{record['mlp_metrics'].get('rmse_mm'):.4f}, "
        f"MAE_mm = {record['mlp_metrics'].get('mae_mm'):.4f}"
    )
    return record


def reaggregate_all_folds(output_dir: Path) -> None:
    """Collect every fold_summary.json under output_dir and rebuild per_fold.csv."""
    records: list[dict] = []
    fold_dirs = sorted(output_dir.glob("fold_*"))
    for fd in fold_dirs:
        fs = fd / "fold_summary.json"
        if fs.exists():
            records.append(json.loads(fs.read_text(encoding="utf-8")))
        else:
            print(f"[warn] {fs} missing — fold not included in aggregate.")

    (output_dir / "all_folds.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    aggregate(records, output_dir)
    print(f"\n[ok] Rebuilt per_fold.csv with {len(records)} folds at {output_dir}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=None,
        help="Skip these holdouts (already recovered).",
    )
    ap.add_argument(
        "--only-aggregate",
        action="store_true",
        help="Skip eval re-runs; just rebuild per_fold.csv from existing fold_summary.json files.",
    )
    args = ap.parse_args()

    if not args.only_aggregate:
        for holdout, paths in FAILED_FOLDS.items():
            if args.skip and holdout in args.skip:
                print(f"[skip] {holdout}")
                continue
            recover_one(holdout, paths, device=args.device)

    reaggregate_all_folds(SWEEP_ROOT)


if __name__ == "__main__":
    main()
