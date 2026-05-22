"""Leave-One-Nozzle-Out (LONO) validation pipeline.

For each unique experiment_name (nozzle), train Stage-1 + Stage-2 +
Stage-3 anchor_off variant with that nozzle held out. Evaluate the held-out
nozzle's data against HA / NS baselines on the same (filtered) point set.

Wall clock: ~13 min/fold on RTX 5090. 5 folds ≈ 65 min total.
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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from engineered_feature_common import (
    DEFAULT_STAGE1_CONFIG,
    build_all_stage_tables,
    build_dataset_registry,
    build_feature_matrix_np,
    infer_feature_family,
    load_run_artifacts,
    split_mu_logvar,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLP_ROOT = PROJECT_ROOT / "MLP"
TRAINING_ROOT = MLP_ROOT / "MLP_training"
RUNS_ROOT = MLP_ROOT / "runs_mlp"
EVAL_SCRIPT = MLP_ROOT / "eval" / "inference_rmse_on_series.py"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

STAGE1_SCRIPT = TRAINING_ROOT / "train_stage1_mse.py"
STAGE2_SCRIPT = TRAINING_ROOT / "train_stage2_nll.py"
STAGE3_SUITE_SCRIPT = TRAINING_ROOT / "run_stage3_ablation_suite.py"
STAGE3_SUITE_CONFIG = TRAINING_ROOT / "stage3_ablation_suite_config.json"

HA_POINTS_CSV = (
    PROJECT_ROOT
    / "MLP" / "baseline" / "Hiroyasu_Arai" / "outputs"
    / "20260509_020253_ha_calibrated_grouped_condition_all_all_clean_diagnostic_20260509_review"
    / "points.csv"
)
NS_POINTS_CSV = (
    PROJECT_ROOT
    / "MLP" / "baseline" / "Naber_Siebers" / "outputs"
    / "20260509_004452_ns_delay_grouped_condition_all_clean_diagnostic_20260509"
    / "points.csv"
)

UNCENSORED_POINTS_CSV = (
    PROJECT_ROOT / "MLP" / "synthetic_data" / "cdf_right_censoring_points" / "cdf_points_uncensored.csv"
)
Q1_ORACLE_CSV = (
    PROJECT_ROOT / "MLP" / "synthetic_data" / "p50_q1_oracle" / "p50_q1_fit_metrics.csv"
)

SAVED_RUN_DIR_RE = re.compile(r"Saved run_dir:\s*(.+)$")
SUITE_SUMMARY_RE = re.compile(r"Suite summary saved to:\s*(.+)$")
EVAL_OUTPUT_RE = re.compile(r"Wrote RMSE evaluation to\s+(.+)$", re.MULTILINE)


def python_exe() -> str:
    return str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))


def run_subprocess_streaming(cmd: list[str], log_path: Path) -> tuple[int, str]:
    """Run cmd, stream stdout/stderr to console + log, return (rc, captured_text)."""
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
        rc = int(proc.returncode)
    return rc, "".join(captured)


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


def identify_nozzle_families(*, data_dir: str, registry, min_groups: int = 50) -> list[str]:
    """Build representative table once and group by experiment_name."""
    print("Building canonical feature table to enumerate nozzle families ...")
    stage_tables = build_all_stage_tables(
        data_dir, registry,
        comparison_time_s=float(DEFAULT_STAGE1_CONFIG["comparison_time_s"]),
        max_curves=None,
        output_dir=None,  # do NOT write the precheck plot here
    )
    rep = stage_tables.representative
    holdout_col = "experiment_name" if "experiment_name" in rep.columns else "source_dataset_name"
    if holdout_col not in rep.columns:
        raise KeyError("representative table missing experiment_name/source_dataset_name; required for LONO.")
    counts = rep.groupby(holdout_col)["sample_group_id"].nunique().sort_values(ascending=False)
    print("\nPer-nozzle sample_group_id counts:")
    print(counts.to_string())
    valid = counts[counts >= min_groups]
    print(f"\nKeeping {len(valid)} nozzles with >= {min_groups} sample_group_ids")
    return [str(name) for name in valid.index]


def find_named_run_dir(suite_summary_path: Path, name: str) -> Path:
    """Pull the named variant's run_dir from the suite summary JSON."""
    summary = json.loads(suite_summary_path.read_text(encoding="utf-8"))
    for entry in summary.get("results", []):
        if str(entry.get("name")) == name:
            run_dir = entry.get("run_dir")
            if run_dir:
                return Path(run_dir)
    # Fallback: selection.best
    best = (summary.get("selection") or {}).get("best") or {}
    if str(best.get("name")) == name and best.get("run_dir"):
        return Path(best["run_dir"])
    raise RuntimeError(f"Could not locate {name!r} run_dir in {suite_summary_path}")


def metrics_from_points(df: pd.DataFrame) -> dict[str, float]:
    truth = df["pen_true_mm"].to_numpy(dtype=float)
    pred = df["pen_pred_mm"].to_numpy(dtype=float)
    std = df["pen_std_mm"].to_numpy(dtype=float)
    resid = pred - truth
    abs_err = np.abs(resid)
    std_safe = np.maximum(std, 1e-12)
    return {
        "n_points": int(len(df)),
        "rmse_mm": float(np.sqrt(np.mean(resid ** 2))) if len(df) else float("nan"),
        "mae_mm": float(np.mean(abs_err)) if len(df) else float("nan"),
        "bias_mm": float(np.mean(resid)) if len(df) else float("nan"),
        "p95_abs_err_mm": float(np.quantile(abs_err, 0.95)) if len(df) else float("nan"),
        "coverage_1sigma": float(np.mean(abs_err <= std_safe)) if len(df) else float("nan"),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * std_safe)) if len(df) else float("nan"),
        "mean_pred_std_mm": float(np.mean(std)) if len(df) else float("nan"),
    }


def filter_points(points_csv: Path, holdout: str, experiment_col: str) -> pd.DataFrame:
    df = pd.read_csv(points_csv, low_memory=False)
    if experiment_col not in df.columns:
        raise KeyError(f"{points_csv} missing column {experiment_col!r}")
    return df.loc[df[experiment_col].astype(str) == str(holdout)].copy()


def _predict_points_batched(
    *,
    artifacts,
    features: np.ndarray,
    a_scale: np.ndarray,
    batch_size: int = 262144,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(artifacts.model.parameters()).device
    family = infer_feature_family(artifacts.train_config["feature_columns"])
    mu_chunks, std_chunks = [], []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            feat_t = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            scale_t = torch.as_tensor(a_scale[start:start + batch_size, None], dtype=torch.float32, device=device)
            out = artifacts.model(feat_t)
            mu_hat, log_var_hat = split_mu_logvar(out)
            log_var_hat = torch.clamp(log_var_hat, min=-20.0, max=20.0)
            if family == "engineered_v2":
                mu = scale_t * mu_hat
                std = scale_t * torch.exp(0.5 * log_var_hat)
            else:
                mu = mu_hat
                std = torch.exp(0.5 * log_var_hat)
            std_floor = float(artifacts.train_config.get("std_clamp_min", 0.0))
            std = torch.clamp(std, min=std_floor)
            mu_chunks.append(mu.detach().cpu().numpy().reshape(-1))
            std_chunks.append(std.detach().cpu().numpy().reshape(-1))
    return np.concatenate(mu_chunks), np.concatenate(std_chunks)


def eval_on_uncensored_points(
    anchor_run: Path,
    holdout: str,
    *,
    uncensored_csv: Path = UNCENSORED_POINTS_CSV,
    device: str = "cpu",
    batch_size: int = 262144,
) -> dict[str, float]:
    """Run MLP inference on right-censored point dataset, held-out nozzle only."""
    if not uncensored_csv.exists():
        print(f"[warn] uncensored points CSV not found: {uncensored_csv}")
        return {}
    df = pd.read_csv(uncensored_csv, low_memory=False)
    df = df[df["experiment_name"].astype(str) == str(holdout)].reset_index(drop=True)
    if len(df) == 0:
        print(f"[warn] No uncensored points for holdout={holdout!r}")
        return {}

    artifacts = load_run_artifacts(anchor_run, device=device)
    registry = build_dataset_registry()
    feature_columns = list(artifacts.train_config["feature_columns"])
    time_feature = str(artifacts.train_config.get("time_feature", "time_norm_0_5ms"))

    feat_blocks, scale_blocks, truth_blocks = [], [], []
    for _, grp in df.groupby("condition_id", sort=False):
        row0 = grp.iloc[0]
        raw = {
            "umbrella_angle_deg": float(row0["umbrella_angle_deg"]),
            "plumes": float(row0["plumes"]),
            "diameter_mm": float(row0["diameter_mm"]),
            "injection_duration_us": float(row0["injection_duration_us"]),
            "injection_pressure_bar": float(row0["injection_pressure_bar"]),
            "control_backpressure_bar": float(row0["control_backpressure_bar"]),
            "chamber_pressure_bar": float(row0["chamber_pressure_bar"]),
            "dataset_key": str(row0["experiment_name"]),
        }
        time_ms = grp["time_ms"].to_numpy(dtype=np.float32)
        try:
            features_np, a_scale_np, _ = build_feature_matrix_np(
                raw, time_ms, artifacts.scaler_state, feature_columns, registry,
                time_feature=time_feature,
            )
        except Exception as exc:
            print(f"[warn] feature build failed for condition {row0.get('condition_id')}: {exc}")
            continue
        feat_blocks.append(features_np)
        scale_blocks.append(a_scale_np.reshape(-1))
        truth_blocks.append(grp["penetration_mm"].to_numpy(dtype=np.float32))

    if not feat_blocks:
        return {}

    features_all = np.concatenate(feat_blocks)
    a_scale_all = np.concatenate(scale_blocks)
    truth_all = np.concatenate(truth_blocks)

    pred_all, std_all = _predict_points_batched(
        artifacts=artifacts,
        features=features_all,
        a_scale=a_scale_all,
        batch_size=batch_size,
    )
    pts_df = pd.DataFrame({
        "pen_pred_mm": pred_all,
        "pen_true_mm": truth_all.astype(float),
        "pen_std_mm": std_all,
    })
    return metrics_from_points(pts_df)


def oracle_metrics_for_holdout(
    holdout: str,
    *,
    oracle_csv: Path = Q1_ORACLE_CSV,
    uncensored_csv: Path = UNCENSORED_POINTS_CSV,
) -> dict[str, float]:
    """Per-condition q1 oracle RMSE/MAE/bias averaged over held-out nozzle conditions."""
    if not oracle_csv.exists() or not uncensored_csv.exists():
        return {}
    fm = pd.read_csv(oracle_csv)
    cond_to_exp = (
        pd.read_csv(uncensored_csv, low_memory=False, usecols=["condition_id", "experiment_name"])
        .drop_duplicates()
    )
    fm = fm.merge(cond_to_exp, on="condition_id", how="left")
    sub = fm[fm["experiment_name"].astype(str) == str(holdout)]
    if sub.empty:
        return {}
    return {
        "n_conditions": int(len(sub)),
        "rmse_mm": float(sub["rmse"].mean()),
        "mae_mm": float(sub["mae"].mean()),
        "bias_mm": float(sub["bias"].mean()),
        "rmse_mm_median": float(sub["rmse"].median()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=str, default=DEFAULT_STAGE1_CONFIG["data_dir"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Limit to N largest nozzles (default: 5, matching the planned main LONO sweep).")
    p.add_argument("--min-groups", type=int, default=50)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--skip-folds", type=str, nargs="*", default=None,
                   help="Skip these experiment_name values (e.g. already-running fold).")
    p.add_argument("--stage3-config", type=Path, default=None,
                   help="Override Stage-3 suite config (default: stage3_ablation_suite_config.json).")
    p.add_argument("--stage3-only", type=str, default="anchor_off",
                   help="Stage-3 suite ablation name to keep (default: anchor_off).")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_one_fold(
    *,
    holdout: str,
    fold_dir: Path,
    seed: int,
    device: str,
    dry_run: bool,
    stage3_config: Path,
    stage3_only: str,
) -> dict[str, Any]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    # ── Stage 1 ──
    s1_log = fold_dir / "stage1.log"
    s1_cmd = [
        python_exe(), str(STAGE1_SCRIPT),
        "--variant", "a_only",
        "--device", device,
        "--seed", str(seed),
        "--lono-holdout", holdout,
        "--allow-failed-precheck",
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s1_cmd)}")
        stage1_run = Path("DRY_RUN/stage1")
    else:
        rc, out = run_subprocess_streaming(s1_cmd, s1_log)
        if rc != 0:
            raise RuntimeError(f"Stage-1 failed (rc={rc}). See {s1_log}.")
        stage1_run = parse_saved_run_dir(out)
        if stage1_run is None:
            raise RuntimeError("Could not parse Stage-1 run_dir.")
    timing["stage1_s"] = time.time() - t0

    # ── Stage 2 ──
    s2_log = fold_dir / "stage2.log"
    s2_cmd = [
        python_exe(), str(STAGE2_SCRIPT),
        str(stage1_run),
        "--device", device,
        "--seed", str(seed),
        "--lono-holdout", holdout,
        "--allow-failed-precheck",
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s2_cmd)}")
        stage2_run = Path("DRY_RUN/stage2")
    else:
        rc, out = run_subprocess_streaming(s2_cmd, s2_log)
        if rc != 0:
            raise RuntimeError(f"Stage-2 failed (rc={rc}). See {s2_log}.")
        stage2_run = parse_saved_run_dir(out)
        if stage2_run is None:
            raise RuntimeError("Could not parse Stage-2 run_dir.")
    timing["stage2_s"] = time.time() - t0

    # ── Stage 3 (anchor_off only) ──
    s3_log = fold_dir / "stage3.log"
    s3_cmd = [
        python_exe(), str(STAGE3_SUITE_SCRIPT),
        "--config", str(stage3_config),
        "--teacher-run", str(stage2_run),
        "--device", device,
        "--seed", str(seed),
        "--only", stage3_only,
        "--lono-holdout", holdout,
    ]
    t0 = time.time()
    if dry_run:
        print(f"[dry-run] {' '.join(s3_cmd)}")
        suite_summary = Path("DRY_RUN/suite_summary.json")
        anchor_run = Path(f"DRY_RUN/{stage3_only}")
    else:
        rc, out = run_subprocess_streaming(s3_cmd, s3_log)
        if rc != 0:
            raise RuntimeError(f"Stage-3 suite failed (rc={rc}). See {s3_log}.")
        suite_summary = parse_suite_summary_path(out)
        if suite_summary is None:
            raise RuntimeError("Could not parse suite_summary path.")
        anchor_run = find_named_run_dir(suite_summary, stage3_only)
    timing["stage3_s"] = time.time() - t0

    record: dict[str, Any] = {
        "holdout": holdout,
        "stage1_run": str(stage1_run),
        "stage2_run": str(stage2_run),
        "stage3_suite_summary": str(suite_summary),
        "winner_run": str(anchor_run),
        "timing": timing,
    }

    # ── External eval (n=795k, save_points=True) ──
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
    if dry_run:
        print(f"[dry-run] {' '.join(eval_cmd)}")
        eval_dir = Path(f"DRY_RUN/eval_{holdout}")
    else:
        rc, out = run_subprocess_streaming(eval_cmd, eval_log)
        if rc != 0:
            raise RuntimeError(f"External eval failed (rc={rc}). See {eval_log}.")
        match = EVAL_OUTPUT_RE.search(out)
        if match:
            eval_dir = Path(match.group(1).strip())
        else:
            # Fallback: glob the eval root for the latest dir matching tag.
            eval_root = MLP_ROOT / "eval"
            candidates = sorted(eval_root.glob(f"rmse_eval_clean_*_lono_{holdout}"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise RuntimeError(f"Could not locate eval dir for fold {holdout}.")
            eval_dir = candidates[0]
    record["eval_dir"] = str(eval_dir)
    record["timing"]["eval_s"] = time.time() - t0

    # ── Per-fold MLP metrics on held-out subset ──
    if not dry_run:
        mlp_points_csv = eval_dir / "points.csv"
        if not mlp_points_csv.exists():
            raise FileNotFoundError(f"Missing {mlp_points_csv}")
        mlp_pts = filter_points(mlp_points_csv, holdout, "folder")
        record["mlp_metrics"] = metrics_from_points(mlp_pts)
        # HA / NS on the same held-out subset
        if HA_POINTS_CSV.exists():
            record["ha_metrics"] = metrics_from_points(
                filter_points(HA_POINTS_CSV, holdout, "experiment_name"))
        if NS_POINTS_CSV.exists():
            record["ns_metrics"] = metrics_from_points(
                filter_points(NS_POINTS_CSV, holdout, "experiment_name"))

        # ── Uncensored-point eval (raw ground truth, held-out nozzle) ──
        print(f"[fold {holdout}] Running eval on uncensored points ...", flush=True)
        record["mlp_uncensored_metrics"] = eval_on_uncensored_points(
            anchor_run, holdout, device=device)

        # ── q1 oracle baseline (per-condition fit on p50) ──
        record["q1_oracle_metrics"] = oracle_metrics_for_holdout(holdout)
    else:
        record["mlp_metrics"] = {}
        record["ha_metrics"] = {}
        record["ns_metrics"] = {}
        record["mlp_uncensored_metrics"] = {}
        record["q1_oracle_metrics"] = {}

    (fold_dir / "fold_summary.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def aggregate(records: list[dict[str, Any]], output_dir: Path) -> None:
    metric_keys = ["rmse_mm", "mae_mm", "bias_mm", "p95_abs_err_mm",
                   "coverage_1sigma", "coverage_2sigma", "mean_pred_std_mm"]
    rows = []
    for rec in records:
        for model_name, key in [
            ("MLP_series", "mlp_metrics"),
            ("MLP_uncensored", "mlp_uncensored_metrics"),
            ("HA", "ha_metrics"),
            ("NS", "ns_metrics"),
        ]:
            m = rec.get(key, {})
            if not m:
                continue
            row = {"holdout": rec["holdout"], "model": model_name}
            for k in metric_keys + ["n_points"]:
                row[k] = m.get(k)
            rows.append(row)
        # q1 oracle: only has rmse_mm / mae_mm / bias_mm / n_conditions
        q1 = rec.get("q1_oracle_metrics", {})
        if q1:
            row = {"holdout": rec["holdout"], "model": "q1_oracle_p50"}
            row["rmse_mm"] = q1.get("rmse_mm")
            row["mae_mm"] = q1.get("mae_mm")
            row["bias_mm"] = q1.get("bias_mm")
            row["n_points"] = q1.get("n_conditions")
            rows.append(row)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(output_dir / "per_fold.csv", index=False)

    # Aggregate: per-model fold mean ± std
    agg_rows = []
    for model_name in ["MLP_series", "MLP_uncensored", "HA", "NS", "q1_oracle_p50"]:
        sub = per_fold.loc[per_fold["model"] == model_name]
        if sub.empty:
            continue
        agg = {"model": model_name, "n_folds": len(sub)}
        for k in metric_keys:
            vals = pd.to_numeric(sub[k], errors="coerce").dropna().to_numpy()
            agg[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            agg[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(agg)
    pd.DataFrame(agg_rows).to_csv(output_dir / "aggregate.csv", index=False)

    # Markdown headline
    md = ["# LONO 5-fold result (mean ± std across folds)\n"]
    md.append("| model | rmse_mm | mae_mm | bias_mm | p95_mm | cov_1σ | cov_2σ |")
    md.append("|---|---|---|---|---|---|---|")
    def _fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else "—"

    for agg in agg_rows:
        md.append(
            f"| {agg['model']} | "
            f"{_fmt(agg['rmse_mm_mean'])} ± {_fmt(agg['rmse_mm_std'])} | "
            f"{_fmt(agg['mae_mm_mean'])} ± {_fmt(agg['mae_mm_std'])} | "
            f"{_fmt(agg['bias_mm_mean'])} ± {_fmt(agg['bias_mm_std'])} | "
            f"{_fmt(agg['p95_abs_err_mm_mean'])} ± {_fmt(agg['p95_abs_err_mm_std'])} | "
            f"{_fmt(agg['coverage_1sigma_mean'])} ± {_fmt(agg['coverage_1sigma_std'])} | "
            f"{_fmt(agg['coverage_2sigma_mean'])} ± {_fmt(agg['coverage_2sigma_std'])} |"
        )
    md.append("\n## Per fold (held-out nozzle)\n")
    md.append("| holdout | model | rmse_mm | mae_mm | bias_mm | p95_mm | cov_1σ | cov_2σ |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in per_fold.iterrows():
        md.append(
            f"| {r['holdout']} | {r['model']} | "
            f"{_fmt(r['rmse_mm'])} | {_fmt(r['mae_mm'])} | {_fmt(r['bias_mm'])} | "
            f"{_fmt(r.get('p95_abs_err_mm', float('nan')))} | "
            f"{_fmt(r.get('coverage_1sigma', float('nan')))} | "
            f"{_fmt(r.get('coverage_2sigma', float('nan')))} |"
        )
    (output_dir / "headline_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n=== Aggregate written ===\n")
    print((output_dir / "headline_comparison.md").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else RUNS_ROOT / f"lono_pipeline_{timestamp}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"LONO output directory: {output_dir}")

    registry = build_dataset_registry()
    nozzles = identify_nozzle_families(
        data_dir=args.data_dir, registry=registry, min_groups=args.min_groups,
    )
    if args.skip_folds:
        nozzles = [n for n in nozzles if n not in set(args.skip_folds)]
    if args.n_folds is not None and args.n_folds < len(nozzles):
        nozzles = nozzles[: args.n_folds]
    print(f"\nWill run {len(nozzles)} folds: {nozzles}\n")

    (output_dir / "lono_config.json").write_text(json.dumps({
        "seed": args.seed,
        "device": args.device,
        "min_groups": args.min_groups,
        "n_folds": len(nozzles),
        "nozzles": nozzles,
        "skip_folds": args.skip_folds,
        "stage3_config": str(args.stage3_config) if args.stage3_config else None,
        "stage3_only": str(args.stage3_only),
    }, indent=2), encoding="utf-8")

    stage3_config = args.stage3_config if args.stage3_config is not None else STAGE3_SUITE_CONFIG
    stage3_only = str(args.stage3_only)
    print(f"Stage-3 suite config: {stage3_config}")
    print(f"Stage-3 ablation name: {stage3_only}")

    records: list[dict[str, Any]] = []
    for i, holdout in enumerate(nozzles, start=1):
        print(f"\n========== Fold {i}/{len(nozzles)}: holdout={holdout} ==========")
        fold_dir = output_dir / f"fold_{i:02d}_{holdout}"
        try:
            rec = run_one_fold(
                holdout=holdout, fold_dir=fold_dir,
                seed=args.seed, device=args.device, dry_run=args.dry_run,
                stage3_config=stage3_config, stage3_only=stage3_only,
            )
            records.append(rec)
        except Exception as e:
            print(f"!! Fold {holdout} FAILED: {e}")
            (fold_dir / "fold_FAILED.json").write_text(
                json.dumps({"error": str(e)}, indent=2), encoding="utf-8"
            )

    (output_dir / "all_folds.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    if records and not args.dry_run:
        aggregate(records, output_dir)
    print(f"\nLONO pipeline complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
