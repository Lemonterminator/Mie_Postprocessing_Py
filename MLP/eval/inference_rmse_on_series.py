"""Evaluate a Stage-3 refinement run against exported penetration series.

This module is intentionally importable by
``MLP.training.train_stage3_distillation_plus_raw_series`` after training and
also runnable as a standalone script.  It reconstructs the same point-level
RMSE/coverage audit used by the thesis figures from the saved refinement run
and the ``MLP/synthetic_data`` CDF wide-series exports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.MLP_training.engineered_feature_common import (
    build_dataset_registry,
    build_feature_matrix_np,
    infer_feature_family,
    load_run_artifacts,
    normalize_dataset_key,
    split_mu_logvar,
)
from MLP.MLP_training.train_stage3_distillation_plus_raw_series import (
    build_teacher_raw_dict,
    extract_prefixed_matrix,
    load_source_table,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "MLP" / "eval"


def _finite_metrics(truth: np.ndarray, pred: np.ndarray, std: np.ndarray) -> dict[str, float | int]:
    resid = pred - truth
    abs_err = np.abs(resid)
    std_safe = np.maximum(std, 1e-12)
    return {
        "n_points": int(truth.size),
        "rmse_mm": float(np.sqrt(np.mean(resid**2))) if truth.size else float("nan"),
        "mae_mm": float(np.mean(abs_err)) if truth.size else float("nan"),
        "bias_mm": float(np.mean(resid)) if truth.size else float("nan"),
        "median_abs_err_mm": float(np.median(abs_err)) if truth.size else float("nan"),
        "p90_abs_err_mm": float(np.quantile(abs_err, 0.90)) if truth.size else float("nan"),
        "p95_abs_err_mm": float(np.quantile(abs_err, 0.95)) if truth.size else float("nan"),
        "coverage_1sigma": float(np.mean(abs_err <= std_safe)) if truth.size else float("nan"),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * std_safe)) if truth.size else float("nan"),
        "mean_pred_std_mm": float(np.mean(std)) if truth.size else float("nan"),
    }


def _predict_points(
    *,
    artifacts,
    features: np.ndarray,
    a_scale: np.ndarray,
    batch_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(artifacts.model.parameters()).device
    family = infer_feature_family(artifacts.train_config["feature_columns"])
    mu_chunks: list[np.ndarray] = []
    std_chunks: list[np.ndarray] = []

    n = len(features)
    batch_points = max(int(batch_points), 1)
    with torch.no_grad():
        for start in range(0, n, batch_points):
            stop = min(start + batch_points, n)
            feat_t = torch.as_tensor(features[start:stop], dtype=torch.float32, device=device)
            scale_t = torch.as_tensor(a_scale[start:stop, None], dtype=torch.float32, device=device)
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


def _plot_pred_vs_actual(points_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    sample = points_df
    if len(sample) > 80000:
        sample = sample.sample(80000, random_state=42)
    ax.scatter(sample["pen_true_mm"], sample["pen_pred_mm"], s=2, alpha=0.18, linewidths=0)
    low = min(float(points_df["pen_true_mm"].min()), float(points_df["pen_pred_mm"].min()))
    high = max(float(points_df["pen_true_mm"].max()), float(points_df["pen_pred_mm"].max()))
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Measured penetration [mm]")
    ax.set_ylabel("Predicted penetration [mm]")
    ax.set_title("Stage-3 prediction vs measurement")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_residual_hist(points_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=160)
    ax.hist(points_df["resid_mm"], bins=80, color="#4c78a8", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual [mm]")
    ax.set_ylabel("Count")
    ax.set_title("Stage-3 residual distribution")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_per_folder(per_folder: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    ordered = per_folder.sort_values("rmse_mean_mm")
    ax.barh(ordered["folder"], ordered["rmse_mean_mm"], color="#4c78a8", alpha=0.85)
    ax.set_xlabel("Mean trajectory RMSE [mm]")
    ax.set_ylabel("")
    ax.set_title("RMSE by campaign folder")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_trajectory(row: pd.Series, points: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.plot(points["time_ms"], points["pen_true_mm"], "o", markersize=3, label="measured")
    ax.plot(points["time_ms"], points["pen_pred_mm"], "-", linewidth=1.4, label="predicted")
    ax.fill_between(
        points["time_ms"].to_numpy(dtype=float),
        (points["pen_pred_mm"] - points["pen_std_mm"]).to_numpy(dtype=float),
        (points["pen_pred_mm"] + points["pen_std_mm"]).to_numpy(dtype=float),
        alpha=0.18,
        label="1 sigma",
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Penetration [mm]")
    ax.set_title(f"{row['folder']} | {row['test_name']} | plume {int(row['plume_idx'])}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_rmse_evaluation(
    *,
    refinement_run: Path | str,
    split: str = "clean",
    filter_experiment: str | None = None,
    device: torch.device | str | None = None,
    t_min_ms: float = 0.0,
    t_max_ms: float = 5.0,
    rel_err_floor_mm: float = 5.0,
    output_root: Path | str | None = None,
    tag: str | None = None,
    batch_points: int = 65536,
    fast: bool = False,
    save_points: bool = True,
    save_plots: bool = True,
    max_traj_plots: int | None = 30,
) -> tuple[Path, dict[str, Any]]:
    run_path = Path(refinement_run).expanduser().resolve()
    artifacts = load_run_artifacts(run_path, device=device)
    registry = build_dataset_registry()
    cdf_wide_df = load_source_table("cdf", split=split)
    if filter_experiment is not None:
        if "experiment_name" not in cdf_wide_df.columns:
            raise KeyError("CDF wide table missing experiment_name column required by filter_experiment.")
        mask = cdf_wide_df["experiment_name"].astype(str) == str(filter_experiment)
        n_rows = int(mask.sum())
        if n_rows == 0:
            raise ValueError(f"filter_experiment={filter_experiment!r} matched 0 CDF rows.")
        cdf_wide_df = cdf_wide_df.loc[mask].reset_index(drop=True)
        print(f"Filtered CDF rows to experiment_name={filter_experiment!r}: {n_rows} trajectories")

    time_cols = [c for c in cdf_wide_df.columns if c.startswith("time_ms_")]
    time_cols.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    frame_ids = [int(c.rsplit("_", 1)[1]) for c in time_cols]
    pen_cols = [f"penetration_mm_{frame_id:03d}" for frame_id in frame_ids]

    feature_blocks: list[np.ndarray] = []
    a_scale_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    meta_blocks: list[pd.DataFrame] = []
    traj_meta_rows: list[dict[str, Any]] = []

    for row_idx, row in cdf_wide_df.iterrows():
        time_vals = extract_prefixed_matrix(cdf_wide_df.iloc[[row_idx]], "time_ms_", frame_ids).reshape(-1)
        pen_vals = extract_prefixed_matrix(cdf_wide_df.iloc[[row_idx]], "penetration_mm_", frame_ids).reshape(-1)
        valid = np.isfinite(time_vals) & np.isfinite(pen_vals) & (time_vals >= t_min_ms) & (time_vals <= t_max_ms)
        if not np.any(valid):
            continue

        time_valid = time_vals[valid].astype(np.float32)
        truth_valid = pen_vals[valid].astype(np.float32)
        raw = build_teacher_raw_dict(row)
        features_np, a_scale_np, _ = build_feature_matrix_np(
            raw,
            time_valid,
            artifacts.scaler_state,
            list(artifacts.train_config["feature_columns"]),
            registry,
            time_feature=str(artifacts.train_config.get("time_feature", "time_norm_0_5ms")),
        )

        folder = str(row.get("experiment_name", "unknown"))
        try:
            dataset_key = normalize_dataset_key(folder)
        except Exception:
            dataset_key = folder
        test_name = str(Path(str(row.get("file_path", row.get("file_name", row_idx)))).parent.name)
        traj_key = f"{folder}|{test_name}|{row.get('file_name', row_idx)}|plume={int(row.get('plume_idx', -1))}"

        feature_blocks.append(features_np)
        a_scale_blocks.append(a_scale_np.reshape(-1))
        truth_blocks.append(truth_valid)
        meta_blocks.append(
            pd.DataFrame(
                {
                    "folder": folder,
                    "dataset_key": dataset_key,
                    "test_name": test_name,
                    "traj_key": traj_key,
                    "row_idx": int(row_idx),
                    "plume_idx": int(row.get("plume_idx", -1)),
                    "injection_duration_us": float(row.get("injection_duration_us", np.nan)),
                    "time_ms": time_valid,
                }
            )
        )
        traj_meta_rows.append(
            {
                "traj_key": traj_key,
                "folder": folder,
                "dataset_key": dataset_key,
                "test_name": test_name,
                "row_idx": int(row_idx),
                "plume_idx": int(row.get("plume_idx", -1)),
                "injection_duration_us": float(row.get("injection_duration_us", np.nan)),
                "max_observed_time_ms": float(np.max(time_valid)),
            }
        )

        if fast and len(traj_meta_rows) >= 512:
            break

    if not feature_blocks:
        raise RuntimeError("No finite CDF points found for RMSE evaluation.")

    features = np.vstack(feature_blocks).astype(np.float32)
    a_scale = np.concatenate(a_scale_blocks).astype(np.float32)
    truth = np.concatenate(truth_blocks).astype(np.float32)
    meta_df = pd.concat(meta_blocks, ignore_index=True)
    pred, std = _predict_points(
        artifacts=artifacts,
        features=features,
        a_scale=a_scale,
        batch_points=batch_points,
    )

    points_df = meta_df.copy()
    points_df["pen_true_mm"] = truth
    points_df["pen_pred_mm"] = pred
    points_df["pen_std_mm"] = std
    points_df["resid_mm"] = pred - truth

    traj_meta = pd.DataFrame(traj_meta_rows)
    per_traj_rows: list[dict[str, Any]] = []
    for traj_key, g in points_df.groupby("traj_key", dropna=False):
        resid = g["resid_mm"].to_numpy(dtype=float)
        abs_resid = np.abs(resid)
        std_safe = np.maximum(g["pen_std_mm"].to_numpy(dtype=float), 1e-12)
        per_traj_rows.append(
            {
                "traj_key": traj_key,
                "n_points": int(len(g)),
                "rmse_mm": float(np.sqrt(np.mean(resid**2))),
                "mae_mm": float(np.mean(abs_resid)),
                "mean_pen_mm": float(np.mean(g["pen_true_mm"])),
                "max_pen_mm": float(np.max(g["pen_true_mm"])),
                "coverage_1sigma": float(np.mean(abs_resid <= std_safe)),
                "coverage_2sigma": float(np.mean(abs_resid <= 2.0 * std_safe)),
            }
        )
    per_traj = pd.DataFrame(per_traj_rows)
    per_traj = traj_meta.merge(per_traj, on="traj_key", how="inner")
    per_traj["is_censored"] = per_traj["max_observed_time_ms"] < (float(t_max_ms) - 1e-6)
    per_traj = per_traj.drop(columns=["max_observed_time_ms"])

    per_folder = (
        per_traj.groupby("folder", dropna=False)
        .agg(
            n_traj=("traj_key", "count"),
            rmse_mean_mm=("rmse_mm", "mean"),
            rmse_median_mm=("rmse_mm", "median"),
            mae_mean_mm=("mae_mm", "mean"),
            coverage_1sigma=("coverage_1sigma", "mean"),
            coverage_2sigma=("coverage_2sigma", "mean"),
        )
        .reset_index()
    )

    metrics = _finite_metrics(truth, pred, std)
    rel_abs = np.abs(points_df["resid_mm"].to_numpy(dtype=float)) / np.maximum(
        np.abs(points_df["pen_true_mm"].to_numpy(dtype=float)),
        float(rel_err_floor_mm),
    )
    truth_range = float(np.max(truth) - np.min(truth))
    overall = {
        **metrics,
        "n_trajectories": int(len(per_traj)),
        "mean_rel_err": float(np.mean(rel_abs)),
        "median_rel_err": float(np.median(rel_abs)),
        "truth_min_mm": float(np.min(truth)),
        "truth_max_mm": float(np.max(truth)),
        "truth_range_mm": truth_range,
        "nrmse_range": float(metrics["rmse_mm"] / truth_range) if truth_range > 0 else float("nan"),
    }

    tag = tag or run_path.name
    out_root = Path(output_root or DEFAULT_OUTPUT_ROOT)
    out_dir = out_root / f"rmse_eval_{split}_{pd.Timestamp.now():%Y%m%d_%H%M%S}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=False)

    if save_points:
        points_df.to_csv(out_dir / "points.csv", index=False)
    per_traj.to_csv(out_dir / "per_trajectory.csv", index=False)
    per_folder.to_csv(out_dir / "per_folder.csv", index=False)

    summary = {
        "refinement_run": str(run_path),
        "split": split,
        "filter_experiment": filter_experiment,
        "t_window_ms": [float(t_min_ms), float(t_max_ms)],
        "rel_err_floor_mm": float(rel_err_floor_mm),
        "n_csv_files": int(cdf_wide_df["file_path"].nunique()) if "file_path" in cdf_wide_df else int(len(cdf_wide_df)),
        "eval_mode": {
            "fast": bool(fast),
            "save_points": bool(save_points),
            "save_plots": bool(save_plots),
            "max_traj_plots": max_traj_plots,
            "batch_points": int(batch_points),
        },
        "feature_columns": list(artifacts.train_config["feature_columns"]),
        "overall": overall,
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if save_plots:
        _plot_pred_vs_actual(points_df, out_dir / "pred_vs_actual.png")
        _plot_residual_hist(points_df, out_dir / "residual_histogram.png")
        _plot_per_folder(per_folder, out_dir / "per_folder_rmse.png")
        traj_plot_dir = out_dir / "traj_plots"
        traj_plot_dir.mkdir(exist_ok=True)
        plot_rows = per_traj.sort_values("rmse_mm", ascending=False)
        if max_traj_plots is not None:
            plot_rows = plot_rows.head(int(max_traj_plots))
        for _, traj_row in plot_rows.iterrows():
            g = points_df.loc[points_df["traj_key"] == traj_row["traj_key"]].sort_values("time_ms")
            safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(traj_row["traj_key"]))[:160]
            _plot_trajectory(traj_row, g, traj_plot_dir / f"{safe_name}.png")

    return out_dir, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement-run", required=True, type=Path)
    parser.add_argument("--split", choices=("clean", "all"), default="clean")
    parser.add_argument("--filter-experiment", default=None,
                        help="If set, evaluate only CDF rows with experiment_name equal to this value.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--t-min-ms", type=float, default=0.0)
    parser.add_argument("--t-max-ms", type=float, default=5.0)
    parser.add_argument("--rel-err-floor-mm", type=float, default=5.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--batch-points", type=int, default=65536)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-save-points", action="store_true")
    parser.add_argument("--no-save-plots", action="store_true")
    parser.add_argument("--max-traj-plots", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir, summary = run_rmse_evaluation(
        refinement_run=args.refinement_run,
        split=args.split,
        filter_experiment=args.filter_experiment,
        device=None if args.device is None or str(args.device).lower() == "auto" else args.device,
        t_min_ms=args.t_min_ms,
        t_max_ms=args.t_max_ms,
        rel_err_floor_mm=args.rel_err_floor_mm,
        output_root=args.output_root,
        tag=args.tag,
        batch_points=args.batch_points,
        fast=args.fast,
        save_points=not args.no_save_points,
        save_plots=not args.no_save_plots,
        max_traj_plots=args.max_traj_plots,
    )
    print(f"Wrote RMSE evaluation to {out_dir}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
