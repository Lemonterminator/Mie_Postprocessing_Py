"""Evaluate a trained SVGP checkpoint on uncensored CDF points.

This bridges the GP-baseline LONO runs to the MLP-LONO evaluation protocol.
The MLP `run_lono_pipeline.py` reports metrics on
``MLP/synthetic_data/cdf_right_censoring_points/cdf_points_uncensored.csv``
(density-drop + FOV-saturation truncated). The GP's built-in ``--external-eval
--external-split clean`` evaluator instead consumes the clean CDF wide table,
which retains cap-hit frames and therefore yields a different point population.

This script makes the two evaluations point-set-identical so the LONO RMSEs are
directly comparable. It loads the trained GP from ``per_seed/seed_<S>/model.pt``,
builds features for each uncensored CDF point through the same
``build_feature_matrix_np`` path the GP training itself uses, calls
``predict_physical`` for (mu, sigma) in mm, and writes ``metrics_summary.json`` /
``per_trajectory.csv`` next to (or under) the run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.MLP_training.engineered_feature_common import (  # noqa: E402
    TIME_FEATURE,
    build_dataset_registry,
    build_feature_matrix_np,
)
from MLP.MLP_training.run_gp_baseline import (  # noqa: E402
    choose_device,
    finite_metrics,
    load_gp_artifacts,
    predict_physical,
)
from MLP.MLP_training.train_stage3_distillation_plus_raw_series import (  # noqa: E402
    build_teacher_raw_dict,
)

DEFAULT_UNCENSORED_CSV = (
    PROJECT_ROOT
    / "MLP" / "synthetic_data" / "cdf_right_censoring_points"
    / "cdf_points_uncensored.csv"
)


def _gaussian_nll(truth: np.ndarray, pred: np.ndarray, std: np.ndarray) -> float:
    std_safe = np.maximum(std.astype(float), 1e-6)
    z = (truth.astype(float) - pred.astype(float)) / std_safe
    return float(np.mean(0.5 * (np.log(2 * np.pi) + 2.0 * np.log(std_safe) + z * z)))


def evaluate_run(
    run_dir: Path,
    *,
    uncensored_csv: Path,
    lono_holdout: str | None,
    seed: int | None,
    batch_points: int,
    device_arg: str,
    output_dir: Path | None,
) -> dict[str, Any]:
    config_path = run_dir / "gp_config_resolved.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    config = json.loads(config_path.read_text())
    if lono_holdout is None:
        lono_holdout = config.get("lono_holdout")
    if not lono_holdout:
        raise ValueError(
            "lono_holdout must be set (either via --lono-holdout or in the run's gp_config_resolved.json)."
        )

    seeds_available = (
        [int(s["seed"]) for s in (config.get("feature_spec_by_seed") or [])]
        or [int(s["seed"]) for s in (config.get("per_seed") or [])]
        or [int(s) for s in (config.get("seeds") or [])]
    )
    if seed is None:
        if len(seeds_available) != 1:
            raise ValueError(
                f"--seed not provided and run has multiple seeds {seeds_available}; pick one explicitly."
            )
        seed = int(seeds_available[0])
    if seed not in seeds_available:
        raise ValueError(f"Seed {seed} not in run's seed list {seeds_available}.")

    checkpoint_path = run_dir / "per_seed" / f"seed_{seed}" / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint_path}")

    device = choose_device(device_arg)
    artifacts = load_gp_artifacts(checkpoint_path, device=device)
    feature_columns = list(artifacts.config["feature_columns"])
    time_feature = str(artifacts.config.get("time_feature", TIME_FEATURE))
    t_min_ms = float(artifacts.config.get("time_min_ms", 0.0))
    t_max_ms = float(artifacts.config.get("time_max_ms", 5.0))

    print(f"Loading uncensored CDF points from {uncensored_csv}")
    df = pd.read_csv(uncensored_csv)
    n_total = len(df)
    df = df.loc[df["experiment_name"].astype(str) == str(lono_holdout)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"No uncensored CDF rows match experiment_name == {lono_holdout!r}; "
            f"available experiments: {sorted(df['experiment_name'].unique().tolist())}"
        )
    df = df.loc[(df["time_ms"] >= t_min_ms) & (df["time_ms"] <= t_max_ms)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"All rows for {lono_holdout!r} fall outside [{t_min_ms}, {t_max_ms}] ms.")
    print(
        f"  total rows: {n_total}; after LONO filter ({lono_holdout}): {len(df)}; "
        f"unique trajectories: {df['traj_key'].nunique()}"
    )

    registry = build_dataset_registry()

    feature_blocks: list[np.ndarray] = []
    a_scale_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    meta_blocks: list[pd.DataFrame] = []

    for traj_key, group in df.groupby("traj_key", sort=False):
        group = group.sort_values("time_ms")
        time_valid = group["time_ms"].to_numpy(dtype=np.float32)
        truth_valid = group["penetration_mm"].to_numpy(dtype=np.float32)
        raw = build_teacher_raw_dict(group.iloc[0])
        features_np, a_scale_np, _ = build_feature_matrix_np(
            raw,
            time_valid,
            artifacts.scaler_state,
            feature_columns,
            registry,
            time_feature=time_feature,
        )
        folder = str(group.iloc[0].get("experiment_name", "unknown"))
        test_name = str(group.iloc[0].get("test_name", "unknown"))
        plume_idx = int(group.iloc[0].get("plume_idx", -1))
        feature_blocks.append(features_np)
        a_scale_blocks.append(a_scale_np.reshape(-1))
        truth_blocks.append(truth_valid)
        meta_blocks.append(
            pd.DataFrame(
                {
                    "traj_key": str(traj_key),
                    "folder": folder,
                    "test_name": test_name,
                    "plume_idx": plume_idx,
                    "time_ms": time_valid,
                }
            )
        )

    if not feature_blocks:
        raise RuntimeError("No usable trajectories after filtering.")

    features = np.vstack(feature_blocks).astype(np.float32)
    a_scale = np.concatenate(a_scale_blocks).astype(np.float32)
    truth = np.concatenate(truth_blocks).astype(np.float32)
    meta_df = pd.concat(meta_blocks, ignore_index=True)

    print(f"Predicting {len(truth)} points across {len(feature_blocks)} trajectories...")
    pred, std, _, _ = predict_physical(
        artifacts,
        features,
        a_scale,
        batch_points=int(batch_points),
        include_mean_posterior_var=bool(artifacts.config.get("include_mean_posterior_var", False)),
    )

    metrics = finite_metrics(truth, pred, std)
    nll = _gaussian_nll(truth, pred, std)

    points_df = meta_df.copy()
    points_df["pen_true_mm"] = truth
    points_df["pen_pred_mm"] = pred
    points_df["pen_std_mm"] = std
    points_df["resid_mm"] = pred - truth

    per_traj_rows: list[dict[str, Any]] = []
    for traj_key, group in points_df.groupby("traj_key", dropna=False):
        resid = group["resid_mm"].to_numpy(dtype=float)
        abs_resid = np.abs(resid)
        std_safe = np.maximum(group["pen_std_mm"].to_numpy(dtype=float), 1e-12)
        per_traj_rows.append(
            {
                "traj_key": traj_key,
                "folder": str(group["folder"].iloc[0]),
                "test_name": str(group["test_name"].iloc[0]),
                "plume_idx": int(group["plume_idx"].iloc[0]),
                "n_points": int(len(group)),
                "rmse_mm": float(np.sqrt(np.mean(resid**2))),
                "mae_mm": float(np.mean(abs_resid)),
                "bias_mm": float(np.mean(resid)),
                "mean_pen_mm": float(np.mean(group["pen_true_mm"])),
                "max_pen_mm": float(np.max(group["pen_true_mm"])),
                "coverage_1sigma": float(np.mean(abs_resid <= std_safe)),
                "coverage_2sigma": float(np.mean(abs_resid <= 2.0 * std_safe)),
            }
        )
    per_traj = pd.DataFrame(per_traj_rows)

    truth_range = float(np.max(truth) - np.min(truth))
    overall = {
        **metrics,
        "n_trajectories": int(len(per_traj)),
        "truth_min_mm": float(np.min(truth)),
        "truth_max_mm": float(np.max(truth)),
        "truth_range_mm": truth_range,
        "nrmse_range": float(metrics["rmse_mm"] / truth_range) if truth_range > 0 else float("nan"),
        "nll_physical": nll,
    }

    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = run_dir / "external_eval_uncensored" / f"uncensored_{lono_holdout}_seed_{seed}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(checkpoint_path),
        "uncensored_csv": str(uncensored_csv),
        "lono_holdout": lono_holdout,
        "seed": seed,
        "feature_columns": feature_columns,
        "time_window_ms": [t_min_ms, t_max_ms],
        "overall": overall,
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    per_traj.to_csv(output_dir / "per_trajectory.csv", index=False)
    print(f"\nWrote: {output_dir / 'metrics_summary.json'}")
    print("Overall metrics:")
    for key, val in overall.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gp-run-dir", type=Path, required=True,
                   help="Path to gp_baseline_*/ run directory (must contain gp_config_resolved.json and per_seed/).")
    p.add_argument("--uncensored-csv", type=Path, default=DEFAULT_UNCENSORED_CSV)
    p.add_argument("--lono-holdout", type=str, default=None,
                   help="experiment_name filter; defaults to value stored in gp_config_resolved.json.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed to evaluate; auto-resolves if the run has a single seed.")
    p.add_argument("--batch-points", type=int, default=65536)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_run(
        args.gp_run_dir,
        uncensored_csv=args.uncensored_csv,
        lono_holdout=args.lono_holdout,
        seed=args.seed,
        batch_points=args.batch_points,
        device_arg=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
