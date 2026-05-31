"""Point-table evaluator for additive residual multi-task SVGP.

Search terms: RESIDUAL_SVGP, point-table eval, family routing.

This mirrors the MLP point-table evaluator but swaps in a GP prediction adapter.
The evaluator rebuilds the canonical feature vector only from physical/meta
columns, then derives ``family_id`` separately for residual routing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.GP_training.residual_multitask_svgp import (  # noqa: E402
    load_residual_svgp_artifacts,
    predict_residual_physical,
)
from MLP.GP_training.run_gp_baseline import choose_device, resolve_path  # noqa: E402
from MLP.MLP_training.engineered_feature_common import (  # noqa: E402
    build_dataset_registry,
    build_feature_matrix_np,
    family_id_from_name,
)
from MLP.eval.inference_rmse_on_point_tables import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POINT_EVAL_SPECS,
    DEFAULT_PRIMARY_EVAL_SET,
    META_COLS_FOR_POINTS,
    _ensure_time_bin_column,
    _feature_group_col,
    _group_metrics,
    _load_eval_table,
    _metrics_from_points,
    _probabilistic_headline_fields,
    _raw_from_meta_row,
    _slice_summaries,
    _write_probabilistic_diagnostics,
    point_eval_specs_for_synthetic_root,
)


def _resolve_checkpoint(path: Path | str, *, seed: int) -> Path:
    resolved = resolve_path(path)
    if resolved.is_file():
        return resolved
    candidate = resolved / "per_seed" / f"seed_{int(seed)}" / "model.pt"
    if candidate.exists():
        return candidate
    flat = resolved / "model.pt"
    if flat.exists():
        return flat
    raise FileNotFoundError(f"Could not resolve residual SVGP checkpoint from {path}.")


def _build_points_predictions(
    *,
    artifacts,
    registry: dict,
    df: pd.DataFrame,
    spec: dict[str, Any],
    batch_points: int,
) -> tuple[pd.DataFrame, int]:
    """Build fixed point-table predictions with family id kept out of features."""

    time_col = str(spec["time_col"])
    truth_col = str(spec["truth_col"])
    feature_columns = list(artifacts.config["feature_columns"])
    time_feature = str(artifacts.config.get("time_feature", "time_norm_0_5ms"))

    sort_cols = [col for col in ["experiment_name", "condition_id", "traj_key", time_col, "frame_pos"] if col in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.reset_index(drop=True)
    group_col = _feature_group_col(df, spec)
    group_iter: Iterable[tuple[Any, pd.DataFrame]]
    group_iter = df.groupby(group_col, sort=False, dropna=False) if group_col else [(None, df)]

    feature_blocks: list[np.ndarray] = []
    a_scale_blocks: list[np.ndarray] = []
    family_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    meta_blocks: list[pd.DataFrame] = []
    skipped_groups = 0

    for _, group in group_iter:
        group = group.sort_values([col for col in [time_col, "frame_pos"] if col in group.columns]).reset_index(drop=True)
        valid = np.isfinite(pd.to_numeric(group[time_col], errors="coerce").to_numpy(dtype=float))
        valid &= np.isfinite(pd.to_numeric(group[truth_col], errors="coerce").to_numpy(dtype=float))
        if not np.any(valid):
            skipped_groups += 1
            continue
        group = group.loc[valid].reset_index(drop=True)
        time_ms = group[time_col].to_numpy(dtype=np.float32)
        truth = group[truth_col].to_numpy(dtype=np.float32)
        try:
            raw = _raw_from_meta_row(group.iloc[0])
            features_np, a_scale_np, _ = build_feature_matrix_np(
                raw,
                time_ms,
                artifacts.scaler_state,
                feature_columns,
                registry,
                time_feature=time_feature,
            )
            # Routing key only: do not add this label to ``features_np``.
            family_value = family_id_from_name(str(raw.get("experiment_name") or raw.get("dataset_key") or ""))
        except Exception as exc:
            skipped_groups += 1
            print(f"[warn] feature build failed for {spec['name']} group={group_col}: {exc}")
            continue

        meta_cols = [col for col in META_COLS_FOR_POINTS if col in group.columns]
        meta = group.loc[:, meta_cols].copy()
        meta["time_ms"] = time_ms
        feature_blocks.append(features_np)
        a_scale_blocks.append(a_scale_np.reshape(-1))
        family_blocks.append(np.full(len(time_ms), int(family_value), dtype=np.int64))
        truth_blocks.append(truth)
        meta_blocks.append(meta)

    if not feature_blocks:
        raise RuntimeError(f"No usable points found for eval set {spec['name']!r}.")

    features = np.vstack(feature_blocks).astype(np.float32)
    a_scale = np.concatenate(a_scale_blocks).astype(np.float32)
    family_id = np.concatenate(family_blocks).astype(np.int64)
    truth = np.concatenate(truth_blocks).astype(np.float32)
    meta_df = pd.concat(meta_blocks, ignore_index=True)
    pred, std, _, _ = predict_residual_physical(
        artifacts,
        features,
        a_scale,
        family_id,
        batch_points=batch_points,
        include_mean_posterior_var=bool(artifacts.config.get("include_mean_posterior_var", False)),
    )

    points_df = meta_df.copy()
    points_df["family_id"] = family_id
    points_df["pen_true_mm"] = truth
    points_df["pen_pred_mm"] = pred
    points_df["pen_std_mm"] = std
    points_df["resid_mm"] = pred - truth
    return points_df, skipped_groups


def evaluate_point_table(
    *,
    artifacts,
    registry: dict,
    spec: dict[str, Any],
    out_dir: Path,
    filter_experiment: str | None,
    t_min_ms: float,
    t_max_ms: float,
    rel_err_floor_mm: float,
    batch_points: int,
    save_points: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_eval_table(
        spec,
        filter_experiment=filter_experiment,
        t_min_ms=t_min_ms,
        t_max_ms=t_max_ms,
    )
    points_df, skipped_groups = _build_points_predictions(
        artifacts=artifacts,
        registry=registry,
        df=df,
        spec=spec,
        batch_points=batch_points,
    )
    points_df = _ensure_time_bin_column(points_df)

    if save_points:
        points_df.to_csv(out_dir / "points.csv", index=False)
    per_condition = _group_metrics(points_df, ["condition_id"], rel_err_floor_mm=rel_err_floor_mm)
    if not per_condition.empty:
        per_condition.to_csv(out_dir / "per_condition.csv", index=False)
    per_experiment = _group_metrics(points_df, ["experiment_name"], rel_err_floor_mm=rel_err_floor_mm)
    if not per_experiment.empty:
        per_experiment.to_csv(out_dir / "per_experiment.csv", index=False)
    per_trajectory = _group_metrics(points_df, ["traj_key"], rel_err_floor_mm=rel_err_floor_mm)
    if not per_trajectory.empty:
        per_trajectory.to_csv(out_dir / "per_trajectory.csv", index=False)
    per_time_bin = _group_metrics(points_df, ["time_bin"], rel_err_floor_mm=rel_err_floor_mm)
    if not per_time_bin.empty:
        per_time_bin.to_csv(out_dir / "per_time_bin.csv", index=False)
    per_condition_time_bin = _group_metrics(points_df, ["condition_id", "time_bin"], rel_err_floor_mm=rel_err_floor_mm)
    if not per_condition_time_bin.empty:
        per_condition_time_bin.to_csv(out_dir / "per_condition_time_bin.csv", index=False)

    prob_summary = _write_probabilistic_diagnostics(points_df, eval_set=str(spec["name"]), out_dir=out_dir)
    slices = _slice_summaries(points_df, rel_err_floor_mm=rel_err_floor_mm)
    if slices:
        pd.DataFrame(
            [{"eval_set": name, **payload["overall"]} for name, payload in slices.items()]
        ).to_csv(out_dir / "slice_metrics.csv", index=False)

    summary = {
        "name": str(spec["name"]),
        "description": str(spec.get("description", "")),
        "path": str(spec["path"]),
        "time_col": str(spec["time_col"]),
        "truth_col": str(spec["truth_col"]),
        "filter_experiment": filter_experiment,
        "t_window_ms": [float(t_min_ms), float(t_max_ms)],
        "n_input_rows": int(len(df)),
        "skipped_feature_groups": int(skipped_groups),
        "overall": _metrics_from_points(points_df, rel_err_floor_mm=rel_err_floor_mm),
        "probabilistic": prob_summary,
        "slices": slices,
        "outputs": {
            "points": str(out_dir / "points.csv") if save_points else None,
            "per_condition": str(out_dir / "per_condition.csv") if not per_condition.empty else None,
            "per_experiment": str(out_dir / "per_experiment.csv") if not per_experiment.empty else None,
            "per_trajectory": str(out_dir / "per_trajectory.csv") if not per_trajectory.empty else None,
            "per_time_bin": str(out_dir / "per_time_bin.csv") if not per_time_bin.empty else None,
            "per_condition_time_bin": str(out_dir / "per_condition_time_bin.csv") if not per_condition_time_bin.empty else None,
            "probabilistic_summary": str(out_dir / "probabilistic_summary.json") if prob_summary else None,
            "reliability_curve": str(out_dir / "reliability_curve.csv") if prob_summary else None,
            "pit_histogram": str(out_dir / "pit_histogram.csv") if prob_summary else None,
            "crps_summary": str(out_dir / "crps_summary.csv") if prob_summary else None,
            "slice_metrics": str(out_dir / "slice_metrics.csv") if slices else None,
        },
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def run_point_table_evaluation(
    *,
    residual_gp_run: Path | str,
    synthetic_root: Path | str | None = None,
    seed: int = 42,
    eval_sets: list[str] | None = None,
    primary_eval_set: str = DEFAULT_PRIMARY_EVAL_SET,
    filter_experiment: str | None = None,
    device: torch.device | str | None = None,
    output_root: Path | str | None = None,
    tag: str | None = None,
    t_min_ms: float = 0.0,
    t_max_ms: float = 5.0,
    rel_err_floor_mm: float = 5.0,
    batch_points: int = 65536,
    save_points: bool = True,
) -> tuple[Path, dict[str, Any]]:
    device_obj = choose_device("auto" if device is None else str(device))
    checkpoint = _resolve_checkpoint(residual_gp_run, seed=seed)
    artifacts = load_residual_svgp_artifacts(checkpoint, device=device_obj)
    registry = build_dataset_registry()

    selected = set(eval_sets or [])
    source_specs = point_eval_specs_for_synthetic_root(synthetic_root) if synthetic_root is not None else DEFAULT_POINT_EVAL_SPECS
    specs = [
        dict(spec)
        for spec in source_specs
        if not selected or str(spec["name"]) in selected
    ]
    if not specs:
        raise ValueError(f"No point eval sets selected from {sorted(selected)}.")

    run_path = resolve_path(residual_gp_run)
    eval_tag = tag or run_path.name
    out_root = resolve_path(output_root or DEFAULT_OUTPUT_ROOT)
    out_dir = out_root / f"point_eval_{datetime.now():%Y%m%d_%H%M%S}_{eval_tag}"
    out_dir.mkdir(parents=True, exist_ok=False)

    eval_summaries: dict[str, Any] = {}
    per_run_rows: list[dict[str, Any]] = []
    for spec in specs:
        name = str(spec["name"])
        summary = evaluate_point_table(
            artifacts=artifacts,
            registry=registry,
            spec=spec,
            out_dir=out_dir / name,
            filter_experiment=filter_experiment,
            t_min_ms=t_min_ms,
            t_max_ms=t_max_ms,
            rel_err_floor_mm=rel_err_floor_mm,
            batch_points=int(batch_points),
            save_points=bool(save_points),
        )
        eval_summaries[name] = summary
        per_run_rows.append({
            "eval_set": name,
            **summary["overall"],
            **_probabilistic_headline_fields(summary.get("probabilistic")),
        })
        for slice_name, slice_payload in summary.get("slices", {}).items():
            per_run_rows.append({"eval_set": slice_name, **slice_payload["overall"]})

    if primary_eval_set not in eval_summaries:
        primary_eval_set = next(iter(eval_summaries))
    pd.DataFrame(per_run_rows).to_csv(out_dir / "per_run_metrics.csv", index=False)
    summary = {
        "residual_gp_run": str(run_path),
        "synthetic_root": None if synthetic_root is None else str(resolve_path(synthetic_root)),
        "checkpoint": str(checkpoint),
        "eval_kind": "residual_svgp_point_tables",
        "primary_eval_set": primary_eval_set,
        "filter_experiment": filter_experiment,
        "t_window_ms": [float(t_min_ms), float(t_max_ms)],
        "rel_err_floor_mm": float(rel_err_floor_mm),
        "batch_points": int(batch_points),
        "save_points": bool(save_points),
        "feature_columns": list(artifacts.config["feature_columns"]),
        "trained_family_ids": list(artifacts.config.get("trained_family_ids", [])),
        "overall": eval_summaries[primary_eval_set]["overall"],
        "eval_sets": eval_summaries,
        "outputs": {"per_run_metrics": str(out_dir / "per_run_metrics.csv")},
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return out_dir, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-gp-run", required=True, type=Path)
    parser.add_argument("--synthetic-root", type=Path, default=None,
                        help="Synthetic-data root containing cdf_right_censoring_points/ and p50_q1_oracle/.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-set", action="append", dest="eval_sets", default=None)
    parser.add_argument("--primary-eval-set", default=DEFAULT_PRIMARY_EVAL_SET)
    parser.add_argument("--filter-experiment", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--t-min-ms", type=float, default=0.0)
    parser.add_argument("--t-max-ms", type=float, default=5.0)
    parser.add_argument("--rel-err-floor-mm", type=float, default=5.0)
    parser.add_argument("--batch-points", type=int, default=65536)
    parser.add_argument("--no-save-points", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir, summary = run_point_table_evaluation(
        residual_gp_run=args.residual_gp_run,
        synthetic_root=args.synthetic_root,
        seed=args.seed,
        eval_sets=args.eval_sets,
        primary_eval_set=args.primary_eval_set,
        filter_experiment=args.filter_experiment,
        device=args.device,
        output_root=args.output_root,
        tag=args.tag,
        t_min_ms=args.t_min_ms,
        t_max_ms=args.t_max_ms,
        rel_err_floor_mm=args.rel_err_floor_mm,
        batch_points=args.batch_points,
        save_points=not args.no_save_points,
    )
    print(f"Wrote residual-SVGP point-table evaluation to {out_dir}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
