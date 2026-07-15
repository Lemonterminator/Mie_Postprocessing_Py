"""Layer-1 evaluation engine: one (model, dataset, eval set) -> artifacts.

Artifacts written under ``<run_dir>/models/<label>/<dataset>/<eval_set>/``:

* ``points.parquet``/``.csv``   canonical per-point predictions (optional)
* ``metrics.json``              overall + slice metrics (full protocol)
* ``per_condition.csv`` / ``per_experiment.csv`` / ``per_time_bin.csv`` /
  ``per_condition_time_bin.csv`` / ``per_trajectory.csv``
* ``reliability_curve.csv`` / ``pit_histogram.csv`` / ``coverage_curve.csv`` /
  ``sigma_bin_calibration.csv``  (probabilistic eval sets only)

No figures here — plotting is Layer 2 (``make_figures.py``) and reads only
these artifacts.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from MLP.eval_pipeline import metrics as M
from MLP.eval_pipeline.common import jsonable, write_json, write_points
from MLP.eval_pipeline.datasets import (
    PROBABILISTIC_EVAL_SETS,
    WEIGHT_COL_BY_EVAL_SET,
    DatasetSpec,
    EvalSetSpec,
    censoring_slices,
    load_eval_table,
    observed_window_slices,
)
from MLP.eval_pipeline.predictors import ModelSpec, load_predictor

TIME_BIN_MS = 0.1  # parity with the cdf censoring-point construction

#: Headline metrics that get a bootstrap CI when --bootstrap is enabled.
BOOTSTRAP_METRICS = ("rmse_mm", "mae_mm", "bias_mm")


def _ensure_time_bin(points: pd.DataFrame) -> pd.DataFrame:
    if "time_bin" not in points.columns:
        # Exact passthrough (legacy parity): flooring time_ms/TIME_BIN_MS is not
        # float-exact at 0.1 spacing and silently collides/skips adjacent grid
        # bins (e.g. 0.2 and 0.3 ms both floor to bin 2 on some platforms).
        points = points.copy()
        points["time_bin"] = points["time_ms"]
        points["time_bin_start_ms"] = points["time_ms"]
        points["time_bin_end_ms"] = points["time_ms"] + TIME_BIN_MS
    return points


def _drop_nonfinite_predictions(points: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows with a non-finite truth/pred/std before any metric reduction.

    A single NaN/Inf poisons every np.mean/median/quantile-based aggregate for
    the whole eval set (parity with the legacy HA/NS/fixed-table evaluators,
    which apply this exact filter before scoring).
    """
    mask = (
        np.isfinite(pd.to_numeric(points["pen_true_mm"], errors="coerce"))
        & np.isfinite(pd.to_numeric(points["pen_pred_mm"], errors="coerce"))
        & np.isfinite(pd.to_numeric(points["pen_std_mm"], errors="coerce"))
    )
    n_dropped = int((~mask).sum())
    return points.loc[mask].reset_index(drop=True), n_dropped


#: Independence unit for the bootstrap: points within one trajectory/condition
#: share correlated residual bias, so resampling raw points overstates the
#: effective sample size. Prefer the finer-grained key when present.
_CLUSTER_KEY_PRIORITY = ("traj_key", "condition_id")


def _bootstrap_block(points: pd.DataFrame, *, n_boot: int, seed: int,
                     n_workers: int) -> dict[str, Any]:
    resid = (points["pen_pred_mm"] - points["pen_true_mm"]).to_numpy()
    cluster_col = next((c for c in _CLUSTER_KEY_PRIORITY if c in points.columns), None)
    out: dict[str, Any] = {
        "bootstrap_cluster_key": cluster_col,
        "bootstrap_n": int(n_boot),
        "bootstrap_workers": int(n_workers),
    }
    if cluster_col is not None:
        out["bootstrap_method"] = "cluster_sufficient_stats"
        intervals = M.cluster_bootstrap_headline_ci(
            resid, points[cluster_col].to_numpy(), n_boot=n_boot, seed=seed,
            n_workers=n_workers)
    else:
        out["bootstrap_method"] = "point_resample"
        intervals = M.bootstrap_headline_ci(
            resid, n_boot=n_boot, seed=seed, n_workers=n_workers)
    for name in BOOTSTRAP_METRICS:
        lo, hi = intervals[name]
        out[f"{name}_ci95_lo"] = lo
        out[f"{name}_ci95_hi"] = hi
    return out


def evaluate_eval_set(
    predictor,
    model: ModelSpec,
    dataset: DatasetSpec,
    spec: EvalSetSpec,
    out_dir: Path,
    *,
    filter_experiment: str | None,
    t_min_ms: float,
    t_max_ms: float,
    rel_err_floor_mm: float,
    batch_points: int,
    save_points: bool,
    limit_conditions: int | None,
    bootstrap: bool,
    bootstrap_n: int = 2000,
    bootstrap_workers: int = 1,
) -> dict[str, Any]:
    """Run inference + full metric protocol for one eval set; write artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    df = load_eval_table(
        spec, filter_experiment=filter_experiment,
        t_min_ms=t_min_ms, t_max_ms=t_max_ms, limit_conditions=limit_conditions,
    )
    points, skipped_groups = predictor.predict(df, spec, batch_points=batch_points)
    points = _ensure_time_bin(points)
    points, n_nonfinite_dropped = _drop_nonfinite_predictions(points)
    infer_seconds = time.perf_counter() - t0
    if n_nonfinite_dropped:
        print(f"[warn] {model.label} | {dataset.name} | {spec.name}: dropped "
              f"{n_nonfinite_dropped} non-finite prediction(s) before scoring")
    if points.empty:
        raise RuntimeError(
            f"All {n_nonfinite_dropped} points had non-finite predictions for {spec.name!r}.")

    probabilistic = spec.name in PROBABILISTIC_EVAL_SETS
    weight_col = WEIGHT_COL_BY_EVAL_SET.get(spec.name)

    overall = M.summarize_points(
        points, rel_err_floor_mm=rel_err_floor_mm,
        weight_col=weight_col, probabilistic=probabilistic,
    )
    bootstrap_seconds: float | None = None
    if bootstrap:
        tb = time.perf_counter()
        overall.update(_bootstrap_block(
            points, n_boot=bootstrap_n, seed=20260521,
            n_workers=bootstrap_workers))
        bootstrap_seconds = time.perf_counter() - tb
        overall["bootstrap_seconds"] = round(bootstrap_seconds, 3)

    slices: dict[str, dict[str, Any]] = {}
    for label, sub in observed_window_slices(points).items():
        slices[label] = M.summarize_points(
            sub, rel_err_floor_mm=rel_err_floor_mm, probabilistic=probabilistic,
        )
    if spec.name == "full_clean":
        for label, sub in censoring_slices(points).items():
            slices[label] = M.summarize_points(
                sub, rel_err_floor_mm=rel_err_floor_mm, probabilistic=probabilistic,
            )

    # --- grouped tables -----------------------------------------------------
    grouped_specs = {
        "per_condition": [c for c in ("condition_id", "condition_key", "experiment_name")
                          if c in points.columns][:2] or None,
        "per_experiment": ["experiment_name"] if "experiment_name" in points.columns else None,
        "per_time_bin": ["time_bin"] if "time_bin" in points.columns else None,
        "per_condition_time_bin": (["condition_id", "time_bin"]
                                   if {"condition_id", "time_bin"} <= set(points.columns) else None),
        "per_trajectory": ["traj_key"] if "traj_key" in points.columns else None,
        "per_censor_status": ["is_right_censored"] if "is_right_censored" in points.columns else None,
    }
    for name, cols in grouped_specs.items():
        if not cols:
            continue
        table = M.grouped_point_metrics(points, cols, rel_err_floor_mm=rel_err_floor_mm)
        if not table.empty:
            table.to_csv(out_dir / f"{name}.csv", index=False)

    # --- probabilistic diagnostics -------------------------------------------
    if probabilistic:
        truth = points["pen_true_mm"].to_numpy()
        mu = points["pen_pred_mm"].to_numpy()
        sigma = points["pen_std_mm"].to_numpy()
        weights = None
        if weight_col and weight_col in points.columns:
            weights = pd.to_numeric(points[weight_col], errors="coerce").to_numpy()
        M.reliability_table(truth, mu, sigma, weights=weights).to_csv(
            out_dir / "reliability_curve.csv", index=False)
        M.pit_histogram(truth, mu, sigma, weights=weights).to_csv(
            out_dir / "pit_histogram.csv", index=False)
        M.coverage_curve(truth, mu, sigma).to_csv(
            out_dir / "coverage_curve.csv", index=False)
        M.sigma_bin_calibration(truth, mu, sigma).to_csv(
            out_dir / "sigma_bin_calibration.csv", index=False)

    points_path: Path | None = None
    if save_points:
        points_path = write_points(points, out_dir / "points")

    payload = {
        "model_label": model.label,
        "model_family": model.family or model.label,
        "model_kind": model.kind,
        "checkpoint_path": str(model.path),
        "seed": model.seed,
        "model_meta": dict(model.meta),
        "dataset": dataset.name,
        "dataset_root": str(dataset.root),
        "eval_set": spec.name,
        "eval_table": str(spec.path),
        "t_window_ms": [t_min_ms, t_max_ms],
        "rel_err_floor_mm": rel_err_floor_mm,
        "limit_conditions": limit_conditions,
        "n_skipped_groups": skipped_groups,
        "n_nonfinite_dropped": n_nonfinite_dropped,
        "infer_seconds": round(infer_seconds, 3),
        "bootstrap_seconds": (round(bootstrap_seconds, 3)
                              if bootstrap_seconds is not None else None),
        "points_file": str(points_path) if points_path else None,
        "overall": overall,
        "slices": slices,
    }
    write_json(out_dir / "metrics.json", payload)
    return payload


def evaluate_model(
    model: ModelSpec,
    dataset: DatasetSpec,
    models_root: Path,
    *,
    device: str,
    filter_experiment: str | None,
    t_min_ms: float,
    t_max_ms: float,
    rel_err_floor_mm: float,
    batch_points_mlp: int,
    batch_points_gp: int,
    save_points: bool,
    limit_conditions: int | None,
    bootstrap: bool,
    bootstrap_n: int,
    bootstrap_workers: int,
) -> list[dict[str, Any]]:
    """Evaluate one model over every eval set of one dataset."""
    try:
        predictor = load_predictor(model, device=device)
    except Exception:
        # A broken checkpoint must not abort the rest of the roster.
        print(f"[FAIL] {model.label}: checkpoint load failed ({model.path})")
        traceback.print_exc()
        error = traceback.format_exc(limit=6)
        return [{
            "model_label": model.label,
            "model_kind": model.kind,
            "dataset": dataset.name,
            "eval_set": spec.name,
            "error": error,
        } for spec in dataset.eval_sets]
    batch = batch_points_mlp if model.kind in ("mlp", "ha", "ns") else batch_points_gp
    payloads: list[dict[str, Any]] = []
    for spec in dataset.eval_sets:
        out_dir = models_root / model.label / dataset.name / spec.name
        try:
            payload = evaluate_eval_set(
                predictor, model, dataset, spec, out_dir,
                filter_experiment=filter_experiment,
                t_min_ms=t_min_ms, t_max_ms=t_max_ms,
                rel_err_floor_mm=rel_err_floor_mm,
                batch_points=batch,
                save_points=save_points,
                limit_conditions=limit_conditions,
                bootstrap=bootstrap,
                bootstrap_n=bootstrap_n,
                bootstrap_workers=bootstrap_workers,
            )
            payloads.append(payload)
            rmse = payload["overall"].get("rmse_mm")
            rmse_txt = f"{rmse:.3f}" if isinstance(rmse, (int, float)) and np.isfinite(rmse) else "nan"
            boot_txt = (f", bootstrap={payload['bootstrap_seconds']}s"
                        if payload.get("bootstrap_seconds") is not None else "")
            print(f"[ok] {model.label} | {dataset.name} | {spec.name}: rmse={rmse_txt} mm "
                  f"({payload['infer_seconds']}s{boot_txt})")
        except Exception:
            print(f"[FAIL] {model.label} | {dataset.name} | {spec.name}")
            traceback.print_exc()
            payloads.append({
                "model_label": model.label,
                "model_kind": model.kind,
                "dataset": dataset.name,
                "eval_set": spec.name,
                "error": traceback.format_exc(limit=6),
            })
    return payloads


def flatten_payloads(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate metrics.json payloads into one wide row per model/dataset/slice."""
    id_cols = ("model_label", "model_family", "model_kind", "checkpoint_path",
               "seed", "dataset", "eval_set")
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        if "error" in payload:
            rows.append({k: payload.get(k) for k in id_cols} | {"slice": "overall",
                                                                "error": payload["error"]})
            continue
        base = {k: jsonable(payload.get(k)) for k in id_cols}
        for key, value in dict(payload.get("model_meta") or {}).items():
            base[f"meta_{key}"] = jsonable(value)
        rows.append(base | {"slice": "overall"} | dict(payload["overall"]))
        for slice_name, slice_metrics in (payload.get("slices") or {}).items():
            rows.append(base | {"slice": slice_name} | dict(slice_metrics))
    return pd.DataFrame(rows)
