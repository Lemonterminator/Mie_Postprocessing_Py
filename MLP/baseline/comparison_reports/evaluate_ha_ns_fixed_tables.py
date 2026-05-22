"""Evaluate fitted H-A/N-S baselines on the fixed Stage-3 diagnostic tables.

This complements ``MLP/eval/evaluate_stage3_fixed_tables.py``.  The neural
evaluator handles MLP/SVGP checkpoints; this script reuses already fitted
physics-baseline run directories and merges their fixed-table metrics into the
same report layout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.baseline.Hiroyasu_Arai.data_io import (  # noqa: E402
    add_canonical_features as add_ha_canonical_features,
    build_dataset_registry as build_ha_dataset_registry,
)
from MLP.baseline.Hiroyasu_Arai.fit import (  # noqa: E402
    ResidualUncertainty as HAResidualUncertainty,
    parameter_sigma_for_points as ha_parameter_sigma_for_points,
    residual_sigma_for_points as ha_residual_sigma_for_points,
)
from MLP.baseline.Hiroyasu_Arai.models import HAParams, predict as predict_ha  # noqa: E402
from MLP.baseline.Naber_Siebers.data_io import (  # noqa: E402
    add_canonical_features as add_ns_canonical_features,
    build_dataset_registry as build_ns_dataset_registry,
)
from MLP.baseline.Naber_Siebers.fit import (  # noqa: E402
    ResidualUncertainty as NSResidualUncertainty,
    apply_angle_policy,
    parameter_sigma_for_points as ns_parameter_sigma_for_points,
    residual_sigma_for_points as ns_residual_sigma_for_points,
)
from MLP.baseline.Naber_Siebers.models import NSParams, predict as predict_ns  # noqa: E402


UNCENSORED_CSV = PROJECT_ROOT / "MLP" / "synthetic_data" / "cdf_right_censoring_points" / "cdf_points_uncensored.csv"
P50_Q1_ROOT = PROJECT_ROOT / "MLP" / "synthetic_data" / "p50_q1_oracle"
P50_Q1_OBSERVED_CSV = P50_Q1_ROOT / "p50_q1_observed_fit_points.csv"
P50_Q1_GRID_CSV = P50_Q1_ROOT / "p50_q1_predictions.csv"
DEFAULT_REFERENCE_REPORT = PROJECT_ROOT / "MLP" / "baseline" / "comparison_reports" / "stage3_fixed_table_eval_20260521"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "MLP" / "baseline" / "comparison_reports"


@dataclass(frozen=True)
class BaselineSpec:
    kind: str
    group: str
    label: str
    run_dir: Path


@dataclass
class PredictionResult:
    points: pd.DataFrame
    overall: dict[str, Any]
    per_condition: pd.DataFrame
    per_experiment: pd.DataFrame


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path: Path | str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_uncertainty(
    run_dir: Path,
    *,
    summary: Mapping[str, Any],
    cls: type[HAResidualUncertainty] | type[NSResidualUncertainty],
) -> HAResidualUncertainty | NSResidualUncertainty:
    unc = dict(summary.get("uncertainty", {}))
    bins_path = run_dir / "residual_uncertainty_by_time.csv"
    if bins_path.exists():
        bins = pd.read_csv(bins_path)
    else:
        bins = pd.DataFrame(columns=["time_bin_center_ms", "sigma_resid_mm", "n_calibration_points"])
    return cls(
        mode=str(unc.get("residual_mode", "time_binned")),
        global_sigma_mm=float(unc.get("global_sigma_mm", np.nan)),
        time_bin_ms=float(unc.get("time_bin_ms", 0.1)),
        sigma_floor_mm=float(unc.get("sigma_floor_mm", 0.5)),
        bin_centers_ms=pd.to_numeric(bins.get("time_bin_center_ms", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .to_numpy(dtype=float),
        bin_sigma_mm=pd.to_numeric(bins.get("sigma_resid_mm", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .to_numpy(dtype=float),
        bin_count=pd.to_numeric(bins.get("n_calibration_points", pd.Series(dtype=float)), errors="coerce")
        .fillna(0)
        .to_numpy(dtype=int),
    )


def load_ha_params(run_dir: Path) -> tuple[HAParams, list[HAParams], HAResidualUncertainty, dict[str, Any]]:
    summary = read_json(run_dir / "metrics_summary.json")
    payload = read_json(run_dir / "model_params.json")
    params = HAParams(
        kv=float(payload["kv"]),
        kp=float(payload["kp"]),
        kbt=float(payload["kbt"]),
        variant=str(payload["variant"]),
        use_zhou=bool(payload.get("use_zhou", False)),
    )
    boot_params: list[HAParams] = []
    boot_path = run_dir / "bootstrap_params.csv"
    if boot_path.exists():
        boot = pd.read_csv(boot_path)
        if "success" in boot.columns:
            boot = boot.loc[boot["success"].astype(bool)]
        for _, row in boot.iterrows():
            boot_params.append(
                HAParams(
                    kv=float(row["kv"]),
                    kp=float(row["kp"]),
                    kbt=float(row["kbt"]),
                    variant=params.variant,
                    use_zhou=params.use_zhou,
                )
            )
    unc = load_uncertainty(run_dir, summary=summary, cls=HAResidualUncertainty)
    return params, boot_params, unc, summary


def load_ns_params(run_dir: Path) -> tuple[NSParams, list[NSParams], NSResidualUncertainty, dict[str, Any], dict[str, Any]]:
    summary = read_json(run_dir / "metrics_summary.json")
    config = read_json(run_dir / "config_effective.json")
    payload = read_json(run_dir / "model_params.json")
    params = NSParams(
        k=float(payload["k"]),
        delay_s=float(payload["delay_s"]),
        variant=str(payload["variant"]),
        use_angle_factor=bool(payload.get("use_angle_factor", False)),
    )
    boot_params: list[NSParams] = []
    boot_path = run_dir / "bootstrap_params.csv"
    if boot_path.exists():
        boot = pd.read_csv(boot_path)
        if "success" in boot.columns:
            boot = boot.loc[boot["success"].astype(bool)]
        for _, row in boot.iterrows():
            boot_params.append(
                NSParams(
                    k=float(row["k"]),
                    delay_s=float(row["delay_s"]),
                    variant=params.variant,
                    use_angle_factor=params.use_angle_factor,
                )
            )
    unc = load_uncertainty(run_dir, summary=summary, cls=NSResidualUncertainty)
    return params, boot_params, unc, summary, config


def prepare_ha_table(df: pd.DataFrame, time_col: str, truth_col: str) -> pd.DataFrame:
    out = df.copy()
    if "time_s" not in out.columns:
        out["time_s"] = pd.to_numeric(out[time_col], errors="coerce") * 1e-3
    out["time_ms"] = pd.to_numeric(out[time_col], errors="coerce")
    out["pen_true_mm"] = pd.to_numeric(out[truth_col], errors="coerce")
    registry = build_ha_dataset_registry()
    return add_ha_canonical_features(out, registry)


def prepare_ns_table(df: pd.DataFrame, time_col: str, truth_col: str, config: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if "time_s" not in out.columns:
        out["time_s"] = pd.to_numeric(out[time_col], errors="coerce") * 1e-3
    out["time_ms"] = pd.to_numeric(out[time_col], errors="coerce")
    out["pen_true_mm"] = pd.to_numeric(out[truth_col], errors="coerce")
    registry = build_ns_dataset_registry()
    out = add_ns_canonical_features(out, registry)
    return apply_angle_policy(out, dict(config), angle_lookup=None)


def metrics_from_points(points: pd.DataFrame) -> dict[str, Any]:
    truth = points["truth_mm"].to_numpy(dtype=float)
    pred = points["pred_mu_mm"].to_numpy(dtype=float)
    std = points["pred_std_mm"].to_numpy(dtype=float)
    finite = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(std)
    truth = truth[finite]
    pred = pred[finite]
    std = std[finite]
    if truth.size == 0:
        return {
            "n_points": 0,
            "rmse_mm": float("nan"),
            "mae_mm": float("nan"),
            "bias_mm": float("nan"),
            "median_abs_err_mm": float("nan"),
            "p90_abs_err_mm": float("nan"),
            "p95_abs_err_mm": float("nan"),
            "coverage_1sigma": float("nan"),
            "coverage_2sigma": float("nan"),
            "mean_pred_std_mm": float("nan"),
        }
    resid = pred - truth
    abs_err = np.abs(resid)
    std_safe = np.maximum(std, 1e-12)
    return {
        "n_points": int(truth.size),
        "rmse_mm": float(np.sqrt(np.mean(resid**2))),
        "mae_mm": float(np.mean(abs_err)),
        "bias_mm": float(np.mean(resid)),
        "median_abs_err_mm": float(np.median(abs_err)),
        "p90_abs_err_mm": float(np.quantile(abs_err, 0.90)),
        "p95_abs_err_mm": float(np.quantile(abs_err, 0.95)),
        "coverage_1sigma": float(np.mean(abs_err <= std_safe)),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * std_safe)),
        "mean_pred_std_mm": float(np.mean(std)),
    }


def grouped_metrics(points: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, grp in points.groupby(list(group_cols), sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        row.update(metrics_from_points(grp))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(points: pd.DataFrame) -> PredictionResult:
    per_condition_cols = [c for c in ["condition_id", "condition_key", "experiment_name"] if c in points.columns]
    per_experiment_cols = [c for c in ["experiment_name"] if c in points.columns]
    return PredictionResult(
        points=points,
        overall=metrics_from_points(points),
        per_condition=grouped_metrics(points, per_condition_cols) if per_condition_cols else pd.DataFrame(),
        per_experiment=grouped_metrics(points, per_experiment_cols) if per_experiment_cols else pd.DataFrame(),
    )


def metadata_frame(df: pd.DataFrame, truth: np.ndarray) -> pd.DataFrame:
    keep = [
        "experiment_name",
        "condition_id",
        "condition_key",
        "traj_key",
        "test_name",
        "file_name",
        "plume_idx",
        "time_ms",
        "time_bin",
        "is_observed_window",
        "n_points_in_p50_bin",
        "n_traces_in_bin",
        "plumes",
        "diameter_mm",
        "umbrella_angle_deg",
        "chamber_pressure_bar",
        "injection_duration_us",
        "injection_pressure_bar",
        "control_backpressure_bar",
    ]
    meta = df.loc[:, [c for c in keep if c in df.columns]].copy()
    meta["truth_mm"] = truth
    return meta


def predict_baseline(spec: BaselineSpec, df: pd.DataFrame, time_col: str, truth_col: str) -> PredictionResult:
    if spec.kind == "ha":
        params, boot_params, unc, _summary = load_ha_params(spec.run_dir)
        points = prepare_ha_table(df, time_col, truth_col)
        pred = predict_ha(points, params)
        sigma_resid = ha_residual_sigma_for_points(points, unc)
        sigma_param = ha_parameter_sigma_for_points(points, boot_params)
    elif spec.kind == "ns":
        params, boot_params, unc, _summary, config = load_ns_params(spec.run_dir)
        points = prepare_ns_table(df, time_col, truth_col, config)
        pred = predict_ns(
            points,
            params,
            angle_factor_floor=float(config.get("angle_factor_floor", 0.25)),
            angle_factor_ceiling=float(config.get("angle_factor_ceiling", 4.0)),
        )
        sigma_resid = ns_residual_sigma_for_points(points, unc)
        sigma_param = ns_parameter_sigma_for_points(points, boot_params, config)
    else:
        raise ValueError(f"Unknown baseline kind: {spec.kind}")

    truth = pd.to_numeric(points["pen_true_mm"], errors="coerce").to_numpy(dtype=float)
    out = metadata_frame(points, truth)
    out["pred_mu_mm"] = pred
    out["pred_std_mm"] = np.sqrt(np.square(sigma_resid) + np.square(sigma_param))
    out["resid_mm"] = out["pred_mu_mm"] - out["truth_mm"]
    finite = np.isfinite(out["truth_mm"]) & np.isfinite(out["pred_mu_mm"]) & np.isfinite(out["pred_std_mm"])
    return summarise(out.loc[finite].reset_index(drop=True))


def add_condition_summary(metrics: Mapping[str, Any], per_condition: pd.DataFrame) -> dict[str, Any]:
    out = dict(metrics)
    if not per_condition.empty:
        out["n_conditions"] = int(len(per_condition))
        out["condition_rmse_mean_mm"] = float(per_condition["rmse_mm"].mean())
        out["condition_rmse_median_mm"] = float(per_condition["rmse_mm"].median())
        out["condition_mae_mean_mm"] = float(per_condition["mae_mm"].mean())
        out["condition_bias_mean_mm"] = float(per_condition["bias_mm"].mean())
    return out


def row_from_metrics(eval_set: str, spec: BaselineSpec, metrics: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "eval_set": eval_set,
        "model_group": spec.group,
        "model_label": spec.label,
        "model_kind": spec.kind,
        "model_path": str(spec.run_dir),
    }
    row.update(metrics)
    return row


def aggregate_group_rows(per_run: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "n_points",
        "n_conditions",
        "rmse_mm",
        "mae_mm",
        "bias_mm",
        "median_abs_err_mm",
        "p90_abs_err_mm",
        "p95_abs_err_mm",
        "coverage_1sigma",
        "coverage_2sigma",
        "mean_pred_std_mm",
        "condition_rmse_mean_mm",
        "condition_rmse_median_mm",
        "condition_mae_mean_mm",
        "condition_bias_mean_mm",
    ]
    rows: list[dict[str, Any]] = []
    for (eval_set, group), sub in per_run.groupby(["eval_set", "model_group"], sort=False):
        out: dict[str, Any] = {
            "eval_set": eval_set,
            "model_group": group,
            "n_runs": int(len(sub)),
        }
        rmse = pd.to_numeric(sub["rmse_mm"], errors="coerce")
        if len(sub):
            best_idx = rmse.idxmin()
            out["best_label_by_rmse"] = sub.loc[best_idx, "model_label"]
            out["best_rmse_mm"] = float(sub.loc[best_idx, "rmse_mm"])
        for col in metric_cols:
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if vals.empty:
                continue
            out[f"{col}_mean"] = float(vals.mean())
            out[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(out)
    return pd.DataFrame(rows)


def format_value(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
    except Exception:
        return "" if value is None else str(value)
    if not math.isfinite(v):
        return ""
    return f"{v:.{digits}f}"


def write_markdown(out_dir: Path, per_run: pd.DataFrame, group_summary: pd.DataFrame, specs: Sequence[BaselineSpec]) -> None:
    lines: list[str] = []
    lines.append("# Stage-3 Fixed-Table Evaluation with HA/NS\n")
    lines.append("Evaluation tables:")
    lines.append(f"- CDF uncensored: `{UNCENSORED_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- P50 observed: `{P50_Q1_OBSERVED_CSV.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- Q1 grid: `{P50_Q1_GRID_CSV.relative_to(PROJECT_ROOT)}`\n")
    lines.append("Physics baseline runs:")
    for spec in specs:
        lines.append(f"- {spec.group}: `{spec.run_dir.relative_to(PROJECT_ROOT)}`")
    lines.append("")

    display_sets = [
        ("cdf_uncensored", "CDF Uncensored Point-Level"),
        ("p50_observed", "P50 Observed Points"),
        ("q1_grid_all", "Q1 Oracle Grid, 0-5 ms"),
        ("q1_grid_extrapolated", "Q1 Oracle Grid, Extrapolated Region"),
    ]
    for eval_set, title in display_sets:
        sub = group_summary.loc[group_summary["eval_set"] == eval_set].copy()
        if sub.empty:
            continue
        lines.append(f"## {title}\n")
        lines.append("| model_group | n_runs | rmse | mae | bias | p95 | cov1 | cov2 | cond_rmse_mean | best |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, row in sub.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("model_group", "")),
                        str(int(row.get("n_runs", 0))),
                        format_value(row.get("rmse_mm_mean")),
                        format_value(row.get("mae_mm_mean")),
                        format_value(row.get("bias_mm_mean")),
                        format_value(row.get("p95_abs_err_mm_mean")),
                        format_value(row.get("coverage_1sigma_mean")),
                        format_value(row.get("coverage_2sigma_mean")),
                        format_value(row.get("condition_rmse_mean_mm_mean")),
                        f"{row.get('best_label_by_rmse', '')} ({format_value(row.get('best_rmse_mm'))})",
                    ]
                )
                + " |"
            )
        lines.append("")

    keep = [
        "eval_set",
        "model_group",
        "model_label",
        "n_points",
        "n_conditions",
        "rmse_mm",
        "mae_mm",
        "bias_mm",
        "p95_abs_err_mm",
        "coverage_1sigma",
        "coverage_2sigma",
        "condition_rmse_mean_mm",
    ]
    lines.append("## Per-Run Headline\n")
    shown = per_run.loc[:, [c for c in keep if c in per_run.columns]].copy()
    try:
        lines.append(shown.to_markdown(index=False))
    except ImportError:
        lines.append("```")
        lines.append(shown.to_string(index=False))
        lines.append("```")
    lines.append("")
    (out_dir / "main_table.md").write_text("\n".join(lines), encoding="utf-8")


def read_reference_frame(reference_report: Path, name: str) -> pd.DataFrame:
    path = reference_report / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def evaluate_all(
    *,
    specs: Sequence[BaselineSpec],
    reference_report: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    uncensored = pd.read_csv(UNCENSORED_CSV, low_memory=False)
    p50_observed = pd.read_csv(P50_Q1_OBSERVED_CSV, low_memory=False)
    q1_grid = pd.read_csv(P50_Q1_GRID_CSV, low_memory=False)

    per_run_rows: list[dict[str, Any]] = []
    per_condition_frames: list[pd.DataFrame] = []
    per_experiment_frames: list[pd.DataFrame] = []

    eval_tables = [
        ("cdf_uncensored", uncensored, "time_ms", "penetration_mm"),
        ("p50_observed", p50_observed, "time_ms", "penetration_mm"),
        ("q1_grid_all", q1_grid, "time_ms", "q1_fit_mm"),
    ]
    for spec in specs:
        print(f"[model] {spec.group}/{spec.label} ({spec.kind})", flush=True)
        for eval_set, df, time_col, truth_col in eval_tables:
            print(f"  [eval] {eval_set}: n_rows={len(df)}", flush=True)
            result = predict_baseline(spec, df, time_col, truth_col)
            metrics = add_condition_summary(result.overall, result.per_condition)
            per_run_rows.append(row_from_metrics(eval_set, spec, metrics))

            if not result.per_condition.empty:
                pc = result.per_condition.copy()
                pc.insert(0, "eval_set", eval_set)
                pc.insert(1, "model_group", spec.group)
                pc.insert(2, "model_label", spec.label)
                pc.insert(3, "model_kind", spec.kind)
                per_condition_frames.append(pc)
            if not result.per_experiment.empty:
                pe = result.per_experiment.copy()
                pe.insert(0, "eval_set", eval_set)
                pe.insert(1, "model_group", spec.group)
                pe.insert(2, "model_label", spec.label)
                pe.insert(3, "model_kind", spec.kind)
                per_experiment_frames.append(pe)

            if eval_set == "q1_grid_all" and "is_observed_window" in result.points.columns:
                for flag, name in [(True, "q1_grid_observed_window"), (False, "q1_grid_extrapolated")]:
                    sub = result.points.loc[result.points["is_observed_window"].astype(bool) == flag]
                    sub_result = summarise(sub)
                    sub_metrics = add_condition_summary(sub_result.overall, sub_result.per_condition)
                    per_run_rows.append(row_from_metrics(name, spec, sub_metrics))

    physics_per_run = pd.DataFrame(per_run_rows)
    physics_per_run.to_csv(out_dir / "physics_per_run_metrics.csv", index=False)
    if per_condition_frames:
        pd.concat(per_condition_frames, ignore_index=True).to_csv(out_dir / "physics_per_condition_metrics.csv", index=False)
    if per_experiment_frames:
        pd.concat(per_experiment_frames, ignore_index=True).to_csv(out_dir / "physics_per_experiment_metrics.csv", index=False)

    reference_per_run = read_reference_frame(reference_report, "per_run_metrics.csv")
    reference_per_condition = read_reference_frame(reference_report, "per_condition_metrics.csv")
    reference_per_experiment = read_reference_frame(reference_report, "per_experiment_metrics.csv")

    combined_per_run = pd.concat([reference_per_run, physics_per_run], ignore_index=True, sort=False)
    combined_group = aggregate_group_rows(combined_per_run)
    combined_per_run.to_csv(out_dir / "per_run_metrics.csv", index=False)
    combined_group.to_csv(out_dir / "group_summary.csv", index=False)
    if not reference_per_condition.empty or per_condition_frames:
        combined_condition = pd.concat(
            [reference_per_condition, *per_condition_frames],
            ignore_index=True,
            sort=False,
        )
        combined_condition.to_csv(out_dir / "per_condition_metrics.csv", index=False)
    if not reference_per_experiment.empty or per_experiment_frames:
        combined_experiment = pd.concat(
            [reference_per_experiment, *per_experiment_frames],
            ignore_index=True,
            sort=False,
        )
        combined_experiment.to_csv(out_dir / "per_experiment_metrics.csv", index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reference_report": str(reference_report),
        "tables": {
            "cdf_uncensored": {"path": str(UNCENSORED_CSV), "n_rows": int(len(uncensored))},
            "p50_observed": {"path": str(P50_Q1_OBSERVED_CSV), "n_rows": int(len(p50_observed))},
            "q1_grid": {"path": str(P50_Q1_GRID_CSV), "n_rows": int(len(q1_grid))},
        },
        "physics_specs": [spec.__dict__ for spec in specs],
    }
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    write_markdown(out_dir, combined_per_run, combined_group, specs)
    print(f"[done] wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ha-run-dir", type=Path, required=True)
    parser.add_argument("--ns-run-dir", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, default=DEFAULT_REFERENCE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.output_dir) if args.output_dir else (
        DEFAULT_OUTPUT_ROOT / f"stage3_fixed_table_eval_with_ha_ns_{datetime.now():%Y%m%d_%H%M%S}"
    )
    specs = [
        BaselineSpec(
            kind="ha",
            group="hiroyasu_arai_calibrated",
            label="ha_calibrated",
            run_dir=resolve_path(args.ha_run_dir),
        ),
        BaselineSpec(
            kind="ns",
            group="naber_siebers_delay",
            label="ns_delay",
            run_dir=resolve_path(args.ns_run_dir),
        ),
    ]
    evaluate_all(specs=specs, reference_report=resolve_path(args.reference_report), out_dir=out_dir)


if __name__ == "__main__":
    main()
