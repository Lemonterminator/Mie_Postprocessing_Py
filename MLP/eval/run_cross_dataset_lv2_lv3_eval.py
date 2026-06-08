"""Run fixed point-table evaluations across lv2 and lv3 QC-gated data roots.

This module intentionally passes synthetic roots explicitly. Do not route these
checks through MLP/synthetic_data because that path may be a junction.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "MLP" / "eval" / "cross_dataset_lv2_lv3"
DEFAULT_EVAL_SETS = (
    "cdf_uncensored",
    "p50_observed",
    "q1_grid_all",
)
ALL_REPORTED_EVAL_SETS = (
    "cdf_uncensored",
    "p50_observed",
    "q1_grid_all",
    "q1_grid_observed_window",
    "q1_grid_extrapolated",
)
REQUIRED_DATASET_FILES = (
    Path("cdf_right_censoring_points") / "cdf_points_uncensored.csv",
    Path("p50_q1_oracle") / "p50_q1_observed_fit_points.csv",
    Path("p50_q1_oracle") / "p50_q1_predictions.csv",
)
AGGREGATE_COLUMNS = (
    "dataset_version",
    "training_dataset",
    "model_family",
    "model_label",
    "checkpoint_path",
    "eval_set",
    "n_points",
    "rmse_mm",
    "mae_mm",
    "bias_mm",
    "p95_abs_err_mm",
    "coverage_1sigma",
    "coverage_2sigma",
    "mean_pred_std_mm",
    "prob_ece",
    "prob_crps_mean",
    "eval_dir",
)


@dataclass(frozen=True)
class DatasetSpec:
    version: str
    root: Path


@dataclass(frozen=True)
class CheckpointSpec:
    model_label: str
    model_family: str
    kind: str
    path: Path
    training_dataset: str
    seed: int | None = None


@dataclass(frozen=True)
class EvalResult:
    dataset_version: str
    model_label: str
    eval_dir: Path


def repo_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def dataset_specs() -> tuple[DatasetSpec, ...]:
    return (
        DatasetSpec("lv2", repo_path("MLP", "synthetic_data_clean_lv2")),
        DatasetSpec("lv3_qc_gated", repo_path("MLP", "synthetic_data_clean_lv3_qc_gated")),
    )


def curated_checkpoints() -> tuple[CheckpointSpec, ...]:
    thesis_root = repo_path(
        "MLP", "best_models", "thesis_baselines", "production_mlp_modeA_5seed"
    )
    thesis_mlp = tuple(
        CheckpointSpec(
            model_label=f"thesis_mlp_seed_{seed}",
            model_family="thesis_mlp_modeA",
            kind="mlp",
            path=thesis_root / f"seed_{seed}",
            training_dataset="pre_qc_thesis",
            seed=int(seed),
        )
        for seed in ("07", "17", "42", "99", "2024")
    )
    return thesis_mlp + (
        CheckpointSpec(
            model_label="thesis_svgp_seed_42",
            model_family="thesis_svgp_singleoutput",
            kind="single_svgp",
            path=repo_path(
                "MLP",
                "best_models",
                "thesis_baselines",
                "svgp_stage3_singleoutput",
                "per_seed",
                "seed_42",
                "model.pt",
            ),
            training_dataset="pre_qc_thesis",
            seed=42,
        ),
        CheckpointSpec(
            model_label="family_head_baseline",
            model_family="family_head_baseline",
            kind="mlp",
            path=repo_path("MLP", "best_models", "residual_study", "family_head_baseline"),
            training_dataset="pre_qc_production",
            seed=42,
        ),
        CheckpointSpec(
            model_label="residual_fh",
            model_family="residual_fh",
            kind="mlp",
            path=repo_path("MLP", "best_models", "residual_study", "residual_fh"),
            training_dataset="pre_qc_production",
            seed=42,
        ),
        CheckpointSpec(
            model_label="residual_film",
            model_family="residual_film",
            kind="mlp",
            path=repo_path("MLP", "best_models", "residual_study", "residual_film"),
            training_dataset="pre_qc_production",
            seed=42,
        ),
        CheckpointSpec(
            model_label="residual_svgp",
            model_family="residual_svgp",
            kind="residual_svgp",
            path=repo_path("MLP", "best_models", "residual_study", "residual_svgp"),
            training_dataset="pre_qc_production",
            seed=42,
        ),
        CheckpointSpec(
            model_label="qc_retrained_residual_fh",
            model_family="residual_fh",
            kind="mlp",
            path=repo_path("MLP", "best_models", "qc_gated_retrains", "residual_fh"),
            training_dataset="lv3_qc_gated",
            seed=42,
        ),
        CheckpointSpec(
            model_label="qc_retrained_residual_film",
            model_family="residual_film",
            kind="mlp",
            path=repo_path("MLP", "best_models", "qc_gated_retrains", "residual_film"),
            training_dataset="lv3_qc_gated",
            seed=42,
        ),
        CheckpointSpec(
            model_label="qc_retrained_residual_svgp",
            model_family="residual_svgp",
            kind="residual_svgp",
            path=repo_path("MLP", "best_models", "qc_gated_retrains", "residual_svgp"),
            training_dataset="lv3_qc_gated",
            seed=42,
        ),
    )


def smoke_checkpoints() -> tuple[CheckpointSpec, ...]:
    labels = {"residual_film", "residual_svgp"}
    return tuple(spec for spec in curated_checkpoints() if spec.model_label in labels)


def rel_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def check_dataset_roots(datasets: Iterable[DatasetSpec]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for dataset in datasets:
        files: dict[str, Any] = {}
        if dataset.root.absolute() == repo_path("MLP", "synthetic_data").absolute():
            errors.append(f"{dataset.version} resolves to MLP/synthetic_data; explicit root required")
        for required in REQUIRED_DATASET_FILES:
            path = dataset.root / required
            if not path.exists():
                errors.append(f"Missing required table: {rel_to_repo(path)}")
                files[required.as_posix()] = {"exists": False, "rows": None}
                continue
            try:
                rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
            except UnicodeDecodeError:
                rows = sum(1 for _ in path.open("r")) - 1
            files[required.as_posix()] = {"exists": True, "rows": max(rows, 0)}
        report[dataset.version] = {
            "root": rel_to_repo(dataset.root),
            "required_files": files,
        }
    if errors:
        raise FileNotFoundError("\n".join(errors))
    return report


def check_checkpoints(checkpoints: Iterable[CheckpointSpec]) -> None:
    errors: list[str] = []
    for spec in checkpoints:
        if spec.kind == "single_svgp":
            expected = spec.path
        elif spec.kind == "residual_svgp":
            expected = spec.path / "per_seed" / f"seed_{spec.seed or 42}" / "model.pt"
        else:
            expected = spec.path / "best_model_refinement.pt"
        if not expected.exists():
            errors.append(f"Missing checkpoint for {spec.model_label}: {rel_to_repo(expected)}")
    if errors:
        raise FileNotFoundError("\n".join(errors))


def run_mlp_eval(
    dataset: DatasetSpec,
    spec: CheckpointSpec,
    eval_sets: tuple[str, ...],
    output_root: Path,
    device: str,
    batch_points: int,
    save_points: bool,
    save_plots: bool,
) -> Path:
    from MLP.eval.inference_rmse_on_point_tables import (
        run_point_table_evaluation as run_mlp_point_table_evaluation,
    )

    tag = f"{dataset.version}__{spec.model_label}"
    out_dir, _summary = run_mlp_point_table_evaluation(
        refinement_run=spec.path,
        synthetic_root=dataset.root,
        eval_sets=eval_sets,
        primary_eval_set="cdf_uncensored",
        device=None if device == "auto" else device,
        output_root=output_root,
        tag=tag,
        batch_points=batch_points,
        save_points=save_points,
        save_plots=save_plots,
    )
    return Path(out_dir)


def run_residual_svgp_eval(
    dataset: DatasetSpec,
    spec: CheckpointSpec,
    eval_sets: tuple[str, ...],
    output_root: Path,
    device: str,
    batch_points: int,
    save_points: bool,
) -> Path:
    from MLP.GP_training.eval_residual_multitask_svgp_on_point_tables import (
        run_point_table_evaluation as run_residual_svgp_point_table_evaluation,
    )

    tag = f"{dataset.version}__{spec.model_label}"
    out_dir, _summary = run_residual_svgp_point_table_evaluation(
        residual_gp_run=spec.path,
        synthetic_root=dataset.root,
        seed=spec.seed or 42,
        eval_sets=eval_sets,
        primary_eval_set="cdf_uncensored",
        device=device,
        output_root=output_root,
        tag=tag,
        batch_points=batch_points,
        save_points=save_points,
    )
    return Path(out_dir)


def run_single_svgp_eval(
    dataset: DatasetSpec,
    spec: CheckpointSpec,
    eval_sets: tuple[str, ...],
    output_root: Path,
    device: str,
    batch_points: int,
) -> Path:
    if set(eval_sets) != set(DEFAULT_EVAL_SETS):
        unsupported = set(eval_sets) - set(DEFAULT_EVAL_SETS)
        if unsupported:
            raise ValueError(f"Unsupported single-output SVGP eval sets: {sorted(unsupported)}")

    sys.modules.setdefault(
        "MLP.MLP_training.run_gp_baseline",
        importlib.import_module("MLP.GP_training.run_gp_baseline"),
    )
    from MLP.eval import evaluate_stage3_fixed_tables as fixed

    tag = f"{dataset.version}__{spec.model_label}"
    out_dir = output_root / f"fixed_eval_{datetime.now():%Y%m%d_%H%M%S}_{tag}"
    old_paths = {
        "UNCENSORED_CSV": fixed.UNCENSORED_CSV,
        "P50_Q1_ROOT": fixed.P50_Q1_ROOT,
        "P50_Q1_FIT_METRICS_CSV": fixed.P50_Q1_FIT_METRICS_CSV,
        "P50_Q1_OBSERVED_CSV": fixed.P50_Q1_OBSERVED_CSV,
        "P50_Q1_GRID_CSV": fixed.P50_Q1_GRID_CSV,
    }
    try:
        p50_root = dataset.root / "p50_q1_oracle"
        fixed.UNCENSORED_CSV = (
            dataset.root / "cdf_right_censoring_points" / "cdf_points_uncensored.csv"
        )
        fixed.P50_Q1_ROOT = p50_root
        fixed.P50_Q1_FIT_METRICS_CSV = p50_root / "p50_q1_fit_metrics.csv"
        fixed.P50_Q1_OBSERVED_CSV = p50_root / "p50_q1_observed_fit_points.csv"
        fixed.P50_Q1_GRID_CSV = p50_root / "p50_q1_predictions.csv"
        fixed.evaluate_all(
            specs=[
                fixed.ModelSpec(
                    kind="gp",
                    group=spec.model_family,
                    label=spec.model_label,
                    path=spec.path,
                )
            ],
            out_dir=out_dir,
            device=fixed.choose_device(device),
            batch_points=batch_points,
        )
    finally:
        fixed.UNCENSORED_CSV = old_paths["UNCENSORED_CSV"]
        fixed.P50_Q1_ROOT = old_paths["P50_Q1_ROOT"]
        fixed.P50_Q1_FIT_METRICS_CSV = old_paths["P50_Q1_FIT_METRICS_CSV"]
        fixed.P50_Q1_OBSERVED_CSV = old_paths["P50_Q1_OBSERVED_CSV"]
        fixed.P50_Q1_GRID_CSV = old_paths["P50_Q1_GRID_CSV"]
    return out_dir


def run_one(
    dataset: DatasetSpec,
    spec: CheckpointSpec,
    eval_sets: tuple[str, ...],
    output_root: Path,
    device: str,
    batch_points_mlp: int,
    batch_points_gp: int,
    save_points: bool,
    save_plots: bool,
) -> EvalResult:
    if spec.kind == "mlp":
        eval_dir = run_mlp_eval(
            dataset,
            spec,
            eval_sets,
            output_root,
            device,
            batch_points_mlp,
            save_points,
            save_plots,
        )
    elif spec.kind == "residual_svgp":
        eval_dir = run_residual_svgp_eval(
            dataset,
            spec,
            eval_sets,
            output_root,
            device,
            batch_points_gp,
            save_points,
        )
    elif spec.kind == "single_svgp":
        eval_dir = run_single_svgp_eval(
            dataset,
            spec,
            eval_sets,
            output_root,
            device,
            batch_points_gp,
        )
    else:
        raise ValueError(f"Unknown checkpoint kind: {spec.kind}")
    return EvalResult(dataset.version, spec.model_label, eval_dir)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def skipped_feature_groups(eval_dir: Path, eval_set: str) -> int | float:
    summary = read_json(eval_dir / "metrics_summary.json")
    try:
        value = summary.get("eval_sets", {}).get(eval_set, {}).get("skipped_feature_groups")
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return math.nan


def normalize_metric_frame(
    result: EvalResult,
    checkpoints_by_label: dict[str, CheckpointSpec],
) -> pd.DataFrame:
    metrics_path = result.eval_dir / "per_run_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {rel_to_repo(metrics_path)}")
    spec = checkpoints_by_label[result.model_label]
    frame = pd.read_csv(metrics_path)
    if "model_kind" in frame.columns:
        frame = frame[frame["model_kind"].astype(str).str.lower() != "oracle"].copy()
    if "model_group" in frame.columns:
        frame = frame[frame["model_group"].astype(str).str.lower() != "q1_oracle"].copy()
    frame["dataset_version"] = result.dataset_version
    frame["training_dataset"] = spec.training_dataset
    frame["model_family"] = spec.model_family
    frame["model_label"] = spec.model_label
    frame["checkpoint_path"] = rel_to_repo(spec.path)
    frame["eval_dir"] = rel_to_repo(result.eval_dir)
    if "eval_set" not in frame.columns:
        frame["eval_set"] = "unknown"
    if "p95_abs_err_mm" not in frame.columns and "p95_abs_error_mm" in frame.columns:
        frame["p95_abs_err_mm"] = frame["p95_abs_error_mm"]
    for column in AGGREGATE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["skipped_feature_groups"] = [
        skipped_feature_groups(result.eval_dir, str(eval_set)) for eval_set in frame["eval_set"]
    ]
    passthrough = [
        "median_abs_err_mm",
        "p90_abs_err_mm",
        "prob_sharpness_mm",
        "skipped_feature_groups",
    ]
    keep = list(AGGREGATE_COLUMNS) + [col for col in passthrough if col in frame.columns]
    for column in (
        "n_points",
        "rmse_mm",
        "mae_mm",
        "bias_mm",
        "p95_abs_err_mm",
        "coverage_1sigma",
        "coverage_2sigma",
        "mean_pred_std_mm",
        "prob_ece",
        "prob_crps_mean",
        "median_abs_err_mm",
        "p90_abs_err_mm",
        "prob_sharpness_mm",
        "skipped_feature_groups",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.loc[:, keep]


def first_metric_row(
    frame: pd.DataFrame,
    dataset_version: str,
    model_label: str,
    eval_set: str,
) -> pd.Series | None:
    matches = frame[
        (frame["dataset_version"] == dataset_version)
        & (frame["model_label"] == model_label)
        & (frame["eval_set"] == eval_set)
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def metric_delta(
    row: dict[str, Any],
    name: str,
    before: pd.Series | None,
    after: pd.Series | None,
    before_prefix: str,
    after_prefix: str,
) -> None:
    if before is None or after is None:
        row[f"{before_prefix}_{name}"] = pd.NA
        row[f"{after_prefix}_{name}"] = pd.NA
        row[f"delta_{name}"] = pd.NA
        row[f"relative_delta_{name}_pct"] = pd.NA
        return
    before_value = pd.to_numeric(pd.Series([before.get(name)]), errors="coerce").iloc[0]
    after_value = pd.to_numeric(pd.Series([after.get(name)]), errors="coerce").iloc[0]
    row[f"{before_prefix}_{name}"] = before_value
    row[f"{after_prefix}_{name}"] = after_value
    if pd.isna(before_value) or pd.isna(after_value):
        row[f"delta_{name}"] = pd.NA
        row[f"relative_delta_{name}_pct"] = pd.NA
    else:
        row[f"delta_{name}"] = after_value - before_value
        row[f"relative_delta_{name}_pct"] = (
            100.0 * (after_value - before_value) / before_value if before_value else pd.NA
        )


def build_post_training_gain(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = "family_head_baseline"
    targets = ("residual_fh", "residual_film", "residual_svgp")
    for eval_set in ALL_REPORTED_EVAL_SETS:
        baseline_row = first_metric_row(frame, "lv2", baseline, eval_set)
        for target in targets:
            target_row = first_metric_row(frame, "lv2", target, eval_set)
            row: dict[str, Any] = {
                "dataset_version": "lv2",
                "eval_set": eval_set,
                "baseline_model_label": baseline,
                "model_label": target,
            }
            for metric in (
                "rmse_mm",
                "mae_mm",
                "p95_abs_err_mm",
                "prob_ece",
                "prob_crps_mean",
            ):
                metric_delta(row, metric, baseline_row, target_row, "baseline", "model")
            rows.append(row)
    return pd.DataFrame(rows)


def build_qc_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = (
        ("residual_fh", "qc_retrained_residual_fh"),
        ("residual_film", "qc_retrained_residual_film"),
        ("residual_svgp", "qc_retrained_residual_svgp"),
    )
    metrics = ("rmse_mm", "mae_mm", "p95_abs_err_mm", "prob_ece", "prob_crps_mean")
    for eval_set in ALL_REPORTED_EVAL_SETS:
        for old_label, qc_label in pairs:
            old_lv2 = first_metric_row(frame, "lv2", old_label, eval_set)
            old_lv3 = first_metric_row(frame, "lv3_qc_gated", old_label, eval_set)
            qc_lv3 = first_metric_row(frame, "lv3_qc_gated", qc_label, eval_set)
            qc_lv2 = first_metric_row(frame, "lv2", qc_label, eval_set)
            row: dict[str, Any] = {
                "model_family": old_label,
                "old_model_label": old_label,
                "qc_retrained_model_label": qc_label,
                "eval_set": eval_set,
            }
            for metric in metrics:
                old_lv2_value = (
                    pd.to_numeric(pd.Series([old_lv2.get(metric)]), errors="coerce").iloc[0]
                    if old_lv2 is not None
                    else pd.NA
                )
                old_lv3_value = (
                    pd.to_numeric(pd.Series([old_lv3.get(metric)]), errors="coerce").iloc[0]
                    if old_lv3 is not None
                    else pd.NA
                )
                qc_lv3_value = (
                    pd.to_numeric(pd.Series([qc_lv3.get(metric)]), errors="coerce").iloc[0]
                    if qc_lv3 is not None
                    else pd.NA
                )
                qc_lv2_value = (
                    pd.to_numeric(pd.Series([qc_lv2.get(metric)]), errors="coerce").iloc[0]
                    if qc_lv2 is not None
                    else pd.NA
                )
                row[f"old_lv2_{metric}"] = old_lv2_value
                row[f"old_lv3_{metric}"] = old_lv3_value
                row[f"qc_retrained_lv3_{metric}"] = qc_lv3_value
                row[f"qc_retrained_lv2_{metric}"] = qc_lv2_value
                if pd.isna(old_lv2_value) or pd.isna(old_lv3_value):
                    row[f"target_eval_cleanup_delta_{metric}"] = pd.NA
                else:
                    row[f"target_eval_cleanup_delta_{metric}"] = old_lv3_value - old_lv2_value
                if pd.isna(old_lv3_value) or pd.isna(qc_lv3_value):
                    row[f"qc_retrain_delta_{metric}"] = pd.NA
                else:
                    row[f"qc_retrain_delta_{metric}"] = qc_lv3_value - old_lv3_value
                if pd.isna(old_lv2_value) or pd.isna(qc_lv3_value):
                    row[f"total_old_lv2_to_qc_retrained_lv3_delta_{metric}"] = pd.NA
                else:
                    row[f"total_old_lv2_to_qc_retrained_lv3_delta_{metric}"] = (
                        qc_lv3_value - old_lv2_value
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def write_aggregates(
    results: Iterable[EvalResult],
    checkpoints: Iterable[CheckpointSpec],
    output_root: Path,
) -> dict[str, Any]:
    checkpoints_by_label = {spec.model_label: spec for spec in checkpoints}
    frames = [normalize_metric_frame(result, checkpoints_by_label) for result in results]
    if not frames:
        raise RuntimeError("No evaluation results to aggregate")
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics = all_metrics.sort_values(
        ["dataset_version", "eval_set", "model_family", "model_label"],
        kind="stable",
    )
    all_path = output_root / "all_model_dataset_metrics.csv"
    headline_path = output_root / "headline_cdf_uncensored.csv"
    gain_path = output_root / "post_training_gain_lv2.csv"
    qc_path = output_root / "qc_decomposition.csv"
    all_metrics.to_csv(all_path, index=False)
    headline = all_metrics[all_metrics["eval_set"] == "cdf_uncensored"].sort_values(
        ["dataset_version", "rmse_mm"], kind="stable"
    )
    headline.to_csv(headline_path, index=False)
    post_gain = build_post_training_gain(all_metrics)
    post_gain.to_csv(gain_path, index=False)
    qc_decomp = build_qc_decomposition(all_metrics)
    qc_decomp.to_csv(qc_path, index=False)
    skipped = all_metrics[
        pd.to_numeric(all_metrics["skipped_feature_groups"], errors="coerce").fillna(0) > 0
    ]
    return {
        "all_model_dataset_metrics": rel_to_repo(all_path),
        "headline_cdf_uncensored": rel_to_repo(headline_path),
        "post_training_gain_lv2": rel_to_repo(gain_path),
        "qc_decomposition": rel_to_repo(qc_path),
        "n_rows": int(len(all_metrics)),
        "n_headline_rows": int(len(headline)),
        "nonzero_skipped_feature_group_rows": int(len(skipped)),
        "skipped_feature_group_rows": skipped[
            ["dataset_version", "model_label", "eval_set", "skipped_feature_groups", "eval_dir"]
        ].to_dict(orient="records"),
    }


def known_number_checks(frame_path: Path, strict: bool) -> list[dict[str, Any]]:
    frame = pd.read_csv(frame_path)
    checks = (
        ("residual_film", "lv3_qc_gated", "cdf_uncensored", 4.401, 0.08),
        ("qc_retrained_residual_svgp", "lv3_qc_gated", "cdf_uncensored", 3.848, 0.08),
    )
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for model_label, dataset_version, eval_set, expected, tolerance in checks:
        row = first_metric_row(frame, dataset_version, model_label, eval_set)
        if row is None:
            ok = False
            actual = pd.NA
        else:
            actual = pd.to_numeric(pd.Series([row.get("rmse_mm")]), errors="coerce").iloc[0]
            ok = bool(not pd.isna(actual) and abs(float(actual) - expected) <= tolerance)
        result = {
            "model_label": model_label,
            "dataset_version": dataset_version,
            "eval_set": eval_set,
            "metric": "rmse_mm",
            "actual": None if pd.isna(actual) else float(actual),
            "expected": expected,
            "tolerance": tolerance,
            "ok": ok,
        }
        results.append(result)
        if not ok:
            failures.append(
                f"{model_label}/{dataset_version}/{eval_set}: expected {expected} +/- "
                f"{tolerance}, got {actual}"
            )
    if strict and failures:
        raise RuntimeError("Known-number checks failed:\n" + "\n".join(failures))
    return results


def write_readme(output_root: Path, aggregate_report: dict[str, Any]) -> None:
    readme = output_root / "README.md"
    lines = [
        "# lv2 / lv3 Fixed-Point Re-evaluation",
        "",
        "This folder contains controlled point-table re-evaluations of curated ML checkpoints.",
        "",
        "Dataset roots are passed explicitly:",
        "",
        "- lv2: `MLP/synthetic_data_clean_lv2`",
        "- lv3_qc_gated: `MLP/synthetic_data_clean_lv3_qc_gated`",
        "",
        "Source-of-truth aggregate CSVs:",
        "",
        f"- `{aggregate_report['all_model_dataset_metrics']}`",
        f"- `{aggregate_report['headline_cdf_uncensored']}`",
        f"- `{aggregate_report['post_training_gain_lv2']}`",
        f"- `{aggregate_report['qc_decomposition']}`",
        "",
        "Interpretation:",
        "",
        "- Read post-training gain within the same dataset root, especially lv2.",
        "- Decompose QC as target/eval cleanup plus retrain gain.",
        "- Treat lv3_qc_gated as sensitivity evidence, not the only final proof.",
        "- Use cdf_uncensored for headlines and P50/Q1 slices for behavior details.",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-points-mlp", type=int, default=262_144)
    parser.add_argument("--batch-points-gp", type=int, default=65_536)
    parser.add_argument(
        "--eval-set",
        dest="eval_sets",
        action="append",
        choices=DEFAULT_EVAL_SETS,
        help="Repeat to limit top-level eval sets. Default runs all top-level sets.",
    )
    parser.add_argument(
        "--model-label",
        dest="model_labels",
        action="append",
        help="Repeat to run only selected model labels.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run one MLP and one SVGP only")
    parser.add_argument("--save-points", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--strict-known-checks", action="store_true")
    parser.add_argument(
        "--aggregate-from-manifest",
        type=Path,
        help="Reuse eval_dirs from a previous run_manifest.json and only rewrite aggregates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    eval_output_root = output_root / "eval_dirs"
    output_root.mkdir(parents=True, exist_ok=True)
    eval_output_root.mkdir(parents=True, exist_ok=True)

    datasets = dataset_specs()
    checkpoints = smoke_checkpoints() if args.smoke else curated_checkpoints()
    if args.model_labels:
        wanted = set(args.model_labels)
        checkpoints = tuple(spec for spec in checkpoints if spec.model_label in wanted)
        missing = wanted - {spec.model_label for spec in checkpoints}
        if missing:
            raise ValueError(f"Unknown model labels: {sorted(missing)}")
    eval_sets = tuple(args.eval_sets) if args.eval_sets else DEFAULT_EVAL_SETS
    if args.smoke:
        eval_sets = ("cdf_uncensored",)

    if args.aggregate_from_manifest:
        manifest = read_json(args.aggregate_from_manifest)
        manifest_results = manifest.get("results", [])
        checkpoints = tuple(
            spec
            for spec in curated_checkpoints()
            if spec.model_label in {item["model_label"] for item in manifest_results}
        )
        results = [
            EvalResult(
                dataset_version=item["dataset_version"],
                model_label=item["model_label"],
                eval_dir=repo_path(*Path(item["eval_dir"]).parts)
                if not Path(item["eval_dir"]).is_absolute()
                else Path(item["eval_dir"]),
            )
            for item in manifest_results
        ]
        aggregate_report = write_aggregates(results, checkpoints, output_root)
        known_checks = []
        if not args.smoke and {spec.model_label for spec in checkpoints} == {
            spec.model_label for spec in curated_checkpoints()
        }:
            known_checks = known_number_checks(
                output_root / "all_model_dataset_metrics.csv",
                strict=args.strict_known_checks,
            )
        manifest["aggregates"] = aggregate_report
        manifest["known_number_checks"] = known_checks
        manifest["regenerated_aggregates_at"] = datetime.now().isoformat(timespec="seconds")
        manifest_path = output_root / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        write_readme(output_root, aggregate_report)
        print(json.dumps(aggregate_report, indent=2), flush=True)
        if known_checks:
            print(json.dumps({"known_number_checks": known_checks}, indent=2), flush=True)
        return

    dataset_report = check_dataset_roots(datasets)
    check_checkpoints(checkpoints)

    results: list[EvalResult] = []
    for dataset in datasets:
        for spec in checkpoints:
            print(
                f"[cross-eval] {dataset.version} :: {spec.model_label} "
                f"({spec.kind}) on {', '.join(eval_sets)}",
                flush=True,
            )
            results.append(
                run_one(
                    dataset=dataset,
                    spec=spec,
                    eval_sets=eval_sets,
                    output_root=eval_output_root,
                    device=args.device,
                    batch_points_mlp=args.batch_points_mlp,
                    batch_points_gp=args.batch_points_gp,
                    save_points=args.save_points,
                    save_plots=args.save_plots,
                )
            )

    aggregate_report = write_aggregates(results, checkpoints, output_root)
    known_checks = []
    if not args.smoke and {spec.model_label for spec in checkpoints} == {
        spec.model_label for spec in curated_checkpoints()
    }:
        known_checks = known_number_checks(
            output_root / "all_model_dataset_metrics.csv",
            strict=args.strict_known_checks,
        )
    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": rel_to_repo(PROJECT_ROOT),
        "smoke": args.smoke,
        "device": args.device,
        "eval_sets": eval_sets,
        "dataset_report": dataset_report,
        "checkpoints": [asdict(spec) | {"path": rel_to_repo(spec.path)} for spec in checkpoints],
        "results": [asdict(result) | {"eval_dir": rel_to_repo(result.eval_dir)} for result in results],
        "aggregates": aggregate_report,
        "known_number_checks": known_checks,
    }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    write_readme(output_root, aggregate_report)
    print(json.dumps(run_manifest["aggregates"], indent=2), flush=True)
    if known_checks:
        print(json.dumps({"known_number_checks": known_checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
