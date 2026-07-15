"""Eval-set definitions and table loading for a synthetic-data root.

The canonical point tables (``cdf_uncensored``, ``full_clean``,
``p50_observed``, ``q1_grid_all``) live at fixed relative paths under every
dataset root
(lv2 / lv3_qc_gated share an identical schema).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from MLP.eval_pipeline.common import guard_dataset_root, resolve_path

#: Feature columns every eval table must provide to rebuild model inputs.
REQUIRED_CONDITION_COLS = (
    "umbrella_angle_deg",
    "plumes",
    "diameter_mm",
    "injection_duration_us",
    "injection_pressure_bar",
    "chamber_pressure_bar",
    "control_backpressure_bar",
)

#: Metadata columns propagated into the points table when present.
META_COLS_FOR_POINTS = (
    "condition_id",
    "condition_key",
    "traj_key",
    "experiment_name",
    "test_name",
    "file_path",
    "file_name",
    "file_stem",
    "plume_idx",
    "frame_pos",
    "time_bin",
    "time_bin_start_ms",
    "time_bin_end_ms",
    "is_observed_window",
    "is_right_censored",
    "is_right_censored_bin",
    "censor_start_reason",
    "b_censor_start",
    "b_density_start",
    "b_fov_start",
    "n_points_in_bin",
    "n_points_in_p50_bin",
    "n_traces_in_bin",
)

#: Eval sets that get full probabilistic diagnostics (matches legacy behavior).
PROBABILISTIC_EVAL_SETS = {"cdf_uncensored", "full_clean", "p50_observed"}

#: Weight column for the weighted probabilistic variant, per eval set.
WEIGHT_COL_BY_EVAL_SET = {"p50_observed": "n_points_in_p50_bin"}


@dataclass(frozen=True)
class EvalSetSpec:
    name: str
    path: Path
    time_col: str
    truth_col: str
    group_col: str
    description: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    eval_sets: tuple[EvalSetSpec, ...] = field(default_factory=tuple)


def eval_set_specs(root: Path | str, names: list[str] | None = None) -> tuple[EvalSetSpec, ...]:
    """Canonical eval-set specs rooted at an explicit dataset root."""
    resolved = guard_dataset_root(resolve_path(root))
    all_specs = (
        EvalSetSpec(
            name="cdf_uncensored",
            path=resolved / "cdf_right_censoring_points" / "cdf_points_uncensored.csv",
            time_col="time_ms",
            truth_col="penetration_mm",
            group_col="condition_id",
            description="CDF points after density/FOV right-censoring removal",
        ),
        EvalSetSpec(
            name="full_clean",
            path=resolved / "cdf_right_censoring_points" / "cdf_points_all.csv",
            time_col="time_ms",
            truth_col="penetration_mm",
            group_col="condition_id",
            description="All clean CDF points, including right-censored regions",
        ),
        EvalSetSpec(
            name="p50_observed",
            path=resolved / "p50_q1_oracle" / "p50_q1_observed_fit_points.csv",
            time_col="time_ms",
            truth_col="penetration_mm",
            group_col="condition_id",
            description="Observed per-condition P50 points used by the q1 oracle fit",
        ),
        EvalSetSpec(
            name="q1_grid_all",
            path=resolved / "p50_q1_oracle" / "p50_q1_predictions.csv",
            time_col="time_ms",
            truth_col="q1_fit_mm",
            group_col="condition_id",
            description="Q1 oracle prediction grid over 0-5 ms",
        ),
    )
    if names is None:
        return all_specs
    by_name = {s.name: s for s in all_specs}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise KeyError(f"Unknown eval set(s) {unknown}; choose from {sorted(by_name)}")
    return tuple(by_name[n] for n in names)


def make_dataset_spec(name: str, root: Path | str,
                      eval_set_names: list[str] | None = None) -> DatasetSpec:
    resolved = guard_dataset_root(resolve_path(root))
    return DatasetSpec(name=name, root=resolved, eval_sets=eval_set_specs(resolved, eval_set_names))


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y"})


def load_eval_table(spec: EvalSetSpec, *, filter_experiment: str | None = None,
                    t_min_ms: float = 0.0, t_max_ms: float = 5.0,
                    limit_conditions: int | None = None) -> pd.DataFrame:
    """Load + validate one eval table, filtered to finite rows in the time window.

    ``limit_conditions`` keeps only the first N condition groups — smoke-test
    hook so a full pipeline pass can be exercised in seconds.
    """
    if not spec.path.exists():
        raise FileNotFoundError(f"Eval table not found: {spec.path}")
    df = pd.read_csv(spec.path, low_memory=False)
    required = {spec.time_col, spec.truth_col, *REQUIRED_CONDITION_COLS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{spec.path} missing required columns: {missing}")
    if filter_experiment is not None:
        if "experiment_name" not in df.columns:
            raise KeyError(f"{spec.path} lacks experiment_name needed by --filter-experiment.")
        df = df.loc[df["experiment_name"].astype(str) == str(filter_experiment)].copy()
        if df.empty:
            raise ValueError(f"{spec.name}: 0 rows for experiment_name={filter_experiment!r}.")
    time_vals = pd.to_numeric(df[spec.time_col], errors="coerce").to_numpy(dtype=float)
    truth_vals = pd.to_numeric(df[spec.truth_col], errors="coerce").to_numpy(dtype=float)
    mask = (
        np.isfinite(time_vals)
        & np.isfinite(truth_vals)
        & (time_vals >= float(t_min_ms))
        & (time_vals <= float(t_max_ms))
    )
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{spec.name} has no finite rows in [{t_min_ms}, {t_max_ms}] ms.")
    if limit_conditions is not None and spec.group_col in df.columns:
        keep = df[spec.group_col].drop_duplicates().head(int(limit_conditions))
        df = df.loc[df[spec.group_col].isin(keep)].reset_index(drop=True)
    return df


def feature_group_col(df: pd.DataFrame, spec: EvalSetSpec) -> str | None:
    """Column identifying rows that share one injection condition."""
    if spec.group_col in df.columns:
        return spec.group_col
    for candidate in ("condition_id", "traj_key", "file_path"):
        if candidate in df.columns:
            return candidate
    return None


def observed_window_slices(points: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a q1-grid points table into observed-window vs extrapolated rows."""
    if "is_observed_window" not in points.columns:
        return {}
    observed = coerce_bool_series(points["is_observed_window"])
    out: dict[str, pd.DataFrame] = {}
    for label, mask in (("q1_grid_observed_window", observed), ("q1_grid_extrapolated", ~observed)):
        sub = points.loc[mask]
        if not sub.empty:
            out[label] = sub
    return out


def censoring_slices(points: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a full-clean CDF table into uncensored vs right-censored rows."""
    if "is_right_censored" not in points.columns:
        return {}
    censored = coerce_bool_series(points["is_right_censored"])
    out: dict[str, pd.DataFrame] = {}
    for label, mask in (("uncensored_rows", ~censored), ("right_censored_rows", censored)):
        sub = points.loc[mask]
        if not sub.empty:
            out[label] = sub
    return out
