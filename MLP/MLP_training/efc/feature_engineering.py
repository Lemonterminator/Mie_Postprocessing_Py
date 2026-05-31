from __future__ import annotations

"""Data preparation and feature engineering for the spray-penetration MLP.

Loads cleaned CSV files produced by the Mie-scattering post-processing pipeline,
selects representative or filtered rows per injection condition, applies the
A-scale transform and z-score standardisation, and builds the train/val/test
split DataFrames consumed by Stage-1 and Stage-2 trainers.

Key functions:
    iter_clean_csv_files        — find all clean/*.csv files under a data root.
    build_all_stage_tables      — top-level entry: build StageTables from data_dir.
    build_canonical_feature_table — canonicalise chamber state, compute A_scale,
                                    z-score features for a single split DataFrame.
    build_variant_feature_table — wrapper for Stage-1: selects variant columns and
                                   fits fresh z-score params.
    apply_saved_scaler_state    — reapply Stage-1 scaler to the Stage-2 filtered split
                                   without refitting (guarantees identical feature space).
    select_representative_row   — one row per unique injection condition (Stage-1 training).
    select_filtered_rows        — all rows passing quality cuts (Stage-2 training).
    assign_splits_by_group      — random group-stratified train/val/test split.
    assign_splits_leave_one_out — LONO split: one experiment_name held out as test.
    run_pretrain_collapse_check — verify Stage-1 trajectories do not collapse;
                                   produces diagnostic plots in the run directory.
    compute_a_scale             — A_scale = ΔP^exp_dp · ρ^exp_rho · d^exp_d.
"""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import re

from .data_io import DatasetMeta, RunArtifacts, StageTables, _load_json, build_dataset_registry, merge_config, normalize_dataset_key
from .feature_registry import (
    A_SCALE_DP_EXP_BY_VARIANT,
    BASE_STATIC_FEATURE_COLUMNS,
    BASE_STATIC_ZSCORE_COLUMNS,
    DEFAULT_A_SCALE_DENSITY_EXP,
    DEFAULT_A_SCALE_DP_EXP,
    DEFAULT_A_SCALE_DIAMETER_EXP,
    DEFAULT_RUNS_ROOT,
    DEFAULT_SYNTHETIC_DATA_ROOT,
    DEFAULT_TEST_MATRIX_ROOT,
    DIAMETER_FEATURE_COLUMNS,
    DIAMETER_ZSCORE_COLUMNS,
    FEATURE_COLUMNS_BY_VARIANT,
    LEGACY_FEATURE_COLUMNS,
    LEGACY_FILL_DEFAULTS,
    MIN_TIME_SHIFT_S,
    PRESSURE_FEATURE_COLUMNS,
    PRESSURE_ZSCORE_COLUMNS,
    REFERENCE_PRESSURE_BAR,
    TARGET_SCALE_MODE_BY_VARIANT,
    TIME_FEATURE,
    ZSCORE_BASE_COLUMNS_BY_VARIANT,
    resolve_tilt_angle_radian,
    umbrella_to_tilt_radian,
)
from .models import (
    reconstruct_penetration_series,
    spray_penetration_model_sigmoid,
    spray_penetration_model_sigmoid_d_dt,
)

REFERENCE_DENSITY_KG_M3 = 5.0


def infer_feature_family(feature_columns: Sequence[str]) -> str:
    if LEGACY_FEATURE_COLUMNS.issubset(set(feature_columns)):
        return "legacy_raw"
    return "engineered_v2"


# Tier-3A / Tier-3B routing channels. family_id is a 2-way split (factory-fresh
# vs modified), nozzle_id is the 6-way split (Nozzle0..Nozzle5). Both are
# derived from the experiment_name / dataset_key strings, never z-scored, and
# only appended to feature_columns when an ablation explicitly opts in.
NOZZLE_NAME_TO_ID: dict[str, int] = {f"Nozzle{i}": i for i in range(9)}
_NOZZLE_ID_RE = re.compile(r"Nozzle(\d+)", re.IGNORECASE)


def family_id_from_name(name: str) -> int:
    """0 for factory-fresh Nozzle0, 1 for any modified nozzle."""
    m = _NOZZLE_ID_RE.search(str(name))
    return 0 if (m is not None and int(m.group(1)) == 0) else 1


def nozzle_id_from_name(name: str, *, default: int = -1) -> int:
    """Extract the nozzle index from a full experiment name (e.g. 'BC20220627_HZ_Nozzle0' -> 0).

    Handles both short ('Nozzle3') and long ('BC20241014_HZ_Nozzle3') forms.
    Returns `default` (-1) for strings that contain no Nozzle<N> pattern.
    """
    m = _NOZZLE_ID_RE.search(str(name))
    return int(m.group(1)) if m is not None else default


def _add_routing_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add family_id and nozzle_id columns if they aren't already present.

    Cheap (int8 each) and harmless to add unconditionally: if no ablation
    opts in via feature_columns the model never reads them.
    """
    name_source = next(
        (c for c in ("experiment_name", "source_dataset_name", "folder", "dataset_key") if c in df.columns),
        None,
    )
    if name_source is None:
        return df
    if "family_id" not in df.columns:
        df["family_id"] = df[name_source].astype(str).map(family_id_from_name).astype("int8")
    if "nozzle_id" not in df.columns:
        df["nozzle_id"] = df[name_source].astype(str).map(nozzle_id_from_name).astype("int8")
    return df


def iter_clean_csv_files(data_root: Path | str) -> list[Path]:
    root = Path(data_root)
    return sorted(p for p in root.rglob("*.csv") if p.parent.name == "clean")


def infer_dataset_name_from_csv_path(csv_path: Path) -> str:
    for candidate in reversed(csv_path.parts[:-1]):
        try:
            normalize_dataset_key(candidate)
        except KeyError:
            continue
        return candidate
    raise KeyError(f"Could not infer dataset name from path: {csv_path}")


def build_sample_group_id(dataset_key: str, csv_path: Path, df: pd.DataFrame) -> str:
    if "injection_duration_us" in df.columns:
        duration_values = pd.to_numeric(df["injection_duration_us"], errors="coerce").dropna().unique()
        if len(duration_values) == 1:
            return f"{dataset_key}|{csv_path}|dur={float(duration_values[0]):.6f}"
    return f"{dataset_key}|{csv_path}"


def select_representative_row(
    clean_folder: Path,
    csv_path: Path,
    fitted_df: pd.DataFrame,
    compare_time_s: float,
) -> dict[str, Any] | None:
    required_cols = ["log_k_sqrt", "log_k_quarter", "log_t0", "log_s"]
    if fitted_df.empty or any(col not in fitted_df.columns for col in required_cols):
        return None

    values = [pd.to_numeric(fitted_df[col], errors="coerce").to_numpy() for col in required_cols]
    penetration = spray_penetration_model_sigmoid(values, compare_time_s)
    if not np.isfinite(penetration).any():
        return None

    median_penetration = float(np.nanmedian(penetration))
    idx = int(np.nanargmin(np.abs(penetration - median_penetration)))
    row = fitted_df.iloc[idx].to_dict()
    row["source_file"] = str(csv_path)
    row["source_folder"] = str(clean_folder)
    row["comparison_time_s"] = float(compare_time_s)
    row["penetration_at_comparison_time"] = float(penetration[idx])
    row["median_penetration_in_group"] = median_penetration
    return row


def select_filtered_rows(
    clean_folder: Path,
    csv_path: Path,
    fitted_df: pd.DataFrame,
    compare_time_s: float,
    z_score_threshold: float = 3.0,
) -> list[dict[str, Any]]:
    required_cols = ["log_k_sqrt", "log_k_quarter", "log_t0", "log_s"]
    if fitted_df.empty or any(col not in fitted_df.columns for col in required_cols):
        return []

    values = [pd.to_numeric(fitted_df[col], errors="coerce").to_numpy() for col in required_cols]
    penetration = spray_penetration_model_sigmoid(values, compare_time_s)
    d_dt = spray_penetration_model_sigmoid_d_dt(values, compare_time_s)
    finite_mask = np.isfinite(penetration) & np.isfinite(d_dt)
    if not finite_mask.any():
        return []

    valid = penetration[finite_mask]
    med = float(np.nanmedian(valid))
    mad = float(np.nanmedian(np.abs(valid - med)))
    sigma = 1.4826 * mad
    sigma_safe = sigma if sigma > 0 else 1.0

    z = np.full(len(fitted_df), np.nan, dtype=float)
    z[finite_mask] = (penetration[finite_mask] - med) / sigma_safe
    mask_z = finite_mask & (np.abs(z) > float(z_score_threshold))
    mask_dt = finite_mask & (np.abs(d_dt) < 1e-6)
    selected_mask = finite_mask & ~mask_z & ~mask_dt
    rows: list[dict[str, Any]] = []
    for idx in np.flatnonzero(selected_mask):
        row = fitted_df.iloc[int(idx)].to_dict()
        row["source_file"] = str(csv_path)
        row["source_folder"] = str(clean_folder)
        row["comparison_time_s"] = float(compare_time_s)
        row["penetration_at_comparison_time"] = float(penetration[idx])
        row["d_dt_penetration_at_comparison_time"] = float(d_dt[idx])
        row["median_penetration_in_group"] = med
        row["robust_sigma_in_group"] = sigma
        row["penetration_z_score"] = float(z[idx])
        row["is_penetration_z_score_outlier"] = bool(mask_z[idx])
        row["is_flat_at_comparison_time"] = bool(mask_dt[idx])
        rows.append(row)
    return rows


def _split_frames_for_selection(dataset_key: str, fitted_df: pd.DataFrame) -> list[pd.DataFrame]:
    if dataset_key == "ds300" and "injection_duration_us" in fitted_df.columns:
        return [group.copy() for _, group in fitted_df.groupby("injection_duration_us", dropna=False)]
    return [fitted_df]


def collect_selected_rows(
    data_root: Path | str,
    registry: Mapping[str, DatasetMeta],
    *,
    compare_time_s: float,
    selection_mode: str,
    max_curves: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for csv_path in iter_clean_csv_files(Path(data_root)):
        dataset_name = infer_dataset_name_from_csv_path(csv_path)
        dataset_key = normalize_dataset_key(dataset_name)
        clean_folder = csv_path.parent
        fitted_df = pd.read_csv(csv_path)
        for group_df in _split_frames_for_selection(dataset_key, fitted_df):
            sample_group_id = build_sample_group_id(dataset_key, csv_path, group_df)
            if selection_mode == "representative":
                row = select_representative_row(clean_folder, csv_path, group_df, compare_time_s)
                if row is None:
                    continue
                row["selection_mode"] = "representative"
                row["dataset_key"] = dataset_key
                row["source_dataset_name"] = dataset_name
                row["sample_group_id"] = sample_group_id
                rows.append(row)
            elif selection_mode == "filtered":
                for row in select_filtered_rows(clean_folder, csv_path, group_df, compare_time_s):
                    row["selection_mode"] = "filtered"
                    row["dataset_key"] = dataset_key
                    row["source_dataset_name"] = dataset_name
                    row["sample_group_id"] = sample_group_id
                    rows.append(row)
            else:
                raise ValueError(f"Unsupported selection_mode '{selection_mode}'")

    df = pd.DataFrame(rows).reset_index(drop=True)
    if max_curves is not None and len(df) > int(max_curves):
        df = df.sample(n=int(max_curves), random_state=42).sort_index().reset_index(drop=True)
    return df


def _require_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' missing from row table.")
    values = pd.to_numeric(df[col], errors="coerce").astype(float)
    if values.isna().any():
        raise ValueError(f"Column '{col}' contains NaN after numeric coercion.")
    return values


def linear_density_from_pressure(pressure_bar: float) -> float:
    return REFERENCE_DENSITY_KG_M3 * float(pressure_bar) / REFERENCE_PRESSURE_BAR


def linear_pressure_from_density(density_kg_m3: float) -> float:
    return REFERENCE_PRESSURE_BAR * float(density_kg_m3) / REFERENCE_DENSITY_KG_M3


def compute_a_scale(
    delta_pressure_bar: Any,
    ambient_density_kg_m3: Any,
    diameter_mm: Any,
    *,
    delta_pressure_exp: float = DEFAULT_A_SCALE_DP_EXP,
) -> Any:
    return (
        np.power(delta_pressure_bar, float(delta_pressure_exp))
        * np.power(ambient_density_kg_m3, DEFAULT_A_SCALE_DENSITY_EXP)
        * np.power(diameter_mm, DEFAULT_A_SCALE_DIAMETER_EXP)
    )


def apply_a_scale_transform(df_in: pd.DataFrame, *, delta_pressure_exp: float) -> pd.DataFrame:
    df = df_in.copy()
    required = ["delta_pressure_bar_phys", "ambient_density_kg_m3", "diameter_mm"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Cannot compute A_scale; missing columns: {missing}")
    df["A_scale"] = compute_a_scale(
        pd.to_numeric(df["delta_pressure_bar_phys"], errors="coerce").astype(float),
        pd.to_numeric(df["ambient_density_kg_m3"], errors="coerce").astype(float),
        pd.to_numeric(df["diameter_mm"], errors="coerce").astype(float),
        delta_pressure_exp=float(delta_pressure_exp),
    )
    df["log_A"] = np.log(df["A_scale"])
    df["a_scale_delta_pressure_exp"] = float(delta_pressure_exp)
    return df


def variant_a_scale_dp_exp(variant: str) -> float:
    return float(A_SCALE_DP_EXP_BY_VARIANT.get(str(variant).lower(), DEFAULT_A_SCALE_DP_EXP))


def variant_target_scale_mode(variant: str) -> str:
    return str(TARGET_SCALE_MODE_BY_VARIANT.get(str(variant).lower(), "a_scale"))


def scaler_a_scale_dp_exp(scaler_state: Mapping[str, Any]) -> float:
    target = scaler_state.get("target", {}) if isinstance(scaler_state, Mapping) else {}
    return float(target.get("a_scale_delta_pressure_exp", DEFAULT_A_SCALE_DP_EXP))


def scaler_target_scale_mode(scaler_state: Mapping[str, Any], feature_columns: Sequence[str] | None = None) -> str:
    target = scaler_state.get("target", {}) if isinstance(scaler_state, Mapping) else {}
    if "scale_mode" in target:
        return str(target["scale_mode"])
    if feature_columns is not None and infer_feature_family(feature_columns) == "legacy_raw":
        return "none"
    return "a_scale"


def canonicalize_chamber_state(
    dataset_key: str,
    chamber_state_raw: float,
    registry: Mapping[str, DatasetMeta],
) -> tuple[float, float, str]:
    meta = registry[dataset_key]
    raw_value = float(chamber_state_raw)
    if meta.chamber_mode == "pressure":
        pressure = raw_value
        density = linear_density_from_pressure(pressure)
        return pressure, density, "raw_pressure_bar"

    density_keys = np.array(sorted(meta.chamber_density_to_pressure.keys()), dtype=float)
    pressure_keys = np.array(sorted(meta.chamber_pressure_to_density.keys()), dtype=float)
    if density_keys.size == 0 or pressure_keys.size == 0:
        pressure = raw_value
        density = linear_density_from_pressure(pressure)
        return pressure, density, "fallback_linear_pressure"

    density_match = density_keys[int(np.argmin(np.abs(density_keys - raw_value)))]
    if np.isclose(raw_value, density_match, atol=1e-6):
        density = float(density_match)
        pressure = float(meta.chamber_density_to_pressure[density])
        return pressure, density, "density_label_from_test_matrix"

    pressure_match = pressure_keys[int(np.argmin(np.abs(pressure_keys - raw_value)))]
    if np.isclose(raw_value, pressure_match, atol=1e-6):
        pressure = float(pressure_match)
        density = float(meta.chamber_pressure_to_density[pressure])
        return pressure, density, "physical_pressure_from_test_matrix"

    pressure = linear_pressure_from_density(raw_value)
    density = raw_value
    return pressure, density, "fallback_linear_density"


def build_canonical_feature_table(
    df_in: pd.DataFrame,
    registry: Mapping[str, DatasetMeta],
    *,
    fill_defaults: Mapping[str, float] | None = None,
    a_scale_delta_pressure_exp: float = DEFAULT_A_SCALE_DP_EXP,
) -> pd.DataFrame:
    if df_in.empty:
        raise ValueError("Input row table is empty.")

    defaults = dict(LEGACY_FILL_DEFAULTS)
    if fill_defaults:
        defaults.update(fill_defaults)

    df = df_in.copy().reset_index(drop=True)
    df["chamber_state_raw"] = _require_numeric(df, "chamber_pressure_bar")
    df["injection_duration_us"] = _require_numeric(df, "injection_duration_us")
    df["diameter_mm"] = _require_numeric(df, "diameter_mm")
    df["plumes"] = _require_numeric(df, "plumes")

    inj = pd.to_numeric(df.get("injection_pressure_bar", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["injection_pressure_bar"] = inj.fillna(float(defaults["injection_pressure_bar"])).astype(float)
    cb = pd.to_numeric(df.get("control_backpressure_bar", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["control_backpressure_bar"] = cb.fillna(float(defaults["control_backpressure_bar"])).astype(float)

    if "umbrella_angle_deg" in df.columns:
        umbrella = pd.to_numeric(df["umbrella_angle_deg"], errors="coerce").astype(float)
    else:
        umbrella = df["dataset_key"].map(lambda key: registry[key].nozzle_properties.get("umbrella_angle_deg", 140.0))
    df["umbrella_angle_deg"] = umbrella.astype(float)
    df["tilt_angle_radian"] = np.deg2rad((180.0 - df["umbrella_angle_deg"]) / 2.0)

    canonical_rows = [
        canonicalize_chamber_state(str(dataset_key), float(chamber_state_raw), registry)
        for dataset_key, chamber_state_raw in zip(df["dataset_key"], df["chamber_state_raw"])
    ]
    df["ambient_pressure_bar_phys"] = [item[0] for item in canonical_rows]
    df["ambient_density_kg_m3"] = [item[1] for item in canonical_rows]
    df["chamber_state_source"] = [item[2] for item in canonical_rows]

    delta_pressure = df["injection_pressure_bar"] - df["ambient_pressure_bar_phys"]
    if (delta_pressure <= 0).any():
        bad = df.loc[delta_pressure <= 0, ["dataset_key", "injection_pressure_bar", "ambient_pressure_bar_phys"]].head()
        raise ValueError(f"Found non-positive delta_pressure_bar_phys rows:\n{bad}")
    df["delta_pressure_bar_phys"] = delta_pressure.astype(float)
    df = apply_a_scale_transform(df, delta_pressure_exp=float(a_scale_delta_pressure_exp))
    df["log_injection_pressure_bar"] = np.log(df["injection_pressure_bar"].astype(float))
    df["log_chamber_pressure_bar"] = np.log(
        np.maximum(df["ambient_pressure_bar_phys"].astype(float), 1e-6)
    )
    df["log_delta_pressure_bar"] = np.log(df["delta_pressure_bar_phys"])
    df = _add_routing_id_columns(df)
    return df


def assign_splits_by_group(
    df_in: pd.DataFrame,
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    if "sample_group_id" not in df_in.columns:
        raise KeyError("sample_group_id column is required for grouped split assignment.")
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or (val_ratio + test_ratio) >= 1:
        raise ValueError("Invalid validation/test split ratios.")

    rng = np.random.default_rng(int(seed))
    unique_groups = np.asarray(pd.Series(df_in["sample_group_id"]).drop_duplicates().tolist(), dtype=object)
    rng.shuffle(unique_groups)
    n_groups = len(unique_groups)
    n_test = int(np.floor(test_ratio * n_groups))
    n_val = int(np.floor(val_ratio * n_groups))

    test_groups = set(unique_groups[:n_test].tolist())
    val_groups = set(unique_groups[n_test : n_test + n_val].tolist())

    def label(group_id: str) -> str:
        if group_id in test_groups:
            return "test"
        if group_id in val_groups:
            return "val"
        return "train"

    df = df_in.copy()
    df["sample_split"] = df["sample_group_id"].astype(str).map(label)
    return df


def assign_splits_leave_one_out(
    df_in: pd.DataFrame,
    *,
    holdout_value: str,
    holdout_column: str = "experiment_name",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """LONO split: one experiment_name → entire test split.

    Within the remaining trajectories, carve out val_ratio for validation
    by `sample_group_id` (so a trajectory's curves stay together). Train
    gets the rest.
    """
    df = df_in.copy()
    if holdout_column not in df.columns:
        fallback_columns = ("source_dataset_name", "folder", "dataset_key")
        fallback = next((col for col in fallback_columns if col in df.columns), None)
        if fallback is None:
            raise KeyError(
                f"Missing holdout column {holdout_column!r}; no fallback among {fallback_columns}."
            )
        holdout_column = fallback
    is_holdout = df[holdout_column].astype(str) == str(holdout_value)
    df_holdout = df.loc[is_holdout].copy()
    df_remaining = df.loc[~is_holdout].copy()
    if df_holdout.empty:
        raise ValueError(f"No rows match {holdout_column}={holdout_value!r}.")

    # Re-use sample_group_id-based splitting on the remaining rows for val.
    rng = np.random.default_rng(int(seed))
    groups = np.asarray(
        pd.Series(df_remaining["sample_group_id"]).astype(str).drop_duplicates().tolist(),
        dtype=object,
    )
    rng.shuffle(groups)
    n_val = int(np.floor(val_ratio * len(groups)))
    val_groups = set(groups[:n_val].tolist())

    df_remaining["sample_split"] = df_remaining["sample_group_id"].astype(str).map(
        lambda g: "val" if g in val_groups else "train"
    )
    df_holdout["sample_split"] = "test"
    return pd.concat([df_remaining, df_holdout], ignore_index=True)


def fit_zscore_params(train_df: pd.DataFrame, base_cols: Sequence[str]) -> dict[str, dict[str, float]]:
    params: dict[str, dict[str, float]] = {}
    for base_col in base_cols:
        values = pd.to_numeric(train_df[base_col], errors="coerce").astype(float)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        params[f"{base_col}_z"] = {"mean": mean, "std": std}
    return params


def apply_zscore(df_in: pd.DataFrame, zscore_params: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    df = df_in.copy()
    for z_col, stats in zscore_params.items():
        base_col = z_col[:-2] if z_col.endswith("_z") else z_col
        df[z_col] = (pd.to_numeric(df[base_col], errors="coerce") - float(stats["mean"])) / (float(stats["std"]) + 1e-12)
    return df


def apply_saved_scaler_state(df_in: pd.DataFrame, scaler_state: Mapping[str, Any]) -> pd.DataFrame:
    if "zscore" not in scaler_state:
        raise KeyError("scaler_state is missing the 'zscore' block.")
    df = apply_a_scale_transform(df_in, delta_pressure_exp=scaler_a_scale_dp_exp(scaler_state))
    df = apply_zscore(df, scaler_state["zscore"])
    # Re-derive routing channels (family_id, nozzle_id) for Tier-3A/3B in case
    # the upstream table was built before these helpers existed.
    df = _add_routing_id_columns(df)
    return df


def build_variant_feature_table(
    df_in: pd.DataFrame,
    *,
    variant: str,
    time_min_ms: float,
    time_max_ms: float,
    add_family_id: bool = False,
    add_nozzle_id: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Build the variant feature table + scaler state + feature_columns list.

    add_family_id (Tier-3A) appends a 0/1 family_id channel.
    add_nozzle_id (Tier-3B) appends a 0..5 nozzle_id channel.
    Both are kept OUT of z-scoring and appended to the END of feature_columns
    so FamilyAwarePenetrationMLP / learnable-d2 can strip them off cleanly.
    """
    token = str(variant).lower()
    if token not in FEATURE_COLUMNS_BY_VARIANT:
        raise KeyError(f"Unsupported variant '{variant}'.")

    a_scale_dp_exp = variant_a_scale_dp_exp(token)
    target_scale_mode = variant_target_scale_mode(token)
    df = apply_a_scale_transform(df_in, delta_pressure_exp=a_scale_dp_exp)
    train_df = df.loc[df["sample_split"] == "train"].reset_index(drop=True)
    zscore_params = fit_zscore_params(train_df, ZSCORE_BASE_COLUMNS_BY_VARIANT[token])
    df = apply_zscore(df, zscore_params)
    df = _add_routing_id_columns(df)
    scaler_state = {
        "time": {
            "type": "fixed_minmax",
            "min_ms": float(time_min_ms),
            "max_ms": float(time_max_ms),
        },
        "zscore": zscore_params,
        "fill_defaults": LEGACY_FILL_DEFAULTS.copy(),
        "canonicalization": {
            "reference_pressure_bar": REFERENCE_PRESSURE_BAR,
            "reference_density_kg_m3": REFERENCE_DENSITY_KG_M3,
        },
        "target": {
            "scale_mode": target_scale_mode,
            "a_scale_delta_pressure_exp": float(a_scale_dp_exp),
            "a_scale_density_exp": DEFAULT_A_SCALE_DENSITY_EXP,
            "a_scale_diameter_exp": DEFAULT_A_SCALE_DIAMETER_EXP,
        },
        "feature_variant": token,
        "extra_routing_channels": {
            "family_id": bool(add_family_id),
            "nozzle_id": bool(add_nozzle_id),
        },
    }
    feature_columns = list(FEATURE_COLUMNS_BY_VARIANT[token])
    if add_family_id:
        feature_columns.append("family_id")
    if add_nozzle_id:
        feature_columns.append("nozzle_id")
    return df, scaler_state, feature_columns


def build_all_stage_tables(
    data_root: Path | str,
    registry: Mapping[str, DatasetMeta],
    *,
    comparison_time_s: float,
    max_curves: int | None = None,
    output_dir: Path | None = None,
    a_scale_delta_pressure_exp: float = DEFAULT_A_SCALE_DP_EXP,
) -> StageTables:
    representative_raw = collect_selected_rows(
        data_root,
        registry,
        compare_time_s=comparison_time_s,
        selection_mode="representative",
        max_curves=max_curves,
    )
    filtered_raw = collect_selected_rows(
        data_root,
        registry,
        compare_time_s=comparison_time_s,
        selection_mode="filtered",
        max_curves=max_curves,
    )

    representative = build_canonical_feature_table(
        representative_raw,
        registry,
        a_scale_delta_pressure_exp=float(a_scale_delta_pressure_exp),
    )
    filtered = build_canonical_feature_table(
        filtered_raw,
        registry,
        a_scale_delta_pressure_exp=float(a_scale_delta_pressure_exp),
    )
    precheck = run_pretrain_collapse_check(representative, output_dir=output_dir)
    return StageTables(representative=representative, filtered=filtered, representative_precheck=precheck)


def _linear_regression_r2(X: np.ndarray, y: np.ndarray) -> float:
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
    X_valid = X_arr[mask]
    y_valid = y_arr[mask]
    if len(y_valid) < 2:
        return float("nan")
    design = np.column_stack([np.ones(len(X_valid)), X_valid])
    coef, *_ = np.linalg.lstsq(design, y_valid, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y_valid - pred) ** 2))
    ss_tot = float(np.sum((y_valid - np.mean(y_valid)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0
    return 1.0 - ss_res / ss_tot


def run_pretrain_collapse_check(
    representative_df: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    sample_times_ms: Sequence[float] | None = None,
) -> dict[str, Any]:
    if representative_df.empty:
        raise ValueError("Representative row table is empty; cannot run pretrain check.")

    times_ms = np.asarray(sample_times_ms if sample_times_ms is not None else np.linspace(0.5, 5.0, 10), dtype=float)
    time_s = times_ms * 1e-3
    fit_cols = ["log_k_sqrt", "log_k_quarter", "log_t0", "log_s"]
    for col in fit_cols:
        if col not in representative_df.columns:
            raise KeyError(f"Required fit column missing from representative table: {col}")

    trajectories = np.stack(
        [
            reconstruct_penetration_series(
                representative_df["log_k_sqrt"].to_numpy(dtype=float),
                representative_df["log_k_quarter"].to_numpy(dtype=float),
                representative_df["log_t0"].to_numpy(dtype=float),
                representative_df["log_s"].to_numpy(dtype=float),
                t_value,
            )
            for t_value in time_s
        ],
        axis=1,
    )
    a_scale = representative_df["A_scale"].to_numpy(dtype=float)[:, None]
    scaled_trajectories = trajectories / a_scale
    sparse_X = representative_df[["diameter_mm", "injection_pressure_bar", "ambient_pressure_bar_phys"]].to_numpy(dtype=float)

    collapse_rows: list[dict[str, float]] = []
    for idx, time_ms in enumerate(times_ms):
        s_values = trajectories[:, idx]
        s_scaled_values = scaled_trajectories[:, idx]
        std_raw = float(np.nanstd(s_values))
        std_scaled = float(np.nanstd(s_scaled_values))
        collapse_rows.append(
            {
                "time_ms": float(time_ms),
                "std_physical": std_raw,
                "std_scaled": std_scaled,
                "collapse_ratio": float(std_scaled / std_raw) if std_raw > 1e-12 else np.nan,
                "r2_sparse_physical": _linear_regression_r2(sparse_X, s_values),
                "r2_sparse_scaled": _linear_regression_r2(sparse_X, s_scaled_values),
            }
        )
    collapse_df = pd.DataFrame(collapse_rows)
    post_mask = collapse_df["time_ms"] >= 0.5
    median_collapse = float(np.nanmedian(collapse_df.loc[post_mask, "collapse_ratio"]))
    mean_r2_phys = float(np.nanmean(collapse_df["r2_sparse_physical"]))
    mean_r2_scaled = float(np.nanmean(collapse_df["r2_sparse_scaled"]))
    explainability_ratio = float(mean_r2_scaled / mean_r2_phys) if mean_r2_phys > 1e-12 else np.nan
    all_collapse_lt_one = bool(np.all(collapse_df.loc[post_mask, "collapse_ratio"].to_numpy(dtype=float) < 1.0))
    passed = bool(all_collapse_lt_one and median_collapse <= 0.8 and explainability_ratio <= 0.5)

    report = {
        "times_ms": collapse_df["time_ms"].tolist(),
        "median_post_0p5ms_collapse_ratio": median_collapse,
        "mean_sparse_r2_physical": mean_r2_phys,
        "mean_sparse_r2_scaled": mean_r2_scaled,
        "sparse_r2_ratio_scaled_over_physical": explainability_ratio,
        "all_post_0p5ms_collapse_lt_one": all_collapse_lt_one,
        "passed": passed,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        collapse_df.to_csv(output_dir / "pretrain_collapse_metrics.csv", index=False)
        with (output_dir / "pretrain_collapse_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plot_count = min(len(trajectories), 160)
        sample_idx = np.linspace(0, len(trajectories) - 1, plot_count, dtype=int) if len(trajectories) > plot_count else np.arange(len(trajectories))
        axes[0].plot(times_ms, trajectories[sample_idx].T, alpha=0.12, color="#1f77b4")
        axes[0].set_title("Representative physical curves")
        axes[0].set_xlabel("Time [ms]")
        axes[0].set_ylabel("S")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(times_ms, scaled_trajectories[sample_idx].T, alpha=0.12, color="#2ca02c")
        axes[1].set_title("Representative A-scaled curves")
        axes[1].set_xlabel("Time [ms]")
        axes[1].set_ylabel("penetration_scaled")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(collapse_df["time_ms"], collapse_df["collapse_ratio"], marker="o", label="Collapse ratio")
        axes[2].plot(collapse_df["time_ms"], collapse_df["r2_sparse_physical"], marker="s", label="R2 sparse -> physical")
        axes[2].plot(collapse_df["time_ms"], collapse_df["r2_sparse_scaled"], marker="^", label="R2 sparse -> scaled")
        axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        axes[2].axhline(0.8, color="tab:red", linestyle=":", linewidth=1, alpha=0.6)
        axes[2].set_title("Pretrain collapse and explainability")
        axes[2].set_xlabel("Time [ms]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "pretrain_collapse_summary.png", dpi=180)
        plt.close(fig)

    return report
