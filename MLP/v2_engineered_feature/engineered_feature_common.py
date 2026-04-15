from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYNTHETIC_DATA_ROOT = PROJECT_ROOT / "MLP" / "synthetic_data"
DEFAULT_TEST_MATRIX_ROOT = PROJECT_ROOT / "test_matrix_json"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "MLP" / "runs_mlp"

REFERENCE_PRESSURE_BAR = 4.42
REFERENCE_DENSITY_KG_M3 = 5.0
MIN_TIME_SHIFT_S = 0.0
EPS = 1e-12

TIME_FEATURE = "time_norm_0_5ms"
BASE_STATIC_FEATURE_COLUMNS = [
    "tilt_angle_radian_z",
    "plumes_z",
    "injection_duration_us_z",
    "control_backpressure_bar_z",
]
FEATURE_COLUMNS_BY_VARIANT = {
    "a_only": [TIME_FEATURE, *BASE_STATIC_FEATURE_COLUMNS],
    "a_plus_log_a": [TIME_FEATURE, *BASE_STATIC_FEATURE_COLUMNS, "log_A_z"],
}
ZSCORE_BASE_COLUMNS_BY_VARIANT = {
    "a_only": [
        "tilt_angle_radian",
        "plumes",
        "injection_duration_us",
        "control_backpressure_bar",
    ],
    "a_plus_log_a": [
        "tilt_angle_radian",
        "plumes",
        "injection_duration_us",
        "control_backpressure_bar",
        "log_A",
    ],
}
LEGACY_FEATURE_COLUMNS = {
    "diameter_mm_z",
    "log_injection_pressure_bar_z",
    "log_chamber_pressure_bar_z",
    "log_delta_pressure_bar_z",
}
LEGACY_FILL_DEFAULTS = {
    "injection_pressure_bar": 2000.0,
    "control_backpressure_bar": 4.0,
}

DEFAULT_STAGE1_CONFIG = {
    "seed": 42,
    "comparison_time_s": 5e-3,
    "data_dir": str(DEFAULT_SYNTHETIC_DATA_ROOT),
    "runs_root": str(DEFAULT_RUNS_ROOT),
    "variant": "a_only",
    "batch_size": 128,
    "hidden_dims": [512, 512, 128],
    "dropout": 0.3,
    "activation": "gelu",
    "learning_rate": 4e-3,
    "weight_decay": 2e-4,
    "epochs": 300,
    "grad_clip_norm": 1.0,
    "num_workers": 0,
    "shuffle_train": True,
    "n_points": 512,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "time_min_ms": 0.0,
    "time_max_ms": 5.0,
    "early_stopping_patience": 40,
    "early_stopping_min_delta": 1e-5,
    "log_interval": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "var_reg_weight": 1e-3,
    "log_var_prior": -2.0,
    "log_var_bounds": (-10.0, 6.0),
    "d1_positive_weight": 5e-5,
    "d2_concave_weight": 5e-4,
    "d2_start_ms": 0.9,
    "d2_transition_ms": 0.05,
    "std_clamp_min": 1e-3,
    "row_selection_mode": "representative",
    "allow_failed_precheck": False,
    "max_curves": None,
}

DEFAULT_STAGE2_CONFIG = {
    "seed": 42,
    "comparison_time_s": 5e-3,
    "data_dir": str(DEFAULT_SYNTHETIC_DATA_ROOT),
    "runs_root": str(DEFAULT_RUNS_ROOT),
    "variant": "a_only",
    "batch_size": 96,
    "hidden_dims": [512, 512, 128],
    "dropout": 0.3,
    "activation": "gelu",
    "learning_rate": 8e-4,
    "weight_decay": 1e-4,
    "epochs": 220,
    "grad_clip_norm": 1.0,
    "num_workers": 0,
    "precompute_dataset": True,
    "persistent_workers": False,
    "prefetch_factor": 2,
    "shuffle_train": True,
    "n_points": 512,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "time_min_ms": 0.0,
    "time_max_ms": 5.0,
    "early_stopping_patience": 35,
    "early_stopping_min_delta": 1e-5,
    "log_interval": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_var_bounds": (-10.0, 6.0),
    "nll_eps": 1e-12,
    "d1_positive_weight": 5e-5,
    "d2_concave_weight": 5e-5,
    "d2_start_ms": 0.5,
    "d2_transition_ms": 0.05,
    "std_clamp_min": 1e-3,
    "row_selection_mode": "filtered",
    "allow_failed_precheck": False,
    "max_curves": None,
}


@dataclass(frozen=True)
class DatasetMeta:
    key: str
    display_name: str
    chamber_mode: str
    chamber_density_to_pressure: dict[float, float]
    chamber_pressure_to_density: dict[float, float]
    nozzle_properties: dict[str, float]


@dataclass(frozen=True)
class StageTables:
    representative: pd.DataFrame
    filtered: pd.DataFrame
    representative_precheck: dict[str, Any]


@dataclass(frozen=True)
class RunArtifacts:
    model: nn.Module
    train_config: dict[str, Any]
    scaler_state: dict[str, Any]
    run_dir: Path
    model_path: Path


def merge_config(base: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(base)
    if overrides:
        merged.update(overrides)
    return merged


def normalize_dataset_key(text: str) -> str:
    lower = str(text).lower()
    if "ds300" in lower:
        return "ds300"
    match = re.search(r"nozzle[_\s-]*([1-8])", lower)
    if match:
        return f"nozzle{match.group(1)}"
    raise KeyError(f"Could not infer dataset key from: {text}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset_registry(test_matrix_root: Path | None = None) -> dict[str, DatasetMeta]:
    root = Path(test_matrix_root or DEFAULT_TEST_MATRIX_ROOT)
    registry: dict[str, DatasetMeta] = {}
    for json_path in sorted(root.glob("*.json")):
        payload = _load_json(json_path)
        key = normalize_dataset_key(json_path.stem)
        nozzle_props = {
            name: float(value)
            for name, value in payload.get("nozzle_properties", {}).items()
            if isinstance(value, (int, float))
        }
        test_matrix = payload.get("test_matrix", {})
        density_to_pressure: dict[float, float] = {}
        pressure_to_density: dict[float, float] = {}
        chamber_mode = "pressure"
        if "groups" in test_matrix:
            for group in test_matrix["groups"]:
                densities = group.get("chamber_density_kg_per_m3")
                pressures = group.get("chamber_pressures_bar")
                if densities is not None and pressures is not None:
                    chamber_mode = "density_label"
                    for density, pressure in zip(densities, pressures):
                        density_to_pressure[float(density)] = float(pressure)
                        pressure_to_density[float(pressure)] = float(density)
        elif "chamber_density_kg_per_m3" in test_matrix and "chamber_pressures_bar" in test_matrix:
            chamber_mode = "density_label"
            for density, pressure in zip(
                test_matrix["chamber_density_kg_per_m3"],
                test_matrix["chamber_pressures_bar"],
            ):
                density_to_pressure[float(density)] = float(pressure)
                pressure_to_density[float(pressure)] = float(density)

        registry[key] = DatasetMeta(
            key=key,
            display_name=str(payload.get("name", key)),
            chamber_mode=chamber_mode,
            chamber_density_to_pressure=density_to_pressure,
            chamber_pressure_to_density=pressure_to_density,
            nozzle_properties=nozzle_props,
        )
    return registry


def make_activation(name: str) -> nn.Module:
    token = (name or "relu").lower()
    if token == "relu":
        return nn.ReLU()
    if token == "gelu":
        return nn.GELU()
    if token == "tanh":
        return nn.Tanh()
    if token == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}'")


class PenetrationMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    make_activation(activation),
                ]
            )
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(config: Mapping[str, Any]) -> PenetrationMLP:
    return PenetrationMLP(
        input_dim=int(config["input_dim"]),
        hidden_dims=[int(x) for x in config["hidden_dims"]],
        output_dim=int(config["output_dim"]),
        activation=str(config.get("activation", "gelu")),
        dropout=float(config.get("dropout", 0.0)),
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def spray_penetration_model_sigmoid(params: Sequence[np.ndarray | float], t: np.ndarray | float) -> np.ndarray:
    log_k_sqrt, log_k_quarter, log_t0, log_s = params
    k_sqrt = np.exp(log_k_sqrt)
    k_quarter = np.exp(log_k_quarter)
    t0 = np.exp(log_t0) + MIN_TIME_SHIFT_S
    s = np.exp(log_s)

    t_arr = np.clip(np.asarray(t, dtype=float), 1e-9, None)
    sqrt_segment = k_sqrt * np.sqrt(t_arr)
    quarter_segment = k_quarter * np.power(t_arr, 0.25)
    w = sigmoid((t_arr - t0) / np.maximum(s, 1e-12))
    return (1.0 - w) * sqrt_segment + w * quarter_segment


def spray_penetration_model_sigmoid_d_dt(params: Sequence[np.ndarray | float], t: np.ndarray | float) -> np.ndarray:
    log_k_sqrt, log_k_quarter, log_t0, log_s = params
    k_sqrt = np.exp(log_k_sqrt)
    k_quarter = np.exp(log_k_quarter)
    t0 = np.exp(log_t0) + MIN_TIME_SHIFT_S
    s = np.exp(log_s)

    t_arr = np.clip(np.asarray(t, dtype=float), 1e-9, None)
    sqrt_segment = k_sqrt * np.sqrt(t_arr)
    quarter_segment = k_quarter * np.power(t_arr, 0.25)
    w = sigmoid((t_arr - t0) / np.maximum(s, 1e-12))
    dw_dt = w * (1.0 - w) / np.maximum(s, 1e-12)
    d_sqrt_dt = 0.5 * k_sqrt / np.sqrt(t_arr)
    d_quarter_dt = 0.25 * k_quarter * np.power(t_arr, -0.75)
    return (1.0 - w) * d_sqrt_dt + w * d_quarter_dt + dw_dt * (quarter_segment - sqrt_segment)


def reconstruct_penetration_series(
    log_k_sqrt: float,
    log_k_quarter: float,
    log_t0: float,
    log_s: float,
    time_s: np.ndarray,
) -> np.ndarray:
    return spray_penetration_model_sigmoid([log_k_sqrt, log_k_quarter, log_t0, log_s], time_s)


def iter_clean_csv_files(data_root: Path | str) -> list[Path]:
    root = Path(data_root)
    return sorted(p for p in root.rglob("*.csv") if p.parent.name == "clean")


def infer_dataset_name_from_csv_path(csv_path: Path) -> str:
    parts = csv_path.parts
    if len(parts) >= 4 and parts[-3].lower() == "cdf":
        return parts[-4]
    if len(parts) >= 2:
        return parts[-2]
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
    df["A_scale"] = np.power(df["delta_pressure_bar_phys"] / df["ambient_density_kg_m3"], 0.25) * np.sqrt(df["diameter_mm"])
    df["log_A"] = np.log(df["A_scale"])
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
    return apply_zscore(df_in, scaler_state["zscore"])


def build_variant_feature_table(
    df_in: pd.DataFrame,
    *,
    variant: str,
    time_min_ms: float,
    time_max_ms: float,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    token = str(variant).lower()
    if token not in FEATURE_COLUMNS_BY_VARIANT:
        raise KeyError(f"Unsupported variant '{variant}'.")

    df = df_in.copy()
    train_df = df.loc[df["sample_split"] == "train"].reset_index(drop=True)
    zscore_params = fit_zscore_params(train_df, ZSCORE_BASE_COLUMNS_BY_VARIANT[token])
    df = apply_zscore(df, zscore_params)
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
        "feature_variant": token,
    }
    return df, scaler_state, list(FEATURE_COLUMNS_BY_VARIANT[token])


def build_all_stage_tables(
    data_root: Path | str,
    registry: Mapping[str, DatasetMeta],
    *,
    comparison_time_s: float,
    max_curves: int | None = None,
    output_dir: Path | None = None,
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

    representative = build_canonical_feature_table(representative_raw, registry)
    filtered = build_canonical_feature_table(filtered_raw, registry)
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


class PointwisePenetrationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        n_points: int,
        time_min_ms: float,
        time_max_ms: float,
        precompute: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.feature_columns = list(feature_columns)
        self.n_points = int(n_points)
        self.time_min_ms = float(time_min_ms)
        self.time_max_ms = float(time_max_ms)
        self.precompute = bool(precompute)
        self.time_span_ms = max(self.time_max_ms - self.time_min_ms, 1e-12)
        self.time_grid_ms = np.linspace(self.time_min_ms, self.time_max_ms, self.n_points, dtype=np.float32)
        self.time_grid_s = self.time_grid_ms * 1e-3
        self.time_norm = ((self.time_grid_ms - self.time_min_ms) / self.time_span_ms).astype(np.float32)
        self._cached_features: torch.Tensor | None = None
        self._cached_target_scaled: torch.Tensor | None = None
        self._cached_target_physical: torch.Tensor | None = None
        self._cached_a_scale: torch.Tensor | None = None

        required_cols = list(self.feature_columns[1:]) + ["A_scale", "log_k_sqrt", "log_k_quarter", "log_t0", "log_s"]
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise KeyError(f"Dataset missing required columns: {missing}")
        if self.precompute and len(self.df) > 0:
            self._build_cache()

    def __len__(self) -> int:
        return len(self.df)

    def _build_cache(self) -> None:
        n_samples = len(self.df)
        feature_dim = len(self.feature_columns)
        time_axis = np.broadcast_to(self.time_norm.reshape(1, self.n_points, 1), (n_samples, self.n_points, 1))
        if len(self.feature_columns) > 1:
            static_values = self.df[self.feature_columns[1:]].to_numpy(dtype=np.float32)
            static_block = np.broadcast_to(
                static_values[:, None, :],
                (n_samples, self.n_points, static_values.shape[1]),
            )
            features_np = np.concatenate([time_axis, static_block], axis=2)
        else:
            features_np = time_axis

        time_s = self.time_grid_s.astype(np.float64)[None, :]
        log_k_sqrt = self.df["log_k_sqrt"].to_numpy(dtype=np.float64)[:, None]
        log_k_quarter = self.df["log_k_quarter"].to_numpy(dtype=np.float64)[:, None]
        log_t0 = self.df["log_t0"].to_numpy(dtype=np.float64)[:, None]
        log_s = self.df["log_s"].to_numpy(dtype=np.float64)[:, None]

        k_sqrt = np.exp(log_k_sqrt)
        k_quarter = np.exp(log_k_quarter)
        t0 = np.exp(log_t0) + MIN_TIME_SHIFT_S
        sharpness = np.exp(log_s)
        sqrt_segment = k_sqrt * np.sqrt(time_s)
        quarter_segment = k_quarter * np.power(time_s, 0.25)
        blend = sigmoid((time_s - t0) / np.maximum(sharpness, 1e-12))
        penetration = ((1.0 - blend) * sqrt_segment + blend * quarter_segment).astype(np.float32)

        a_scale = self.df["A_scale"].to_numpy(dtype=np.float32)[:, None]
        a_scale_block = np.broadcast_to(a_scale[:, None, :], (n_samples, self.n_points, 1))
        target_physical = penetration[..., None]
        target_scaled = (penetration / a_scale).astype(np.float32)[..., None]

        self._cached_features = torch.from_numpy(np.ascontiguousarray(features_np.reshape(n_samples, self.n_points, feature_dim)))
        self._cached_target_scaled = torch.from_numpy(np.ascontiguousarray(target_scaled))
        self._cached_target_physical = torch.from_numpy(np.ascontiguousarray(target_physical))
        self._cached_a_scale = torch.from_numpy(np.ascontiguousarray(a_scale_block))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._cached_features is not None:
            return {
                "features": self._cached_features[int(idx)],
                "target_scaled": self._cached_target_scaled[int(idx)],
                "target_physical": self._cached_target_physical[int(idx)],
                "a_scale": self._cached_a_scale[int(idx)],
                "sample_idx": torch.full((self.n_points,), int(idx), dtype=torch.long),
            }

        row = self.df.iloc[int(idx)]
        penetration = reconstruct_penetration_series(
            float(row["log_k_sqrt"]),
            float(row["log_k_quarter"]),
            float(row["log_t0"]),
            float(row["log_s"]),
            self.time_grid_s,
        ).astype(np.float32)
        a_scale = float(row["A_scale"])
        target_scaled = (penetration / a_scale).reshape(-1, 1).astype(np.float32)
        target_physical = penetration.reshape(-1, 1).astype(np.float32)
        a_repeat = np.full((self.n_points, 1), a_scale, dtype=np.float32)

        static_vec = row[self.feature_columns[1:]].to_numpy(dtype=np.float32)
        static_mat = np.repeat(static_vec[None, :], self.n_points, axis=0)
        features = np.column_stack([self.time_norm, static_mat]).astype(np.float32)
        return {
            "features": torch.from_numpy(features),
            "target_scaled": torch.from_numpy(target_scaled),
            "target_physical": torch.from_numpy(target_physical),
            "a_scale": torch.from_numpy(a_repeat),
            "sample_idx": torch.full((self.n_points,), int(idx), dtype=torch.long),
        }


def collate_pointwise(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.cat([item[key] for item in batch], dim=0)
        for key in ("features", "target_scaled", "target_physical", "a_scale", "sample_idx")
    }


def make_dataloaders(
    df_in: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    batch_size: int,
    n_points: int,
    time_min_ms: float,
    time_max_ms: float,
    shuffle_train: bool,
    num_workers: int,
    precompute_dataset: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> tuple[dict[str, PointwisePenetrationDataset], dict[str, DataLoader]]:
    datasets = {
        split: PointwisePenetrationDataset(
            df_in.loc[df_in["sample_split"] == split].reset_index(drop=True),
            feature_columns=feature_columns,
            n_points=n_points,
            time_min_ms=time_min_ms,
            time_max_ms=time_max_ms,
            precompute=precompute_dataset,
        )
        for split in ("train", "val", "test")
    }
    common_loader_kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available() and not precompute_dataset,
        "collate_fn": collate_pointwise,
    }
    if int(num_workers) > 0:
        common_loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            common_loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(batch_size),
            shuffle=bool(shuffle_train),
            **common_loader_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=int(batch_size),
            shuffle=False,
            **common_loader_kwargs,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=int(batch_size),
            shuffle=False,
            **common_loader_kwargs,
        ),
    }
    return datasets, dataloaders


def split_mu_logvar(model_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if model_output.shape[-1] == 3:
        mu, log_var, _onset = torch.split(model_output, [1, 1, 1], dim=-1)
        return mu, log_var
    mu, log_var = model_output.chunk(2, dim=-1)
    return mu, log_var


def derivative_physics_penalty(
    mu_physical: torch.Tensor,
    n_points: int,
    *,
    time_max_ms: float,
    d2_start_ms: float,
    d2_transition_ms: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n_points <= 1:
        zero = mu_physical.new_tensor(0.0)
        return zero, zero
    mu_seq = mu_physical.reshape(-1, n_points)
    d1 = mu_seq[:, 1:] - mu_seq[:, :-1]
    d1_negative_penalty = torch.relu(-d1).pow(2).mean()
    if n_points <= 2:
        return d1_negative_penalty, mu_physical.new_tensor(0.0)

    d2 = mu_seq[:, 2:] - 2.0 * mu_seq[:, 1:-1] + mu_seq[:, :-2]
    t_ms = torch.linspace(0.0, float(time_max_ms), int(n_points), device=mu_physical.device)
    t_center_ms = t_ms[1:-1]
    gate = torch.sigmoid((t_center_ms - float(d2_start_ms)) / max(float(d2_transition_ms), 1e-6)).unsqueeze(0)
    d2_positive_penalty = (torch.relu(d2).pow(2) * gate).mean()
    return d1_negative_penalty, d2_positive_penalty


def stage1_objective(
    model_output: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    n_points: int,
    time_max_ms: float,
    var_reg_weight: float,
    log_var_prior: float,
    log_var_bounds: tuple[float, float],
    d1_positive_weight: float,
    d2_concave_weight: float,
    d2_start_ms: float,
    d2_transition_ms: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mu_hat, log_var_hat = split_mu_logvar(model_output)
    log_var_hat = torch.clamp(log_var_hat, min=float(log_var_bounds[0]), max=float(log_var_bounds[1]))
    target_scaled = batch["target_scaled"]
    a_scale = batch["a_scale"]
    mu_physical = a_scale * mu_hat

    mse_scaled = torch.mean((mu_hat - target_scaled) ** 2)
    prior = torch.full_like(log_var_hat, float(log_var_prior))
    var_reg = torch.mean((log_var_hat - prior) ** 2)
    d1_penalty, d2_penalty = derivative_physics_penalty(
        mu_physical,
        n_points=n_points,
        time_max_ms=time_max_ms,
        d2_start_ms=d2_start_ms,
        d2_transition_ms=d2_transition_ms,
    )
    loss = (
        mse_scaled
        + float(var_reg_weight) * var_reg
        + float(d1_positive_weight) * d1_penalty
        + float(d2_concave_weight) * d2_penalty
    )
    physical_mae = torch.mean(torch.abs(mu_physical - batch["target_physical"]))
    metrics = {
        "loss": float(loss.detach().cpu()),
        "mse_scaled": float(mse_scaled.detach().cpu()),
        "physical_mae": float(physical_mae.detach().cpu()),
        "var_reg": float(var_reg.detach().cpu()),
        "d1_penalty": float(d1_penalty.detach().cpu()),
        "d2_penalty": float(d2_penalty.detach().cpu()),
    }
    return loss, metrics


def stage2_objective(
    model_output: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    n_points: int,
    time_max_ms: float,
    log_var_bounds: tuple[float, float],
    nll_eps: float,
    d1_positive_weight: float,
    d2_concave_weight: float,
    d2_start_ms: float,
    d2_transition_ms: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mu_hat, log_var_hat = split_mu_logvar(model_output)
    log_var_hat = torch.clamp(log_var_hat, min=float(log_var_bounds[0]), max=float(log_var_bounds[1]))
    target_scaled = batch["target_scaled"]
    a_scale = batch["a_scale"]
    var_hat = torch.exp(log_var_hat) + float(nll_eps)
    scaled_nll = torch.mean(0.5 * (log_var_hat + (mu_hat - target_scaled) ** 2 / var_hat))

    mu_physical = a_scale * mu_hat
    d1_penalty, d2_penalty = derivative_physics_penalty(
        mu_physical,
        n_points=n_points,
        time_max_ms=time_max_ms,
        d2_start_ms=d2_start_ms,
        d2_transition_ms=d2_transition_ms,
    )
    loss = scaled_nll + float(d1_positive_weight) * d1_penalty + float(d2_concave_weight) * d2_penalty

    std_physical = a_scale * torch.exp(0.5 * log_var_hat)
    physical_mae = torch.mean(torch.abs(mu_physical - batch["target_physical"]))
    metrics = {
        "loss": float(loss.detach().cpu()),
        "nll_scaled": float(scaled_nll.detach().cpu()),
        "physical_mae": float(physical_mae.detach().cpu()),
        "std_physical_mean": float(std_physical.mean().detach().cpu()),
        "d1_penalty": float(d1_penalty.detach().cpu()),
        "d2_penalty": float(d2_penalty.detach().cpu()),
    }
    return loss, metrics


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    objective_name: str,
    objective_kwargs: Mapping[str, Any],
    global_iter_start: int,
    epoch_idx: int,
    log_every: int,
) -> tuple[dict[str, float], list[dict[str, float]], int]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    epoch_totals: dict[str, float] = {}
    iter_logs: list[dict[str, float]] = []
    total_points = 0
    global_iter = int(global_iter_start)

    grad_clip_norm = objective_kwargs.get("grad_clip_norm")
    objective_core = {k: v for k, v in objective_kwargs.items() if k != "grad_clip_norm"}

    for step_idx, batch in enumerate(dataloader, start=1):
        batch_device = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        output = model(batch_device["features"])
        if objective_name == "stage1":
            loss, metrics = stage1_objective(output, batch_device, **objective_core)
        elif objective_name == "stage2":
            loss, metrics = stage2_objective(output, batch_device, **objective_core)
        else:
            raise KeyError(f"Unsupported objective_name '{objective_name}'.")

        if is_train:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()

        n = int(batch_device["features"].shape[0])
        total_points += n
        for key, value in metrics.items():
            epoch_totals[key] = epoch_totals.get(key, 0.0) + float(value) * n

        log_row = {
            "epoch": float(epoch_idx),
            "step": float(step_idx),
            "global_iter": float(global_iter),
            "split": "train" if is_train else "val",
        }
        log_row.update({key: float(value) for key, value in metrics.items()})
        iter_logs.append(log_row)

        if is_train and step_idx % max(1, int(log_every)) == 0:
            summary = " ".join(
                f"{key}={value:.6f}"
                for key, value in metrics.items()
                if key in ("loss", "mse_scaled", "nll_scaled", "physical_mae")
            )
            print(f"[Epoch {epoch_idx:03d}] step={step_idx:04d} iter={global_iter:06d} {summary}")
        global_iter += 1

    denom = max(total_points, 1)
    epoch_metrics = {"epoch": float(epoch_idx), "points": float(total_points)}
    epoch_metrics.update({key: value / denom for key, value in epoch_totals.items()})
    return epoch_metrics, iter_logs, global_iter


def sanitize_config_for_json(config: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        elif isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [str(item) if isinstance(item, Path) else item for item in value]
        elif isinstance(value, dict):
            out[key] = sanitize_config_for_json(value)
    return out


def create_run_dir(runs_root: Path | str, prefix: str, variant: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_root) / f"{prefix}_{variant}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train_with_early_stopping(
    *,
    model: nn.Module,
    dataloaders: Mapping[str, DataLoader],
    device: torch.device,
    objective_name: str,
    objective_kwargs: Mapping[str, Any],
    epochs: int,
    optimizer: torch.optim.Optimizer,
    patience: int,
    min_delta: float,
    log_every: int,
) -> tuple[nn.Module, pd.DataFrame, pd.DataFrame]:
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    iter_history: list[dict[str, float]] = []
    epoch_history: list[dict[str, float]] = []
    global_iter = 0
    no_improve = 0
    start_t = time.time()

    for epoch in range(1, int(epochs) + 1):
        train_metrics, train_iter, global_iter = run_epoch(
            model,
            dataloaders["train"],
            optimizer=optimizer,
            device=device,
            objective_name=objective_name,
            objective_kwargs=objective_kwargs,
            global_iter_start=global_iter,
            epoch_idx=epoch,
            log_every=log_every,
        )
        train_metrics["split"] = "train"

        with torch.no_grad():
            val_metrics, val_iter, global_iter = run_epoch(
                model,
                dataloaders["val"],
                optimizer=None,
                device=device,
                objective_name=objective_name,
                objective_kwargs=objective_kwargs,
                global_iter_start=global_iter,
                epoch_idx=epoch,
                log_every=log_every,
            )
        val_metrics["split"] = "val"
        iter_history.extend(train_iter)
        iter_history.extend(val_iter)
        epoch_history.append(train_metrics)
        epoch_history.append(val_metrics)

        improved = (best_val_loss - float(val_metrics["loss"])) > float(min_delta)
        if improved:
            best_val_loss = float(val_metrics["loss"])
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{int(epochs)} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"no_improve={no_improve}/{int(patience)}"
            )
        if no_improve >= int(patience):
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        test_metrics, _, _ = run_epoch(
            model,
            dataloaders["test"],
            optimizer=None,
            device=device,
            objective_name=objective_name,
            objective_kwargs=objective_kwargs,
            global_iter_start=global_iter,
            epoch_idx=epoch_history[-1]["epoch"] if epoch_history else 0,
            log_every=log_every,
        )
    test_metrics["split"] = "test"
    epoch_history.append(test_metrics)
    elapsed = time.time() - start_t
    print(f"Training completed in {elapsed:.1f}s. Best val loss={best_val_loss:.6f}")
    return model, pd.DataFrame(iter_history), pd.DataFrame(epoch_history)


def save_training_outputs(
    run_dir: Path,
    *,
    model: nn.Module,
    checkpoint_name: str,
    train_config: Mapping[str, Any],
    scaler_state: Mapping[str, Any],
    row_table: pd.DataFrame,
    iter_history: pd.DataFrame,
    epoch_history: pd.DataFrame,
    precheck_report: Mapping[str, Any] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / checkpoint_name)
    iter_history.to_csv(run_dir / "iteration_loss.csv", index=False)
    epoch_history.to_csv(run_dir / "epoch_loss.csv", index=False)
    row_table.to_csv(run_dir / "row_table.csv", index=False)
    with (run_dir / "scaler_state.json").open("w", encoding="utf-8") as f:
        json.dump(scaler_state, f, indent=2)
    with (run_dir / "train_config_used.json").open("w", encoding="utf-8") as f:
        json.dump(sanitize_config_for_json(train_config), f, indent=2)
    if precheck_report is not None:
        with (run_dir / "pretrain_collapse_report.json").open("w", encoding="utf-8") as f:
            json.dump(dict(precheck_report), f, indent=2)


def plot_loss_curves(epoch_history: pd.DataFrame, run_dir: Path, *, objective_name: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for split in ("train", "val", "test"):
        group = epoch_history.loc[epoch_history["split"] == split]
        if group.empty:
            continue
        axes[0].plot(group["epoch"], group["loss"], marker="o", label=split)
        if "physical_mae" in group.columns:
            axes[1].plot(group["epoch"], group["physical_mae"], marker="o", label=split)
        if "d1_penalty" in group.columns:
            axes[2].plot(group["epoch"], group["d1_penalty"], marker="o", label=f"{split} d1")
        if "d2_penalty" in group.columns:
            axes[2].plot(group["epoch"], group["d2_penalty"], marker="x", label=f"{split} d2")
    axes[0].set_title("Epoch loss")
    axes[1].set_title("Epoch physical MAE")
    axes[2].set_title("Epoch shape penalties")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = run_dir / f"{objective_name}_loss_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def unwrap_state_dict(state: Any) -> dict[str, torch.Tensor]:
    if isinstance(state, dict):
        for key in ("state_dict", "model_state", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(state)}")
    return state


def resolve_model_path(run_dir: Path) -> Path:
    for name in ("best_model_refinement.pt", "best_model_stage2.pt", "best_model_stage1.pt"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported checkpoint found under: {run_dir}")


def has_supported_checkpoint(run_dir: Path) -> bool:
    return run_dir.is_dir() and any(
        (run_dir / name).exists()
        for name in ("best_model_refinement.pt", "best_model_stage1.pt", "best_model_stage2.pt")
    )


def has_run_artifacts(run_dir: Path) -> bool:
    return (
        has_supported_checkpoint(run_dir)
        and (run_dir / "train_config_used.json").exists()
        and (run_dir / "scaler_state.json").exists()
    )


def _resolve_teacher_run_dir(run_dir: Path) -> Path | None:
    teacher_path_file = run_dir / "teacher_run_dir.txt"
    if not teacher_path_file.exists():
        return None

    teacher_raw = teacher_path_file.read_text(encoding="utf-8").strip()
    if not teacher_raw:
        return None

    saved_teacher = Path(teacher_raw).expanduser()
    candidate_paths: list[Path] = []
    candidate_paths.append(saved_teacher.resolve())
    if not saved_teacher.is_absolute():
        candidate_paths.append((run_dir / saved_teacher).resolve())

    teacher_name = saved_teacher.name
    if teacher_name:
        candidate_paths.append((run_dir.parent / teacher_name).resolve())
        candidate_paths.append((DEFAULT_RUNS_ROOT / teacher_name).resolve())

        refine_config_path = run_dir / "refine_config.json"
        if refine_config_path.exists():
            refine_config = _load_json(refine_config_path)
            runs_root_raw = str(refine_config.get("runs_root", "")).strip()
            if runs_root_raw:
                runs_root = Path(runs_root_raw).expanduser()
                candidate_paths.append((runs_root / teacher_name).resolve())
                candidate_paths.append((run_dir.parent / runs_root.name / teacher_name).resolve())

    seen: set[Path] = set()
    for candidate in candidate_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        if has_run_artifacts(candidate):
            return candidate
    return None


def _infer_output_dim_from_state(state: Mapping[str, torch.Tensor]) -> int | None:
    last_linear_idx = -1
    output_dim: int | None = None
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or not key.endswith(".weight"):
            continue
        match = re.search(r"\.(\d+)\.weight$", key)
        if match is None:
            continue
        layer_idx = int(match.group(1))
        if layer_idx >= last_linear_idx:
            last_linear_idx = layer_idx
            output_dim = int(tensor.shape[0])
    return output_dim


def _load_run_metadata(run_dir: Path, model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = run_dir / "train_config_used.json"
    scaler_path = run_dir / "scaler_state.json"
    teacher_run_dir = None

    if config_path.exists():
        config = _load_json(config_path)
    else:
        teacher_run_dir = _resolve_teacher_run_dir(run_dir)
        if teacher_run_dir is None:
            raise FileNotFoundError(
                f"Missing train_config_used.json under: {run_dir}. "
                "No portable teacher run could be resolved from teacher_run_dir.txt."
            )
        config = _load_json(teacher_run_dir / "train_config_used.json")

    if scaler_path.exists():
        scaler_state = _load_json(scaler_path)
    else:
        teacher_run_dir = teacher_run_dir or _resolve_teacher_run_dir(run_dir)
        if teacher_run_dir is None:
            raise FileNotFoundError(
                f"Missing scaler_state.json under: {run_dir}. "
                "No portable teacher run could be resolved from teacher_run_dir.txt."
            )
        scaler_state = _load_json(teacher_run_dir / "scaler_state.json")

    if model_path.name == "best_model_refinement.pt":
        config = dict(config)
        config["stage"] = "refinement"
        if teacher_run_dir is not None:
            config["teacher_run_dir"] = str(teacher_run_dir)

    return config, scaler_state


def resolve_run_dir(path_str: str) -> Path:
    base = Path(path_str).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")
    if base.is_file():
        base = base.parent
    if has_supported_checkpoint(base):
        return base

    candidates: list[tuple[float, Path]] = []
    for child in base.iterdir():
        if has_supported_checkpoint(child):
            candidates.append((resolve_model_path(child).stat().st_mtime, child))
    if not candidates:
        raise FileNotFoundError(f"Could not find run artifacts under: {base}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def load_run_artifacts(run_dir: Path | str, device: torch.device | str | None = None) -> RunArtifacts:
    resolved = resolve_run_dir(str(run_dir))
    model_path = resolve_model_path(resolved)
    config, scaler_state = _load_run_metadata(resolved, model_path)
    requested_device = str(device or config.get("device", "cpu"))
    if device is None and requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device_obj = torch.device(requested_device)
    state = unwrap_state_dict(torch.load(model_path, map_location=device_obj))
    inferred_output_dim = _infer_output_dim_from_state(state)
    if inferred_output_dim is not None and int(config.get("output_dim", inferred_output_dim)) != inferred_output_dim:
        config = dict(config)
        config["output_dim"] = inferred_output_dim
    model = build_model(config)
    model.load_state_dict(state)
    model.to(device_obj)
    model.eval()
    return RunArtifacts(
        model=model,
        train_config=config,
        scaler_state=scaler_state,
        run_dir=resolved,
        model_path=model_path,
    )


def zscore_from_state(value: float, z_col: str, scaler_state: Mapping[str, Any]) -> float:
    if z_col not in scaler_state["zscore"]:
        raise KeyError(f"Scaler state is missing z-score statistics for '{z_col}'.")
    stats = scaler_state["zscore"][z_col]
    return (float(value) - float(stats["mean"])) / (float(stats["std"]) + 1e-12)


def infer_feature_family(feature_columns: Sequence[str]) -> str:
    if LEGACY_FEATURE_COLUMNS.issubset(set(feature_columns)):
        return "legacy_raw"
    return "engineered_v2"


def canonicalize_raw_input(
    raw: Mapping[str, Any],
    registry: Mapping[str, DatasetMeta],
    *,
    for_engineered_v2: bool,
) -> dict[str, float]:
    out: dict[str, float] = {
        "tilt_angle_radian": float(raw["tilt_angle_radian"]),
        "plumes": float(raw["plumes"]),
        "diameter_mm": float(raw["diameter_mm"]),
        "injection_duration_us": float(raw["injection_duration_us"]),
        "injection_pressure_bar": float(raw["injection_pressure_bar"]),
        "control_backpressure_bar": float(raw["control_backpressure_bar"]),
    }

    if "ambient_pressure_bar_phys" in raw:
        pressure = float(raw["ambient_pressure_bar_phys"])
        density = float(raw.get("ambient_density_kg_m3", linear_density_from_pressure(pressure)))
        raw_mode = "physical_pressure_direct"
    elif "ambient_density_kg_m3" in raw:
        density = float(raw["ambient_density_kg_m3"])
        pressure = float(raw.get("ambient_pressure_bar_phys", linear_pressure_from_density(density)))
        raw_mode = "ambient_density_direct"
    else:
        chamber_value = float(raw.get("chamber_state_raw", raw.get("chamber_pressure_bar")))
        dataset_key = raw.get("dataset_key")
        if for_engineered_v2 and dataset_key is not None:
            dataset_key = normalize_dataset_key(str(dataset_key))
            pressure, density, raw_mode = canonicalize_chamber_state(dataset_key, chamber_value, registry)
        else:
            pressure = chamber_value
            density = linear_density_from_pressure(pressure)
            raw_mode = "legacy_chamber_pressure_bar"
        out["chamber_state_raw"] = chamber_value

    out["ambient_pressure_bar_phys"] = pressure
    out["ambient_density_kg_m3"] = density
    out["chamber_state_source"] = raw_mode
    delta_pressure = out["injection_pressure_bar"] - out["ambient_pressure_bar_phys"]
    if delta_pressure <= 0:
        raise ValueError("delta_pressure_bar_phys must stay positive during inference.")
    out["delta_pressure_bar_phys"] = delta_pressure
    out["A_scale"] = math.pow(delta_pressure / density, 0.25) * math.sqrt(out["diameter_mm"])
    out["log_A"] = math.log(out["A_scale"])
    return out


def build_feature_matrix_np(
    raw: Mapping[str, Any],
    time_ms: np.ndarray,
    scaler_state: Mapping[str, Any],
    feature_columns: Sequence[str],
    registry: Mapping[str, DatasetMeta],
    *,
    time_feature: str = TIME_FEATURE,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    family = infer_feature_family(feature_columns)
    canonical = canonicalize_raw_input(raw, registry, for_engineered_v2=(family == "engineered_v2"))

    time_min_ms = float(scaler_state["time"]["min_ms"])
    time_max_ms = float(scaler_state["time"]["max_ms"])
    time_norm = np.clip((np.asarray(time_ms, dtype=np.float32) - time_min_ms) / max(time_max_ms - time_min_ms, 1e-12), 0.0, 1.0)
    feature_series: dict[str, np.ndarray] = {
        time_feature: time_norm.astype(np.float32),
    }
    if family == "engineered_v2":
        feature_series.update(
            {
                "tilt_angle_radian_z": np.full_like(time_norm, zscore_from_state(canonical["tilt_angle_radian"], "tilt_angle_radian_z", scaler_state), dtype=np.float32),
                "plumes_z": np.full_like(time_norm, zscore_from_state(canonical["plumes"], "plumes_z", scaler_state), dtype=np.float32),
                "injection_duration_us_z": np.full_like(time_norm, zscore_from_state(canonical["injection_duration_us"], "injection_duration_us_z", scaler_state), dtype=np.float32),
                "control_backpressure_bar_z": np.full_like(time_norm, zscore_from_state(canonical["control_backpressure_bar"], "control_backpressure_bar_z", scaler_state), dtype=np.float32),
            }
        )
        if "log_A_z" in feature_columns:
            feature_series["log_A_z"] = np.full_like(
                time_norm,
                zscore_from_state(canonical["log_A"], "log_A_z", scaler_state),
                dtype=np.float32,
            )
    else:
        raw_chamber_for_legacy = float(raw.get("chamber_pressure_bar", raw.get("chamber_state_raw", canonical["ambient_pressure_bar_phys"])))
        feature_series.update(
            {
                "tilt_angle_radian_z": np.full_like(time_norm, zscore_from_state(canonical["tilt_angle_radian"], "tilt_angle_radian_z", scaler_state), dtype=np.float32),
                "plumes_z": np.full_like(time_norm, zscore_from_state(canonical["plumes"], "plumes_z", scaler_state), dtype=np.float32),
                "diameter_mm_z": np.full_like(time_norm, zscore_from_state(canonical["diameter_mm"], "diameter_mm_z", scaler_state), dtype=np.float32),
                "injection_duration_us_z": np.full_like(time_norm, zscore_from_state(canonical["injection_duration_us"], "injection_duration_us_z", scaler_state), dtype=np.float32),
                "log_injection_pressure_bar_z": np.full_like(time_norm, zscore_from_state(np.log(canonical["injection_pressure_bar"]), "log_injection_pressure_bar_z", scaler_state), dtype=np.float32),
                "log_chamber_pressure_bar_z": np.full_like(time_norm, zscore_from_state(np.log(max(raw_chamber_for_legacy, 1e-6)), "log_chamber_pressure_bar_z", scaler_state), dtype=np.float32),
                "log_delta_pressure_bar_z": np.full_like(time_norm, zscore_from_state(np.log(canonical["delta_pressure_bar_phys"]), "log_delta_pressure_bar_z", scaler_state), dtype=np.float32),
                "control_backpressure_bar_z": np.full_like(time_norm, zscore_from_state(canonical["control_backpressure_bar"], "control_backpressure_bar_z", scaler_state), dtype=np.float32),
            }
        )

    matrix = np.column_stack([feature_series[name] for name in feature_columns]).astype(np.float32)
    a_scale = np.full((len(time_norm), 1), canonical["A_scale"], dtype=np.float32)
    return matrix, a_scale, canonical


def predict_physical_sweep(
    artifacts: RunArtifacts,
    raw: Mapping[str, Any],
    time_ms: np.ndarray,
    registry: Mapping[str, DatasetMeta],
) -> dict[str, Any]:
    feature_columns = list(artifacts.train_config["feature_columns"])
    time_feature = str(artifacts.train_config.get("time_feature", TIME_FEATURE))
    features_np, a_scale_np, canonical = build_feature_matrix_np(
        raw,
        time_ms,
        artifacts.scaler_state,
        feature_columns,
        registry,
        time_feature=time_feature,
    )
    device = next(artifacts.model.parameters()).device
    features = torch.as_tensor(features_np, dtype=torch.float32, device=device)
    a_scale = torch.as_tensor(a_scale_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        output = artifacts.model(features)
        mu_hat, log_var_hat = split_mu_logvar(output)
        family = infer_feature_family(feature_columns)
        if family == "engineered_v2":
            mu_physical = a_scale * mu_hat
            std_physical = a_scale * torch.exp(0.5 * torch.clamp(log_var_hat, min=-20.0, max=20.0))
        else:
            mu_physical = mu_hat
            std_physical = torch.exp(0.5 * torch.clamp(log_var_hat, min=-20.0, max=20.0))

    std_floor = float(artifacts.train_config.get("std_clamp_min", 0.0))
    std_physical = torch.clamp(std_physical, min=std_floor)
    return {
        "time_ms": np.asarray(time_ms, dtype=np.float32),
        "mu_physical": mu_physical.detach().cpu().numpy().reshape(-1),
        "std_physical": std_physical.detach().cpu().numpy().reshape(-1),
        "a_scale": a_scale.detach().cpu().numpy().reshape(-1),
        "canonical": canonical,
        "feature_matrix_shape": tuple(features_np.shape),
    }


def torch_scalar(value: float | torch.Tensor, *, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        if requires_grad and not tensor.requires_grad:
            tensor = tensor.clone().detach().requires_grad_(True)
        return tensor
    return torch.tensor([float(value)], dtype=torch.float32, device=device, requires_grad=requires_grad)


def _torch_zscore(scaler_state: Mapping[str, Any], value: torch.Tensor, z_col: str, device: torch.device) -> torch.Tensor:
    stats = scaler_state["zscore"][z_col]
    mean = torch.tensor(float(stats["mean"]), dtype=torch.float32, device=device)
    std = torch.tensor(float(stats["std"]), dtype=torch.float32, device=device)
    return (value - mean) / (std + 1e-12)


def build_feature_tensor_torch(
    artifacts: RunArtifacts,
    raw_values: Mapping[str, Any],
    time_ms_value: float,
    registry: Mapping[str, DatasetMeta],
    *,
    axis_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(artifacts.model.parameters()).device
    feature_columns = list(artifacts.train_config["feature_columns"])
    family = infer_feature_family(feature_columns)

    tensors: dict[str, Any] = {}
    for key, value in raw_values.items():
        if isinstance(value, (int, float, np.floating)):
            tensors[key] = torch_scalar(value, device=device, requires_grad=(key == axis_name))
        else:
            tensors[key] = value

    time_min_ms = float(artifacts.scaler_state["time"]["min_ms"])
    time_max_ms = float(artifacts.scaler_state["time"]["max_ms"])
    time_norm = torch.clamp((torch_scalar(time_ms_value, device=device) - time_min_ms) / max(time_max_ms - time_min_ms, 1e-12), 0.0, 1.0)

    if family == "engineered_v2":
        if "ambient_pressure_bar_phys" in tensors:
            ambient_pressure = tensors["ambient_pressure_bar_phys"]
            ambient_density = tensors.get(
                "ambient_density_kg_m3",
                torch_scalar(linear_density_from_pressure(float(ambient_pressure.detach().cpu().item())), device=device),
            )
        elif "ambient_density_kg_m3" in tensors:
            ambient_density = tensors["ambient_density_kg_m3"]
            ambient_pressure = tensors.get(
                "ambient_pressure_bar_phys",
                torch_scalar(linear_pressure_from_density(float(ambient_density.detach().cpu().item())), device=device),
            )
        else:
            chamber_raw = float(raw_values.get("chamber_state_raw", raw_values.get("chamber_pressure_bar")))
            dataset_key = normalize_dataset_key(str(raw_values.get("dataset_key", "ds300")))
            pressure, density, _ = canonicalize_chamber_state(dataset_key, chamber_raw, registry)
            ambient_pressure = torch_scalar(pressure, device=device)
            ambient_density = torch_scalar(density, device=device)
        injection_pressure = tensors["injection_pressure_bar"]
        delta_pressure = torch.clamp(injection_pressure - ambient_pressure, min=1e-6)
        diameter = tensors["diameter_mm"]
        a_scale = torch.pow(delta_pressure / ambient_density, 0.25) * torch.sqrt(torch.clamp(diameter, min=1e-9))
        log_a = torch.log(torch.clamp(a_scale, min=1e-9))
        feature_series = {
            str(artifacts.train_config.get("time_feature", TIME_FEATURE)): time_norm,
            "tilt_angle_radian_z": _torch_zscore(artifacts.scaler_state, tensors["tilt_angle_radian"], "tilt_angle_radian_z", device),
            "plumes_z": _torch_zscore(artifacts.scaler_state, tensors["plumes"], "plumes_z", device),
            "injection_duration_us_z": _torch_zscore(artifacts.scaler_state, tensors["injection_duration_us"], "injection_duration_us_z", device),
            "control_backpressure_bar_z": _torch_zscore(artifacts.scaler_state, tensors["control_backpressure_bar"], "control_backpressure_bar_z", device),
        }
        if "log_A_z" in feature_columns:
            feature_series["log_A_z"] = _torch_zscore(artifacts.scaler_state, log_a, "log_A_z", device)
    else:
        injection_pressure = tensors["injection_pressure_bar"]
        chamber_pressure = tensors.get("chamber_pressure_bar", tensors.get("chamber_state_raw"))
        chamber_pressure = chamber_pressure if isinstance(chamber_pressure, torch.Tensor) else torch_scalar(chamber_pressure, device=device)
        delta_pressure = torch.clamp(injection_pressure - chamber_pressure, min=1e-6)
        a_scale = torch.ones_like(delta_pressure)
        feature_series = {
            str(artifacts.train_config.get("time_feature", TIME_FEATURE)): time_norm,
            "tilt_angle_radian_z": _torch_zscore(artifacts.scaler_state, tensors["tilt_angle_radian"], "tilt_angle_radian_z", device),
            "plumes_z": _torch_zscore(artifacts.scaler_state, tensors["plumes"], "plumes_z", device),
            "diameter_mm_z": _torch_zscore(artifacts.scaler_state, tensors["diameter_mm"], "diameter_mm_z", device),
            "injection_duration_us_z": _torch_zscore(artifacts.scaler_state, tensors["injection_duration_us"], "injection_duration_us_z", device),
            "log_injection_pressure_bar_z": _torch_zscore(artifacts.scaler_state, torch.log(torch.clamp(injection_pressure, min=1e-6)), "log_injection_pressure_bar_z", device),
            "log_chamber_pressure_bar_z": _torch_zscore(artifacts.scaler_state, torch.log(torch.clamp(chamber_pressure, min=1e-6)), "log_chamber_pressure_bar_z", device),
            "log_delta_pressure_bar_z": _torch_zscore(artifacts.scaler_state, torch.log(delta_pressure), "log_delta_pressure_bar_z", device),
            "control_backpressure_bar_z": _torch_zscore(artifacts.scaler_state, tensors["control_backpressure_bar"], "control_backpressure_bar_z", device),
        }

    features = torch.column_stack([feature_series[name] for name in feature_columns])
    return features, a_scale.reshape(-1, 1)


def evaluate_physical_point_with_derivatives(
    artifacts: RunArtifacts,
    raw_values: Mapping[str, Any],
    *,
    time_ms_value: float,
    axis_name: str,
    registry: Mapping[str, DatasetMeta],
) -> dict[str, float]:
    device = next(artifacts.model.parameters()).device
    axis_tensor = torch_scalar(raw_values[axis_name], device=device, requires_grad=True)
    raw_with_grad = dict(raw_values)
    raw_with_grad[axis_name] = axis_tensor
    features, a_scale = build_feature_tensor_torch(artifacts, raw_with_grad, time_ms_value, registry, axis_name=axis_name)
    output = artifacts.model(features)
    mu_hat, log_var_hat = split_mu_logvar(output)
    family = infer_feature_family(list(artifacts.train_config["feature_columns"]))
    if family == "engineered_v2":
        mu = (a_scale * mu_hat).reshape(-1)[0]
        std = (a_scale * torch.exp(0.5 * log_var_hat)).reshape(-1)[0]
    else:
        mu = mu_hat.reshape(-1)[0]
        std = torch.exp(0.5 * log_var_hat).reshape(-1)[0]

    grad = torch.autograd.grad(mu, axis_tensor, create_graph=True)[0].reshape(-1)[0]
    curvature = torch.autograd.grad(grad, axis_tensor)[0].reshape(-1)[0]
    return {
        "mu": float(mu.detach().cpu()),
        "std": float(torch.clamp(std, min=float(artifacts.train_config.get("std_clamp_min", 0.0))).detach().cpu()),
        "grad": float(grad.detach().cpu()),
        "curvature": float(curvature.detach().cpu()),
    }
