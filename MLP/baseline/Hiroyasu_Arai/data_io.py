from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_ROOT = PROJECT_ROOT / "MLP" / "synthetic_data"
TEST_MATRIX_ROOT = PROJECT_ROOT / "test_matrix_json"
REFERENCE_PRESSURE_BAR = 4.42
REFERENCE_DENSITY_KG_M3 = 5.0


@dataclass(frozen=True)
class DatasetMeta:
    key: str
    display_name: str
    chamber_mode: str
    chamber_density_to_pressure: dict[float, float]
    chamber_pressure_to_density: dict[float, float]
    nozzle_properties: dict[str, float]


def normalize_dataset_key(text: str) -> str:
    lower = str(text).lower()
    if "ds300" in lower or re.search(r"nozzle[_\s-]*0\b", lower):
        return "ds300"
    match = re.search(r"nozzle[_\s-]*([1-8])", lower)
    if match:
        return f"nozzle{match.group(1)}"
    raise KeyError(f"Could not infer dataset key from: {text}")


def split_dir_names(split: str) -> tuple[str, str]:
    if split == "clean":
        return "series_wide_clean", "series_clean"
    if split == "all":
        return "series_wide_all", "series_all"
    raise ValueError(f"Unsupported split: {split}")


def load_source_table(source: str = "cdf", split: str = "clean", synthetic_root: Path | None = None) -> pd.DataFrame:
    root = Path(synthetic_root or SYNTHETIC_ROOT)
    wide_dir_name, _ = split_dir_names(split)
    frames: list[pd.DataFrame] = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        wide_dir = experiment_dir / source / wide_dir_name
        if not wide_dir.exists():
            continue
        for path in sorted(wide_dir.glob("*.csv")):
            df = pd.read_csv(path)
            df.insert(0, "experiment_name", experiment_dir.name)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No {source!r} {split!r} wide-series files found under {root}. "
            "Run MLP/curve_fit/fit_raw_data.py first."
        )
    return pd.concat(frames, ignore_index=True, sort=False).copy()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset_registry(test_matrix_root: Path | None = None) -> dict[str, DatasetMeta]:
    root = Path(test_matrix_root or TEST_MATRIX_ROOT)
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
                if densities is None or pressures is None:
                    continue
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


def linear_density_from_pressure(pressure_bar: float) -> float:
    return REFERENCE_DENSITY_KG_M3 * float(pressure_bar) / REFERENCE_PRESSURE_BAR


def linear_pressure_from_density(density_kg_m3: float) -> float:
    return REFERENCE_PRESSURE_BAR * float(density_kg_m3) / REFERENCE_DENSITY_KG_M3


def canonicalize_chamber_state(
    dataset_key: str,
    chamber_state_raw: float,
    registry: dict[str, DatasetMeta],
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


def add_canonical_features(df_in: pd.DataFrame, registry: dict[str, DatasetMeta]) -> pd.DataFrame:
    df = df_in.copy()
    if "dataset_key" not in df.columns:
        df["dataset_key"] = df["experiment_name"].map(normalize_dataset_key)
    df["chamber_state_raw"] = pd.to_numeric(df["chamber_pressure_bar"], errors="coerce").astype(float)
    df["diameter_mm"] = pd.to_numeric(df["diameter_mm"], errors="coerce").astype(float)
    df["injection_pressure_bar"] = pd.to_numeric(df["injection_pressure_bar"], errors="coerce").astype(float)
    df["injection_duration_us"] = pd.to_numeric(df["injection_duration_us"], errors="coerce").astype(float)
    df["control_backpressure_bar"] = pd.to_numeric(df["control_backpressure_bar"], errors="coerce").astype(float)
    df["plumes"] = pd.to_numeric(df["plumes"], errors="coerce").astype(float)
    canonical_rows = [
        canonicalize_chamber_state(str(dataset_key), float(chamber_state), registry)
        for dataset_key, chamber_state in zip(df["dataset_key"], df["chamber_state_raw"])
    ]
    df["ambient_pressure_bar_phys"] = [item[0] for item in canonical_rows]
    df["ambient_density_kg_m3"] = [item[1] for item in canonical_rows]
    df["chamber_state_source"] = [item[2] for item in canonical_rows]
    df["delta_pressure_bar_phys"] = df["injection_pressure_bar"] - df["ambient_pressure_bar_phys"]
    bad = df["delta_pressure_bar_phys"] <= 0
    if bad.any():
        sample = df.loc[bad, ["experiment_name", "injection_pressure_bar", "ambient_pressure_bar_phys"]].head()
        raise ValueError(f"Found non-positive pressure difference rows:\n{sample}")
    # H-A-specific SI columns
    df["delta_pressure_pa"] = df["delta_pressure_bar_phys"] * 1e5
    df["diameter_m"] = df["diameter_mm"] / 1000.0
    df["t_inj_s"] = df["injection_duration_us"] / 1e6
    return df


def sorted_frame_ids(df: pd.DataFrame, prefix: str) -> list[int]:
    ids = []
    for col in df.columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix):]
        if suffix.isdigit():
            ids.append(int(suffix))
    return sorted(ids)


def resolve_repo_path(path_text: str) -> Path:
    text = str(path_text)
    if not text:
        return Path(text)
    normalized = text.replace("\\", "/")
    marker = "Mie_scattering_top_view_results/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        return PROJECT_ROOT / "Mie_scattering_top_view_results" / rel
    marker = "MLP/synthetic_data/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        return PROJECT_ROOT / "MLP" / "synthetic_data" / rel
    return Path(text)


def trajectory_test_name(file_path: str, fallback: str) -> str:
    try:
        text = str(file_path)
        if "\\" in text:
            return PureWindowsPath(text).parent.name
        return Path(text).parent.name
    except Exception:
        return str(fallback)


def assign_grouped_split(df: pd.DataFrame, seed: int, val_frac: float, test_frac: float) -> pd.DataFrame:
    out = df.copy()
    out["split_group_id"] = (
        out["dataset_key"].astype(str)
        + "|"
        + out["test_name"].astype(str)
        + "|dur="
        + out["injection_duration_us"].round(6).astype(str)
    )
    rng = np.random.default_rng(int(seed))
    groups = np.asarray(pd.Series(out["split_group_id"]).drop_duplicates().tolist(), dtype=object)
    rng.shuffle(groups)
    n_test = int(np.floor(float(test_frac) * len(groups)))
    n_val = int(np.floor(float(val_frac) * len(groups)))
    n_test = min(max(n_test, 1), max(len(groups) - 2, 1))
    n_val = min(max(n_val, 1), max(len(groups) - n_test - 1, 1))
    test_groups = set(groups[:n_test].tolist())
    val_groups = set(groups[n_test: n_test + n_val].tolist())
    out["sample_split"] = "train"
    out.loc[out["split_group_id"].isin(val_groups), "sample_split"] = "val"
    out.loc[out["split_group_id"].isin(test_groups), "sample_split"] = "test"
    return out


def assign_row_random_split(df: pd.DataFrame, seed: int, val_frac: float, test_frac: float) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    indices = np.arange(len(df), dtype=int)
    rng.shuffle(indices)
    n_test = int(round(len(df) * float(test_frac)))
    n_val = int(round(len(df) * float(val_frac)))
    n_test = min(max(n_test, 1), max(len(df) - 2, 1))
    n_val = min(max(n_val, 1), max(len(df) - n_test - 1, 1))
    labels = np.full(len(df), "train", dtype=object)
    labels[indices[:n_test]] = "test"
    labels[indices[n_test: n_test + n_val]] = "val"
    out = df.copy()
    out["sample_split"] = labels
    out["split_group_id"] = out["traj_key"]
    return out


def assign_leave_one_nozzle_split(df: pd.DataFrame, holdout_nozzle: str, seed: int, val_frac: float) -> pd.DataFrame:
    holdout_key = normalize_dataset_key(holdout_nozzle)
    out = df.copy()
    out["split_group_id"] = (
        out["dataset_key"].astype(str)
        + "|"
        + out["test_name"].astype(str)
        + "|dur="
        + out["injection_duration_us"].round(6).astype(str)
    )
    out["sample_split"] = "train"
    out.loc[out["dataset_key"] == holdout_key, "sample_split"] = "test"
    train_groups = np.asarray(
        pd.Series(out.loc[out["sample_split"] == "train", "split_group_id"]).drop_duplicates().tolist(),
        dtype=object,
    )
    rng = np.random.default_rng(int(seed))
    rng.shuffle(train_groups)
    n_val = int(np.floor(float(val_frac) * len(train_groups)))
    n_val = min(max(n_val, 1), max(len(train_groups) - 1, 1))
    val_groups = set(train_groups[:n_val].tolist())
    out.loc[out["split_group_id"].isin(val_groups), "sample_split"] = "val"
    return out


def assign_split(
    wide_df: pd.DataFrame,
    *,
    mode: str,
    seed: int,
    val_frac: float,
    test_frac: float,
    holdout_nozzle: str | None = None,
) -> pd.DataFrame:
    if mode == "row_random_stage3":
        return assign_row_random_split(wide_df, seed=seed, val_frac=val_frac, test_frac=test_frac)
    if mode == "grouped_condition":
        return assign_grouped_split(wide_df, seed=seed, val_frac=val_frac, test_frac=test_frac)
    if mode == "leave_one_nozzle":
        if not holdout_nozzle:
            raise ValueError("holdout_nozzle is required when split_mode='leave_one_nozzle'.")
        return assign_leave_one_nozzle_split(wide_df, holdout_nozzle=holdout_nozzle, seed=seed, val_frac=val_frac)
    raise ValueError(f"Unsupported split mode: {mode}")


def wide_to_points(
    wide_df: pd.DataFrame,
    *,
    time_min_ms: float,
    time_max_ms: float,
) -> pd.DataFrame:
    frame_ids = sorted_frame_ids(wide_df, "time_ms_")
    meta_cols = [
        "experiment_name", "dataset_key", "test_name", "traj_key",
        "file_path", "file_name", "file_stem", "plume_idx",
        "sample_split", "split_group_id", "plumes",
        "diameter_mm", "diameter_m", "fps",
        "chamber_pressure_bar", "ambient_pressure_bar_phys", "ambient_density_kg_m3",
        "delta_pressure_bar_phys", "delta_pressure_pa", "chamber_state_source",
        "injection_duration_us", "injection_pressure_bar", "control_backpressure_bar",
        "t_inj_s",
    ]
    meta_cols = [col for col in meta_cols if col in wide_df.columns]
    rows: list[pd.DataFrame] = []
    for frame_id in frame_ids:
        time_col = f"time_ms_{frame_id:03d}"
        pen_col = f"penetration_mm_{frame_id:03d}"
        if time_col not in wide_df.columns or pen_col not in wide_df.columns:
            continue
        block = wide_df.loc[:, meta_cols].copy()
        block["frame_id"] = frame_id
        block["time_ms"] = pd.to_numeric(wide_df[time_col], errors="coerce")
        block["time_s"] = block["time_ms"] * 1e-3
        block["pen_true_mm"] = pd.to_numeric(wide_df[pen_col], errors="coerce")
        valid = (
            np.isfinite(block["time_ms"])
            & np.isfinite(block["pen_true_mm"])
            & (block["time_ms"] >= float(time_min_ms))
            & (block["time_ms"] <= float(time_max_ms))
        )
        if valid.any():
            rows.append(block.loc[valid])
    if not rows:
        raise RuntimeError("No finite points found after wide-to-long conversion.")
    return pd.concat(rows, ignore_index=True, sort=False)


def prepare_wide_table(config: dict[str, Any]) -> pd.DataFrame:
    registry = build_dataset_registry()
    wide = load_source_table(config.get("source", "cdf"), config.get("series_split", "clean")).copy()
    wide["dataset_key"] = wide["experiment_name"].map(normalize_dataset_key)
    wide["test_name"] = [
        trajectory_test_name(path, fallback)
        for path, fallback in zip(
            wide.get("file_path", pd.Series("", index=wide.index)),
            wide.get("file_stem", wide.index.astype(str)),
        )
    ]
    wide["traj_key"] = (
        wide["experiment_name"].astype(str)
        + "|"
        + wide["test_name"].astype(str)
        + "|"
        + wide["file_name"].astype(str)
        + "|plume="
        + wide["plume_idx"].astype(str)
    )
    wide = add_canonical_features(wide, registry)
    return assign_split(
        wide,
        mode=str(config.get("split_mode", "grouped_condition")),
        seed=int(config.get("seed", 42)),
        val_frac=float(config.get("val_frac", 0.15)),
        test_frac=float(config.get("test_frac", 0.15)),
        holdout_nozzle=config.get("holdout_nozzle"),
    )
