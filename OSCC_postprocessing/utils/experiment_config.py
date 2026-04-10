"""Utilities for loading and managing multi-hole spray experiment configurations.

Provides helpers for:
- loading JSON test-matrix configs (``load_experiment_config``)
- accessing nozzle properties (``get_nozzle_properties``)
- expanding cartesian/grouped test matrices (``expand_test_matrix``)
- resolving per-file metadata such as injection duration and group ID
- building tidy results DataFrames (``create_results_dataframe``)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def extract_cine_number(file_path: Path | str) -> int | None:
    """Extract cine number from filename stem (e.g., ``'121.cine'`` → ``121``)."""
    path = Path(file_path)
    match = re.search(r"\d+", path.stem)
    if not match:
        return None
    return int(match.group(0))


def load_experiment_config(json_path: str | Path) -> dict:
    """Load experiment configuration from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nozzle_properties(config: dict) -> dict:
    """Extract nozzle properties from the configuration."""
    return config.get("nozzle_properties", {})


def expand_test_matrix(config: dict) -> list[dict]:
    """Expand the test matrix into a flat list of test conditions.

    Handles both cartesian-expansion and explicitly grouped configurations.
    Each returned dict includes an ``'id'`` key for group identification.

    Parameters
    ----------
    config:
        Configuration dictionary loaded from JSON.

    Returns
    -------
    list of dict
        Each entry has at minimum ``{'id': int, 'chamber_pressure_bar': ..., ...}``.
    """
    from itertools import product

    matrix = config.get("test_matrix", {})
    results = []
    group_id = 1

    if matrix.get("expansion") == "cartesian":
        pressures = matrix.get("chamber_pressures_bar", [])
        durations = matrix.get("injection_durations_us", [])
        for p, d in product(pressures, durations):
            results.append({
                "id": group_id,
                "chamber_pressure_bar": p,
                "injection_duration_us": d,
            })
            group_id += 1

    elif "groups" in matrix:
        for group in matrix["groups"]:
            if "id" in group:
                results.append({
                    "id": group["id"],
                    "chamber_pressure_bar": group.get("chamber_pressure_bar"),
                    "injection_pressure_bar": group.get("injection_pressure_bar"),
                    "control_backpressure": group.get("control_backpressure"),
                })
            elif group.get("expansion") == "cartesian":
                pressures = group.get("chamber_pressures_bar", [])
                durations = group.get("injection_durations_us", [])
                inj_pressure = group.get("injection_pressure_bar")
                for p, d in product(pressures, durations):
                    results.append({
                        "id": group_id,
                        "chamber_pressure_bar": p,
                        "injection_duration_us": d,
                        "injection_pressure_bar": inj_pressure,
                    })
                    group_id += 1

    return results


def get_test_condition_by_id(config: dict, group_id: int) -> dict | None:
    """Get a specific test condition by its group ID.

    Returns ``None`` when the ID is not found in the test matrix.
    """
    conditions = expand_test_matrix(config)
    for cond in conditions:
        if cond.get("id") == group_id:
            return cond
    return None


def compute_injection_duration_us(
    config: dict,
    cine_number: int | None,
    fallback: float | int | None = None,
) -> float | int | None:
    """Compute injection duration from a config lookup formula, if available.

    Falls back to the provided constant when the formula lookup is unavailable
    or the cine number is ``None``.
    """
    if cine_number is None:
        return fallback

    lookup = config.get("injection_duration_lookup", {})
    formula = lookup.get("formula", {})
    block_expr = formula.get("block")
    rules = formula.get("rules", [])

    if not block_expr or not rules:
        return fallback

    safe_globals: dict = {"__builtins__": {}}
    local_vars: dict = {"cine_number": cine_number}
    try:
        block = eval(str(block_expr), safe_globals, local_vars)
        local_vars["block"] = block
        for rule in rules:
            condition_expr = rule.get("condition")
            result_expr = rule.get("result")
            if not condition_expr or result_expr is None:
                continue
            if bool(eval(str(condition_expr), safe_globals, local_vars)):
                return eval(str(result_expr), safe_globals, local_vars)
    except Exception:
        return fallback

    return fallback


def extract_group_id_from_path(path: Path | str) -> int | None:
    """Extract the test group ID from a file or folder path.

    Handles naming conventions such as ``'T01'``, ``'T1'``, ``'Group_01'``,
    or purely numeric folder names.
    """
    path = Path(path)
    patterns = [
        r"[Tt](\d+)",
        r"[Gg]roup[_]?(\d+)",
        r"^(\d+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.stem)
        if match:
            return int(match.group(1))
    return None


def create_results_dataframe(
    config: dict,
    group_id: int,
    file_name: str,
    penetration_data: np.ndarray,
    num_plumes: int,
) -> pd.DataFrame:
    """Create a tidy results DataFrame with experiment conditions and penetration data.

    Parameters
    ----------
    config:
        Configuration dictionary loaded from JSON.
    group_id:
        Test group ID.
    file_name:
        Name of the processed cine file.
    penetration_data:
        Array with shape ``(num_plumes, num_frames)`` or ``(num_frames,)``.
    num_plumes:
        Number of injector holes.

    Returns
    -------
    pd.DataFrame
        One row per frame; columns include experiment conditions and per-plume
        penetration values.
    """
    condition = get_test_condition_by_id(config, group_id) or {}
    nozzle_props = get_nozzle_properties(config)

    num_frames = penetration_data.shape[1] if penetration_data.ndim > 1 else len(penetration_data)

    data: dict = {
        "file": [file_name] * num_frames,
        "frame_idx": list(range(num_frames)),
        "group_id": [group_id] * num_frames,
        "nozzle_name": [config.get("name", "")] * num_frames,
        "plumes": [nozzle_props.get("plumes")] * num_frames,
        "diameter_mm": [nozzle_props.get("diameter_mm")] * num_frames,
        "umbrella_angle_deg": [nozzle_props.get("umbrella_angle_deg")] * num_frames,
        "fps": [nozzle_props.get("fps")] * num_frames,
        "chamber_pressure_bar": [condition.get("chamber_pressure_bar")] * num_frames,
        "injection_duration_us": [condition.get("injection_duration_us")] * num_frames,
        "injection_pressure_bar": [condition.get("injection_pressure_bar")] * num_frames,
    }

    for plume_idx in range(num_plumes):
        if penetration_data.ndim > 1:
            data[f"penetration_plume_{plume_idx}"] = penetration_data[plume_idx]
        else:
            data[f"penetration_plume_{plume_idx}"] = penetration_data

    return pd.DataFrame(data)


def append_to_master_dataframe(
    master_df: pd.DataFrame | None,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    """Append new results to the master DataFrame.

    Returns a copy of ``new_df`` when ``master_df`` is ``None`` or empty.
    """
    if master_df is None or master_df.empty:
        return new_df.copy()
    return pd.concat([master_df, new_df], ignore_index=True)


def save_master_dataframe(df: pd.DataFrame, output_path: str | Path, format: str = "csv"):
    """Save the master DataFrame to disk.

    Parameters
    ----------
    df:
        DataFrame to save.
    output_path:
        Output file path (extension is appended automatically).
    format:
        ``'csv'``, ``'parquet'``, or ``'both'``.  Default is ``'csv'``.
    """
    output_path = Path(output_path)
    if format in ("csv", "both"):
        df.to_csv(output_path.with_suffix(".csv"), index=False)
    if format in ("parquet", "both"):
        df.to_parquet(output_path.with_suffix(".parquet"), index=False)


__all__ = [
    "append_to_master_dataframe",
    "compute_injection_duration_us",
    "create_results_dataframe",
    "expand_test_matrix",
    "extract_cine_number",
    "extract_group_id_from_path",
    "get_nozzle_properties",
    "get_test_condition_by_id",
    "load_experiment_config",
    "save_master_dataframe",
]
