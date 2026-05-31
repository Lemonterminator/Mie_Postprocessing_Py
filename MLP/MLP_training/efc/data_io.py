from __future__ import annotations

"""Core I/O types and configuration dataclasses for the MLP pipeline.

Immutable data types:
    DatasetMeta   — per-nozzle static metadata loaded from JSON test-matrix files.
    StageTables   — pair of DataFrames (representative, filtered) produced by
                    build_all_stage_tables, plus a precheck diagnostic dict.
    RunArtifacts  — loaded checkpoint: model weights, train config, scaler state,
                    and the run directory Path.

Configuration dataclasses:
    Stage1Config  — typed defaults for Stage-1 MSE training.
    Stage2Config  — typed defaults for Stage-2 NLL warm-start training.
    Each exposes to_dict() → plain dict (for merge_config) and
    from_dict(d) → reconstruction from a saved train_config_used.json.

merge_config(base, overrides) applies a flat dict-update: every key in
overrides shadows the corresponding key in base; unspecified keys keep defaults.
"""

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
import torch.nn as nn

from .feature_registry import DEFAULT_RUNS_ROOT, DEFAULT_SYNTHETIC_DATA_ROOT, DEFAULT_TEST_MATRIX_ROOT


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
    if "ds300" in lower or re.search(r"nozzle[_\s-]*0\b", lower):
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


@dataclass
class Stage1Config:
    """Default hyperparameters for Stage-1 MSE training.

    Trains PenetrationMLP to predict ``penetration / A_scale`` via MSE.  The
    log-variance head is regularised toward log_var_prior to keep it well-
    initialised for Stage-2 NLL fine-tuning.  Shape penalties (d1_positive_weight,
    d2_concave_weight) enforce monotone and concave-after-transition trajectories.

    Use to_dict() for a plain dict compatible with merge_config.
    Use from_dict(d) to reconstruct from a saved train_config_used.json.
    """

    seed: int = 42
    comparison_time_s: float = 5e-3
    data_dir: str = field(default_factory=lambda: str(DEFAULT_SYNTHETIC_DATA_ROOT))
    runs_root: str = field(default_factory=lambda: str(DEFAULT_RUNS_ROOT))
    variant: str = "a_dp050_plus_pressures"
    batch_size: int = 128
    hidden_dims: list = field(default_factory=lambda: [512, 512, 128])
    dropout: float = 0.3
    activation: str = "gelu"
    learning_rate: float = 4e-3
    weight_decay: float = 2e-4
    epochs: int = 300
    grad_clip_norm: float | None = 1.0
    num_workers: int = 0
    shuffle_train: bool = True
    n_points: int = 512
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    time_min_ms: float = 0.0
    time_max_ms: float = 5.0
    early_stopping_patience: int = 40
    early_stopping_min_delta: float = 1e-5
    log_interval: int = 50
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    var_reg_weight: float = 1e-3
    log_var_prior: float = -2.0
    log_var_bounds: tuple = field(default_factory=lambda: (-10.0, 6.0))
    d1_positive_weight: float = 5e-5
    d2_concave_weight: float = 5e-4
    d2_start_ms: float = 0.9
    d2_transition_ms: float = 0.05
    std_clamp_min: float = 1e-3
    row_selection_mode: str = "representative"
    allow_failed_precheck: bool = False
    max_curves: int | None = None
    # Tier-2C onset auxiliary head
    onset_aux_head: bool = False
    onset_aux_hidden: int = 64
    onset_aux_weight: float = 0.0
    onset_t_ms_max: float = 0.3
    # Tier-3A family-aware architecture
    architecture_mode: str = "family_head"     # {single, family_head}
    n_families: int = 2
    family_head_dims: list = field(default_factory=lambda: [128])
    fallback_family_id: int = 1
    # Tier-3B learnable per-nozzle d2 weight
    learnable_d2: bool = False
    n_families_for_d2: int = 6
    learnable_d2_floor: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Stage1Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Stage2Config:
    """Default hyperparameters for Stage-2 NLL warm-start training.

    Warm-starts from the Stage-1 checkpoint and optimises Gaussian NLL in
    A-scaled space.  Optional anchor penalties (controlled by stage2_ablation)
    prevent the predicted mean and std from inflating in the first
    anchor_window_ms ms of injection, where raw measurements are sparse.

    stage2_ablation choices:
        "no_anchor"        — pure NLL
        "mu_anchor"        — NLL + lambda_mu_anchor · L_mu (production default)
        "mu_sigma_anchor"  — NLL + lambda_mu_anchor · L_mu + lambda_sigma_anchor · L_sigma

    Use to_dict() for a plain dict compatible with merge_config.
    Use from_dict(d) to reconstruct from a saved train_config_used.json.
    """

    seed: int = 42
    comparison_time_s: float = 5e-3
    data_dir: str = field(default_factory=lambda: str(DEFAULT_SYNTHETIC_DATA_ROOT))
    runs_root: str = field(default_factory=lambda: str(DEFAULT_RUNS_ROOT))
    variant: str = "a_dp050_plus_pressures"
    batch_size: int = 96
    hidden_dims: list = field(default_factory=lambda: [512, 512, 128])
    dropout: float = 0.3
    activation: str = "gelu"
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    epochs: int = 220
    grad_clip_norm: float | None = 1.0
    num_workers: int = 0
    precompute_dataset: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = 2
    shuffle_train: bool = True
    n_points: int = 512
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    time_min_ms: float = 0.0
    time_max_ms: float = 5.0
    early_stopping_patience: int = 35
    early_stopping_min_delta: float = 1e-5
    log_interval: int = 50
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    log_var_bounds: tuple = field(default_factory=lambda: (-10.0, 6.0))
    nll_eps: float = 1e-12
    d1_positive_weight: float = 5e-5
    d2_concave_weight: float = 5e-5
    d2_start_ms: float = 0.5
    d2_transition_ms: float = 0.05
    std_clamp_min: float = 1e-3
    stage2_ablation: str = "mu_anchor"
    lambda_mu_anchor: float = 1e-2
    lambda_sigma_anchor: float = 0.0
    anchor_window_ms: float = 0.15
    sigma_anchor_floor_mm: float = 0.0
    row_selection_mode: str = "filtered"
    allow_failed_precheck: bool = False
    max_curves: int | None = None
    # Tier-2C onset auxiliary head (must match the Stage-1 setting because
    # Stage 2 warm-starts from the Stage-1 checkpoint).
    onset_aux_head: bool = False
    onset_aux_hidden: int = 64
    onset_aux_weight: float = 0.0
    onset_t_ms_max: float = 0.3
    # Tier-3A family-aware architecture (must match Stage-1 for warm-start)
    architecture_mode: str = "family_head"
    n_families: int = 2
    family_head_dims: list = field(default_factory=lambda: [128])
    fallback_family_id: int = 1
    # Tier-3B learnable per-nozzle d2 weight (Stage-2 inherits the dim only)
    learnable_d2: bool = False
    n_families_for_d2: int = 6
    learnable_d2_floor: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Stage2Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


DEFAULT_STAGE1_CONFIG = Stage1Config().to_dict()
DEFAULT_STAGE2_CONFIG = Stage2Config().to_dict()
