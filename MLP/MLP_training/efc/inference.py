from __future__ import annotations

"""Inference utilities: model loading, feature canonicalisation, physical predictions.

load_run_artifacts        — load a trained checkpoint from a run directory into a
                             RunArtifacts namedtuple (model on device, train_config,
                             scaler_state, run_dir, model_path).
resolve_run_dir           — resolve "latest_stage2" or an explicit path to an absolute
                             run directory; used by Stage-3 to locate the teacher.
predict_physical_sweep    — evaluate the model over a time grid and return mu (mm)
                             and std (mm) in physical space, applying the A_scale
                             inverse transform.
build_feature_matrix_np   — assemble the z-scored feature matrix (numpy) for a given
                             set of injection conditions and a time grid.
build_feature_tensor_torch — same as above, returning a torch.Tensor on device.
canonicalize_raw_input    — normalise raw injection-condition dicts (pressure vs density
                             chamber mode, umbrella vs tilt angle) to a canonical form.
evaluate_physical_point_with_derivatives — forward pass + autograd for mu, std, and
                             their time-derivatives at a single point; used in post-hoc
                             physics diagnostics.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data_io import DatasetMeta, RunArtifacts, StageTables, _load_json, merge_config, normalize_dataset_key
from .feature_engineering import (
    apply_saved_scaler_state,
    apply_zscore,
    build_all_stage_tables,
    build_canonical_feature_table,
    build_variant_feature_table,
    canonicalize_chamber_state,
    infer_feature_family,
    linear_density_from_pressure,
    linear_pressure_from_density,
    scaler_a_scale_dp_exp,
    scaler_target_scale_mode,
)
from .feature_registry import (
    FEATURE_COLUMNS_BY_VARIANT,
    ZSCORE_BASE_COLUMNS_BY_VARIANT,
    DEFAULT_A_SCALE_DENSITY_EXP,
    DEFAULT_A_SCALE_DIAMETER_EXP,
    DEFAULT_A_SCALE_DP_EXP,
    DEFAULT_RUNS_ROOT,
    DEFAULT_SYNTHETIC_DATA_ROOT,
    LEGACY_FEATURE_COLUMNS,
    TIME_FEATURE,
    resolve_tilt_angle_radian,
)
from .models import PenetrationMLP, build_model
from .objectives import split_mu_logvar


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
    # Skip output_dim inference for multi-head architectures (FamilyAwarePenetrationMLP
    # has many 1-channel heads — the single-head heuristic infers 1 instead of 2 and
    # then overwrites the correct saved output_dim). For "single" architecture the
    # inference is still a useful fallback for old checkpoints that predate the field.
    arch_mode = str(config.get("architecture_mode", "single")).lower()
    if arch_mode == "single":
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


def canonicalize_raw_input(
    raw: Mapping[str, Any],
    registry: Mapping[str, DatasetMeta],
    *,
    for_engineered_v2: bool,
    a_scale_delta_pressure_exp: float = DEFAULT_A_SCALE_DP_EXP,
) -> dict[str, float]:
    tilt_angle_radian = resolve_tilt_angle_radian(raw)
    umbrella_angle_deg = float(raw["umbrella_angle_deg"]) if "umbrella_angle_deg" in raw else float(180.0 - np.rad2deg(2.0 * tilt_angle_radian))
    out: dict[str, float] = {
        "tilt_angle_radian": tilt_angle_radian,
        "umbrella_angle_deg": umbrella_angle_deg,
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
    out["A_scale"] = (
        math.pow(delta_pressure, float(a_scale_delta_pressure_exp))
        * math.pow(density, DEFAULT_A_SCALE_DENSITY_EXP)
        * math.pow(out["diameter_mm"], DEFAULT_A_SCALE_DIAMETER_EXP)
    )
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
    target_scale_mode = scaler_target_scale_mode(scaler_state, feature_columns)
    canonical = canonicalize_raw_input(
        raw,
        registry,
        for_engineered_v2=(family == "engineered_v2"),
        a_scale_delta_pressure_exp=scaler_a_scale_dp_exp(scaler_state),
    )

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
        if "log_injection_pressure_bar_z" in feature_columns:
            feature_series["log_injection_pressure_bar_z"] = np.full_like(
                time_norm,
                zscore_from_state(
                    float(np.log(max(float(canonical["injection_pressure_bar"]), 1e-6))),
                    "log_injection_pressure_bar_z",
                    scaler_state,
                ),
                dtype=np.float32,
            )
        if "log_chamber_pressure_bar_z" in feature_columns:
            feature_series["log_chamber_pressure_bar_z"] = np.full_like(
                time_norm,
                zscore_from_state(
                    float(np.log(max(float(canonical["ambient_pressure_bar_phys"]), 1e-6))),
                    "log_chamber_pressure_bar_z",
                    scaler_state,
                ),
                dtype=np.float32,
            )
        if "log_delta_pressure_bar_z" in feature_columns:
            feature_series["log_delta_pressure_bar_z"] = np.full_like(
                time_norm,
                zscore_from_state(
                    float(np.log(max(float(canonical["delta_pressure_bar_phys"]), 1e-6))),
                    "log_delta_pressure_bar_z",
                    scaler_state,
                ),
                dtype=np.float32,
            )
        if "diameter_mm_z" in feature_columns:
            feature_series["diameter_mm_z"] = np.full_like(
                time_norm,
                zscore_from_state(canonical["diameter_mm"], "diameter_mm_z", scaler_state),
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

    # Tier-3A/3B routing channels (family_id, nozzle_id) are NOT z-scored and
    # are derived from the dataset_key at inference time. We populate them lazily
    # so legacy feature_columns lists without these channels are unaffected.
    if "family_id" in feature_columns or "nozzle_id" in feature_columns:
        from .feature_engineering import family_id_from_name, nozzle_id_from_name
        dataset_name = str(raw.get("dataset_key") or raw.get("experiment_name") or "")
        if "family_id" in feature_columns:
            feature_series["family_id"] = np.full_like(
                time_norm, float(family_id_from_name(dataset_name)), dtype=np.float32
            )
        if "nozzle_id" in feature_columns:
            feature_series["nozzle_id"] = np.full_like(
                time_norm, float(nozzle_id_from_name(dataset_name, default=-1)), dtype=np.float32
            )

    matrix = np.column_stack([feature_series[name] for name in feature_columns]).astype(np.float32)
    a_scale_value = 1.0 if target_scale_mode == "none" else canonical["A_scale"]
    a_scale = np.full((len(time_norm), 1), a_scale_value, dtype=np.float32)
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
    a_scale_dp_exp = scaler_a_scale_dp_exp(artifacts.scaler_state)

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
        a_scale = (
            torch.pow(delta_pressure, float(a_scale_dp_exp))
            * torch.pow(ambient_density, DEFAULT_A_SCALE_DENSITY_EXP)
            * torch.pow(torch.clamp(diameter, min=1e-9), DEFAULT_A_SCALE_DIAMETER_EXP)
        )
        log_a = torch.log(torch.clamp(a_scale, min=1e-9))
        feature_series = {
            str(artifacts.train_config.get("time_feature", TIME_FEATURE)): time_norm,
            "tilt_angle_radian_z": _torch_zscore(
                artifacts.scaler_state,
                torch_scalar(resolve_tilt_angle_radian(raw_values), device=device),
                "tilt_angle_radian_z",
                device,
            ),
            "plumes_z": _torch_zscore(artifacts.scaler_state, tensors["plumes"], "plumes_z", device),
            "injection_duration_us_z": _torch_zscore(artifacts.scaler_state, tensors["injection_duration_us"], "injection_duration_us_z", device),
            "control_backpressure_bar_z": _torch_zscore(artifacts.scaler_state, tensors["control_backpressure_bar"], "control_backpressure_bar_z", device),
        }
        if "log_A_z" in feature_columns:
            feature_series["log_A_z"] = _torch_zscore(artifacts.scaler_state, log_a, "log_A_z", device)
        if "log_injection_pressure_bar_z" in feature_columns:
            feature_series["log_injection_pressure_bar_z"] = _torch_zscore(
                artifacts.scaler_state,
                torch.log(torch.clamp(injection_pressure, min=1e-6)),
                "log_injection_pressure_bar_z",
                device,
            )
        if "log_chamber_pressure_bar_z" in feature_columns:
            feature_series["log_chamber_pressure_bar_z"] = _torch_zscore(
                artifacts.scaler_state,
                torch.log(torch.clamp(ambient_pressure, min=1e-6)),
                "log_chamber_pressure_bar_z",
                device,
            )
        if "log_delta_pressure_bar_z" in feature_columns:
            feature_series["log_delta_pressure_bar_z"] = _torch_zscore(
                artifacts.scaler_state,
                torch.log(torch.clamp(delta_pressure, min=1e-6)),
                "log_delta_pressure_bar_z",
                device,
            )
        if "diameter_mm_z" in feature_columns:
            feature_series["diameter_mm_z"] = _torch_zscore(
                artifacts.scaler_state,
                diameter,
                "diameter_mm_z",
                device,
            )
    else:
        injection_pressure = tensors["injection_pressure_bar"]
        chamber_pressure = tensors.get("chamber_pressure_bar", tensors.get("chamber_state_raw"))
        chamber_pressure = chamber_pressure if isinstance(chamber_pressure, torch.Tensor) else torch_scalar(chamber_pressure, device=device)
        delta_pressure = torch.clamp(injection_pressure - chamber_pressure, min=1e-6)
        a_scale = torch.ones_like(delta_pressure)
        feature_series = {
            str(artifacts.train_config.get("time_feature", TIME_FEATURE)): time_norm,
            "tilt_angle_radian_z": _torch_zscore(
                artifacts.scaler_state,
                torch_scalar(resolve_tilt_angle_radian(raw_values), device=device),
                "tilt_angle_radian_z",
                device,
            ),
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
