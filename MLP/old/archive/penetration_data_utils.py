from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ArrayLikeMapping = Union[Mapping[str, np.ndarray], MutableMapping[str, np.ndarray]]


@dataclass
class FrameStats:
    """Container for per-frame statistics."""

    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def compute_tilt_degrees(umbrella_angle_deg: float) -> float:
    """Return the tilt angle based on the umbrella angle definition."""
    return (180.0 - float(umbrella_angle_deg)) / 2.0


def compute_correction_factor(umbrella_angle_deg: float) -> float:
    """Geometric correction factor for the umbrella tilt."""
    tilt_deg = compute_tilt_degrees(umbrella_angle_deg)
    return 1.0 / float(np.cos(np.deg2rad(tilt_deg)))


def extract_condition_array(
    npz: ArrayLikeMapping, preferred_key: str = "condition_data_cleaned"
) -> np.ndarray:
    """Extract the 3D condition array (repetition, plume, frames) from an NPZ dict."""
    if preferred_key in npz:
        arr = np.array(npz[preferred_key], copy=False)
    else:
        arr = None
        for key, value in npz.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                arr = np.array(value, copy=False)
                break
        if arr is None:
            raise KeyError(f"No 3D array found in NPZ keys: {list(npz.keys())}")

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (rep, plume, frames); got {arr.shape}")
    return arr


def compute_frame_stats(
    data_3d: np.ndarray, frame_rate_hz: float, correction_factor: float
) -> FrameStats:
    """Compute time, mean, and std for each frame across all repetitions/plumes."""
    if data_3d.ndim != 3:
        raise ValueError(f"data_3d must be 3D; got shape {data_3d.shape}")

    _, _, frames = data_3d.shape
    corrected = data_3d * float(correction_factor)
    flat = corrected.reshape(-1, frames)
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    time_s = np.arange(frames, dtype=float) / float(frame_rate_hz)
    return FrameStats(time_s=time_s, mean=mean, std=std)


def save_frame_stats(
    out_path: Path,
    stats: FrameStats,
    frame_rate_hz: float,
    correction_factor: float,
) -> None:
    """Persist frame-wise stats to NPZ along with metadata."""
    meta = {
        "frame_rate_hz": float(frame_rate_hz),
        "correction_factor": float(correction_factor),
    }
    np.savez(
        str(out_path),
        time_s=stats.time_s.astype(float),
        mean=stats.mean.astype(float),
        std=stats.std.astype(float),
        meta=meta,
    )


def strictly_increasing_filter(seq: Sequence[float]) -> np.ndarray:
    """Replace non-increasing values with NaN while preserving monotonic gains."""
    arr = np.asarray(seq, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)

    filtered = np.full_like(arr, np.nan, dtype=float)
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return filtered

    first_idx = int(np.argmax(valid_mask))
    last_valid = arr[first_idx]
    filtered[first_idx] = last_valid

    for idx in range(first_idx + 1, arr.size):
        value = arr[idx]
        if np.isnan(value):
            continue
        if value > last_valid:
            filtered[idx] = value
            last_valid = value

    return filtered


def _extract_first_right_censored_idx(data: ArrayLikeMapping) -> Optional[int]:
    """Return the first right-censored frame index if present."""
    value = data.get("first_right_censored_idx")
    if value is None:
        return None
    idx = np.array(value).reshape(-1)[0]
    return int(idx) if not np.isnan(idx) else None


def build_testpoint_dataframe(
    data: ArrayLikeMapping,
    testpoint: int,
    fps: float,
    umbrella_angle_deg: float,
    plumes: float,
    diameter_mm: float,
    t_group_to_cond: Mapping[int, Mapping[str, float]],
    dropna: bool = True,
) -> pd.DataFrame:
    """Build a tidy DataFrame for a single test point."""
    condition_array = extract_condition_array(data)
    penetration_series = condition_array.reshape(-1, condition_array.shape[2])
    interval_s = 1.0 / float(fps)
    time_ms = np.arange(penetration_series.shape[1], dtype=float) * interval_s * 1000.0
    tilt_angle = np.deg2rad(compute_tilt_degrees(umbrella_angle_deg))
    censor_idx = _extract_first_right_censored_idx(data)

    cond_meta = t_group_to_cond.get(testpoint, {})
    chamber_pressure = cond_meta.get("chamber_pressure", np.nan)
    injection_duration = cond_meta.get("injection_duration", np.nan)

    frames = []
    for shot in penetration_series:
        filtered = strictly_increasing_filter(shot)
        df = pd.DataFrame(
            {
                "penetration_pixels": filtered,
                "time_ms": time_ms,
                "is_right_censored": _build_censor_flag(
                    filtered.shape[0], censor_idx
                ),
                "tilt_angle_radian": tilt_angle,
                "plumes": plumes,
                "diameter_mm": diameter_mm,
                "chamber_pressure": chamber_pressure,
                "injection_duration": injection_duration,
            }
        )
        frames.append(df.dropna(subset=["penetration_pixels"]) if dropna else df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_censor_flag(length: int, first_idx: Optional[int]) -> np.ndarray:
    flag = np.zeros(length, dtype=int)
    if first_idx is not None and 0 <= first_idx < length:
        flag[first_idx:] = 1
    return flag


def iter_testpoint_condition_files(
    results_dir: Path, condition_filename: str = "condition_01_stats.npz"
) -> Iterator[Tuple[int, Path]]:
    """Yield (testpoint, condition_stats_path) for sub-folders named like T<number>."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or not child.name.upper().startswith("T"):
            continue
        try:
            testpoint = int(child.name[1:])
        except ValueError:
            continue
        candidate = child / condition_filename
        if candidate.exists():
            yield testpoint, candidate

