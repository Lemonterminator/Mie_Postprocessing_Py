"""
Data preparation for spray penetration frame-wise statistics.

For each condition stats file under Cine/penetration_results, this module:
- Loads the 3D penetration array (repetition, plume, frames)
- Applies geometric correction for 20° tilt (multiply by 1/cos(20°))
- Collapses repetition+plume into a single axis and computes per-frame nanmean/nanstd
- Computes time for each frame at 34,000 fps
- Saves results as <condition>_frame_stats.npz next to the source file

Saved keys in the output NPZ:
- time_s: (frames,) float64, time in seconds
- mean:   (frames,) float64, corrected per-frame mean
- std:    (frames,) float64, corrected per-frame std
- meta:   dict with frame_rate_hz and correction_factor
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np

global root

# Constants
'''
Remember to change this import
'''
from test_matrix.Nozzle1 import FPS, UMBRELLA_ANGLE


'''
Remember to change this path
'''
# DS300
# root = Path(r"C:/Users/Jiang/Documents/Mie_Py/Mie_Postprocessing_Py/DS300/penetration_results")
# Nozzle 1
root = Path(r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241003_HZ_Nozzle1")
# Nozzle 4
# root = Path(r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241007_HZ_Nozzle4")
# Nozzle 2
# root = Path(r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241017_HZ_Nozzle2")
# Nozzle 3 
# root = Path(r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241014_HZ_Nozzle3")



TILT_DEG: float = (180-UMBRELLA_ANGLE)/2.0
CORRECTION_FACTOR: float = 1.0 / float(np.cos(np.deg2rad(TILT_DEG)))
FRAME_RATE_HZ: float = FPS

def _load_condition_array(npz_path: Path) -> np.ndarray:
    """Load the 3D penetration array from a condition stats NPZ.

    Expected primary key: 'condition_data_cleaned' with shape (repetition, plume, frames).
    Falls back to the first 3D ndarray if the expected key is missing.
    """
    npz = np.load(str(npz_path))
    if "condition_data_cleaned" in npz.files:
        arr = np.array(npz["condition_data_cleaned"], copy=False)
    else:
        # Fallback: pick the first 3D ndarray
        arr = None
        for k in npz.files:
            v = npz[k]
            if isinstance(v, np.ndarray) and v.ndim == 3:
                arr = np.array(v, copy=False)
                break
        if arr is None:
            raise KeyError(
                f"No 3D array found in {npz_path}. Keys: {list(npz.files)}"
            )
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D array (rep, plume, frames); got shape {arr.shape} from {npz_path}"
        )
    return arr


def compute_frame_stats(
    data_3d: np.ndarray,
    frame_rate_hz: float = FRAME_RATE_HZ,
    correction_factor: float = CORRECTION_FACTOR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame mean and std with geometric correction.

    - data_3d: shape (repetition, plume, frames)
    - Returns: (time_s, mean, std) each with shape (frames,)
    """
    if data_3d.ndim != 3:
        raise ValueError(f"data_3d must be 3D; got {data_3d.shape}")
    rep, plume, frames = data_3d.shape
    # Apply geometric correction
    corrected = data_3d * float(correction_factor)
    # Collapse repetition and plume axes
    flat = corrected.reshape(-1, frames)  # (rep*plume, frames)
    # Nan-aware stats across shots/plumes
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    # Time for each frame index
    time_s = np.arange(frames, dtype=float) / float(frame_rate_hz)
    return time_s, mean, std


def save_frame_stats(
    out_path: Path,
    time_s: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    frame_rate_hz: float = FRAME_RATE_HZ,
    correction_factor: float = CORRECTION_FACTOR,
) -> None:
    """Save frame-wise stats to NPZ with metadata."""
    meta: Dict[str, Any] = {
        "frame_rate_hz": float(frame_rate_hz),
        "correction_factor": float(correction_factor),
    }
    np.savez(
        str(out_path),
        time_s=time_s.astype(float),
        mean=mean.astype(float),
        std=std.astype(float),
        meta=meta,
    )


def derive_output_path(npz_path: Path) -> Path:
    """Derive output file path as <name>_frame_stats.npz in the same directory."""
    stem = npz_path.stem  # e.g., 'condition_01_stats'
    parent = npz_path.parent
    out_stem = stem.replace("_stats", "_frame_stats") if stem.endswith("_stats") else f"{stem}_frame_stats"
    return parent / f"{out_stem}.npz"


def process_condition_file(npz_path: Path, verbose: bool = True) -> Path:
    """Process a single condition stats NPZ and save frame-wise stats."""
    arr3d = _load_condition_array(npz_path)
    # 
    try:
        npz = np.load(str(npz_path))
        mean = np.array(npz['condition_data_cleaned_mean'], copy=False)
        std = np.array(npz['condition_data_std'], copy=False)
        time_s, _, _ = compute_frame_stats(arr3d)
    except Exception:
        time_s, mean, std = compute_frame_stats(arr3d)
    out_path = derive_output_path(npz_path)
    save_frame_stats(out_path, time_s, mean, std)
    if verbose:
        print(
            f"Processed {npz_path.name}: frames={time_s.size}, saved -> {out_path.name}"
        )
    return out_path


def process_all(
    root: Path,
    pattern: str = "condition_*_stats.npz",
    verbose: bool = True,
) -> None:
    """Process all matching NPZ files under the given root recursively."""
    root = Path(root)
    files = sorted(root.rglob(pattern))
    if verbose:
        print(f"Found {len(files)} files under {root}")
    for f in files:
        try:
            process_condition_file(f, verbose=verbose)
        except Exception as e:
            print(f"[WARN] Skipped {f}: {e}")


if __name__ == "__main__":
    process_all(root=root)

