from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Ensure the project root is importable when running from the examples folder.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Prefer GPU if CuPy is available; otherwise use a NumPy-compatible shim.
try:
    import cupy as _cupy  # type: ignore

    _cupy.cuda.runtime.getDeviceCount()
    cp = _cupy
    USING_CUPY = True
except Exception as exc:  # pragma: no cover - hardware dependent
    print(f"CuPy unavailable, falling back to NumPy backend: {exc}")
    USING_CUPY = False

    class _NumpyCompat:
        def __getattr__(self, name):
            return getattr(np, name)

        def asarray(self, a, dtype=None):
            return np.asarray(a, dtype=dtype)

        def asnumpy(self, a):
            return np.asarray(a)

        def get(self, a):
            return a

    cp = _NumpyCompat()  # type: ignore

from OSCC_postprocessing.functions_bw import keep_largest_component
from OSCC_postprocessing.functions_videos import load_cine_video
from OSCC_postprocessing.optical_flow import compute_farneback_flows
from OSCC_postprocessing.video_filters import gaussian_video_cpu, median_filter_video_auto
from OSCC_postprocessing.video_playback import play_video_cv2, play_videos_side_by_side
from OSCC_postprocessing.svd_background_removal import (
    svd_foreground_cuda as svd_foreground,
        godec_like,
        )

# Import rotation utility based on backend availability to avoid hard Cupy dependency
if USING_CUPY:
    from OSCC_postprocessing.rotate_with_alignment import (
        rotate_video_nozzle_at_0_half_cupy as rotate_video_nozzle_at_0_half_backend,
    )
else:
    from OSCC_postprocessing.rotate_with_alignment_cpu import (
        rotate_video_nozzle_at_0_half_numpy as rotate_video_nozzle_at_0_half_backend,
    )

# Default dataset location can be overridden with the SCH_DATA_PATH environment variable.
DATA_ROOT = (
    Path.home()
    / "OneDrive - W\u00E4rtsil\u00E4 Corporation"
    / "Documents"
    / "Nozzle_temp_impact_SCH"
)


def _iter_files(folder: Path) -> Iterable[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file())


def to_numpy(arr):
    return cp.asnumpy(arr) if USING_CUPY else np.asarray(arr)




def _min_max_scale(arr: cp.ndarray) -> cp.ndarray:
    mn = arr.min()
    mx = arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return cp.zeros_like(arr)


def _load_metadata(files: Iterable[Path]) -> tuple[int, float, tuple[float, float]]:
    env_cx = os.getenv("SCH_CENTRE_X")
    env_cy = os.getenv("SCH_CENTRE_Y")
    env_offset = os.getenv("SCH_OFFSET")
    if env_cx and env_cy and env_offset:
        plumes = int(os.getenv("SCH_PLUMES", "0"))
        centre = (float(env_cx), float(env_cy))
        offset = float(env_offset)
        return plumes, offset, centre

    for file in files:
        if file.suffix.lower() == ".json":
            with file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                plumes = int(data["plumes"])
                offset = float(data["offset"])
                centre = (float(data["centre_x"]) + 4.0, float(data["centre_y"]) - 3.0)
                return plumes, offset, centre
    raise FileNotFoundError(
        "No metadata JSON found alongside the cine files. "
        "Provide SCH_CENTRE_X, SCH_CENTRE_Y, and SCH_OFFSET environment variables to run without metadata."
    )


def _prepare_temporal_smoothing(rotated: cp.ndarray, smooth_frames: int) -> tuple[cp.ndarray, np.ndarray]:
    rotated_cpu = to_numpy(rotated)
    smoothed_np = median_filter_video_auto(np.swapaxes(rotated_cpu, 0, 2), smooth_frames, 1)
    smoothed_np = np.swapaxes(smoothed_np, 0, 2)

    min_val = smoothed_np.min()
    max_val = smoothed_np.max()
    if max_val > min_val:
        smoothed_np = (smoothed_np - min_val) / (max_val - min_val)
    else:
        smoothed_np = np.zeros_like(smoothed_np, dtype=np.float32)

    smoothed_cp = cp.asarray(smoothed_np, dtype=cp.float32)
    return smoothed_cp, smoothed_np


def _rotate_align_video_cpu(
    video: np.ndarray,
    nozzle_center: tuple[float, float],
    offset_deg: float,
    *,
    interpolation: str,
    out_shape: tuple[int, int] | None,
    border_mode: str,
    cval: float,
) -> np.ndarray:
    """
    Delegate to the NumPy implementation in OSCC_postprocessing.rotate_with_alignment_cpu.
    Returns only the rotated video (np.ndarray).
    """
    rotated_np, _, _ = rotate_video_nozzle_at_0_half_backend(
        video,
        nozzle_center,
        offset_deg,
        interpolation=interpolation,
        border_mode=border_mode,
        out_shape=out_shape,
        cval=cval,
    )
    return rotated_np.astype(np.float32, copy=False)


def main() -> None:
    folder_path = Path(os.environ.get("SCH_DATA_PATH", DATA_ROOT))
    if not folder_path.exists():
        raise FileNotFoundError(f"Data folder not found: {folder_path}")

    files = _iter_files(folder_path)
    number_of_plumes, offset, centre = _load_metadata(files)
    print(f"Metadata: plumes={number_of_plumes}, offset={offset}, centre={centre}")

    cine_files = [f for f in files if f.suffix.lower() == ".cine"]
    if not cine_files:
        raise FileNotFoundError(f"No .cine files found in {folder_path}")

    for cine_file in cine_files:
        frame_limit = int(os.environ.get("SCH_FRAME_LIMIT", "399"))
        video = load_cine_video(str(cine_file), frame_limit=frame_limit).astype(np.float32) / 4096.0
        F, H, W = video.shape

        if USING_CUPY:
            try:
                rotated_gpu, _, _ = rotate_video_nozzle_at_0_half_backend(
                    video,
                    centre,
                    -45.0,
                    interpolation="bicubic",
                    border_mode="constant",
                    out_shape=(H // 2, W),
                )
                rotated = cp.asarray(rotated_gpu, dtype=cp.float32)
            except Exception as exc:  # pragma: no cover - hardware dependent
                print(f"GPU rotation failed ({exc}), falling back to CPU numpy implementation.")
                rotated_np = _rotate_align_video_cpu(
                    video,
                    centre,
                    -45.0,
                    interpolation="bicubic",
                    border_mode="constant",
                    out_shape=(H // 2, W),
                    cval=0.0,
                )
                rotated = cp.asarray(rotated_np, dtype=cp.float32)
        else:
            rotated_np = _rotate_align_video_cpu(
                video,
                centre,
                -45.0,
                interpolation="bicubic",
                border_mode="constant",
                out_shape=(H // 2, W),
                cval=0.0,
            )
            rotated = cp.asarray(rotated_np, dtype=cp.float32)

        intensity = to_numpy(cp.sum(cp.sum(rotated, axis=2), axis=1))
        _ = intensity  # currently for diagnostic plotting if desired

        rotated = 1.0 - rotated

        smooth_frames = 3
        temporal_smoothing, temporal_smoothing_np = _prepare_temporal_smoothing(rotated, smooth_frames)

        foreground_svd = svd_foreground(temporal_smoothing, 10, bkg_frame_limit=20)
        foreground_godec = godec_like(temporal_smoothing, 10)

        svd_pos = cp.maximum(foreground_svd, 0.0)
        svd_neg = cp.maximum(-foreground_svd, 0.0)

        svd_pos = _min_max_scale(svd_pos)
        svd_neg = _min_max_scale(svd_neg)

        gamma = 1.5
        play_videos_side_by_side(
            (
                to_numpy(cp.swapaxes(rotated, 1, 2)),
                to_numpy(cp.swapaxes(svd_pos, 1, 2)) ** gamma,
                to_numpy(cp.swapaxes(svd_neg, 1, 2)) ** gamma,
            ),
            intv=100,
        )

        tdi_map = to_numpy(cp.sum(svd_pos ** gamma, axis=1))
        plt.imshow(tdi_map.T, cmap="jet", origin="lower")
        plt.title("Time-Distance Intensity Map")
        plt.show(block=False)

        tdi_map_norm = (tdi_map - tdi_map.min()) / (tdi_map.max() - tdi_map.min() + 1e-12)
        tdi_map_norm = np.clip(tdi_map_norm, 0.0, 1.0)
        tdi_map_u8 = (tdi_map_norm * 255.0).astype(np.uint8)
        _, bw = cv2.threshold(tdi_map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw1 = keep_largest_component(bw)
        bw2 = bw1.T

        penetration = bw2.shape[0] - np.argmax(bw2[::-1, :], axis=0)
        penetration[penetration == bw2.shape[0]] = 0
        plt.figure()
        plt.plot(penetration, color="red")
        plt.title("Penetration over time")
        plt.show(block=False)

        rotated_cp = cp.asarray(rotated.copy())
        for f in range(F):
            rotated_cp[f, :, penetration[f]:] *= 1e-1
        play_video_cv2(to_numpy(rotated_cp), intv=100)

        foreground_godec = _min_max_scale(foreground_godec)
        play_videos_side_by_side(
            (
                to_numpy(rotated),
                np.clip(to_numpy(foreground_svd), 0.0, 1.0),
                np.clip(to_numpy(foreground_godec), 0.0, 1.0),
            )
        )

        flows_svd = compute_farneback_flows(to_numpy(svd_pos))
        flows_mag = np.sqrt(flows_svd[:, 0, :, :] ** 2 + flows_svd[:, 1, :, :] ** 2)
        flows_mag = (flows_mag - flows_mag.min()) / (flows_mag.max() - flows_mag.min() + 1e-12)
        play_video_cv2(flows_mag * 10.0)
        tdi_map_flows = np.sum(flows_mag, axis=1)

        tdi_map_flows_scaled = (tdi_map_flows - tdi_map_flows.min()) / (
            tdi_map_flows.max() - tdi_map_flows.min() + 1e-12
        )
        _ = tdi_map_flows_scaled  # placeholder for downstream analysis

        xy_median = median_filter_video_auto(to_numpy(foreground_godec), 5, 5)
        xy_blur = gaussian_video_cpu(xy_median.astype(np.float32), ksize=(7, 7))
        play_videos_side_by_side(
            (
                to_numpy(rotated),
                temporal_smoothing_np,
                to_numpy(foreground_godec),
                xy_median,
                xy_blur,
            )
        )


if __name__ == "__main__":
    main()
