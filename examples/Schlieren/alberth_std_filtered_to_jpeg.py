from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure the project root is importable when running from the examples folder.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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

    cp = _NumpyCompat()  # type: ignore

from OSCC_postprocessing.cine.functions_videos import load_cine_video
from OSCC_postprocessing.filters.stdfilt import stdfilt
from OSCC_postprocessing.io.async_avi_saver import AsyncAVISaver
from OSCC_postprocessing.utils.scaling import robust_scale

if USING_CUPY:
    from OSCC_postprocessing.rotation.rotate_with_alignment import (
        rotate_video_nozzle_at_0_half_cupy as rotate_video_nozzle_at_0_half_backend,
    )
else:
    from OSCC_postprocessing.rotation.rotate_with_alignment_cpu import (
        rotate_video_nozzle_at_0_half_numpy as rotate_video_nozzle_at_0_half_backend,
    )


DEFAULT_CINE = Path(r"G:\MeOH_test\Schlieren\T55_Schlieren Cam_3.cine")
DEFAULT_JSON = Path(r"G:\MeOH_test\Schlieren\config.json")
DEFAULT_OUT_DIR = Path(r"G:\MeOH_test\Schlieren\Processed_Results")
DEFAULT_VIDEO_BITS = 12
DEFAULT_STD_FILT_KSIZE = 21
DEFAULT_JPEG_QUALITY = 95
DEFAULT_FPS = 30


def _load_metadata(json_file: Path) -> tuple[float, tuple[float, float]]:
    with json_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    offset = float(data["offset"])
    centre = (float(data["centre_x"]), float(data["centre_y"]))
    return offset, centre


def _to_numpy(arr):
    return cp.asnumpy(arr) if USING_CUPY else np.asarray(arr)


def _export_jpeg_frames(video_u8: np.ndarray, output_dir: Path, jpeg_quality: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    for idx, frame in enumerate(video_u8):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame_path = output_dir / f"frame_{idx:05d}.jpg"
        ok = cv2.imwrite(str(frame_path), frame_bgr, params)
        if not ok:
            raise RuntimeError(f"Failed to write JPEG frame: {frame_path}")


def _export_mjpg_avi(video_u8: np.ndarray, output_path: Path, fps: int) -> None:
    video_bgr = np.repeat(video_u8[..., None], 3, axis=-1)
    saver = AsyncAVISaver(max_workers=1, default_codec="MJPG")
    try:
        saver.save(output_path, video_bgr, fps=fps, codec="MJPG", is_color=True, auto_normalize=False)
        saver.wait()
    finally:
        saver.shutdown(wait=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the Alberth schlieren pipeline through stdfilt masking, "
            "then robust-scale to [0, 1] and export a JPEG-frame video surrogate."
        )
    )
    parser.add_argument("--cine", type=Path, default=DEFAULT_CINE, help="Input .cine file.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="Metadata JSON file.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output root directory.")
    parser.add_argument("--frame-limit", type=int, default=None, help="Optional max frame count.")
    parser.add_argument("--video-bits", type=int, default=DEFAULT_VIDEO_BITS, help="Source cine bit depth.")
    parser.add_argument(
        "--std-filt-ksize",
        type=int,
        default=DEFAULT_STD_FILT_KSIZE,
        help="Odd kernel size for stdfilt.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG quality 0..100.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Preview AVI frame rate.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.std_filt_ksize <= 0 or args.std_filt_ksize % 2 == 0:
        raise ValueError("--std-filt-ksize must be a positive odd integer.")
    if not (0 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be between 0 and 100.")
    if not args.cine.exists():
        raise FileNotFoundError(f"Cine file not found: {args.cine}")
    if not args.json.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {args.json}")

    offset, centre = _load_metadata(args.json)
    print(f"[info] metadata offset={offset:.3f}, centre={centre}")

    brightness_levels = float(2**args.video_bits)
    video = load_cine_video(str(args.cine), frame_limit=args.frame_limit).astype(np.float32) / brightness_levels
    frame_count, height, _ = video.shape
    print(f"[info] loaded video shape={video.shape}, brightness_levels={brightness_levels:g}")

    out_shape = (height // 2, height)
    segment, _, _ = rotate_video_nozzle_at_0_half_backend(
        video,
        centre,
        offset,
        interpolation="nearest",
        border_mode="constant",
        out_shape=out_shape,
    )
    segment = cp.asarray(robust_scale(segment, 1, 99.9), dtype=cp.float32)
    print(f"[info] segment shape={tuple(int(v) for v in segment.shape)}, backend={'cupy' if USING_CUPY else 'numpy'}")

    segment_normalized = cp.asarray(robust_scale(segment, 1, 99.99), dtype=cp.float32)
    segment_normalized_np = _to_numpy(segment_normalized).astype(np.float32, copy=False)
    segment_normalized_u8 = (segment_normalized_np*255.0).round().astype(np.uint8)

    std_filtered_vid = stdfilt(segment, ksize=args.std_filt_ksize)
    std_filtered_vid[segment == 0.0] = 0.0

    std_filtered_normalized = cp.asarray(robust_scale(std_filtered_vid, 1, 99.99), dtype=cp.float32)
    std_filtered_normalized_np = _to_numpy(std_filtered_normalized).astype(np.float32, copy=False)
    std_filtered_normalized_u8 = (std_filtered_normalized_np * 255.0).round().astype(np.uint8)




    experiment_root = args.out_dir / f"{args.cine.stem}_std_filtered_jpeg"
    jpeg_dir = experiment_root / "jpeg_frames"
    npy_path = experiment_root / "std_filtered_normalized.npy"
    avi_path_std = experiment_root / "std_filtered_normalized_mjpg.avi"
    avi_path_raw = experiment_root / "raw_normalized_mjpg.avi"
    experiment_root.mkdir(parents=True, exist_ok=True)

    np.save(npy_path, std_filtered_normalized_np)
    _export_jpeg_frames(std_filtered_normalized_u8, jpeg_dir, args.jpeg_quality)
    _export_mjpg_avi(segment_normalized_u8, avi_path_raw, args.fps)

    _export_mjpg_avi(std_filtered_normalized_u8, avi_path_std, args.fps)

    print(f"[ok] frames={frame_count}")
    print(f"[ok] saved numpy: {npy_path}")
    print(f"[ok] saved jpeg frames: {jpeg_dir}")
    print(f"[ok] saved mjpg avi: {avi_path_raw}")
    print(f"[ok] saved mjpg avi: {avi_path_std}")



if __name__ == "__main__":
    main()
