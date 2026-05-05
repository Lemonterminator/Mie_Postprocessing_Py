"""Lightweight sensitivity audit for the multi-hole Mie processing chain.

The script runs a small parameter grid on a handful of representative videos
without changing ``mie_multi_hole.py``.  It is designed for the full data PC:
point it at 3-5 ``.cine`` files with their folder-level ``config.json`` files,
or at ``.npy``/``.npz`` video arrays plus explicit geometry arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from OSCC_postprocessing.analysis.mie_multihole import (
    mie_multihole_postprocessing,
    mie_multihole_preprocessing,
    penetration_cdf_all_plumes,
)
from OSCC_postprocessing.binary_ops.masking import generate_ring_mask


DEFAULT_Q_VALUES = [0.990, 0.995, 0.997]
DEFAULT_BINS_VALUES = [360, 720, 1440]
DEFAULT_SOBEL_VALUES = [3, 5]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_video(path: Path, *, frame_limit: int | None, normalize_divisor: float | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        video = np.load(path)
    elif suffix == ".npz":
        payload = np.load(path)
        if not payload.files:
            raise ValueError(f"npz has no arrays: {path}")
        video = payload[payload.files[0]]
    elif suffix == ".cine":
        from OSCC_postprocessing.cine.functions_videos import load_cine_video

        video = load_cine_video(str(path), frame_limit=frame_limit)
    else:
        raise ValueError(f"Unsupported video suffix '{suffix}' for {path}")

    if frame_limit is not None and suffix != ".cine":
        video = video[: int(frame_limit)]
    video = np.asarray(video, dtype=np.float32)
    if normalize_divisor is None:
        finite_max = float(np.nanmax(video)) if video.size else 1.0
        normalize_divisor = 4096.0 if finite_max > 4.0 else 1.0
    if normalize_divisor > 0:
        video = video / float(normalize_divisor)
    return video


def _geometry_from_config(path: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if path is not None and path.exists():
        data.update(_load_json(path))

    def pick(name: str, arg_name: str | None = None):
        value = getattr(args, arg_name or name, None)
        if value is not None:
            return value
        if name in data:
            return data[name]
        raise ValueError(f"Missing geometry value '{name}'. Pass it or provide config.json.")

    return {
        "centre": (float(pick("centre_x")), float(pick("centre_y"))),
        "inner_radius": float(pick("inner_radius")),
        "outer_radius": float(pick("outer_radius")),
        "plumes": int(pick("plumes")),
        "umbrella_angle_deg": float(data.get("umbrella_angle_deg", getattr(args, "umbrella_angle_deg", 180.0))),
    }


def _finite_delta_stats(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(candidate) & np.isfinite(baseline)
    if not np.any(valid):
        return {
            "n_overlap": 0,
            "median_abs_delta_mm": float("nan"),
            "p95_abs_delta_mm": float("nan"),
            "max_abs_delta_mm": float("nan"),
        }
    delta = np.abs(candidate[valid] - baseline[valid])
    return {
        "n_overlap": int(delta.size),
        "median_abs_delta_mm": float(np.median(delta)),
        "p95_abs_delta_mm": float(np.quantile(delta, 0.95)),
        "max_abs_delta_mm": float(np.max(delta)),
    }


def audit_one_video(
    video_path: Path,
    *,
    config_path: Path | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    video = _load_video(video_path, frame_limit=args.frame_limit, normalize_divisor=args.normalize_divisor)
    if video.ndim != 3:
        raise ValueError(f"expected video shape (F,H,W), got {video.shape} for {video_path}")

    geometry = _geometry_from_config(config_path, args)
    frames, height, width = video.shape
    ring_mask = generate_ring_mask(
        height,
        width,
        geometry["centre"],
        geometry["inner_radius"],
        geometry["outer_radius"],
    )

    q_values = [float(x) for x in args.q_values]
    bin_values = [int(x) for x in args.angular_bins]
    sobel_values = [int(x) for x in args.sobel_wsize]
    baseline_key = (args.baseline_q, args.baseline_bins, args.baseline_sobel_wsize)
    cache: dict[tuple[int, int], dict[str, Any]] = {}
    penetration: dict[tuple[float, int, int], np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    for sobel_wsize in sorted(set(sobel_values + [args.baseline_sobel_wsize])):
        foreground, highpass = mie_multihole_preprocessing(
            video,
            ring_mask,
            wsize=sobel_wsize,
            sigma=args.sobel_sigma,
            frames_before_SOI=args.frames_before_soi,
            noise_floor_multiplier=args.noise_floor_multiplier,
            threshold=args.threshold,
            q_min_foreground=args.q_min_foreground,
            q_max_foreground=args.q_max_foreground,
            q_min_highpass=args.q_min_highpass,
            q_max_highpass=args.q_max_highpass,
        )
        for bins in sorted(set(bin_values + [args.baseline_bins])):
            post = mie_multihole_postprocessing(
                foreground,
                highpass,
                geometry["centre"],
                geometry["plumes"],
                geometry["inner_radius"],
                geometry["outer_radius"],
                bins=bins,
                INTERPOLATION=args.interpolation,
                BORDER_MODE=args.border_mode,
            )
            cache[(sobel_wsize, bins)] = post
            for q in sorted(set(q_values + [args.baseline_q])):
                penetration[(q, bins, sobel_wsize)] = penetration_cdf_all_plumes(
                    post["segments_fg"],
                    geometry["inner_radius"],
                    quantile=q,
                    frames_before_SOI=args.frames_before_soi,
                    umbrella_angle=geometry["umbrella_angle_deg"],
                )

    baseline_pen = penetration[baseline_key]
    baseline_offset = float(cache[(args.baseline_sobel_wsize, args.baseline_bins)]["fft_offset_deg"])
    for (q, bins, sobel_wsize), candidate in sorted(penetration.items()):
        post = cache[(sobel_wsize, bins)]
        stats = _finite_delta_stats(candidate, baseline_pen)
        rows.append(
            {
                "video_path": str(video_path),
                "frames": int(frames),
                "height": int(height),
                "width": int(width),
                "q": float(q),
                "angular_bins": int(bins),
                "sobel_wsize": int(sobel_wsize),
                "fft_offset_deg": float(post["fft_offset_deg"]),
                "fft_offset_delta_deg": float(post["fft_offset_deg"]) - baseline_offset,
                "is_baseline": (q, bins, sobel_wsize) == baseline_key,
                **stats,
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path, help="Representative .cine/.npy/.npz videos.")
    parser.add_argument("--config", type=Path, default=None, help="Shared config.json. If omitted, use each video's parent/config.json when present.")
    parser.add_argument("--centre-x", type=float, default=None)
    parser.add_argument("--centre-y", type=float, default=None)
    parser.add_argument("--inner-radius", type=float, default=None)
    parser.add_argument("--outer-radius", type=float, default=None)
    parser.add_argument("--plumes", type=int, default=None)
    parser.add_argument("--umbrella-angle-deg", type=float, default=180.0)
    parser.add_argument("--frame-limit", type=int, default=160)
    parser.add_argument("--normalize-divisor", type=float, default=None)
    parser.add_argument("--q-values", type=float, nargs="+", default=DEFAULT_Q_VALUES)
    parser.add_argument("--angular-bins", type=int, nargs="+", default=DEFAULT_BINS_VALUES)
    parser.add_argument("--sobel-wsize", type=int, nargs="+", default=DEFAULT_SOBEL_VALUES)
    parser.add_argument("--baseline-q", type=float, default=0.995)
    parser.add_argument("--baseline-bins", type=int, default=720)
    parser.add_argument("--baseline-sobel-wsize", type=int, default=3)
    parser.add_argument("--sobel-sigma", type=float, default=1.0)
    parser.add_argument("--frames-before-soi", type=int, default=10)
    parser.add_argument("--noise-floor-multiplier", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--q-min-foreground", type=float, default=5.0)
    parser.add_argument("--q-max-foreground", type=float, default=99.99)
    parser.add_argument("--q-min-highpass", type=float, default=5.0)
    parser.add_argument("--q-max-highpass", type=float, default=99.9999)
    parser.add_argument("--interpolation", default="nearest")
    parser.add_argument("--border-mode", default="constant")
    parser.add_argument("--out-dir", type=Path, default=Path("MLP/eval/mie_sensitivity"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for video_path in args.videos:
        config_path = args.config
        if config_path is None:
            candidate = video_path.parent / "config.json"
            config_path = candidate if candidate.exists() else None
        all_rows.extend(audit_one_video(video_path, config_path=config_path, args=args))

    df = pd.DataFrame(all_rows)
    detail_path = args.out_dir / "mie_sensitivity_detail.csv"
    summary_path = args.out_dir / "mie_sensitivity_summary.csv"
    df.to_csv(detail_path, index=False)
    summary = (
        df.groupby(["q", "angular_bins", "sobel_wsize"], dropna=False)
        .agg(
            n_videos=("video_path", "nunique"),
            median_abs_delta_mm=("median_abs_delta_mm", "median"),
            p95_abs_delta_mm=("p95_abs_delta_mm", "median"),
            median_fft_offset_delta_deg=("fft_offset_delta_deg", "median"),
            p95_abs_fft_offset_delta_deg=("fft_offset_delta_deg", lambda s: float(np.quantile(np.abs(s), 0.95))),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {detail_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
