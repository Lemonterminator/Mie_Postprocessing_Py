"""
Unified utilities for spray penetration processing.

This script now covers two stages:
1. Frame-wise statistics generation for every ``condition_*_stats.npz`` file.
2. Aggregation of per-test-point penetration traces into a single CSV dataset.

Both stages reuse helpers from ``penetration_data_utils.py`` so that
geometric corrections, dataframe construction, and filtering logic live in
a single place.
"""

from __future__ import annotations

import argparse
import os
from importlib import import_module
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - convenient import when run as a module
    from .penetration_data_utils import (
        FrameStats,
        build_testpoint_dataframe,
        compute_correction_factor,
        compute_frame_stats,
        extract_condition_array,
        iter_testpoint_condition_files,
        save_frame_stats,
    )
except ImportError:  # pragma: no cover - fallback when executed as a script
    from penetration_data_utils import (
        FrameStats,
        build_testpoint_dataframe,
        compute_correction_factor,
        compute_frame_stats,
        extract_condition_array,
        iter_testpoint_condition_files,
        save_frame_stats,
    )


def derive_output_path(npz_path: Path) -> Path:
    """Return ``<stem>_frame_stats.npz`` in the same directory as the source."""
    stem = npz_path.stem
    out_stem = stem.replace("_stats", "_frame_stats") if stem.endswith("_stats") else f"{stem}_frame_stats"
    return npz_path.with_name(f"{out_stem}.npz")


def process_condition_file(
    npz_path: Path,
    frame_rate_hz: float,
    correction_factor: float,
    prefer_precomputed: bool = True,
    verbose: bool = True,
) -> Path:
    """Process a single condition NPZ and save the frame-wise stats NPZ."""
    with np.load(str(npz_path)) as npz_data:
        arr3d = extract_condition_array(npz_data)
        stats = compute_frame_stats(arr3d, frame_rate_hz=frame_rate_hz, correction_factor=correction_factor)
        if prefer_precomputed:
            mean = npz_data.get("condition_data_cleaned_mean")
            std = npz_data.get("condition_data_std")
            if mean is not None and std is not None:
                stats = FrameStats(
                    time_s=stats.time_s,
                    mean=np.array(mean, copy=False),
                    std=np.array(std, copy=False),
                )

    out_path = derive_output_path(npz_path)
    save_frame_stats(out_path, stats, frame_rate_hz=frame_rate_hz, correction_factor=correction_factor)
    if verbose:
        print(f"[frame] {npz_path.name} -> {out_path.name} ({stats.time_s.size} frames)")
    return out_path


def process_all_condition_files(
    root: Path,
    frame_rate_hz: float,
    correction_factor: float,
    pattern: str = "condition_*_stats.npz",
    verbose: bool = True,
) -> Iterable[Path]:
    """Process every matching NPZ under ``root`` recursively."""
    root = Path(root)
    files = sorted(root.rglob(pattern))
    if verbose:
        print(f"Found {len(files)} stats files under {root}")

    for npz_path in files:
        try:
            yield process_condition_file(
                npz_path=npz_path,
                frame_rate_hz=frame_rate_hz,
                correction_factor=correction_factor,
                verbose=verbose,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            print(f"[WARN] Skipped {npz_path}: {exc}")


def build_dataframe_for_directory(
    results_dir: Path,
    fps: float,
    umbrella_angle_deg: float,
    plumes: float,
    diameter_mm: float,
    t_group_to_cond: Mapping[int, Mapping[str, float]],
    condition_filename: str = "condition_01_stats.npz",
    dropna: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Create a concatenated DataFrame for every ``T*/condition_01_stats.npz``."""
    results_dir = Path(results_dir)
    frames = []
    for testpoint, npz_path in iter_testpoint_condition_files(results_dir, condition_filename):
        if verbose:
            print(f"[csv] T{testpoint} -> {npz_path.name}")
        with np.load(str(npz_path)) as npz_data:
            df = build_testpoint_dataframe(
                data=npz_data,
                testpoint=testpoint,
                fps=fps,
                umbrella_angle_deg=umbrella_angle_deg,
                plumes=plumes,
                diameter_mm=diameter_mm,
                t_group_to_cond=t_group_to_cond,
                dropna=dropna,
            )
            if df.empty:
                continue
            df["test_point"] = testpoint
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def default_csv_name(results_dir: Path) -> str:
    """Return ``<parent>_penetration_data.csv`` for backward compatibility."""
    results_dir = Path(results_dir)
    parent = results_dir.parent.name if results_dir.parent != results_dir else results_dir.name
    return f"{parent}_penetration_data.csv"


def write_dataframe(df: pd.DataFrame, output_dir: Path, csv_name: Optional[str] = None) -> Path:
    """Persist the combined dataframe to ``output_dir``."""
    output_dir = Path(output_dir)
    csv_name = csv_name or default_csv_name(output_dir)
    out_path = output_dir / csv_name
    df.to_csv(out_path, index=False)
    return out_path


def load_test_matrix(module_name: str):
    """Dynamically import a ``test_matrix`` module."""
    dotted = module_name if "." in module_name else f"MLP.test_matrix.{module_name}"
    module = import_module(dotted)
    required = ("FPS", "UMBRELLA_ANGLE", "PLUMES", "DIAMETER", "T_GROUP_TO_COND")
    missing = [attr for attr in required if not hasattr(module, attr)]
    if missing:
        raise AttributeError(f"{dotted} is missing: {', '.join(missing)}")
    return module


def _autodetect_results_dir(root: Optional[Path]) -> Optional[Path]:
    if root is None:
        return None
    root = Path(root)
    if root.name.lower().endswith("penetration_results"):
        return root
    candidate = root / "penetration_results"
    return candidate if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spray penetration data preparation pipeline.")
    parser.add_argument("--root", type=Path, help="Root directory containing condition_*_stats.npz files.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing T*/ folders; defaults to <root>/penetration_results if present.",
    )
    parser.add_argument(
        "--test-matrix",
        default="Nozzle8",
        help="Name of the test_matrix module (e.g., Nozzle1, Nozzle5, ...).",
    )
    parser.add_argument(
        "--stats-pattern",
        default="condition_*_stats.npz",
        help="Glob pattern for stats files when computing frame-wise data.",
    )
    parser.add_argument(
        "--condition-file",
        default="condition_01_stats.npz",
        help="Filename to load within each T*/ folder for CSV generation.",
    )
    parser.add_argument("--csv-name", help="Optional name for the aggregated CSV output.")
    parser.add_argument("--skip-frame-stats", action="store_true", help="Skip frame-wise stats generation.")
    parser.add_argument("--skip-dataframe", action="store_true", help="Skip CSV aggregation.")
    parser.add_argument("--keep-na", action="store_true", help="Retain NaNs in the penetration_series output.")
    parser.add_argument("--verbose", action="store_true", help="Print progress information.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = args.verbose

    root = args.root
    if root is None:
        env_root = os.getenv("PENETRATION_ROOT")
        if env_root:
            root = Path(env_root)

    results_dir = args.results_dir or _autodetect_results_dir(root)
    test_matrix = load_test_matrix(args.test_matrix)
    fps = float(test_matrix.FPS)
    umbrella_angle = float(test_matrix.UMBRELLA_ANGLE)
    plumes = float(test_matrix.PLUMES)
    diameter_mm = float(test_matrix.DIAMETER)
    t_group_to_cond = test_matrix.T_GROUP_TO_COND
    correction_factor = compute_correction_factor(umbrella_angle)

    if not args.skip_frame_stats:
        if root is None:
            raise SystemExit("Please provide --root (or PENETRATION_ROOT) for frame-wise stats generation.")
        for _ in process_all_condition_files(
            root=root,
            frame_rate_hz=fps,
            correction_factor=correction_factor,
            pattern=args.stats_pattern,
            verbose=verbose,
        ):
            pass

    if not args.skip_dataframe:
        if results_dir is None:
            raise SystemExit(
                "Unable to determine the penetration_results directory. Provide --results-dir explicitly."
            )
        df = build_dataframe_for_directory(
            results_dir=results_dir,
            fps=fps,
            umbrella_angle_deg=umbrella_angle,
            plumes=plumes,
            diameter_mm=diameter_mm,
            t_group_to_cond=t_group_to_cond,
            condition_filename=args.condition_file,
            dropna=not args.keep_na,
            verbose=verbose,
        )
        if df.empty:
            print("No rows were generated for the CSV output; skipping file write.")
        else:
            out_path = write_dataframe(df, results_dir=results_dir, csv_name=args.csv_name)
            if verbose:
                print(f"Saved DataFrame to {out_path}")


if __name__ == "__main__":
    main()
