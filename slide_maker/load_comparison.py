"""
Load Comparison - Load processed data for testpoint comparison
==============================================================

Reads saved data from process_all.py and locates files for comparison.
In "sample" mode, finds the first repetition (rep 1) for each testpoint.
In "average" mode, finds all repetitions for each test point.

Input Structure (created by process_all.py):
    {video_dir}/Processed_Results/
        ├── Rotated_Videos/     # .avi, .npz video files
        └── Postprocessed_Data/ # .csv data files

    {dewe_dir}/Processed_Results/
        └── Postprocessed_Data/ # .csv files from .dxd

Usage:
    python load_comparison.py                    # Use default config.json
    python load_comparison.py --config my.json   # Use custom config file
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from OSCC_postprocessing.dewe.dewe import align_dewe_dataframe_to_soe


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_default_config_path() -> Path:
    return Path(__file__).parent / "config.json"


# =============================================================================
# File Naming Conventions
# =============================================================================

def parse_testpoint_from_filename(filename: str) -> Optional[str]:
    """Extract testpoint number from filename (e.g., 'T2_0001.csv' -> '2')."""
    stem = Path(filename).stem
    if stem.startswith("T"):
        tp = ""
        for ch in stem[1:]:
            if ch.isdigit():
                tp += ch
            else:
                break
        if tp:
            return tp
    return None


def parse_repetition_from_filename(filename: str) -> int:
    """Extract repetition number from filename (e.g., 'T2_0001.csv' -> 1)."""
    stem = Path(filename).stem
    parts = stem.split("_")
    for part in parts:
        if len(part) == 4 and part.isdigit():
            return int(part)
    return 1


def find_files_for_testpoint(
    directory: Path,
    testpoint: str,
    suffix: str = ".csv",
    repetition: Optional[int] = None,
) -> List[Path]:
    """Find all files matching a testpoint, optionally filtered by repetition."""
    if not directory.exists():
        return []
    matches = []
    for f in directory.iterdir():
        if not f.is_file() or f.suffix.lower() != suffix.lower():
            continue
        tp = parse_testpoint_from_filename(f.name)
        if tp == testpoint:
            if repetition is not None:
                rep = parse_repetition_from_filename(f.name)
                if rep != repetition:
                    continue
            matches.append(f)
    return sorted(matches)


def get_related_file_key(path: Path) -> str:
    """Return a repetition-agnostic key for grouping related files."""
    parts = [p for p in path.stem.split("_") if not (len(p) == 4 and p.isdigit())]
    return "_".join(parts)


def group_related_files(files: List[Path]) -> Dict[str, List[Path]]:
    """Group repetition files that represent the same processed output."""
    groups: Dict[str, List[Path]] = {}
    for path in sorted(files):
        groups.setdefault(get_related_file_key(path), []).append(path)
    return groups


def _load_first_npz_array(path: Path) -> np.ndarray:
    with np.load(path) as npz:
        key = npz.files[0]
        return np.asarray(npz[key])


def _average_loaded_arrays(files: List[Path], loader) -> np.ndarray:
    total = np.asarray(loader(files[0]), dtype=np.float32).copy()
    if total.ndim < 2:
        raise ValueError(f"Expected 2D or 3D array, got shape {total.shape}")
    for path in files[1:]:
        arr = np.asarray(loader(path), dtype=np.float32)
        if arr.shape != total.shape:
            raise ValueError(f"Shape mismatch: {files[0].name} {total.shape} vs {path.name} {arr.shape}")
        np.add(total, arr, out=total)
    total /= np.float32(len(files))
    return total


def _load_heatmap_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float32)


def _average_tabular_csvs(files: List[Path]) -> pd.DataFrame:
    base = pd.read_csv(files[0])
    if len(files) == 1:
        return base
    numeric_columns = list(base.select_dtypes(include=[np.number]).columns)
    numeric_total = base[numeric_columns].to_numpy(dtype=np.float32, copy=True) if numeric_columns else None
    for path in files[1:]:
        current = pd.read_csv(path)
        if current.shape != base.shape or list(current.columns) != list(base.columns):
            raise ValueError(f"Layout mismatch: {files[0].name} vs {path.name}")
        if numeric_total is not None:
            np.add(numeric_total, current[numeric_columns].to_numpy(dtype=np.float32), out=numeric_total)
    if numeric_total is not None:
        numeric_total /= np.float32(len(files))
        base.loc[:, numeric_columns] = numeric_total
    return base


def load_related_files(
    file_map: Dict[str, Dict[str, List[Path]]],
    mode: str = "sample",
    align_config: Optional[dict] = None,
) -> Dict[str, Dict[str, object]]:
    """Load located comparison files, averaging repetitions in 'average' mode."""
    mode = mode.lower()
    if mode not in {"sample", "average"}:
        raise ValueError(f"Unsupported mode: {mode!r}")

    loaded: Dict[str, Dict[str, object]] = {data_type: {} for data_type in file_map}
    align_config = align_config or {}

    for data_type, testpoint_map in file_map.items():
        for tp_str, files in testpoint_map.items():
            if not files:
                loaded[data_type][tp_str] = None if data_type == "dewe_csv" else {}
                continue

            if data_type == "dewe_csv":
                df = pd.read_csv(files[0], index_col=0)
                if align_config:
                    df = align_dewe_dataframe_to_soe(
                        df,
                        injection_current_col=align_config.get("injection_current_col",
                                                               "Main Injector - Current Profile"),
                        grad_threshold=align_config.get("grad_threshold", 5),
                        pre_samples=align_config.get("pre_samples", 50),
                        window_ms=align_config.get("window_ms", 10.0),
                    )
                loaded[data_type][tp_str] = df
                continue

            grouped = group_related_files(files)
            loaded_groups: Dict[str, object] = {}
            for group_key, group_files in grouped.items():
                first_path = group_files[0]
                if first_path.suffix.lower() == ".npz":
                    loaded_groups[group_key] = (
                        _average_loaded_arrays(group_files, _load_first_npz_array)
                        if mode == "average" and len(group_files) > 1
                        else _load_first_npz_array(first_path)
                    )
                elif "_heatmap" in first_path.stem.lower():
                    loaded_groups[group_key] = (
                        _average_loaded_arrays(group_files, _load_heatmap_csv)
                        if mode == "average" and len(group_files) > 1
                        else _load_heatmap_csv(first_path)
                    )
                else:
                    loaded_groups[group_key] = (
                        _average_tabular_csvs(group_files)
                        if mode == "average"
                        else pd.read_csv(first_path)
                    )
            loaded[data_type][tp_str] = loaded_groups

    return loaded


# =============================================================================
# Data Loading
# =============================================================================

def locate_comparison_files(
    config: dict,
    testpoints: List[int],
    mode: str = "sample",
) -> Dict[str, Dict[str, List[Path]]]:
    """Locate all files needed for comparison."""
    mode = mode.lower()
    if mode not in {"sample", "average"}:
        raise ValueError(f"Unsupported mode: {mode!r}")

    result = {
        "dewe_csv": {}, "mie_data": {}, "mie_video": {},
        "luminescence_data": {}, "luminescence_video": {},
        "schlieren_video": {},
    }

    repetition = 1 if mode == "sample" else None

    for tp in testpoints:
        tp_str = str(tp)

        dewe_dir = Path(config["directories"].get("dewe", ""))
        if dewe_dir.exists():
            dewe_data_dir = dewe_dir / "Processed_Results" / "Postprocessed_Data"
            result["dewe_csv"][tp_str] = find_files_for_testpoint(dewe_data_dir, tp_str, ".csv", 1)

        mie_dir = Path(config["directories"].get("mie", ""))
        if mie_dir.exists():
            mie_data_dir = mie_dir / "Processed_Results" / "Postprocessed_Data"
            mie_video_dir = mie_dir / "Processed_Results" / "Rotated_Videos"
            result["mie_data"][tp_str] = find_files_for_testpoint(mie_data_dir, tp_str, ".csv", repetition)
            result["mie_video"][tp_str] = find_files_for_testpoint(mie_video_dir, tp_str, ".npz", repetition)

        lum_dir = Path(config["directories"].get("luminescence", ""))
        if lum_dir.exists():
            lum_data_dir = lum_dir / "Processed_Results" / "Postprocessed_Data"
            lum_video_dir = lum_dir / "Processed_Results" / "Rotated_Videos"
            result["luminescence_data"][tp_str] = find_files_for_testpoint(lum_data_dir, tp_str, ".csv", repetition)
            result["luminescence_video"][tp_str] = find_files_for_testpoint(lum_video_dir, tp_str, ".npz", repetition)

        sch_dir = Path(config["directories"].get("schlieren", ""))
        if sch_dir.exists():
            sch_video_dir = sch_dir / "Processed_Results" / "Rotated_Videos"
            result["schlieren_video"][tp_str] = find_files_for_testpoint(sch_video_dir, tp_str, ".npz", repetition)

    return result


def validate_required_files(
    file_map: Dict[str, Dict[str, List[Path]]],
    testpoints: List[int],
    required_types: List[str] = None,
) -> bool:
    if required_types is None:
        required_types = ["dewe_csv"]
    missing = []
    for data_type in required_types:
        if data_type not in file_map:
            continue
        for tp in testpoints:
            if not file_map[data_type].get(str(tp)):
                missing.append(f"{data_type} for testpoint T{tp}")
    if missing:
        raise FileNotFoundError("Missing required files:\n  - " + "\n  - ".join(missing))
    return True


def print_located_files(file_map: Dict[str, Dict[str, List[Path]]]) -> None:
    print("\n" + "="*60)
    print("LOCATED FILES FOR COMPARISON")
    print("="*60)
    for data_type, tp_files in file_map.items():
        if not any(files for files in tp_files.values()):
            continue
        print(f"\n{data_type.upper().replace('_', ' ')}:")
        for tp, files in sorted(tp_files.items()):
            if files:
                for f in files:
                    print(f"  T{tp}: {f}")
            else:
                print(f"  T{tp}: (not found)")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Load processed data for testpoint comparison")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config or get_default_config_path()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    comparison_sets = config.get("comparison_sets", {})
    if not comparison_sets:
        print("Error: No comparison_sets defined in config")
        return 1

    mode = config.get("processing", {}).get("mode", "sample")
    print(f"Mode: {mode}")

    for set_name, testpoints in comparison_sets.items():
        print(f"\n{'='*60}")
        print(f"COMPARISON SET: {set_name}  testpoints={testpoints}")
        print("="*60)
        file_map = locate_comparison_files(config, testpoints, mode)
        print_located_files(file_map)
        try:
            validate_required_files(file_map, testpoints, required_types=["dewe_csv"])
            print("\n[OK] All required Dewesoft files found")
        except FileNotFoundError as e:
            print(f"\n[WARN] {e}")

    print("\n" + "="*60)
    print("[OK] File location complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
