"""
Load Comparison - Load processed data for testpoint comparison
==============================================================

Reads saved data from process_all.py and locates files for comparison.
In "sample" mode, finds the first repetition (rep 1) for each testpoint.

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

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OSCC_postprocessing.dewe.dewe import (
    align_dewe_dataframe_to_soe,
)


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_default_config_path() -> Path:
    """Return path to config.json in the same directory as this script."""
    return Path(__file__).parent / "config.json"


# =============================================================================
# File Naming Conventions
# =============================================================================

def parse_testpoint_from_filename(filename: str) -> Optional[str]:
    """Extract testpoint number from filename.
    
    Examples:
        "T2_0001.csv" -> "2"
        "T56_Mie Camera_1.csv" -> "56"
        "T2_NFL_Cam_5_experiement.csv" -> "2"
    """
    stem = Path(filename).stem
    # Find "T" followed by digits
    if stem.startswith("T"):
        # Extract digits after T until non-digit
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
    """Extract repetition number from filename.
    
    Examples:
        "T2_0001.csv" -> 1
        "T2_0002.csv" -> 2
        "T56_Mie Camera_1.csv" -> 1 (default if no _NNNN pattern)
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    
    # Look for 4-digit repetition number (e.g., "0001")
    for part in parts:
        if len(part) == 4 and part.isdigit():
            return int(part)
    
    # Default to 1 if no repetition found
    return 1


def find_files_for_testpoint(
    directory: Path, 
    testpoint: str, 
    suffix: str = ".csv",
    repetition: Optional[int] = None
) -> List[Path]:
    """Find all files matching a testpoint, optionally filtered by repetition.
    
    Parameters
    ----------
    directory : Path
        Directory to search
    testpoint : str
        Testpoint number (e.g., "2", "56")
    suffix : str
        File suffix to match (e.g., ".csv", ".npz")
    repetition : int, optional
        If specified, only return files with this repetition number
    
    Returns
    -------
    List[Path]
        Matching file paths, sorted by name
    """
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


# =============================================================================
# Data Loading
# =============================================================================

def locate_comparison_files(
    config: dict,
    testpoints: List[int],
    mode: str = "sample"
) -> Dict[str, Dict[str, List[Path]]]:
    """Locate all files needed for comparison.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with directory paths
    testpoints : list of int
        Testpoint numbers to compare (e.g., [2, 56])
    mode : str
        "sample" = first repetition only, "all" = all repetitions
    
    Returns
    -------
    dict
        Nested dict: {data_type: {testpoint: [file_paths]}}
    """
    result = {
        "dewe_csv": {},
        "mie_data": {},
        "mie_video": {},
        "luminescence_data": {},
        "luminescence_video": {},
        "schlieren_data": {},
        "schlieren_video": {},
    }
    
    repetition = 1 if mode == "sample" else None
    
    for tp in testpoints:
        tp_str = str(tp)
        
        # Dewesoft CSV files
        dewe_dir = Path(config["directories"].get("dewe", ""))
        if dewe_dir.exists():
            dewe_data_dir = dewe_dir / "Processed_Results" / "Postprocessed_Data"
            files = find_files_for_testpoint(dewe_data_dir, tp_str, ".csv", repetition)
            result["dewe_csv"][tp_str] = files
        
        # Mie files
        mie_dir = Path(config["directories"].get("mie", ""))
        if mie_dir.exists():
            mie_data_dir = mie_dir / "Processed_Results" / "Postprocessed_Data"
            mie_video_dir = mie_dir / "Processed_Results" / "Rotated_Videos"
            result["mie_data"][tp_str] = find_files_for_testpoint(mie_data_dir, tp_str, ".csv", repetition)
            result["mie_video"][tp_str] = find_files_for_testpoint(mie_video_dir, tp_str, ".npz", repetition)
        
        # Luminescence files
        lum_dir = Path(config["directories"].get("luminescence", ""))
        if lum_dir.exists():
            lum_data_dir = lum_dir / "Processed_Results" / "Postprocessed_Data"
            lum_video_dir = lum_dir / "Processed_Results" / "Rotated_Videos"
            result["luminescence_data"][tp_str] = find_files_for_testpoint(lum_data_dir, tp_str, ".csv", repetition)
            result["luminescence_video"][tp_str] = find_files_for_testpoint(lum_video_dir, tp_str, ".npz", repetition)
        
        # Schlieren files
        sch_dir = Path(config["directories"].get("schlieren", ""))
        if sch_dir.exists():
            sch_data_dir = sch_dir / "Processed_Results" / "Postprocessed_Data"
            sch_video_dir = sch_dir / "Processed_Results" / "Rotated_Videos"
            result["schlieren_data"][tp_str] = find_files_for_testpoint(sch_data_dir, tp_str, ".csv", repetition)
            result["schlieren_video"][tp_str] = find_files_for_testpoint(sch_video_dir, tp_str, ".npz", repetition)
    
    return result


def validate_required_files(
    file_map: Dict[str, Dict[str, List[Path]]],
    testpoints: List[int],
    required_types: List[str] = None
) -> bool:
    """Validate that all required files exist.
    
    Raises FileNotFoundError if any required file is missing.
    """
    if required_types is None:
        required_types = ["dewe_csv"]  # At minimum, need Dewesoft data
    
    missing = []
    for data_type in required_types:
        if data_type not in file_map:
            continue
        for tp in testpoints:
            tp_str = str(tp)
            files = file_map[data_type].get(tp_str, [])
            if not files:
                missing.append(f"{data_type} for testpoint T{tp}")
    
    if missing:
        raise FileNotFoundError(
            f"Missing required files:\n  - " + "\n  - ".join(missing)
        )
    
    return True


def print_located_files(file_map: Dict[str, Dict[str, List[Path]]]) -> None:
    """Print all located files to console."""
    print("\n" + "="*60)
    print("LOCATED FILES FOR COMPARISON")
    print("="*60)
    
    for data_type, tp_files in file_map.items():
        has_files = any(files for files in tp_files.values())
        if not has_files:
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
    parser = argparse.ArgumentParser(
        description="Load processed data for testpoint comparison"
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config JSON file (default: config.json)")
    args = parser.parse_args()

    config_path = args.config or get_default_config_path()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    # Get comparison sets from config
    comparison_sets = config.get("comparison_sets", {})
    if not comparison_sets:
        print("Error: No comparison_sets defined in config")
        return 1

    mode = config.get("processing", {}).get("mode", "sample")
    print(f"Mode: {mode}")

    # Process each comparison set
    for set_name, testpoints in comparison_sets.items():
        print(f"\n{'='*60}")
        print(f"COMPARISON SET: {set_name}")
        print(f"Testpoints: {testpoints}")
        print("="*60)

        # Locate all files
        file_map = locate_comparison_files(config, testpoints, mode)
        
        # Print located files
        print_located_files(file_map)
        
        # Validate required files exist
        try:
            validate_required_files(file_map, testpoints, required_types=["dewe_csv"])
            print("\n✓ All required Dewesoft files found")
        except FileNotFoundError as e:
            print(f"\n✗ Error: {e}")
            return 1

    print("\n" + "="*60)
    print("✓ File location complete!")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
