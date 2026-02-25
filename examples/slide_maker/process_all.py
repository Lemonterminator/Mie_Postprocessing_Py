"""
Process All - Batch Processing for .cine and .dxd Files
========================================================

Processes all .cine (Mie, Luminescence, Schlieren) and .dxd (Dewesoft) files
in the configured directories and saves results.

Output Structure:
    {video_dir}/Processed_Results/
        ├── Rotated_Videos/     # Rotated and filtered video strips (.avi, .npz)
        └── Postprocessed_Data/ # Extracted metrics and data (.csv, .npz)

    {dewe_dir}/Processed_Results/
        └── Postprocessed_Data/ # Converted CSV files from .dxd

Usage:
    python process_all.py                    # Use default config.json
    python process_all.py --config my.json   # Use custom config file
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OSCC_postprocessing.cine.functions_videos import load_cine_video
from OSCC_postprocessing.dewe.dewe import load_dataframe

# Video processing pipelines
from mie_single_hole import mie_single_hole_pipeline
from luminesence import luminescence_pipeline
from examples.archieve.singlehole_pipeline import singlehole_pipeline


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
# Utility Functions
# =============================================================================

def iter_files(directory: Path, suffix: str = None) -> Iterable[Path]:
    """Iterate over files in directory, optionally filtered by suffix."""
    if not directory.exists():
        return []
    files = sorted(p for p in directory.iterdir() if p.is_file())
    if suffix:
        files = [f for f in files if f.suffix.lower() == suffix.lower()]
    return files


def load_video_metadata(directory: Path) -> dict:
    """Load calibration metadata from JSON file (config.json in video directory)."""
    config_file = directory / "config.json"
    if not config_file.exists():
        json_files = [f for f in directory.iterdir() 
                      if f.suffix.lower() == ".json"]
        if json_files:
            config_file = json_files[0]
        else:
            raise FileNotFoundError(
                f"No config.json found in {directory}.\n"
                "Run GUI.py for manual calibration first."
            )

    with config_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        return {
            "plumes": int(data.get("plumes", 1)),
            "offset": float(data.get("offset", 0)),
            "centre": (float(data.get("centre_x", 0)), float(data.get("centre_y", 0))),
            "inner_radius": float(data.get("inner_radius", 50)),
            "outer_radius": float(data.get("outer_radius", 200)),
        }


def setup_output_dirs(base_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """Create and return output directories (results_dir, rotated_dir, data_dir, plots_dir)."""
    results_dir = base_dir / "Processed_Results"
    rotated_dir = results_dir / "Rotated_Videos"
    data_dir = results_dir / "Postprocessed_Data"
    plots_dir = results_dir / "Plots"
    for d in [results_dir, rotated_dir, data_dir, plots_dir]:
        d.mkdir(exist_ok=True)
    return results_dir, rotated_dir, data_dir, plots_dir


# =============================================================================
# Processing Functions
# =============================================================================

def process_video_files(config: dict, video_type: str) -> None:
    """Process all .cine files for a given video type (mie, luminescence, schlieren)."""
    dir_key = video_type.lower()
    video_dir = Path(config["directories"].get(dir_key, ""))
    
    if str(video_dir) == "" or not video_dir.exists():
        print(f"{video_type.capitalize()} directory not specified or doesn't exist, skipping...")
        return

    print(f"\n{'='*60}\nProcessing {video_type.capitalize()} videos: {video_dir}\n{'='*60}")

    # Setup output directories
    _, rotated_dir, data_dir, plots_dir = setup_output_dirs(video_dir)

    # Load metadata
    try:
        metadata = load_video_metadata(video_dir)
        centre = metadata["centre"]
        offset = metadata["offset"]
        inner_radius = metadata["inner_radius"]
        outer_radius = metadata["outer_radius"]
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return

    # Get processing options
    proc_cfg = config.get("processing", {})
    save_intermediate = proc_cfg.get("save_processed_video_strips", False)
    saved_video_fps = proc_cfg.get("saved_video_fps", 20)
    video_bits = proc_cfg.get("video_bits", 12)
    brightness_levels = 2.0 ** video_bits
    frame_limit = proc_cfg.get("frame_limit", None)


    # Process each cine file
    cine_files = list(iter_files(video_dir, ".cine"))
    if not cine_files:
        print(f"  No .cine files found in {video_dir}")
        return

    print(f"  Found {len(cine_files)} .cine files")

    for i, cine_file in enumerate(cine_files, 1):
        print(f"\n  [{i}/{len(cine_files)}] Processing: {cine_file.name}")
        try:
            # Load and normalize video
            video = load_cine_video(str(cine_file), frame_limit=frame_limit)
            video = video.astype(np.float32) / brightness_levels
            file_stem = cine_file.stem
            
            if video_type.lower() == "mie":
                mie_single_hole_pipeline(
                    video=video,
                    file_name=file_stem,
                    centre=centre,
                    rotation_offset=offset,
                    inner_radius=inner_radius,
                    outer_radius=outer_radius,
                    video_out_dir=rotated_dir,
                    data_out_dir=data_dir,
                    save_video_strip=save_intermediate,  # Always save video for later use
                    save_mode="filtered",
                    preview=False,
                )
                
            elif video_type.lower() == "luminescence":
                luminescence_pipeline(
                    video=video,
                    file_name=file_stem,
                    centre=centre,
                    rotation_offset=offset,
                    inner_radius=inner_radius,
                    outer_radius=outer_radius,
                    video_out_dir=rotated_dir,
                    data_out_dir=data_dir,
                    save_video_strip=save_intermediate,
                    save_mode="filtered",
                    preview=False,
                )
                
            elif video_type.lower() == "schlieren":
                singlehole_pipeline(
                    "Schlieren",
                    video,
                    offset,
                    centre,
                    cine_file.name,
                    rotated_dir,
                    data_dir,
                    save_intermediate_results=save_intermediate,
                    saved_video_FPS=saved_video_fps,
                )
            
            print(f"    ✓ Done: {cine_file.name}")
        except Exception as e:
            import traceback
            print(f"    ✗ Error processing {cine_file.name}: {e}")
            traceback.print_exc()


def process_dewe_files(config: dict) -> None:
    """Process all .dxd files and export to CSV."""
    dewe_dir = Path(config["directories"].get("dewe", ""))
    
    if str(dewe_dir) == "" or not dewe_dir.exists():
        print("Dewe directory not specified or doesn't exist, skipping...")
        return

    print(f"\n{'='*60}\nProcessing Dewesoft data: {dewe_dir}\n{'='*60}")

    # Setup output directories
    results_dir = dewe_dir / "Processed_Results"
    data_dir = results_dir / "Postprocessed_Data"
    results_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    # Find all .dxd files
    dxd_files = list(iter_files(dewe_dir, ".dxd"))
    if not dxd_files:
        print(f"  No .dxd files found in {dewe_dir}")
        return

    print(f"  Found {len(dxd_files)} .dxd files")

    for i, dxd_file in enumerate(dxd_files, 1):
        csv_out = data_dir / f"{dxd_file.stem}.csv"
        if csv_out.exists():
            print(f"  [{i}/{len(dxd_files)}] Already exists: {csv_out.name}")
            continue
            
        print(f"  [{i}/{len(dxd_files)}] Converting: {dxd_file.name} -> {csv_out.name}")
        try:
            df = load_dataframe(dxd_file)
            df.to_csv(csv_out)
            print(f"    ✓ Saved: {csv_out.name}")
        except Exception as e:
            print(f"    ✗ Error: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process all .cine and .dxd files in configured directories"
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config JSON file (default: config.json)")
    args = parser.parse_args()

    config_path = args.config or get_default_config_path()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Create a config.json with directory paths first.")
        return 1

    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    # Process all video types
    for video_type in ["mie", "luminescence", "schlieren"]:
        process_video_files(config, video_type)

    # Process Dewesoft data
    process_dewe_files(config)

    print("\n" + "="*60)
    print("✓ All processing complete!")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
