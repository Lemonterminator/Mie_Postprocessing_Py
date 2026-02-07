"""
Slide Maker - Batch Processing for Combustion Chamber Data
===========================================================

Processes Schlieren, Mie, Luminescence, and Dewesoft data for comparison slides.
Configuration is loaded from config.json in the same directory.

Usage:
    python slide_maker.py                    # Use default config.json
    python slide_maker.py --config my.json   # Use custom config file
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, cast

# Configure matplotlib backend before import
os.environ.setdefault("MPLBACKEND", "TkAgg")

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception as exc:
    HAS_MPL = False
    plt = None
    print(f"matplotlib unavailable; skipping plots: {exc}")

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OSCC_postprocessing.dewe.dewe import (
    load_dataframe,
    align_dewe_dataframe_to_soe,
    plot_dataframe,
    plot_reactive,
)
from OSCC_postprocessing.cine.functions_videos import load_cine_video

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


def parse_dewe_filename(name: str) -> Tuple[str, int]:
    """Extract testpoint and repetition from Dewe filename (e.g., 'T2_0001.csv')."""
    stem = Path(name).stem
    parts = stem.split("_")
    testpoint = parts[0].lstrip("T")
    repetition = int(parts[1]) if len(parts) > 1 else 1
    return testpoint, repetition


def setup_output_dirs(base_dir: Path) -> Tuple[Path, Path, Path]:
    """Create and return output directories."""
    results_dir = base_dir / "Processed_Results"
    data_dir = results_dir / "Postprocessed_Data"
    plots_dir = results_dir / "Plots"
    for d in [results_dir, data_dir, plots_dir]:
        d.mkdir(exist_ok=True)
    return results_dir, data_dir, plots_dir


# =============================================================================
# Processing Functions
# =============================================================================

def load_video_metadata(directory: Path) -> dict:
    """Load calibration metadata from JSON file (config.json in video directory).
    
    Returns
    -------
    dict with keys: plumes, offset, centre, inner_radius, outer_radius
    """
    # Look for config.json in the video directory
    config_file = directory / "config.json"
    if not config_file.exists():
        # Try any JSON file
        json_files = [f for f in directory.iterdir() 
                      if f.suffix.lower() == ".json" and f.name != "config.json"]
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


def process_video_data(config: dict, video_type: str) -> None:
    """Process video data (Schlieren, Mie, or Luminescence).
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    video_type : str
        One of "schlieren", "mie", or "luminescence"
    """
    dir_key = video_type.lower()
    video_dir = Path(config["directories"].get(dir_key, ""))
    
    if str(video_dir) == "" or not video_dir.exists():
        print(f"{video_type.capitalize()} directory not specified or doesn't exist, skipping...")
        return

    print(f"\n{'='*60}\nProcessing {video_type.capitalize()} videos: {video_dir}\n{'='*60}")

    # Setup output directories
    results_dir = video_dir / "Processed_Results"
    rotated_dir = results_dir / "Rotated_Videos"
    data_dir = results_dir / "Postprocessed_Data"
    for d in [results_dir, rotated_dir, data_dir]:
        d.mkdir(exist_ok=True)

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
    save_intermediate = config.get("processing", {}).get("save_intermediate_results", False)
    saved_video_fps = config.get("processing", {}).get("saved_video_fps", 20)
    video_bits = config.get("processing", {}).get("video_bits", 12)
    brightness_levels = 2.0 ** video_bits

    # Process each cine file
    cine_files = list(iter_files(video_dir, ".cine"))
    if not cine_files:
        print(f"  No .cine files found in {video_dir}")
        return

    for cine_file in cine_files:
        print(f"  Processing: {cine_file.name}")
        try:
            # Load and normalize video
            video = load_cine_video(str(cine_file))
            video = video.astype(np.float32) / brightness_levels
            
            file_stem = cine_file.stem
            
            if video_type.lower() == "mie":
                # Use mie_single_hole_pipeline
                mie_single_hole_pipeline(
                    video=video,
                    file_name=file_stem,
                    centre=centre,
                    rotation_offset=offset,
                    inner_radius=inner_radius,
                    outer_radius=outer_radius,
                    video_out_dir=rotated_dir,
                    data_out_dir=data_dir,
                    save_video_strip=save_intermediate,
                    preview=False,
                )
                
            elif video_type.lower() == "luminescence":
                # Use luminescence_pipeline
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
                    preview=False,
                )
                
            elif video_type.lower() == "schlieren":
                # Use singlehole_pipeline for Schlieren
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
            
            print(f"    Done: {cine_file.name}")
        except Exception as e:
            import traceback
            print(f"    Error processing {cine_file.name}: {e}")
            traceback.print_exc()


def process_dewe_data(config: dict) -> None:
    """Process Dewesoft data and generate comparison plots."""
    dewe_dir = Path(config["directories"]["dewe"])
    if not dewe_dir.exists() or str(dewe_dir) == "":
        print("Dewe directory not specified or doesn't exist, skipping...")
        return

    print(f"\n{'='*60}\nProcessing Dewesoft data: {dewe_dir}\n{'='*60}")

    _, data_dir, plots_dir = setup_output_dirs(dewe_dir)
    align_cfg = config.get("alignment", {})
    hrr_cfg = config.get("hrr_parameters", {})
    mode = config.get("processing", {}).get("mode", "sample")

    # Export DXD -> CSV (skip if already exists)
    for dxd_file in iter_files(dewe_dir, ".dxd"):
        csv_out = data_dir / f"{dxd_file.stem}.csv"
        if not csv_out.exists():
            print(f"Converting: {dxd_file.name} -> CSV")
            df = load_dataframe(dxd_file)
            df.to_csv(csv_out)

    # Get all CSV files
    csv_files = list(iter_files(data_dir, ".csv"))
    if not csv_files:
        print("No CSV files found in data directory")
        return

    # Parse filenames
    file_info = {f.name: parse_dewe_filename(f.name) for f in csv_files}

    # Process each comparison set
    for set_name, testpoints in config.get("comparison_sets", {}).items():
        print(f"\nProcessing comparison set: {set_name} ({testpoints})")

        # Select files based on mode
        testpoint_strs = [str(tp) for tp in testpoints]
        if mode == "sample":
            selected = [name for name, (tp, rep) in file_info.items()
                        if tp in testpoint_strs and rep == 1]
        else:  # "average all"
            selected = [name for name, (tp, _) in file_info.items()
                        if tp in testpoint_strs]

        if not selected:
            print(f"  No files found for testpoints: {testpoints}")
            continue

        # Load and align dataframes
        aligned_dfs = []
        labels = []
        for name in sorted(selected):
            df = load_dataframe(data_dir / name)
            df.index = df.index - df.index[0]  # Reset time to 0

            df_aligned = align_dewe_dataframe_to_soe(
                df,
                injection_current_col=align_cfg.get("injection_current_col", "Main Injector - Current Profile"),
                grad_threshold=align_cfg.get("grad_threshold", 5),
                pre_samples=align_cfg.get("pre_samples", 50),
                window_ms=align_cfg.get("window_ms", 10.0),
            )
            aligned_dfs.append(df_aligned)
            labels.append(Path(name).stem)
            print(f"  Loaded: {name}")

        # Generate plots
        if HAS_MPL and aligned_dfs:
            # Reactive plot with dual Y-axes
            try:
                fig, ax = plot_reactive(
                    aligned_dfs,
                    testpoint_labels=labels,
                    V_m3=hrr_cfg.get("V_m3", 8.5e-3),
                    gamma=hrr_cfg.get("gamma", 1.35),
                    fc_p=hrr_cfg.get("fc_p", 1000.0),
                    fc_hrr=hrr_cfg.get("fc_hrr", 600.0),
                    title=f"Comparison: {set_name}",
                    figsize=(14, 7),
                )
                fig.savefig(plots_dir / f"{set_name}_reactive.png", dpi=200)
                plt.close(fig)
                print(f"  Saved: {set_name}_reactive.png")
            except Exception as e:
                print(f"  Error creating reactive plot: {e}")

            # Individual overlay plots
            fig_main, ax_main = None, None
            for df, label in zip(aligned_dfs, labels):
                df_plot = df.set_index("time_ms") if "time_ms" in df.columns else df
                fig_main, ax_main = cast(
                    Tuple["Figure", "Axes"],
                    plot_dataframe(
                        df_plot,
                        title=f"Comparison: {set_name}",
                        criteria=["Chamber pressure", "Temperature", "Heat Release"],
                        ax=ax_main,
                        label_prefix=f"{label} | ",
                        return_fig=True,
                    ),
                )
            if fig_main:
                fig_main.savefig(plots_dir / f"{set_name}_overlay.png", dpi=200)
                plt.close(fig_main)
                print(f"  Saved: {set_name}_overlay.png")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Slide Maker - Combustion Data Processing")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config JSON file (default: config.json in script directory)")
    args = parser.parse_args()

    config_path = args.config or get_default_config_path()
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Creating default config.json...")
        # Create minimal config
        default_config = {
            "directories": {"schlieren": "", "mie": "", "luminescence": "", "dewe": ""},
            "comparison_sets": {"example": [1, 2]},
            "processing": {"mode": "sample", "save_intermediate_results": False},
            "hrr_parameters": {"V_m3": 8.5e-3, "gamma": 1.35, "fc_p": 1000.0, "fc_hrr": 600.0},
            "alignment": {"injection_current_col": "Main Injector - Current Profile",
                          "grad_threshold": 5, "pre_samples": 50, "window_ms": 10.0},
        }
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        print(f"Edit {config_path} and run again.")
        return

    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    # Process video data (Schlieren, Mie, Luminescence)
    for video_type in ["schlieren", "mie", "luminescence"]:
        process_video_data(config, video_type)

    # Process Dewesoft data
    process_dewe_data(config)

    print("\n" + "="*60)
    print("Processing complete!")


if __name__ == "__main__":
    main()
