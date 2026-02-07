"""
Comparison Viewer - Visualize testpoint comparisons with plots and videos
=========================================================================

Creates a grid visualization comparing multiple testpoints:

Layout (2-case example):
    Column 1: Plots           | Column 2: T2 Videos    | Column 3: T56 Videos
    ─────────────────────────────────────────────────────────────────────────
    Row 1: P/T/HRR plot       | Mie video T2           | Mie video T56
    Row 2: Mie Area plot      | Schlieren video T2     | Schlieren video T56
    Row 3: Injection current  | Luminescence video T2  | Luminescence video T56
    Row 4: (reserved)         | Luminescence heatmap   | Luminescence heatmap

Usage:
    python comparison_viewer.py                    # Use default config.json
    python comparison_viewer.py --config my.json   # Use custom config
    python comparison_viewer.py --debug            # Show instead of save
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from OSCC_postprocessing.dewe.dewe import align_dewe_dataframe_to_soe
from OSCC_postprocessing.dewe.heat_release_calulation import hrr_calc

# Import the normalization function
from examples.slide_maker.matplotlib_video_displayer import normalize_video


# =============================================================================
# Configuration
# =============================================================================

# Default column name mappings (can be overridden in config.json)
DEFAULT_COLUMN_NAMES = {
    "chamber_pressure": ["Chamber pressure (BarA)", "Chamber pressure", "chamber_pressure"],
    "chamber_temperature": ["Temperature acc. Ideal gas law", "Chamber gas temperature", "chamber_temperature"],
    "heat_release": ["Heat Release", "HRR", "heat_release"],
    "injection_current": ["Main Injector - Current Profile", "Current Profile", "injection_current"],
    "mie_area": ["Area", "area", "Spray Area"],
}


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_default_config_path() -> Path:
    """Return path to config.json in the same directory as this script."""
    return Path(__file__).parent / "config.json"


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name from a list of candidates."""
    for candidate in candidates:
        # Exact match
        if candidate in df.columns:
            return candidate
        # Case-insensitive partial match
        for col in df.columns:
            if candidate.lower() in col.lower():
                return col
    return None


# =============================================================================
# File Location (from load_comparison.py)
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


def find_file_for_testpoint(
    directory: Path, 
    testpoint: str, 
    suffix: str = ".csv",
    repetition: int = 1,
    name_contains: Optional[str] = None
) -> Optional[Path]:
    """Find a single file matching testpoint and repetition.
    
    Parameters
    ----------
    directory : Path
        Directory to search in.
    testpoint : str
        Testpoint number (e.g., "2" for T2).
    suffix : str
        File extension to match.
    repetition : int
        Repetition number to match.
    name_contains : str, optional
        Additional substring the filename must contain (e.g., "_metrics", "_heatmap").
    """
    if not directory.exists():
        return None
    
    for f in directory.iterdir():
        if not f.is_file() or f.suffix.lower() != suffix.lower():
            continue
        # Check name_contains filter
        if name_contains and name_contains not in f.name:
            continue
        tp = parse_testpoint_from_filename(f.name)
        rep = parse_repetition_from_filename(f.name)
        if tp == testpoint and rep == repetition:
            return f
    return None


# =============================================================================
# Data Loading
# =============================================================================

class ComparisonData:
    """Container for all data needed for comparison visualization."""
    
    def __init__(self, config: dict, testpoints: List[int]):
        self.config = config
        self.testpoints = testpoints
        self.testpoint_strs = [str(tp) for tp in testpoints]
        self.num_cases = len(testpoints)
        
        # Data storage
        self.dewe_data: Dict[str, pd.DataFrame] = {}
        self.mie_data: Dict[str, pd.DataFrame] = {}
        self.mie_videos: Dict[str, np.ndarray] = {}
        self.schlieren_videos: Dict[str, np.ndarray] = {}
        self.luminescence_videos: Dict[str, np.ndarray] = {}
        self.luminescence_heatmaps: Dict[str, np.ndarray] = {}  # Pre-computed heatmaps
        
        # Column name mappings from config or defaults
        self.column_names = config.get("column_names", DEFAULT_COLUMN_NAMES)
        
        # Alignment config
        self.align_cfg = config.get("alignment", {})
        self.hrr_cfg = config.get("hrr_parameters", {})
    
    def load_all(self) -> None:
        """Load all data for all testpoints."""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        for tp_str in self.testpoint_strs:
            print(f"\nTestpoint T{tp_str}:")
            self._load_testpoint_data(tp_str)
    
    def _load_testpoint_data(self, tp_str: str) -> None:
        """Load all data for a single testpoint."""
        # Load Dewesoft CSV
        dewe_dir = Path(self.config["directories"].get("dewe", ""))
        if dewe_dir.exists():
            dewe_data_dir = dewe_dir / "Processed_Results" / "Postprocessed_Data"
            dewe_file = find_file_for_testpoint(dewe_data_dir, tp_str, ".csv", 1)
            if dewe_file:
                print(f"  Dewe CSV: {dewe_file}")
                df = pd.read_csv(dewe_file, index_col=0)
                df.index.name = "time_s"
                self.dewe_data[tp_str] = df
            else:
                raise FileNotFoundError(f"Dewe CSV not found for T{tp_str} in {dewe_data_dir}")
        
        # Load Mie data CSV (files are named *_metrics.csv)
        mie_dir = Path(self.config["directories"].get("mie", ""))
        if mie_dir.exists():
            mie_data_dir = mie_dir / "Processed_Results" / "Postprocessed_Data"
            mie_file = find_file_for_testpoint(mie_data_dir, tp_str, ".csv", 1, name_contains="_metrics")
            if mie_file:
                print(f"  Mie CSV: {mie_file}")
                self.mie_data[tp_str] = pd.read_csv(mie_file)
            else:
                print(f"  Mie CSV: NOT FOUND (looking for *_metrics.csv in {mie_data_dir})")
            
            # Load Mie video
            mie_video_dir = mie_dir / "Processed_Results" / "Rotated_Videos"
            mie_video_file = find_file_for_testpoint(mie_video_dir, tp_str, ".npz", 1)
            if mie_video_file:
                print(f"  Mie video: {mie_video_file}")
                npz = np.load(mie_video_file)
                # Get the first array in the npz file
                key = list(npz.keys())[0]
                self.mie_videos[tp_str] = npz[key]
        
        # Load Schlieren video
        sch_dir = Path(self.config["directories"].get("schlieren", ""))
        if sch_dir.exists():
            sch_video_dir = sch_dir / "Processed_Results" / "Rotated_Videos"
            sch_video_file = find_file_for_testpoint(sch_video_dir, tp_str, ".npz", 1)
            if sch_video_file:
                print(f"  Schlieren video: {sch_video_file}")
                npz = np.load(sch_video_file)
                key = list(npz.keys())[0]
                self.schlieren_videos[tp_str] = npz[key]
        
        # Load Luminescence video (optional)
        lum_dir = Path(self.config["directories"].get("luminescence", ""))
        if lum_dir.exists():
            lum_video_dir = lum_dir / "Processed_Results" / "Rotated_Videos"
            lum_video_file = find_file_for_testpoint(lum_video_dir, tp_str, ".npz", 1)
            if lum_video_file:
                print(f"  Luminescence video: {lum_video_file}")
                npz = np.load(lum_video_file)
                key = list(npz.keys())[0]
                self.luminescence_videos[tp_str] = npz[key]
            
            # Load pre-computed luminescence heatmap (if available)
            lum_data_dir = lum_dir / "Processed_Results" / "Postprocessed_Data"
            heatmap_file = find_file_for_testpoint(lum_data_dir, tp_str, ".npz", 1, name_contains="_heatmap")
            if heatmap_file:
                print(f"  Luminescence heatmap: {heatmap_file}")
                npz = np.load(heatmap_file)
                key = list(npz.keys())[0]
                self.luminescence_heatmaps[tp_str] = npz[key]


# =============================================================================
# Style Constants
# =============================================================================

# Colors and line styles for multi-testpoint comparison
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
LINE_STYLES = ['-', '--', ':', '-.', '-']


# =============================================================================
# Visualization Builder - Custom Layout with Shared Axes
# =============================================================================

class ComparisonViewer:
    """Custom viewer with decoupled column 1 (plots) and columns 2+ (videos)."""
    
    def __init__(self, data: ComparisonData, figsize: Tuple[float, float] = None):
        self.data = data
        self.num_cases = data.num_cases
        self.num_rows = 4  # P/T/HRR, Mie Area, Injection, Heatmap
        
        # Figure sizing
        if figsize is None:
            # Column 1 wider for plots, video columns tighter
            figsize = (4 + 3 * self.num_cases, 12)
        
        # Create figure with GridSpec
        self.fig = plt.figure(figsize=figsize, constrained_layout=False)
        
        # Main grid: 2 columns - plot column (wider) | video columns (tighter)
        # width_ratios: make plot column wider
        self.gs_main = GridSpec(
            1, 2, figure=self.fig, 
            width_ratios=[1.2, self.num_cases],
            wspace=0.15
        )
        
        # Sub-grid for column 1 (plots with shared x-axis)
        self.gs_plots = self.gs_main[0, 0].subgridspec(self.num_rows, 1, hspace=0.08)
        
        # Sub-grid for video columns (tighter spacing)
        self.gs_videos = self.gs_main[0, 1].subgridspec(
            self.num_rows, self.num_cases, 
            wspace=0.02, hspace=0.05
        )
        
        # Create axes
        self.plot_axes: List[Axes] = []
        self.video_axes: List[List[Axes]] = []  # [row][col]
        self.video_entries: List[Dict] = []  # For animation
        
        self._setup_axes()
        self._populate_plots()
        self._populate_videos()
    
    def _setup_axes(self) -> None:
        """Create all axes with proper sharing."""
        # Column 1: plots with shared x-axis
        for row in range(self.num_rows):
            if row == 0:
                ax = self.fig.add_subplot(self.gs_plots[row, 0])
            else:
                # Share x-axis with first plot
                ax = self.fig.add_subplot(self.gs_plots[row, 0], sharex=self.plot_axes[0])
            
            # Hide x-tick labels for all but bottom plot
            if row < self.num_rows - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            
            self.plot_axes.append(ax)
        
        # Columns 2+: videos
        for row in range(self.num_rows):
            row_axes = []
            for col in range(self.num_cases):
                ax = self.fig.add_subplot(self.gs_videos[row, col])
                ax.set_xticks([])
                ax.set_yticks([])
                row_axes.append(ax)
            self.video_axes.append(row_axes)
    
    def _populate_plots(self) -> None:
        """Populate column 1 with plots."""
        time_window = self.data.align_cfg.get("window_ms", 10.0)
        
        # Row 0: P/T/HRR
        self._plot_pressure_temp_hrr(self.plot_axes[0])
        
        # Row 1: Mie Area
        self._plot_mie_area(self.plot_axes[1])
        
        # Row 2: Injection Current
        self._plot_injection_current(self.plot_axes[2], time_window)
        
        # Row 3: Reserved (empty or additional plot)
        self._plot_placeholder(self.plot_axes[3], "(reserved)")
        
        # Set shared x-label only on bottom
        self.plot_axes[-1].set_xlabel("Time (ms)")
    
    def _plot_pressure_temp_hrr(self, ax: Axes) -> None:
        """Plot pressure, temperature, HRR on given axes."""
        ax2 = ax.twinx()
        
        for idx, tp_str in enumerate(self.data.testpoint_strs):
            if tp_str not in self.data.dewe_data:
                continue
            
            df = self.data.dewe_data[tp_str]
            style = LINE_STYLES[idx % len(LINE_STYLES)]
            
            # Align to SoE
            inj_col = find_column(df, self.data.column_names["injection_current"])
            df_aligned = align_dewe_dataframe_to_soe(
                df,
                injection_current_col=inj_col or "",
                grad_threshold=self.data.align_cfg.get("grad_threshold", 5),
                pre_samples=self.data.align_cfg.get("pre_samples", 50),
                window_ms=self.data.align_cfg.get("window_ms", 10.0),
            )
            
            x = df_aligned["time_ms"].to_numpy() if "time_ms" in df_aligned.columns else np.arange(len(df_aligned))
            
            # Plot pressure
            p_col = find_column(df_aligned, self.data.column_names["chamber_pressure"])
            if p_col:
                ax.plot(x, df_aligned[p_col], color='red', linestyle=style, 
                       linewidth=1.2, label=f"P (T{tp_str})", alpha=0.8)
            
            # Plot temperature
            t_col = find_column(df_aligned, self.data.column_names["chamber_temperature"])
            if t_col:
                ax.plot(x, df_aligned[t_col] / 10, color='green', linestyle=style,
                       linewidth=1.2, label=f"T/10 (T{tp_str})", alpha=0.8)
            
            # Calculate and plot HRR
            if p_col:
                time_s = x / 1000.0
                hrr_result = hrr_calc(
                    df_aligned[p_col].to_numpy(),
                    time=time_s,
                    V_m3=self.data.hrr_cfg.get("V_m3", 8.5e-3),
                    gamma=self.data.hrr_cfg.get("gamma", 1.35),
                    fc_p=self.data.hrr_cfg.get("fc_p", 1000.0),
                    fc_hrr=self.data.hrr_cfg.get("fc_hrr", 600.0),
                )
                hrr_kw = hrr_result["HRR_W"].to_numpy() / 1000.0
                ax2.plot(x, hrr_kw, color='blue', linestyle=style,
                        linewidth=1.2, label=f"HRR (T{tp_str})", alpha=0.8)
        
        ax.set_ylabel("P (bar) / T (K/10)", fontsize=9)
        ax2.set_ylabel("HRR (kJ/s)", color='blue', fontsize=9)
        ax.set_title("Chamber P/T/HRR", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7)
        ax2.legend(loc='upper right', fontsize=7)
    
    def _plot_mie_area(self, ax: Axes) -> None:
        """Plot Mie spray area."""
        has_data = False
        for idx, tp_str in enumerate(self.data.testpoint_strs):
            if tp_str not in self.data.mie_data:
                continue
            
            df = self.data.mie_data[tp_str]
            style = LINE_STYLES[idx % len(LINE_STYLES)]
            color = PLOT_COLORS[idx % len(PLOT_COLORS)]
            
            # Find area column
            area_col = find_column(df, self.data.column_names.get("mie_area", ["Area"]))
            if area_col:
                has_data = True
                x = np.arange(len(df))
                ax.plot(x, df[area_col], color=color, linestyle=style,
                       linewidth=1.2, label=f"T{tp_str}")
        
        if not has_data:
            ax.text(0.5, 0.5, "Mie Area\n(no data)", ha='center', va='center', 
                   fontsize=10, color='gray', transform=ax.transAxes)
        
        ax.set_ylabel("Area (px²)", fontsize=9)
        ax.set_title("Mie Spray Area", fontsize=10)
        ax.grid(True, alpha=0.3)
        if has_data:
            ax.legend(fontsize=8)
    
    def _plot_injection_current(self, ax: Axes, time_window_ms: float) -> None:
        """Plot injection current."""
        for idx, tp_str in enumerate(self.data.testpoint_strs):
            if tp_str not in self.data.dewe_data:
                continue
            
            df = self.data.dewe_data[tp_str]
            style = LINE_STYLES[idx % len(LINE_STYLES)]
            color = PLOT_COLORS[idx % len(PLOT_COLORS)]
            
            inj_col = find_column(df, self.data.column_names["injection_current"])
            df_aligned = align_dewe_dataframe_to_soe(
                df,
                injection_current_col=inj_col or "",
                grad_threshold=self.data.align_cfg.get("grad_threshold", 5),
                pre_samples=self.data.align_cfg.get("pre_samples", 50),
                window_ms=time_window_ms,
            )
            
            x = df_aligned["time_ms"].to_numpy() if "time_ms" in df_aligned.columns else np.arange(len(df_aligned))
            
            if inj_col and inj_col in df_aligned.columns:
                ax.plot(x, df_aligned[inj_col], color=color, linestyle=style,
                       linewidth=1.2, label=f"T{tp_str}")
        
        ax.set_ylabel("Current (A)", fontsize=9)
        ax.set_title("Injection Current", fontsize=10)
        ax.set_xlim(0, time_window_ms)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_placeholder(self, ax: Axes, text: str) -> None:
        """Plot a placeholder."""
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10, color='gray',
               transform=ax.transAxes)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _populate_videos(self) -> None:
        """Populate video columns."""
        # Row 0: Mie videos
        for col, tp_str in enumerate(self.data.testpoint_strs):
            ax = self.video_axes[0][col]
            self._setup_video_cell(ax, self.data.mie_videos.get(tp_str), f"Mie T{tp_str}")
        
        # Row 1: Schlieren videos
        for col, tp_str in enumerate(self.data.testpoint_strs):
            ax = self.video_axes[1][col]
            self._setup_video_cell(ax, self.data.schlieren_videos.get(tp_str), f"Schlieren T{tp_str}")
        
        # Row 2: Luminescence videos
        for col, tp_str in enumerate(self.data.testpoint_strs):
            ax = self.video_axes[2][col]
            self._setup_video_cell(ax, self.data.luminescence_videos.get(tp_str), f"Lum. T{tp_str}")
        
        # Row 3: Luminescence heatmaps (static)
        for col, tp_str in enumerate(self.data.testpoint_strs):
            ax = self.video_axes[3][col]
            self._setup_heatmap_cell(ax, tp_str)
    
    def _setup_video_cell(self, ax: Axes, video: Optional[np.ndarray], title: str) -> None:
        """Setup a video cell (animated or placeholder)."""
        ax.set_title(title, fontsize=9)
        
        if video is None:
            ax.text(0.5, 0.5, "(not available)", ha='center', va='center', 
                   fontsize=8, color='gray', transform=ax.transAxes)
            return
        
        # Normalize video for display
        video = normalize_video(video, quantize_bits=8)
        
        # Create imshow and store for animation
        im = ax.imshow(video[0], cmap='gray', vmin=0, vmax=1, animated=True)
        self.video_entries.append({"data": video, "im": im})
    
    def _setup_heatmap_cell(self, ax: Axes, tp_str: str) -> None:
        """Setup a heatmap cell (static)."""
        heatmap = None
        
        # First try pre-computed heatmap
        if tp_str in self.data.luminescence_heatmaps:
            heatmap = self.data.luminescence_heatmaps[tp_str].astype(np.float32)
        # Fallback: compute from video
        elif tp_str in self.data.luminescence_videos:
            video = self.data.luminescence_videos[tp_str]
            heatmap = np.sum(video.astype(np.float32), axis=1)
        
        if heatmap is not None:
            # Normalize
            h_min, h_max = heatmap.min(), heatmap.max()
            if h_max - h_min > 1e-8:
                heatmap = (heatmap - h_min) / (h_max - h_min)
            ax.imshow(heatmap, origin='lower', cmap='hot', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f"Heatmap T{tp_str}", fontsize=9)
        else:
            ax.text(0.5, 0.5, "Heatmap\n(N/A)", ha='center', va='center',
                   fontsize=8, color='gray', transform=ax.transAxes)
            ax.set_title(f"Heatmap T{tp_str}", fontsize=9)
    
    def _update_frame(self, frame_idx: int):
        """Animation update function."""
        artists = []
        for entry in self.video_entries:
            video = entry["data"]
            idx = frame_idx % video.shape[0]
            entry["im"].set_array(video[idx])
            artists.append(entry["im"])
        return artists
    
    def _get_max_frames(self) -> int:
        """Get maximum number of frames across all videos."""
        max_frames = 1
        for entry in self.video_entries:
            max_frames = max(max_frames, entry["data"].shape[0])
        return max_frames
    
    def show(self) -> None:
        """Show interactive animation."""
        if self.video_entries:
            self.anim = FuncAnimation(
                self.fig, self._update_frame,
                frames=self._get_max_frames(),
                interval=50, blit=True
            )
        plt.tight_layout()
        plt.show()
    
    def save_video(self, output_path: str, fps: int = 20) -> None:
        """Save animation to video file."""
        from matplotlib.animation import FFMpegWriter
        
        if not self.video_entries:
            print("No video entries to animate. Saving static figure.")
            self.fig.savefig(output_path.replace('.mp4', '.png'), dpi=150)
            return
        
        self.anim = FuncAnimation(
            self.fig, self._update_frame,
            frames=self._get_max_frames(),
            interval=1000 // fps, blit=True
        )
        
        writer = FFMpegWriter(fps=fps, metadata={'title': 'Comparison'})
        self.anim.save(output_path, writer=writer)
        print(f"Video saved: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize testpoint comparisons with plots and videos"
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config JSON file (default: config.json)")
    parser.add_argument("--debug", action="store_true",
                        help="Show visualization instead of saving video")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output video file path")
    parser.add_argument("--fps", type=int, default=20,
                        help="Output video FPS (default: 20)")
    args = parser.parse_args()

    config_path = args.config or get_default_config_path()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    # Get first comparison set
    comparison_sets = config.get("comparison_sets", {})
    if not comparison_sets:
        print("Error: No comparison_sets defined in config")
        return 1

    set_name, testpoints = next(iter(comparison_sets.items()))
    print(f"Comparison set: {set_name}")
    print(f"Testpoints: {testpoints}")

    # Load all data
    data = ComparisonData(config, testpoints)
    data.load_all()

    # Build visualization
    print("\n" + "="*60)
    print("BUILDING VISUALIZATION")
    print("="*60)
    print(f"Layout: 4 rows x {1 + len(testpoints)} cols (plots | videos)")
    
    viewer = ComparisonViewer(data)

    # Show or save
    if args.debug:
        print("\nShowing visualization (debug mode)...")
        viewer.show()
    else:
        output_path = args.output or Path(f"comparison_{set_name}.mp4")
        if not output_path.is_absolute():
            output_path = Path(__file__).parent / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving video to: {output_path}")
        viewer.save_video(str(output_path), fps=args.fps)

    print("\n✓ Visualization complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
