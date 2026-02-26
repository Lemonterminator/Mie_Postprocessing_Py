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
import shutil
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

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
    
    def __init__(self, data: ComparisonData, figsize: Tuple[float, float] = None, title: Optional[str] = None):
        self.data = data
        self.num_cases = data.num_cases
        self.num_plot_rows = 4  # Keep plot column fixed and decoupled
        self.video_row_kinds = ["mie", "schlieren", "luminescence", "heatmap"]
        self.active_video_rows = [k for k in self.video_row_kinds if self._is_row_available(k)]
        self.num_video_rows = len(self.active_video_rows)
        if self.num_video_rows == 0:
            # Keep one placeholder row to avoid creating an empty GridSpec.
            self.active_video_rows = ["none"]
            self.num_video_rows = 1
        
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
        self.gs_plots = self.gs_main[0, 0].subgridspec(self.num_plot_rows, 1, hspace=0.08)
        
        # Sub-grid for video columns (tighter spacing)
        self.gs_videos = self.gs_main[0, 1].subgridspec(
            self.num_video_rows, self.num_cases,
            wspace=0.02, hspace=0.05
        )
        
        # Create axes
        self.plot_axes: List[Axes] = []
        self.video_axes: List[List[Axes]] = []  # [row][col]
        self.video_entries: List[Dict] = []  # For animation
        self.video_scale_cache: Dict[int, Tuple[float, float]] = {}
        
        self._setup_axes()
        self._populate_plots()
        self._populate_videos()
        if title:
            self.fig.suptitle(title, fontsize=14, y=0.995)

    def _is_row_available(self, video_kind: str) -> bool:
        """Return True if this video row has at least one available case."""
        if video_kind == "mie":
            return any(self.data.mie_videos.get(tp) is not None for tp in self.data.testpoint_strs)
        if video_kind == "schlieren":
            return any(self.data.schlieren_videos.get(tp) is not None for tp in self.data.testpoint_strs)
        if video_kind == "luminescence":
            return any(self.data.luminescence_videos.get(tp) is not None for tp in self.data.testpoint_strs)
        if video_kind == "heatmap":
            return any(
                (self.data.luminescence_heatmaps.get(tp) is not None)
                or (self.data.luminescence_videos.get(tp) is not None)
                for tp in self.data.testpoint_strs
            )
        return False
    
    def _setup_axes(self) -> None:
        """Create all axes with proper sharing."""
        # Column 1: plots with shared x-axis
        for row in range(self.num_plot_rows):
            if row == 0:
                ax = self.fig.add_subplot(self.gs_plots[row, 0])
            else:
                # Share x-axis with first plot
                ax = self.fig.add_subplot(self.gs_plots[row, 0], sharex=self.plot_axes[0])
            
            # Hide x-tick labels for all but bottom plot
            if row < self.num_plot_rows - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            
            self.plot_axes.append(ax)
        
        # Columns 2+: videos
        for row in range(self.num_video_rows):
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

        # Keep left-most plot column fixed and decoupled from video row availability.
        self._plot_pressure_temp_hrr(self.plot_axes[0])
        self._plot_mie_area(self.plot_axes[1])
        self._plot_injection_current(self.plot_axes[2], time_window)
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
        for row_idx, video_kind in enumerate(self.active_video_rows):
            for col, tp_str in enumerate(self.data.testpoint_strs):
                ax = self.video_axes[row_idx][col]
                if video_kind == "mie":
                    self._setup_video_cell(ax, self.data.mie_videos.get(tp_str), f"Mie T{tp_str}")
                elif video_kind == "schlieren":
                    self._setup_video_cell(ax, self.data.schlieren_videos.get(tp_str), f"Schlieren T{tp_str}")
                elif video_kind == "luminescence":
                    self._setup_video_cell(ax, self.data.luminescence_videos.get(tp_str), f"Lum. T{tp_str}")
                elif video_kind == "heatmap":
                    self._setup_heatmap_cell(ax, tp_str)
                else:
                    ax.text(0.5, 0.5, "(not available)", ha='center', va='center',
                            fontsize=8, color='gray', transform=ax.transAxes)
    
    def _setup_video_cell(self, ax: Axes, video: Optional[np.ndarray], title: str) -> None:
        """Setup a video cell (animated or placeholder)."""
        ax.set_title(title, fontsize=9)
        
        if video is None:
            ax.text(0.5, 0.5, "(not available)", ha='center', va='center', 
                   fontsize=8, color='gray', transform=ax.transAxes)
            return
        
        # Create imshow and store raw video with global 3D min/max for consistent scaling.
        mn, mx = self._get_video_global_minmax(video, use_cuda=False)
        first_frame = self._normalize_frame_with_bounds(video[0], mn, mx, use_cuda=False)
        im = ax.imshow(first_frame, cmap='gray', vmin=0, vmax=1, animated=True)
        self.video_entries.append({"data": video, "im": im, "mn": mn, "mx": mx})
    
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
            entry["im"].set_array(
                self._normalize_frame_with_bounds(video[idx], entry["mn"], entry["mx"], use_cuda=False)
            )
            artists.append(entry["im"])
        return artists

    def _is_cuda_usable(self) -> bool:
        if cp is None:
            return False
        try:
            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False

    def _get_video_global_minmax(self, video: np.ndarray, use_cuda: bool = False) -> Tuple[float, float]:
        """Compute/cached global min/max over full 3D video array."""
        key = id(video)
        cached = self.video_scale_cache.get(key)
        if cached is not None:
            return cached

        arr = video
        if use_cuda and self._is_cuda_usable():
            try:
                gpu = cp.asarray(arr, dtype=cp.float32)
                mn = float(cp.min(gpu).item())
                mx = float(cp.max(gpu).item())
                self.video_scale_cache[key] = (mn, mx)
                return mn, mx
            except Exception:
                pass

        np_arr = np.asarray(arr, dtype=np.float32)
        mn, mx = float(np_arr.min()), float(np_arr.max())
        self.video_scale_cache[key] = (mn, mx)
        return mn, mx

    def _normalize_frame_with_bounds(
        self, frame: np.ndarray, mn: float, mx: float, use_cuda: bool = False
    ) -> np.ndarray:
        """Normalize one frame using fixed global bounds (min-max scale)."""
        if not (mx > mn):
            return np.zeros_like(np.asarray(frame, dtype=np.float32), dtype=np.float32)

        arr = frame
        denom = mx - mn
        if use_cuda and self._is_cuda_usable():
            try:
                gpu = cp.asarray(arr, dtype=cp.float32)
                gpu = (gpu - mn) / denom
                return cp.asnumpy(gpu).astype(np.float32, copy=False)
            except Exception:
                pass

        np_arr = np.asarray(arr, dtype=np.float32)
        out = (np_arr - mn) / denom
        return out.astype(np.float32, copy=False)

    def _ffmpeg_has_encoder(self, ffmpeg_bin: str, encoder_name: str) -> bool:
        try:
            proc = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-encoders"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            return encoder_name in (proc.stdout or "")
        except Exception:
            return False

    def _prepare_static_background(self) -> None:
        """Draw once and cache static background for blit-based frame rendering."""
        if not isinstance(self.fig.canvas, FigureCanvasAgg):
            self.fig.set_canvas(FigureCanvasAgg(self.fig))
        self.fig.canvas.draw()
        self._cached_bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _render_frame_rgb(self, frame_idx: int, use_cuda: bool = False) -> np.ndarray:
        """Render a single frame as RGB uint8 using cached background + dynamic artists."""
        if not hasattr(self, "_cached_bg"):
            self._prepare_static_background()
        self.fig.canvas.restore_region(self._cached_bg)
        for entry in self.video_entries:
            video = entry["data"]
            idx = frame_idx % video.shape[0]
            im = entry["im"]
            im.set_array(
                self._normalize_frame_with_bounds(video[idx], entry["mn"], entry["mx"], use_cuda=use_cuda)
            )
            im.axes.draw_artist(im)
        self.fig.canvas.blit(self.fig.bbox)
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return np.ascontiguousarray(rgba[:, :, :3])
    
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
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
    
    def save_video(
        self,
        output_path: str,
        fps: int = 20,
        use_cuda: bool = True,
        prefer_nvenc: bool = True,
        fallback_cpu: bool = True,
        encode_preset: str = "p4",
        crf_or_cq: int = 23,
    ) -> None:
        """Save animation using explicit frame rendering and ffmpeg encode."""
        from matplotlib.animation import PillowWriter

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.video_entries:
            print("No video entries to animate. Saving static figure.")
            static_path = out_path.with_suffix(".png")
            self.fig.savefig(static_path, dpi=150)
            print(f"Static figure saved: {static_path}")
            return

        total_frames = self._get_max_frames()
        gpu_ok = use_cuda and self._is_cuda_usable()
        ffmpeg_bin = shutil.which("ffmpeg")
        has_ffmpeg = ffmpeg_bin is not None
        has_nvenc = bool(has_ffmpeg and self._ffmpeg_has_encoder(ffmpeg_bin, "h264_nvenc"))
        use_nvenc = bool(prefer_nvenc and gpu_ok and has_nvenc)

        self._prepare_static_background()
        h, w = self._render_frame_rgb(0, use_cuda=gpu_ok).shape[:2]

        def _encode_with_ffmpeg(encoder: str) -> None:
            if encoder == "h264_nvenc":
                codec_args = ["-c:v", "h264_nvenc", "-preset", encode_preset, "-cq", str(crf_or_cq)]
            else:
                codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf_or_cq)]

            cmd = [
                ffmpeg_bin, "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}",
                "-r", str(fps),
                "-i", "-",
                "-an",
                *codec_args,
                "-pix_fmt", "yuv420p",
                str(out_path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            assert proc.stdin is not None
            start = time.time()
            for i in range(total_frames):
                if i % 50 == 0 or i == total_frames - 1:
                    print(f"Rendering frame {i + 1}/{total_frames}")
                rgb = self._render_frame_rgb(i, use_cuda=gpu_ok)
                if i % 50 == 0 or i == total_frames - 1:
                    print(f"Encoding frame {i + 1}/{total_frames}")
                proc.stdin.write(rgb.tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read() if proc.stderr is not None else b""
            proc.wait()
            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="ignore")
                raise RuntimeError(err[-1200:])
            print(f"Encoding completed in {time.time() - start:.2f}s")

        if has_ffmpeg:
            try:
                selected = "h264_nvenc" if use_nvenc else "libx264"
                print(f"Using ffmpeg encoder: {selected}")
                _encode_with_ffmpeg(selected)
                print(f"Video saved: {out_path}")
                return
            except Exception as exc:
                print(f"ffmpeg encode failed ({exc}).")
                if fallback_cpu and has_ffmpeg:
                    try:
                        print("Falling back to CPU encoder: libx264")
                        _encode_with_ffmpeg("libx264")
                        print(f"Video saved: {out_path}")
                        return
                    except Exception as exc2:
                        print(f"CPU fallback failed ({exc2}). Falling back to GIF.")
                else:
                    print("CPU fallback disabled. Falling back to GIF.")

        # Final fallback: GIF
        self.anim = FuncAnimation(
            self.fig, self._update_frame,
            frames=total_frames,
            interval=1000 // max(1, fps), blit=True
        )
        gif_path = out_path.with_suffix(".gif")
        writer = PillowWriter(fps=fps)
        self.anim.save(str(gif_path), writer=writer)
        print(f"Video saved (GIF fallback): {gif_path}")


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

    # Process all comparison sets
    comparison_sets = config.get("comparison_sets", {})
    if not comparison_sets:
        print("Error: No comparison_sets defined in config")
        return 1

    def _safe_filename(name: str) -> str:
        s = re.sub(r'[<>:\"/\\|?*]+', "_", name.strip())
        s = re.sub(r"\s+", " ", s).strip()
        return s or "comparison"

    items = list(comparison_sets.items())
    for idx, (set_name, testpoints) in enumerate(items, start=1):
        print("\n" + "#" * 60)
        print(f"[{idx}/{len(items)}] Comparison set: {set_name}")
        print(f"Testpoints: {testpoints}")

        # Load all data
        data = ComparisonData(config, testpoints)
        data.load_all()

        # Build visualization
        print("\n" + "=" * 60)
        print("BUILDING VISUALIZATION")
        print("=" * 60)
        viewer = ComparisonViewer(data, title=set_name)
        print(
            f"Layout: plots=4 rows (fixed), videos={viewer.num_video_rows} rows (dynamic), "
            f"cols={1 + len(testpoints)}"
        )

        # Show or save
        if args.debug:
            print("\nShowing visualization (debug mode)...")
            viewer.show()
        else:
            if args.output is not None and len(items) == 1:
                output_path = args.output
                if not output_path.is_absolute():
                    output_path = Path.cwd() / output_path
            else:
                out_name = f"comparison_{_safe_filename(set_name)}.mp4"
                output_path = Path.cwd() / "video" / out_name

            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving video to: {output_path}")
            viewer.save_video(str(output_path), fps=args.fps)
            plt.close(viewer.fig)

    print("\nVisualization complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
