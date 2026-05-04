"""
Comparison Viewer - Visualize testpoint comparisons with plots and videos
=========================================================================

Creates a grid visualization comparing multiple testpoints:

Layout (2-case example):
    Column 1: Plots           | Column 2: T2 Videos    | Column 3: T56 Videos
    ─────────────────────────────────────────────────────────────────────────
    Plot rows (see PLOT_ROW_KINDS below)
    Video rows (see VIDEO_ROW_KINDS below)

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
except Exception:
    cp = None

# Path setup
_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent
for _p in [str(_REPO_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from OSCC_postprocessing.dewe.dewe import align_dewe_dataframe_to_soe
from OSCC_postprocessing.dewe.heat_release_calulation import hrr_calc
from matplotlib_video_displayer import normalize_video


# =============================================================================
# LAYOUT CONFIGURATION — edit these lists to change what appears in the output
# =============================================================================

# Plot rows (left column) — remove, reorder, or add entries.
# Available keys: "pressure_temp_hrr", "mie_area", "injection_current"
PLOT_ROW_KINDS: List[str] = [
    "pressure_temp_hrr",
    "mie_area",
    "injection_current",
]

# Video rows (right columns) — remove, reorder, or add entries.
# Available keys: "mie", "schlieren", "luminescence", "heatmap"
# Rows with no data for any testpoint are automatically hidden.
VIDEO_ROW_KINDS: List[str] = [
    "mie",
    "schlieren",
    "luminescence",
    "heatmap",
]


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_COLUMN_NAMES = {
    "chamber_pressure": ["Chamber pressure (BarA)", "Chamber pressure", "chamber_pressure"],
    "chamber_temperature": ["Temperature acc. Ideal gas law", "Chamber gas temperature", "chamber_temperature"],
    "heat_release": ["Heat Release", "HRR", "heat_release"],
    "injection_current": ["Main Injector - Current Profile", "Current Profile", "injection_current"],
    "mie_area": ["Area", "area", "Spray Area"],
}

PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
LINE_STYLES = ['-', '--', ':', '-.', '-']

# Plot rows that use a shared time_ms x-axis.
# "mie_area" is excluded because the mie metrics CSV has no time column —
# it uses frame index instead.
_TIME_SHARED_KINDS = {"pressure_temp_hrr", "injection_current"}


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_default_config_path() -> Path:
    return Path(__file__).parent / "config.json"


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        for col in df.columns:
            if candidate.lower() in col.lower():
                return col
    return None


# =============================================================================
# File Location
# =============================================================================

def parse_testpoint_from_filename(filename: str) -> Optional[str]:
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
    name_contains: Optional[str] = None,
) -> Optional[Path]:
    if not directory.exists():
        return None
    for f in directory.iterdir():
        if not f.is_file() or f.suffix.lower() != suffix.lower():
            continue
        if name_contains and name_contains not in f.name:
            continue
        tp = parse_testpoint_from_filename(f.name)
        rep = parse_repetition_from_filename(f.name)
        if tp == testpoint and rep == repetition:
            return f
    return None


def find_files_for_testpoint(
    directory: Path,
    testpoint: str,
    suffix: str = ".csv",
    name_contains: Optional[str] = None,
) -> List[Path]:
    if not directory.exists():
        return []
    matches: List[Path] = []
    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.suffix.lower() != suffix.lower():
            continue
        if name_contains and name_contains not in f.name:
            continue
        tp = parse_testpoint_from_filename(f.name)
        if tp == testpoint:
            matches.append(f)
    return matches


def load_first_npz_array(path: Path) -> np.ndarray:
    with np.load(path) as npz:
        key = npz.files[0]
        return np.asarray(npz[key])


def average_arrays(paths: List[Path], loader) -> np.ndarray:
    """Average arrays, truncating to the shortest leading dimension."""
    if not paths:
        raise ValueError("average_arrays requires at least one path")

    loaded = [(path, np.asarray(loader(path), dtype=np.float32)) for path in paths]
    ndim_values = {arr.ndim for _, arr in loaded}
    if len(ndim_values) != 1:
        raise ValueError("Cannot average arrays with different ranks: "
                         + ", ".join(f"{p.name} {a.shape}" for p, a in loaded))

    shape_groups: Dict[Tuple[int, ...], List] = {}
    for path, arr in loaded:
        trailing = tuple(arr.shape[1:]) if arr.ndim >= 2 else ()
        shape_groups.setdefault(trailing, []).append((path, arr))

    if len(shape_groups) > 1:
        selected_shape, selected_group = max(shape_groups.items(), key=lambda x: (len(x[1]), x[0]))
        skipped = [f"{p.name} {a.shape}" for s, g in shape_groups.items()
                   if s != selected_shape for p, a in g]
        print(f"  Warning: skipping {len(skipped)} files with non-matching shape: {', '.join(skipped)}")
        loaded = selected_group

    min_len = min(a.shape[0] for _, a in loaded)
    max_len = max(a.shape[0] for _, a in loaded)
    if min_len != max_len:
        print(f"  Warning: truncating arrays to {min_len} frames for averaging")
        loaded = [(p, a[:min_len].copy()) for p, a in loaded]

    total = loaded[0][1].copy()
    for _, current in loaded[1:]:
        np.add(total, current, out=total)
    total /= np.float32(len(loaded))
    return total


def average_dataframes(paths: List[Path], *, index_col: Optional[int] = None) -> pd.DataFrame:
    if not paths:
        raise ValueError("average_dataframes requires at least one path")
    frames = [pd.read_csv(p, index_col=index_col) for p in paths]
    row_counts = [f.shape[0] for f in frames]
    min_rows = min(row_counts)
    if min_rows != max(row_counts):
        print(f"  Warning: truncating CSV files to {min_rows} rows for averaging")
        frames = [f.iloc[:min_rows].copy() for f in frames]

    result = frames[0].copy()
    all_cols = list(dict.fromkeys(col for f in frames for col in f.columns))
    for col in all_cols:
        series_list = [f[col].to_numpy(dtype=np.float32) for f in frames
                       if col in f.columns and pd.api.types.is_numeric_dtype(f[col])]
        if series_list:
            result[col] = np.stack(series_list).mean(axis=0)
    return result


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
        self.mode = str(config.get("processing", {}).get("mode", "sample")).lower()
        if self.mode not in {"sample", "average"}:
            raise ValueError(f"Unsupported processing mode: {self.mode!r}")

        self.dewe_data: Dict[str, pd.DataFrame] = {}
        self.mie_data: Dict[str, pd.DataFrame] = {}
        self.mie_videos: Dict[str, np.ndarray] = {}
        self.schlieren_videos: Dict[str, np.ndarray] = {}
        self.luminescence_videos: Dict[str, np.ndarray] = {}
        self.luminescence_heatmaps: Dict[str, np.ndarray] = {}

        self.column_names = config.get("column_names", DEFAULT_COLUMN_NAMES)
        self.align_cfg = config.get("alignment", {})
        self.hrr_cfg = config.get("hrr_parameters", {})

    def load_all(self) -> None:
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        for tp_str in self.testpoint_strs:
            print(f"\nTestpoint T{tp_str}:")
            self._load_testpoint_data(tp_str)

    def _load_testpoint_data(self, tp_str: str) -> None:
        # Dewesoft CSV
        dewe_dir = Path(self.config["directories"].get("dewe", ""))
        if dewe_dir.exists():
            dewe_data_dir = dewe_dir / "Processed_Results" / "Postprocessed_Data"
            dewe_files = (
                find_files_for_testpoint(dewe_data_dir, tp_str, ".csv")
                if self.mode == "average"
                else ([f] if (f := find_file_for_testpoint(dewe_data_dir, tp_str, ".csv", 1)) else [])
            )
            if dewe_files:
                print(f"  Dewe CSV ({self.mode}): {len(dewe_files)} file(s)")
                df = (average_dataframes(dewe_files, index_col=0)
                      if self.mode == "average"
                      else pd.read_csv(dewe_files[0], index_col=0))
                df.index.name = "time_s"
                self.dewe_data[tp_str] = df
            else:
                print(f"  Warning: Dewe CSV not found for T{tp_str} in {dewe_data_dir} — plots skipped for this testpoint")

        # Mie data CSV
        mie_dir = Path(self.config["directories"].get("mie", ""))
        if mie_dir.exists():
            mie_data_dir = mie_dir / "Processed_Results" / "Postprocessed_Data"
            mie_files = (
                find_files_for_testpoint(mie_data_dir, tp_str, ".csv", name_contains="_metrics")
                if self.mode == "average"
                else ([f] if (f := find_file_for_testpoint(mie_data_dir, tp_str, ".csv", 1, name_contains="_metrics")) else [])
            )
            if mie_files:
                print(f"  Mie CSV ({self.mode}): {len(mie_files)} file(s)")
                self.mie_data[tp_str] = (average_dataframes(mie_files)
                                          if self.mode == "average"
                                          else pd.read_csv(mie_files[0]))
            else:
                print(f"  Mie CSV: NOT FOUND (looking for *_metrics.csv in {mie_data_dir})")

            # Mie video
            mie_video_dir = mie_dir / "Processed_Results" / "Rotated_Videos"
            mie_video_files = (
                find_files_for_testpoint(mie_video_dir, tp_str, ".npz")
                if self.mode == "average"
                else ([f] if (f := find_file_for_testpoint(mie_video_dir, tp_str, ".npz", 1)) else [])
            )
            if mie_video_files:
                print(f"  Mie video ({self.mode}): {len(mie_video_files)} file(s)")
                self.mie_videos[tp_str] = (
                    average_arrays(mie_video_files, load_first_npz_array)
                    if self.mode == "average"
                    else load_first_npz_array(mie_video_files[0])
                )

        # Schlieren video (raw, unprocessed — output of manual_segment.py)
        sch_dir = Path(self.config["directories"].get("schlieren", ""))
        if sch_dir.exists():
            sch_video_dir = sch_dir / "Processed_Results" / "Rotated_Videos"
            sch_video_files = (
                find_files_for_testpoint(sch_video_dir, tp_str, ".npz")
                if self.mode == "average"
                else ([f] if (f := find_file_for_testpoint(sch_video_dir, tp_str, ".npz", 1)) else [])
            )
            if sch_video_files:
                print(f"  Schlieren video ({self.mode}): {len(sch_video_files)} file(s)")
                self.schlieren_videos[tp_str] = (
                    average_arrays(sch_video_files, load_first_npz_array)
                    if self.mode == "average"
                    else load_first_npz_array(sch_video_files[0])
                )

        # Luminescence video
        lum_dir = Path(self.config["directories"].get("luminescence", ""))
        if lum_dir.exists():
            lum_video_dir = lum_dir / "Processed_Results" / "Rotated_Videos"
            lum_video_files = (
                find_files_for_testpoint(lum_video_dir, tp_str, ".npz")
                if self.mode == "average"
                else ([f] if (f := find_file_for_testpoint(lum_video_dir, tp_str, ".npz", 1)) else [])
            )
            if lum_video_files:
                print(f"  Luminescence video ({self.mode}): {len(lum_video_files)} file(s)")
                self.luminescence_videos[tp_str] = (
                    average_arrays(lum_video_files, load_first_npz_array)
                    if self.mode == "average"
                    else load_first_npz_array(lum_video_files[0])
                )

            # Pre-computed heatmap
            lum_data_dir = lum_dir / "Processed_Results" / "Postprocessed_Data"
            heatmap_files = (
                find_files_for_testpoint(lum_data_dir, tp_str, ".npz", name_contains="_heatmap")
                if self.mode == "average"
                else ([f] if (f := find_file_for_testpoint(lum_data_dir, tp_str, ".npz", 1, name_contains="_heatmap")) else [])
            )
            if heatmap_files:
                print(f"  Luminescence heatmap ({self.mode}): {len(heatmap_files)} file(s)")
                self.luminescence_heatmaps[tp_str] = (
                    average_arrays(heatmap_files, load_first_npz_array)
                    if self.mode == "average"
                    else load_first_npz_array(heatmap_files[0])
                )


# =============================================================================
# Visualization Builder
# =============================================================================

class ComparisonViewer:
    """Grid viewer: left column = plots, right columns = per-testpoint videos."""

    def __init__(self, data: ComparisonData, figsize: Tuple[float, float] = None, title: Optional[str] = None):
        self.data = data
        self.num_cases = data.num_cases

        self.plot_row_kinds = list(PLOT_ROW_KINDS)
        self.num_plot_rows = len(self.plot_row_kinds)

        self.video_row_kinds = [k for k in VIDEO_ROW_KINDS if self._is_video_row_available(k)]
        self.num_video_rows = max(len(self.video_row_kinds), 1)
        if not self.video_row_kinds:
            self.video_row_kinds = ["none"]

        if figsize is None:
            figsize = (4 + 3 * self.num_cases, 3 * max(self.num_plot_rows, self.num_video_rows))

        self.fig = plt.figure(figsize=figsize, constrained_layout=False)
        self.gs_main = GridSpec(1, 2, figure=self.fig,
                                width_ratios=[1.2, self.num_cases], wspace=0.15)
        self.gs_plots = self.gs_main[0, 0].subgridspec(self.num_plot_rows, 1, hspace=0.08)
        self.gs_videos = self.gs_main[0, 1].subgridspec(
            self.num_video_rows, self.num_cases, wspace=0.02, hspace=0.05)

        self.plot_axes: List[Axes] = []
        self.video_axes: List[List[Axes]] = []
        self.video_entries: List[Dict] = []
        self.video_scale_cache: Dict[int, Tuple[float, float]] = {}

        self._setup_axes()
        self._populate_plots()
        self._populate_videos()
        if title:
            self.fig.suptitle(title, fontsize=14, y=0.995)

    def _is_video_row_available(self, video_kind: str) -> bool:
        tps = self.data.testpoint_strs
        if video_kind == "mie":
            return any(self.data.mie_videos.get(tp) is not None for tp in tps)
        if video_kind == "schlieren":
            return any(self.data.schlieren_videos.get(tp) is not None for tp in tps)
        if video_kind == "luminescence":
            return any(self.data.luminescence_videos.get(tp) is not None for tp in tps)
        if video_kind == "heatmap":
            return any(
                self.data.luminescence_heatmaps.get(tp) is not None
                or self.data.luminescence_videos.get(tp) is not None
                for tp in tps
            )
        return False

    def _setup_axes(self) -> None:
        # Plot axes: pressure/injection_current share time_ms x-axis;
        # mie_area is kept independent (frame-indexed, no time conversion).
        shared_time_ax = None
        for row, kind in enumerate(self.plot_row_kinds):
            if kind in _TIME_SHARED_KINDS:
                if shared_time_ax is None:
                    ax = self.fig.add_subplot(self.gs_plots[row, 0])
                    shared_time_ax = ax
                else:
                    ax = self.fig.add_subplot(self.gs_plots[row, 0], sharex=shared_time_ax)
            else:
                ax = self.fig.add_subplot(self.gs_plots[row, 0])
            self.plot_axes.append(ax)

        # Hide x-tick labels for all but the last plot row
        for ax in self.plot_axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        # Video axes
        for row in range(self.num_video_rows):
            row_axes = []
            for col in range(self.num_cases):
                ax = self.fig.add_subplot(self.gs_videos[row, col])
                ax.set_xticks([])
                ax.set_yticks([])
                row_axes.append(ax)
            self.video_axes.append(row_axes)

    def _populate_plots(self) -> None:
        time_window = self.data.align_cfg.get("window_ms", 10.0)
        dispatch = {
            "pressure_temp_hrr": lambda ax: self._plot_pressure_temp_hrr(ax),
            "mie_area":          lambda ax: self._plot_mie_area(ax),
            "injection_current": lambda ax: self._plot_injection_current(ax, time_window),
        }
        for row, kind in enumerate(self.plot_row_kinds):
            fn = dispatch.get(kind)
            if fn:
                fn(self.plot_axes[row])
            else:
                self._plot_placeholder(self.plot_axes[row], f"({kind})")

        last_kind = self.plot_row_kinds[-1] if self.plot_row_kinds else ""
        self.plot_axes[-1].set_xlabel("Frame" if last_kind == "mie_area" else "Time (ms)")

    def _plot_pressure_temp_hrr(self, ax: Axes) -> None:
        ax2 = ax.twinx()
        for idx, tp_str in enumerate(self.data.testpoint_strs):
            if tp_str not in self.data.dewe_data:
                continue
            df = self.data.dewe_data[tp_str]
            style = LINE_STYLES[idx % len(LINE_STYLES)]
            inj_col = find_column(df, self.data.column_names["injection_current"])
            df_aligned = align_dewe_dataframe_to_soe(
                df,
                injection_current_col=inj_col or "",
                grad_threshold=self.data.align_cfg.get("grad_threshold", 5),
                pre_samples=self.data.align_cfg.get("pre_samples", 50),
                window_ms=self.data.align_cfg.get("window_ms", 10.0),
            )
            x = df_aligned["time_ms"].to_numpy() if "time_ms" in df_aligned.columns else np.arange(len(df_aligned))
            p_col = find_column(df_aligned, self.data.column_names["chamber_pressure"])
            if p_col:
                ax.plot(x, df_aligned[p_col], color='red', linestyle=style,
                        linewidth=1.2, label=f"P (T{tp_str})", alpha=0.8)
            t_col = find_column(df_aligned, self.data.column_names["chamber_temperature"])
            if t_col:
                ax.plot(x, df_aligned[t_col] / 10, color='green', linestyle=style,
                        linewidth=1.2, label=f"T/10 (T{tp_str})", alpha=0.8)
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
        """Plot Mie spray area vs frame index.

        The mie metrics CSV has no time column (it is indexed by frame number).
        This plot therefore uses frame index on the x-axis and does NOT share
        the time_ms x-axis used by the pressure/current plots.
        """
        has_data = False
        for idx, tp_str in enumerate(self.data.testpoint_strs):
            if tp_str not in self.data.mie_data:
                continue
            df = self.data.mie_data[tp_str]
            style = LINE_STYLES[idx % len(LINE_STYLES)]
            color = PLOT_COLORS[idx % len(PLOT_COLORS)]
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
        ax.set_xlabel("Frame", fontsize=9)
        ax.set_title("Mie Spray Area", fontsize=10)
        ax.grid(True, alpha=0.3)
        if has_data:
            ax.legend(fontsize=8)

    def _plot_injection_current(self, ax: Axes, time_window_ms: float) -> None:
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
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10, color='gray',
                transform=ax.transAxes)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

    def _populate_videos(self) -> None:
        for row_idx, video_kind in enumerate(self.video_row_kinds):
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
        ax.set_title(title, fontsize=9)
        if video is None:
            ax.text(0.5, 0.5, "(not available)", ha='center', va='center',
                    fontsize=8, color='gray', transform=ax.transAxes)
            return
        mn, mx = self._get_video_global_minmax(video)
        first_frame = self._normalize_frame(video[0], mn, mx)
        im = ax.imshow(first_frame, cmap='gray', vmin=0, vmax=1, animated=True)
        self.video_entries.append({"data": video, "im": im, "mn": mn, "mx": mx})

    def _setup_heatmap_cell(self, ax: Axes, tp_str: str) -> None:
        heatmap = None
        if tp_str in self.data.luminescence_heatmaps:
            heatmap = self.data.luminescence_heatmaps[tp_str].astype(np.float32)
        elif tp_str in self.data.luminescence_videos:
            video = self.data.luminescence_videos[tp_str]
            heatmap = np.sum(video.astype(np.float32), axis=1)
        if heatmap is not None:
            h_min, h_max = heatmap.min(), heatmap.max()
            if h_max - h_min > 1e-8:
                heatmap = (heatmap - h_min) / (h_max - h_min)
            ax.imshow(heatmap, origin='lower', cmap='hot', aspect='auto', vmin=0, vmax=1)
        else:
            ax.text(0.5, 0.5, "Heatmap\n(N/A)", ha='center', va='center',
                    fontsize=8, color='gray', transform=ax.transAxes)
        ax.set_title(f"Heatmap T{tp_str}", fontsize=9)

    def _update_frame(self, frame_idx: int):
        artists = []
        for entry in self.video_entries:
            video = entry["data"]
            idx = frame_idx % video.shape[0]
            entry["im"].set_array(self._normalize_frame(video[idx], entry["mn"], entry["mx"]))
            artists.append(entry["im"])
        return artists

    def _get_video_global_minmax(self, video: np.ndarray) -> Tuple[float, float]:
        key = id(video)
        if key in self.video_scale_cache:
            return self.video_scale_cache[key]
        if cp is not None:
            try:
                gpu = cp.asarray(video, dtype=cp.float32)
                mn, mx = float(cp.min(gpu).item()), float(cp.max(gpu).item())
                self.video_scale_cache[key] = (mn, mx)
                return mn, mx
            except Exception:
                pass
        arr = np.asarray(video, dtype=np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        self.video_scale_cache[key] = (mn, mx)
        return mn, mx

    def _normalize_frame(self, frame: np.ndarray, mn: float, mx: float) -> np.ndarray:
        if not (mx > mn):
            return np.zeros_like(np.asarray(frame, dtype=np.float32))
        arr = np.asarray(frame, dtype=np.float32)
        return ((arr - mn) / (mx - mn)).astype(np.float32, copy=False)

    def _ffmpeg_has_encoder(self, ffmpeg_bin: str, encoder_name: str) -> bool:
        try:
            proc = subprocess.run([ffmpeg_bin, "-hide_banner", "-encoders"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, check=False)
            return encoder_name in (proc.stdout or "")
        except Exception:
            return False

    def _prepare_static_background(self) -> None:
        if not isinstance(self.fig.canvas, FigureCanvasAgg):
            self.fig.set_canvas(FigureCanvasAgg(self.fig))
        self.fig.canvas.draw()
        self._cached_bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _render_frame_rgb(self, frame_idx: int) -> np.ndarray:
        if not hasattr(self, "_cached_bg"):
            self._prepare_static_background()
        self.fig.canvas.restore_region(self._cached_bg)
        for entry in self.video_entries:
            video = entry["data"]
            idx = frame_idx % video.shape[0]
            im = entry["im"]
            im.set_array(self._normalize_frame(video[idx], entry["mn"], entry["mx"]))
            im.axes.draw_artist(im)
        self.fig.canvas.blit(self.fig.bbox)
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return np.ascontiguousarray(rgba[:, :, :3])

    def _get_max_frames(self) -> int:
        return max((e["data"].shape[0] for e in self.video_entries), default=1)

    def show(self) -> None:
        if self.video_entries:
            self.anim = FuncAnimation(self.fig, self._update_frame,
                                      frames=self._get_max_frames(), interval=50, blit=True)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

    def save_video(
        self,
        output_path: str,
        fps: int = 20,
        prefer_nvenc: bool = True,
        fallback_cpu: bool = True,
        encode_preset: str = "p4",
        crf_or_cq: int = 23,
    ) -> None:
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
        ffmpeg_bin = shutil.which("ffmpeg")
        has_ffmpeg = ffmpeg_bin is not None
        has_nvenc = bool(has_ffmpeg and self._ffmpeg_has_encoder(ffmpeg_bin, "h264_nvenc"))
        use_nvenc = bool(prefer_nvenc and has_nvenc)

        self._prepare_static_background()
        h, w = self._render_frame_rgb(0).shape[:2]

        def _encode_with_ffmpeg(encoder: str) -> None:
            if encoder == "h264_nvenc":
                codec_args = ["-c:v", "h264_nvenc", "-preset", encode_preset, "-cq", str(crf_or_cq)]
            else:
                codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf_or_cq)]
            cmd = [ffmpeg_bin, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an",
                   *codec_args, "-pix_fmt", "yuv420p", str(out_path)]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            assert proc.stdin is not None
            start = time.time()
            for i in range(total_frames):
                if i % 50 == 0 or i == total_frames - 1:
                    print(f"  Frame {i + 1}/{total_frames}")
                proc.stdin.write(self._render_frame_rgb(i).tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read() if proc.stderr else b""
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(stderr.decode("utf-8", errors="ignore")[-1200:])
            print(f"  Encoding done in {time.time() - start:.2f}s")

        if has_ffmpeg:
            try:
                selected = "h264_nvenc" if use_nvenc else "libx264"
                print(f"Using encoder: {selected}")
                _encode_with_ffmpeg(selected)
                print(f"Video saved: {out_path}")
                return
            except Exception as exc:
                print(f"Encoder {selected} failed: {exc}")
                if fallback_cpu and has_ffmpeg and use_nvenc:
                    try:
                        print("Falling back to libx264")
                        _encode_with_ffmpeg("libx264")
                        print(f"Video saved: {out_path}")
                        return
                    except Exception as exc2:
                        print(f"CPU fallback failed: {exc2}. Falling back to GIF.")

        gif_path = out_path.with_suffix(".gif")
        self.anim = FuncAnimation(self.fig, self._update_frame,
                                  frames=total_frames, interval=1000 // max(1, fps), blit=True)
        self.anim.save(str(gif_path), writer=PillowWriter(fps=fps))
        print(f"Video saved (GIF): {gif_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize testpoint comparisons with plots and videos")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", help="Show visualization instead of saving")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=20)
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

    def _safe_filename(name: str) -> str:
        s = re.sub(r'[<>:\"/\\|?*]+', "_", name.strip())
        return re.sub(r"\s+", " ", s).strip() or "comparison"

    items = list(comparison_sets.items())
    for idx, (set_name, testpoints) in enumerate(items, start=1):
        print("\n" + "#" * 60)
        print(f"[{idx}/{len(items)}] {set_name}  testpoints={testpoints}")

        data = ComparisonData(config, testpoints)
        data.load_all()

        print("\n" + "=" * 60)
        print("BUILDING VISUALIZATION")
        viewer = ComparisonViewer(data, title=set_name)
        print(f"  Plot rows: {viewer.plot_row_kinds}")
        print(f"  Video rows: {viewer.video_row_kinds}")

        if args.debug:
            viewer.show()
        else:
            if args.output is not None and len(items) == 1:
                output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output
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
