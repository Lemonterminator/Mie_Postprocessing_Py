"""Per-folder cleaned/flagged/raw plot helpers for the raw-fit pipeline.

The plotters recompute aligned series from the raw CSV via
``prepare_cleaned_series`` (cached per CSV path) and overlay the fitted q1
curve from the row's log-parameters. Multi-duration nozzles (e.g.
``Nozzle0`` with T340/T560) are split into per-duration PNGs by
``_dur_groups``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .config import (
    DIFF_THRESHOLD_LOWER,
    DIFF_THRESHOLD_UPPER,
    PLOT_EXTRAP_FACTOR,
    PLOT_NUM_POINTS,
    PLOT_YLIM_MM,
)
from .q1_model import spray_penetration_model_quarter_only
from .series_io import _read_csv_with_expanded_static_meta, prepare_cleaned_series


def _dur_groups(df):
    """Yield ``(filename_suffix, sub_df)`` for each injection-duration group.

    Single duration → one group with empty suffix.
    Multiple durations (e.g. Nozzle0) → one group per duration so that each
    injection condition gets its own plot file (``_T340``, ``_T560``, …).
    """
    col = "injection_duration_us"
    if col not in df.columns or df.empty:
        yield "", df
        return
    durs = sorted(pd.to_numeric(df[col], errors="coerce").dropna().unique())
    if len(durs) <= 1:
        yield "", df
        return
    dur_series = pd.to_numeric(df[col], errors="coerce")
    for d in durs:
        yield f"_T{int(d)}", df[dur_series == d]


def _build_csv_resolvers(csv_files):
    name_to_paths = {}
    stem_to_paths = {}
    for p in csv_files:
        name_to_paths.setdefault(p.name, []).append(p)
        stem_to_paths.setdefault(p.stem, []).append(p)

    def resolve_csv(row):
        row_file_path = getattr(row, "file_path", "")
        if isinstance(row_file_path, str) and row_file_path != "":
            p = Path(row_file_path)
            if p.exists():
                return p
        row_file_name = str(getattr(row, "file_name", ""))
        cands = name_to_paths.get(row_file_name, [])
        if len(cands) == 1:
            return cands[0]
        if len(cands) == 0:
            row_file_stem = str(getattr(row, "file_stem", ""))
            cands = stem_to_paths.get(row_file_stem, [])
            if len(cands) == 1:
                return cands[0]
        return None

    return resolve_csv


def save_fit_plot(
    folder,
    plot_df,
    csv_files,
    out_plot_dir,
    plot_kind,
    penetration_source,
    mm_per_px_scale,
    fps_default,
    max_hydraulic_delay_frames,
    delay_clip_half_window,
):
    """Plot cleaned traces with fitted q1 curves for one folder and source."""
    cache = {}  # csv_path -> (time_s, time_ms, cleaned_series, inj_dur_s)
    resolve_csv = _build_csv_resolvers(csv_files)

    legend_handles = [
        Line2D([0], [0], color="gray", linewidth=1.2, linestyle="-", label="raw trace"),
        Line2D([0], [0], color="gray", linewidth=1.2, linestyle="--", label="q1 fit (∜t only)"),
    ]
    title_suffix = "solid=raw  --=q1"

    saved_paths = []
    for _suffix, _group_df in _dur_groups(plot_df):
        rng = np.random.default_rng()
        plt.figure(figsize=(10, 6))
        has_curve = False

        for row in _group_df.itertuples(index=False):
            csv_path = resolve_csv(row)
            if csv_path is None:
                continue

            cache_key = str(csv_path.resolve())
            if cache_key not in cache:
                df_file = _read_csv_with_expanded_static_meta(csv_path)
                time_s, time_ms, cleaned_series, _, _, _ = prepare_cleaned_series(
                    df_file,
                    mm_per_px_scale=mm_per_px_scale,
                    fps_default=fps_default,
                    max_hydraulic_delay_frames=max_hydraulic_delay_frames,
                    delay_clip_half_window=delay_clip_half_window,
                    penetration_source=penetration_source,
                    replace_negative_with_zero=penetration_source["replace_negative_with_zero"],
                    diff_threshold_lower=DIFF_THRESHOLD_LOWER,
                    diff_threshold_upper=DIFF_THRESHOLD_UPPER,
                )
                inj_dur_s = float(df_file["injection_duration_us"].iloc[0]) * 1e-6
                cache[cache_key] = (time_s, time_ms, cleaned_series, inj_dur_s)

            time_s, time_ms, cleaned_series, _inj_dur_s = cache[cache_key]
            plume_idx = int(row.plume_idx)
            if plume_idx < 0 or plume_idx >= cleaned_series.shape[0]:
                continue

            ts = time_s[plume_idx]
            tms = time_ms[plume_idx]
            raw_series = cleaned_series[plume_idx]
            valid_raw = np.isfinite(tms) & np.isfinite(raw_series)
            if not np.any(valid_raw):
                continue

            color = rng.random(3)
            plt.plot(tms[valid_raw], raw_series[valid_raw], alpha=0.65, linewidth=1.0, color=color)
            t_end = float(np.nanmax(ts) * PLOT_EXTRAP_FACTOR)
            t_extrap_s = np.linspace(0.0, t_end, PLOT_NUM_POINTS)
            draw_fit = (
                bool(getattr(row, "success", False))
                and np.isfinite(getattr(row, "log_k_quarter", np.nan))
                and np.isfinite(getattr(row, "log_t0", np.nan))
                and np.isfinite(getattr(row, "log_s", np.nan))
            )
            if draw_fit:
                log_params = [row.log_k_quarter, row.log_t0, row.log_s]
                y_extrap = spray_penetration_model_quarter_only(log_params, t_extrap_s)
                plt.plot(
                    1e3 * t_extrap_s,
                    y_extrap,
                    linestyle="--",
                    alpha=0.45,
                    linewidth=1.0,
                    color=color,
                )
            has_curve = True

        if not has_curve:
            plt.text(0.5, 0.5, f"No {plot_kind} traces for this folder", ha="center", va="center")

        plt.legend(handles=legend_handles, fontsize=7, loc="upper left")
        dur_label = f" | T={_suffix[2:]}µs" if _suffix else ""
        plt.title(
            f"{folder.name}{dur_label}: {penetration_source['label']} {plot_kind}  [{title_suffix}]",
            fontsize=9,
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Penetration (mm)")
        plt.grid(alpha=0.25)
        plt.ylim(0, PLOT_YLIM_MM)

        out_plot_path = out_plot_dir / f"{folder.name}{_suffix}.png"
        plt.tight_layout()
        plt.savefig(out_plot_path, dpi=140)
        plt.close()
        saved_paths.append(out_plot_path)

    return saved_paths


def save_raw_plot(
    folder,
    plot_df,
    csv_files,
    out_plot_dir,
    penetration_source,
    mm_per_px_scale,
    fps_default,
    max_hydraulic_delay_frames,
    delay_clip_half_window,
):
    """Plot delay-aligned raw/cleaned traces over the early 0-2 ms window."""
    cache = {}
    resolve_csv = _build_csv_resolvers(csv_files)

    saved_paths = []
    for _suffix, _group_df in _dur_groups(plot_df):
        rng = np.random.default_rng()
        plt.figure(figsize=(10, 6))
        has_curve = False

        for row in _group_df.itertuples(index=False):
            csv_path = resolve_csv(row)
            if csv_path is None:
                continue

            cache_key = str(csv_path.resolve())
            if cache_key not in cache:
                df_file = _read_csv_with_expanded_static_meta(csv_path)
                time_s, time_ms, cleaned_series, _, _, _ = prepare_cleaned_series(
                    df_file,
                    mm_per_px_scale=mm_per_px_scale,
                    fps_default=fps_default,
                    max_hydraulic_delay_frames=max_hydraulic_delay_frames,
                    delay_clip_half_window=delay_clip_half_window,
                    penetration_source=penetration_source,
                    replace_negative_with_zero=penetration_source["replace_negative_with_zero"],
                    diff_threshold_lower=DIFF_THRESHOLD_LOWER,
                    diff_threshold_upper=DIFF_THRESHOLD_UPPER,
                )
                cache[cache_key] = (time_s, time_ms, cleaned_series)

            time_s, time_ms, cleaned_series = cache[cache_key]
            plume_idx = int(row.plume_idx)
            if plume_idx < 0 or plume_idx >= cleaned_series.shape[0]:
                continue

            tms = time_ms[plume_idx]
            raw_series = cleaned_series[plume_idx]

            mask_time = tms <= 2.0
            valid_raw = np.isfinite(tms) & np.isfinite(raw_series) & mask_time
            if not np.any(valid_raw):
                continue

            color = rng.random(3)
            plt.plot(tms[valid_raw], raw_series[valid_raw], alpha=0.65, linewidth=1.0, color=color)
            has_curve = True

        if not has_curve:
            plt.text(1.0, 0.5, "No traces for this folder", ha="center", va="center")

        dur_label = f" | T={_suffix[2:]}µs" if _suffix else ""
        plt.title(
            f"{folder.name}{dur_label}: {penetration_source['label']} aligned raw traces (0-2ms)"
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Penetration (mm)")
        plt.grid(alpha=0.25)
        plt.xlim(0, 2.0)
        plt.ylim(0, PLOT_YLIM_MM)

        out_plot_path = out_plot_dir / f"{folder.name}{_suffix}.png"
        plt.tight_layout()
        plt.savefig(out_plot_path, dpi=140)
        plt.close()
        saved_paths.append(out_plot_path)

    return saved_paths
