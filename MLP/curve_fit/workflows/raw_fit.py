"""Raw Mie trajectory fit production workflow.

This is the entry point for the curve-fit production pipeline. For each
configured nozzle dataset and each penetration source (CDF, bw_x,
bw_polar), it:

1. reads raw per-frame plume measurements plus companion ``.meta.json`` files;
2. resolves physical units, hydraulic delay, plume alignment, and cutoff
   filters;
3. fits the q1 quarter-root model to each plume trace;
4. writes per-folder fit tables, clean/flagged splits, long and wide series
   tables, plots, and a compact ``fit_report.csv``.

Run directly to chain dataset summaries, filter-survival summaries, fit
diagnostics, spatial-censoring audit, CDF censoring points, and the
P50-q1 oracle unless ``--no-chain`` is passed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from MLP.curve_fit.core import config
from MLP.curve_fit.core.config import (
    DIFF_THRESHOLD_LOWER,
    DIFF_THRESHOLD_UPPER,
    FIT_MODEL_NAME,
    K_SQRT_SENTINEL,
    LOG_K_SQRT_SENTINEL,
    MASK_GROUP_COLS,
    META_COLS,
    MIN_SERIES_POINTS,
    RMSE_SUCCESS_THRESHOLD_MM,
    get_enabled_penetration_sources,
)
from MLP.curve_fit.core.filter_masking import apply_filter_masking
from MLP.curve_fit.core.plotting import save_fit_plot, save_raw_plot
from MLP.curve_fit.core.q1_model import (
    fit_quarter_only,
    spray_penetration_model_quarter_only,
)
from MLP.curve_fit.core.series_io import (
    _read_csv_with_expanded_static_meta,
    build_wide_series_df,
    collect_series_rows,
    filter_series_df,
    get_dataset_settings,
    prepare_cleaned_series,
)


def process_folder(
    folder,
    out_all_dir,
    out_clean_dir,
    out_series_all_dir,
    out_series_clean_dir,
    out_series_wide_all_dir,
    out_series_wide_clean_dir,
    out_plots_clean_dir,
    out_plots_flagged_dir,
    out_plots_raw_all_dir,
    penetration_source,
    mm_per_px_scale,
    fps_default,
    max_hydraulic_delay_frames,
    delay_clip_half_window,
):
    """Process one condition folder for one penetration source."""
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"Skip {folder.name}: no csv files.")
        return None

    rows = []
    series_rows = []
    for file_path in csv_files:
        df_file = _read_csv_with_expanded_static_meta(file_path)
        time_s, time_ms, cleaned_series, delays_raw, delays_used, delay_sources = prepare_cleaned_series(
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
        series_rows.extend(
            collect_series_rows(
                file_path=file_path,
                time_s=time_s,
                time_ms=time_ms,
                cleaned_series=cleaned_series,
                delays_raw=delays_raw,
                delays_used=delays_used,
                delay_sources=delay_sources,
            )
        )

        meta = {}
        for col in META_COLS:
            meta[col] = df_file[col].iloc[0] if col in df_file.columns else np.nan

        inj_dur_s = float(meta["injection_duration_us"]) * 1e-6
        x0_q1 = np.log([1.0, max(2.0 * inj_dur_s, 1e-9), 1.0])

        number_of_plumes = cleaned_series.shape[0]
        for plume_idx in range(number_of_plumes):
            series = cleaned_series[plume_idx]
            ts = time_s[plume_idx]
            if int((np.isfinite(ts) & np.isfinite(series)).sum()) < MIN_SERIES_POINTS:
                continue
            valid = np.isfinite(ts) & np.isfinite(series)
            t_max_s = float(np.nanmax(ts)) if np.any(np.isfinite(ts)) else float("nan")
            fit = fit_quarter_only(ts, series, x0_q1)
            log_params = fit["log_params"]
            if fit["success"] and np.all(np.isfinite(log_params)) and np.any(valid):
                y_true = series[valid]
                y_hat = spray_penetration_model_quarter_only(log_params, ts[valid])
                rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
                ss_res = float(np.sum((y_true - y_hat) ** 2))
                ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
            else:
                rmse = np.nan
                r2 = np.nan
                log_params = np.full(3, np.nan)

            log_k_quarter, log_t0, log_s = log_params

            row_dict = {
                "fit_model": FIT_MODEL_NAME,
                "penetration_source": penetration_source["key"],
                "file_path": str(file_path.resolve()),
                "file_name": file_path.name,
                "file_stem": file_path.stem,
                "plume_idx": plume_idx,
                "delay_frames": delays_used[plume_idx],
                "delay_frames_raw": delays_raw[plume_idx],
                "delay_frames_used": delays_used[plume_idx],
                "delay_source": delay_sources[plume_idx],
                "k_sqrt": K_SQRT_SENTINEL,
                "k_quarter": fit["k_quarter"],
                "t0": fit["t0"],
                "s": fit["s"],
                "cost": fit["cost"],
                "success": fit["success"],
                "n": fit["n"],
                "rmse": rmse,
                "r2": r2,
                "log_k_sqrt": LOG_K_SQRT_SENTINEL,
                "log_k_quarter": log_k_quarter,
                "log_t0": log_t0,
                "log_s": log_s,
                "t_max_s": t_max_s,
                "nfev": fit["nfev"],
                "optimality": fit["optimality"],
                "status": fit["status"],
                "std_log_k_quarter": fit["std_log_k_quarter"],
                "std_log_t0": fit["std_log_t0"],
                "std_log_s": fit["std_log_s"],
                "corr_logk_logt0": fit["corr_logk_logt0"],
                "corr_logk_logs": fit["corr_logk_logs"],
                "corr_logt0_logs": fit["corr_logt0_logs"],
                **meta,
            }
            rows.append(row_dict)

    results_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    masked_df, clean_df, flagged_df = apply_filter_masking(results_df, group_cols=MASK_GROUP_COLS)
    series_df = pd.DataFrame(series_rows).replace([np.inf, -np.inf], np.nan)
    series_clean_df = filter_series_df(series_df, clean_df)
    series_wide_all_df = build_wide_series_df(series_df, masked_df)
    series_wide_clean_df = build_wide_series_df(series_clean_df, clean_df)

    out_all_path = out_all_dir / f"{folder.name}.csv"
    out_clean_path = out_clean_dir / f"{folder.name}.csv"
    out_flagged_path = out_all_dir / f"{folder.name}_flagged.csv"
    out_series_all_path = out_series_all_dir / f"{folder.name}.csv"
    out_series_clean_path = out_series_clean_dir / f"{folder.name}.csv"
    out_series_wide_all_path = out_series_wide_all_dir / f"{folder.name}.csv"
    out_series_wide_clean_path = out_series_wide_clean_dir / f"{folder.name}.csv"
    clean_plot_paths = save_fit_plot(
        folder,
        clean_df,
        csv_files,
        out_plots_clean_dir,
        "clean",
        penetration_source,
        mm_per_px_scale=mm_per_px_scale,
        fps_default=fps_default,
        max_hydraulic_delay_frames=max_hydraulic_delay_frames,
        delay_clip_half_window=delay_clip_half_window,
    )
    flagged_plot_paths = save_fit_plot(
        folder,
        flagged_df,
        csv_files,
        out_plots_flagged_dir,
        "flagged",
        penetration_source,
        mm_per_px_scale=mm_per_px_scale,
        fps_default=fps_default,
        max_hydraulic_delay_frames=max_hydraulic_delay_frames,
        delay_clip_half_window=delay_clip_half_window,
    )
    raw_plot_paths = save_raw_plot(
        folder,
        results_df,
        csv_files,
        out_plots_raw_all_dir,
        penetration_source,
        mm_per_px_scale=mm_per_px_scale,
        fps_default=fps_default,
        max_hydraulic_delay_frames=max_hydraulic_delay_frames,
        delay_clip_half_window=delay_clip_half_window,
    )

    masked_df.to_csv(out_all_path, index=False)
    clean_df.to_csv(out_clean_path, index=False)
    flagged_df.to_csv(out_flagged_path, index=False)
    series_df.to_csv(out_series_all_path, index=False)
    series_clean_df.to_csv(out_series_clean_path, index=False)
    series_wide_all_df.to_csv(out_series_wide_all_path, index=False)
    series_wide_clean_df.to_csv(out_series_wide_clean_path, index=False)

    _fmt_paths = lambda paths: ", ".join(p.name for p in paths)
    print(
        f"[{penetration_source['key']}] Saved {out_all_path.name} ({len(masked_df)} total rows), "
        f"{out_clean_path.name} ({len(clean_df)} clean), "
        f"{out_flagged_path.name} ({len(flagged_df)} flagged), "
        f"{out_series_all_path.name} ({len(series_df)} series rows), "
        f"{out_series_clean_path.name} ({len(series_clean_df)} clean series rows), "
        f"{out_series_wide_all_path.name} ({len(series_wide_all_df)} wide rows), "
        f"{out_series_wide_clean_path.name} ({len(series_wide_clean_df)} clean wide rows), "
        f"{_fmt_paths(clean_plot_paths)} (clean-curve plot(s)), "
        f"{_fmt_paths(flagged_plot_paths)} (flagged-curve plot(s)), "
        f"{_fmt_paths(raw_plot_paths)} (raw-all plot(s)) from {len(csv_files)} files"
    )

    _n = len(masked_df)
    stats = {
        "nozzle": folder.parent.name,
        "folder": folder.name,
        "penetration_source": penetration_source["key"],
        "fit_model": FIT_MODEL_NAME,
        "n_total": _n,
        "n_clean": len(clean_df),
        "n_flagged": len(flagged_df),
        "success_main": np.nan,
        "success_rate_main_pct": np.nan,
        "rmse_clean_main_mm": np.nan,
        "rmse_flagged_main_mm": np.nan,
    }
    if _n > 0:
        _rmse_main = pd.to_numeric(masked_df["rmse"], errors="coerce")
        _ok = int(
            ((~masked_df["flag_bad_fit"].fillna(True)) & (_rmse_main < RMSE_SUCCESS_THRESHOLD_MM)).sum()
        )
        _pct = 100.0 * _ok / _n
        _rmse_c = clean_df["rmse"].median() if len(clean_df) > 0 else float("nan")
        _rmse_f = flagged_df["rmse"].median() if len(flagged_df) > 0 else float("nan")
        stats.update({
            "success_main": _ok,
            "success_rate_main_pct": round(_pct, 1),
            "rmse_clean_main_mm": round(_rmse_c, 3) if np.isfinite(_rmse_c) else np.nan,
            "rmse_flagged_main_mm": round(_rmse_f, 3) if np.isfinite(_rmse_f) else np.nan,
        })
        print(
            f"  [fit-report] main/{FIT_MODEL_NAME}  success {_ok}/{_n} ({_pct:.1f}%)  "
            f"RMSE median  clean {_rmse_c:.2f} mm  flagged {_rmse_f:.2f} mm"
        )
    return stats


def _process_folder_worker(kwargs):
    """Module-level wrapper so ProcessPoolExecutor can pickle the task."""
    return process_folder(**kwargs)


def _run_python_module(module: str, args: list[str], *, label: str, optional: bool = False) -> None:
    """Run ``python -m <module>`` with the current fit roots in its environment."""
    env = os.environ.copy()
    env["FIT_OUTPUT_ROOT"] = str(config.DATA_OUT_DIR)
    env["FIT_INPUT_ROOT"] = str(config.DATA_ROOT)
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n[fit_raw_data] {label}")
    completed = subprocess.run(cmd, env=env, check=False)
    if completed.returncode != 0:
        msg = f"{label} failed with exit code {completed.returncode}"
        if optional:
            print(f"[{label}] skipped: {msg}")
            return
        raise RuntimeError(msg)


def _run_default_postprocessing(args) -> None:
    """Generate diagnostics, CDF censoring points, and the p50-q1 oracle source."""
    _run_python_module(
        "MLP.curve_fit.reports.summarize_dataset",
        [],
        label="dataset summary",
        optional=True,
    )
    _run_python_module(
        "MLP.curve_fit.reports.summarize_filter_survival",
        [
            "--fit-report", str(config.DATA_OUT_DIR / "fit_report.csv"),
            "--out-dir", str(config.DATA_OUT_DIR / "fit_survival_report"),
        ],
        label="filter survival summary",
        optional=True,
    )
    _run_python_module(
        "MLP.curve_fit.reports.fit_diagnostics",
        [],
        label="fit diagnostics",
        optional=True,
    )
    _run_python_module(
        "MLP.curve_fit.reports.audit_cdf_spatial_censoring",
        [
            "--synthetic-root", str(config.DATA_OUT_DIR),
            "--out-dir", str(config.DATA_OUT_DIR / "spatial_censoring_audit"),
        ],
        label="spatial censoring audit",
        optional=True,
    )

    censoring_out = config.DATA_OUT_DIR / "cdf_right_censoring_points"
    if not args.no_cdf_censoring_points:
        from MLP.curve_fit.workflows.cdf_censoring_points import run_cdf_censoring_points

        if censoring_out.exists():
            shutil.rmtree(censoring_out)
        summary = run_cdf_censoring_points(
            synthetic_root=config.DATA_OUT_DIR,
            out_dir=censoring_out,
            make_plots=True,
            condition_plots=True,
        )
        print(f"[cdf_right_censoring_points] wrote {summary['out_dir']}")

    if not args.no_p50_q1:
        points_uncensored = censoring_out / "cdf_points_uncensored.csv"
        if not points_uncensored.exists():
            print(f"[p50_q1_oracle] skipped: missing {points_uncensored}")
            return
        from MLP.curve_fit.workflows.p50_q1_oracle import run_p50_q1_oracle

        summary = run_p50_q1_oracle(
            points_uncensored=points_uncensored,
            synthetic_root=config.DATA_OUT_DIR,
            out_dir=config.DATA_OUT_DIR / "p50_q1_oracle",
            source_key=args.p50_source_key,
            min_bins=args.p50_min_bins,
            min_traces_per_bin=args.p50_min_traces_per_bin,
            extrapolate_t_max_ms=args.p50_extrapolate_t_max_ms,
        )
        print(f"[p50_q1_oracle] fitted {summary['n_fit_conditions']} conditions")


def main():
    """CLI entry point: collect folder tasks, fit them, and write the report."""
    _p = argparse.ArgumentParser(add_help=False)
    _p.add_argument("--output-root", type=Path, default=None)
    _p.add_argument("--input-root", type=Path, default=None)
    _p.add_argument("--n-workers", type=int, default=None)
    _p.add_argument("--no-chain", action="store_true", default=False)
    _p.add_argument("--no-cdf-censoring-points", action="store_true", default=False)
    _p.add_argument("--no-p50-q1", action="store_true", default=False)
    _p.add_argument("--p50-source-key", default="cdf_p50_q1")
    _p.add_argument("--p50-min-bins", type=int, default=5)
    _p.add_argument("--p50-min-traces-per-bin", type=int, default=4)
    _p.add_argument("--p50-extrapolate-t-max-ms", type=float, default=5.0)
    _p.add_argument("--nozzle-filter", type=str, default=None,
                    help="Restrict to one nozzle name suffix (for smoke-testing).")
    _args, _ = _p.parse_known_args()

    if _args.output_root is not None:
        config.DATA_OUT_DIR = _args.output_root
    if _args.input_root is not None:
        config.DATA_ROOT = _args.input_root
    if _args.n_workers is not None:
        config.N_WORKERS = _args.n_workers
    if _args.nozzle_filter is not None:
        config.NOZZLE_NAMES = [
            n for n in config.NOZZLE_NAMES if n == _args.nozzle_filter or n.endswith(_args.nozzle_filter)
        ]

    # Propagate to worker subprocess environments before spawning the pool.
    os.environ["FIT_OUTPUT_ROOT"] = str(config.DATA_OUT_DIR)
    os.environ["FIT_INPUT_ROOT"] = str(config.DATA_ROOT)
    os.environ["FIT_N_WORKERS"] = str(config.N_WORKERS)
    if _args.nozzle_filter is not None:
        os.environ["FIT_NOZZLE_FILTER"] = _args.nozzle_filter

    enabled_sources = get_enabled_penetration_sources()
    if not enabled_sources:
        raise ValueError("At least one penetration-series switch must be enabled.")

    # --- Pass 1: collect tasks and pre-create all output directories ---
    tasks = []
    for name in config.NOZZLE_NAMES:
        settings = get_dataset_settings(name)
        mm_per_px_scale = 90.0 / settings["or_mm_per_px_reference"]
        root = config.DATA_ROOT / name
        out_dir = config.DATA_OUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        if not root.exists():
            print(f"Skipping missing dataset root: {root}")
            continue

        subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if not subdirs:
            raise FileNotFoundError(f"No subdirs found in {root}")

        print(f"Found {len(subdirs)} subdirs in {root}")
        for folder in subdirs:
            for penetration_source in enabled_sources:
                metric_out_dir = out_dir / penetration_source["key"]
                out_all_dir = metric_out_dir / "all"
                out_clean_dir = metric_out_dir / "clean"
                out_series_all_dir = metric_out_dir / "series_all"
                out_series_clean_dir = metric_out_dir / "series_clean"
                out_series_wide_all_dir = metric_out_dir / "series_wide_all"
                out_series_wide_clean_dir = metric_out_dir / "series_wide_clean"
                out_plots_clean_dir = metric_out_dir / "plots_clean"
                out_plots_flagged_dir = metric_out_dir / "plots_flagged"
                out_plots_raw_all_dir = metric_out_dir / "plots_raw_all"
                for d in (
                    out_all_dir, out_clean_dir,
                    out_series_all_dir, out_series_clean_dir,
                    out_series_wide_all_dir, out_series_wide_clean_dir,
                    out_plots_clean_dir, out_plots_flagged_dir, out_plots_raw_all_dir,
                ):
                    d.mkdir(parents=True, exist_ok=True)

                tasks.append({
                    "folder": folder,
                    "out_all_dir": out_all_dir,
                    "out_clean_dir": out_clean_dir,
                    "out_series_all_dir": out_series_all_dir,
                    "out_series_clean_dir": out_series_clean_dir,
                    "out_series_wide_all_dir": out_series_wide_all_dir,
                    "out_series_wide_clean_dir": out_series_wide_clean_dir,
                    "out_plots_clean_dir": out_plots_clean_dir,
                    "out_plots_flagged_dir": out_plots_flagged_dir,
                    "out_plots_raw_all_dir": out_plots_raw_all_dir,
                    "penetration_source": penetration_source,
                    "mm_per_px_scale": mm_per_px_scale,
                    "fps_default": settings["fps_default"],
                    "max_hydraulic_delay_frames": settings["max_hydraulic_delay_frames"],
                    "delay_clip_half_window": settings["delay_clip_half_window"],
                })

    # --- Pass 2: dispatch ---
    n_workers = config.N_WORKERS or os.cpu_count()
    if n_workers == 1:
        results = [_process_folder_worker(kwargs) for kwargs in tasks]
    else:
        print(f"Launching {n_workers} workers for {len(tasks)} folder tasks …")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_process_folder_worker, tasks))

    # --- Pass 3: write fit report ---
    all_stats = [s for s in results if isinstance(s, dict)]
    if all_stats:
        report_df = pd.DataFrame(all_stats)
        report_path = config.DATA_OUT_DIR / "fit_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\nFit report saved -> {report_path}  ({len(report_df)} rows)")

    if not _args.no_chain:
        _run_default_postprocessing(_args)


if __name__ == "__main__":
    main()
