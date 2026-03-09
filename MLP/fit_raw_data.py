from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit
import matplotlib.pyplot as plt



# Input/output roots
data_root = Path(r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py")
data_out_dir = Path(r"C:\Users\Jiang\Documents\Mie_Postprocessing_Py\MLP\synthetic_data")


names = [
    "BC20241003_HZ_Nozzle1",
    "BC20241017_HZ_Nozzle2",
    "BC20241014_HZ_Nozzle3",
    "BC20241007_HZ_Nozzle4",
    "BC20241010_HZ_Nozzle5",
    "BC20241011_HZ_Nozzle6",
    "BC20241015_HZ_Nozzle7",
    "BC20241016_HZ_Nozzle8",
    "BC20220627 - Heinzman DS300 - Mie Top view"
]


# Filtering/masking settings (from notebook prototype defaults)
MASK_GROUP_COLS = ("file_name",)
MASK_Z_THRESH = 3.0
MASK_MIN_N = 10
MASK_S_UPPER = 1e-3  # < 1 ms
MASK_T0_UPPER = 0.8e-3 
MASK_FAR_TIME_MS = 5.0
MASK_PENETRATION_LOWER_MM = 25.0
MASK_PENETRATION_UPPER_MM = 300.0

PLOT_EXTRAP_FACTOR = 1.6
PLOT_NUM_POINTS = 300
PLOT_YLIM_MM = 200.0




DIFF_THRESHOLD_LOWER = 2.0  # px
DIFF_THRESHOLD_UPPER = 40.0  # px
MIN_TI = 0.0



META_COLS = [
    "plumes",
    "diameter_mm",
    "fps",
    "chamber_pressure_bar",
    "injection_duration_us",
    "injection_pressure_bar",
]


def robust_z(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index, dtype=float)
    med = s.median(skipna=True)
    mad = (s - med).abs().median(skipna=True)
    if not np.isfinite(mad) or mad < 1e-12:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return 0.6745 * (s - med) / (mad + 1e-12)


def apply_filter_masking(df, group_cols=MASK_GROUP_COLS, z_thresh=MASK_Z_THRESH):
    out = df.copy()
    if out.empty:
        out["cost_per_point"] = np.nan
        out["penetration_far_mm"] = np.nan
        out["z_t0"] = np.nan
        out["z_rmse"] = np.nan
        out["z_cost"] = np.nan
        out["mask_basic"] = False
        out["mask_penetration_far"] = False
        out["mask_outlier"] = False
        out["flag_bad_fit"] = False
        return out, out.copy(), out.copy()

    out["cost_per_point"] = 2.0 * pd.to_numeric(out["cost"], errors="coerce") / pd.to_numeric(
        out["n"], errors="coerce"
    ).clip(lower=1)
    t_far_s = MASK_FAR_TIME_MS * 1e-3
    log_params_far = out[["log_k_sqrt", "log_k_quarter", "log_t0", "log_s"]].apply(
        pd.to_numeric, errors="coerce"
    )
    out["penetration_far_mm"] = np.nan
    valid_far = (
        out["success"].fillna(False)
        & np.isfinite(log_params_far["log_k_sqrt"])
        & np.isfinite(log_params_far["log_k_quarter"])
        & np.isfinite(log_params_far["log_t0"])
        & np.isfinite(log_params_far["log_s"])
    )
    if valid_far.any():
        lp_far = log_params_far.loc[
            valid_far, ["log_k_sqrt", "log_k_quarter", "log_t0", "log_s"]
        ].to_numpy(dtype=float)
        k_sqrt_far = np.exp(lp_far[:, 0])
        k_quarter_far = np.exp(lp_far[:, 1])
        t0_far = np.exp(lp_far[:, 2]) + MIN_TI
        s_far = np.exp(lp_far[:, 3])
        w_far = expit((t_far_s - t0_far) / s_far)
        far_vals = (1.0 - w_far) * (k_sqrt_far * np.sqrt(t_far_s)) + w_far * (
            k_quarter_far * np.power(t_far_s, 0.25)
        )
        out.loc[valid_far, "penetration_far_mm"] = np.asarray(far_vals, dtype=float)

    if "t_max_s" in out.columns:
        t_max = pd.to_numeric(out["t_max_s"], errors="coerce")
    else:
        t0_numeric = pd.to_numeric(out["t0"], errors="coerce")
        fallback_tmax = np.nan if t0_numeric.notna().sum() == 0 else np.nanmax(t0_numeric)
        t_max = pd.Series(np.full(len(out), fallback_tmax), index=out.index)

    # hard checks from the notebook prototype
    out["mask_basic"] = (
        out["success"].fillna(False)
        & np.isfinite(pd.to_numeric(out["t0"], errors="coerce"))
        & np.isfinite(pd.to_numeric(out["rmse"], errors="coerce"))
        & np.isfinite(pd.to_numeric(out["cost_per_point"], errors="coerce"))
        & (pd.to_numeric(out["n"], errors="coerce") >= MASK_MIN_N)
        & (pd.to_numeric(out["t0"], errors="coerce") > 0)
        & (pd.to_numeric(out["t0"], errors="coerce") < t_max)
        & (pd.to_numeric(out["s"], errors="coerce") > 0)
        & (pd.to_numeric(out["s"], errors="coerce") < MASK_S_UPPER)
        & (pd.to_numeric(out["t0"], errors="coerce") < MASK_T0_UPPER)
    )
    out["mask_penetration_far"] = (
        np.isfinite(pd.to_numeric(out["penetration_far_mm"], errors="coerce"))
        & pd.to_numeric(out["penetration_far_mm"], errors="coerce").between(
            MASK_PENETRATION_LOWER_MM, MASK_PENETRATION_UPPER_MM
        )
    )

    # robust outlier checks (prototype: grouped by file_name)
    out["z_t0"] = out.groupby(list(group_cols), dropna=False)["t0"].transform(robust_z)
    out["z_rmse"] = out.groupby(list(group_cols), dropna=False)["rmse"].transform(
        lambda s: robust_z(np.log1p(pd.to_numeric(s, errors="coerce")))
    )
    out["z_cost"] = out.groupby(list(group_cols), dropna=False)["cost_per_point"].transform(
        lambda s: robust_z(np.log1p(pd.to_numeric(s, errors="coerce")))
    )

    out["mask_outlier"] = (
        out["z_t0"].abs().gt(z_thresh)
        | out["z_rmse"].abs().gt(z_thresh)
        | out["z_cost"].abs().gt(z_thresh)
    ).fillna(False)

    out["flag_bad_fit"] = (~out["mask_basic"]) | (~out["mask_penetration_far"]) | out["mask_outlier"]

    clean_df = out.loc[~out["flag_bad_fit"]].copy()
    flagged_df = out.loc[out["flag_bad_fit"]].copy()
    return out, clean_df, flagged_df


def penetration_cleaning(
    arr,
    scaling_factor,
    diff_threshold_lower=1.0,
    diff_threshold_upper=np.inf,
    hd_upper_lim=15,
):
    arr = np.asarray(arr, dtype=float).copy()
    penetration_delay = 0

    if arr.size <= 1:
        return arr * scaling_factor, penetration_delay

    scan_limit = min(hd_upper_lim, arr.size - 1)
    for f in range(scan_limit):
        if arr[f + 1] == 0 or np.isnan(arr[f + 1]):
            penetration_delay += 1
            arr[f] = np.nan

    if penetration_delay > 0:
        # Shift left without wraparound; fill tail with NaN.
        arr_shifted = np.full_like(arr, np.nan, dtype=float)
        if penetration_delay < arr.size:
            arr_shifted[:-penetration_delay] = arr[penetration_delay:]
        arr = arr_shifted

    arr_diff = np.diff(arr)
    lower_cut_idx = np.where(arr_diff < diff_threshold_lower)[0]
    if lower_cut_idx.size > 0:
        arr = arr[: lower_cut_idx[0].item()]

    # Remove frames onward if a sudden unrealistically large jump appears.
    arr_diff = np.diff(arr)
    upper_cut_idx = np.where(arr_diff > diff_threshold_upper)[0]
    if upper_cut_idx.size > 0:
        arr = arr[: upper_cut_idx[0].item()]

    arr *= scaling_factor
    return arr, penetration_delay


def spray_penetration_model_sigmoid(params, t):
    """
    params (log-space): [log_k_sqrt, log_k_quarter, log_t0, log_s]
    """
    log_k_sqrt, log_k_quarter, log_t0, log_s = params

    k_sqrt = np.exp(log_k_sqrt)
    k_quarter = np.exp(log_k_quarter)
    t0 = np.exp(log_t0) + MIN_TI
    s = np.exp(log_s)

    t = np.clip(np.asarray(t, dtype=float), 1e-9, None)
    sqrt_segment = k_sqrt * np.sqrt(t)
    quarter_root_segment = k_quarter * np.power(t, 0.25)
    w = expit((t - t0) / s)

    return (1.0 - w) * sqrt_segment + w * quarter_root_segment


def fit_sigmoid(t, y, ti, x0):
    valid = np.isfinite(t) & np.isfinite(y)
    if valid.sum() < 4:
        return {
            "log_params": np.full(4, np.nan),
            "k_sqrt": np.nan,
            "k_quarter": np.nan,
            "t0": np.nan,
            "s": np.nan,
            "cost": np.inf,
            "success": False,
            "n": int(valid.sum()),
        }

    t_fit = t[valid]
    y_fit = y[valid]

    def residuals(params):
        y_hat = spray_penetration_model_sigmoid(params, t_fit)
        r = y_hat - y_fit
        if not np.all(np.isfinite(r)):
            return np.full_like(y_fit, 1e6, dtype=float)
        return r

    res = least_squares(residuals, x0, method="trf", loss="huber", f_scale=1.0)
    log_k_sqrt, log_k_quarter, log_t0, log_s = res.x

    return {
        "log_params": res.x,
        "k_sqrt": float(np.exp(log_k_sqrt)),
        "k_quarter": float(np.exp(log_k_quarter)),
        "t0": float(np.exp(log_t0) + MIN_TI),
        "s": float(np.exp(log_s)),
        "cost": float(res.cost),
        "success": bool(res.success),
        "n": int(valid.sum()),
    }


def prepare_cleaned_series(
    df_file,
    mm_per_px_scale,
    fps_default,
    max_hydraulic_delay_frames,
    delay_clip_half_window,
    diff_threshold_lower=DIFF_THRESHOLD_LOWER,
    diff_threshold_upper=DIFF_THRESHOLD_UPPER,
):
    number_of_plumes = int(df_file["plumes"].iloc[0])
    fps = float(df_file["fps"].iloc[0])
    if np.isnan(fps):
        fps = fps_default
    frame_idx = np.asarray(df_file["frame_idx"]).astype(int)

    time_s = frame_idx / fps
    time_ms = time_s * 1e3

    tilt_ang = (180.0 - float(df_file["umbrella_angle_deg"].iloc[0])) / 2.0
    umbrella_angle_correction = 1.0 / np.cos(np.deg2rad(tilt_ang))
    pen_correction = mm_per_px_scale * umbrella_angle_correction

    max_len = int(frame_idx.max()) + 1
    cleaned_series = np.full((number_of_plumes, max_len), np.nan)
    delays_raw = np.full(number_of_plumes, np.nan, dtype=float)
    temp_series = [None] * number_of_plumes

    for plume_idx in range(number_of_plumes):
        col = f"penetration_highpass_bw_plume_{plume_idx}"
        if col not in df_file.columns:
            continue

        arr = np.asarray(df_file[col], dtype=float).copy()
        cleaned_serie, delay = penetration_cleaning(
            arr,
            pen_correction,
            diff_threshold_lower=diff_threshold_lower,
            diff_threshold_upper=diff_threshold_upper,
            hd_upper_lim=max_hydraulic_delay_frames,
        )
        delays_raw[plume_idx] = delay
        temp_series[plume_idx] = np.asarray(cleaned_serie, dtype=float)

    valid_delays = delays_raw[np.isfinite(delays_raw)]
    median_delay = int(np.round(np.nanmedian(valid_delays))) if valid_delays.size else 0
    lower_bound = median_delay - int(delay_clip_half_window)
    upper_bound = median_delay + int(delay_clip_half_window)
    delays_used = np.full(number_of_plumes, median_delay, dtype=float)
    valid_raw_mask = np.isfinite(delays_raw)
    if np.any(valid_raw_mask):
        delays_used[valid_raw_mask] = np.clip(
            np.round(delays_raw[valid_raw_mask]),
            lower_bound,
            upper_bound,
        )

    # Align each plume using clipped raw delay to limit outlier shifts while
    # preserving plume-level delay variability.
    for plume_idx in range(number_of_plumes):
        series = temp_series[plume_idx]
        if series is None:
            continue

        plume_delay_raw = (
            int(np.round(delays_raw[plume_idx])) if np.isfinite(delays_raw[plume_idx]) else median_delay
        )
        plume_delay_used = int(np.round(delays_used[plume_idx]))
        delta = plume_delay_raw - plume_delay_used

        if delta > 0:
            # This plume was shifted too far left; shift right by delta.
            aligned = np.full_like(series, np.nan, dtype=float)
            if delta < series.size:
                aligned[delta:] = series[:-delta]
        elif delta < 0:
            # This plume was not shifted enough; shift left by -delta.
            shift_left = -delta
            aligned = series[shift_left:]
        else:
            aligned = series

        n = min(aligned.size, max_len)
        if n > 0:
            cleaned_series[plume_idx, :n] = aligned[:n]

    return time_s, time_ms, cleaned_series, delays_raw, delays_used


def save_fit_plot(
    folder,
    plot_df,
    csv_files,
    out_plot_dir,
    plot_kind,
    mm_per_px_scale,
    fps_default,
    max_hydraulic_delay_frames,
    delay_clip_half_window,
):
    cache = {}  # csv_path -> (time_s, time_ms, cleaned_series, inj_dur_s)

    # map filename and stem to path for fallback resolution
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

    plt.figure(figsize=(10, 6))
    has_curve = False
    rng = np.random.default_rng()

    for row in plot_df.itertuples(index=False):
        csv_path = resolve_csv(row)
        if csv_path is None:
            continue

        cache_key = str(csv_path.resolve())
        if cache_key not in cache:
            df_file = pd.read_csv(csv_path)
            time_s, time_ms, cleaned_series, _, _ = prepare_cleaned_series(
                df_file,
                mm_per_px_scale=mm_per_px_scale,
                fps_default=fps_default,
                max_hydraulic_delay_frames=max_hydraulic_delay_frames,
                delay_clip_half_window=delay_clip_half_window,
                diff_threshold_lower=DIFF_THRESHOLD_LOWER,
                diff_threshold_upper=DIFF_THRESHOLD_UPPER,
            )
            inj_dur_s = float(df_file["injection_duration_us"].iloc[0]) * 1e-6
            cache[cache_key] = (time_s, time_ms, cleaned_series, inj_dur_s)

        time_s, time_ms, cleaned_series, inj_dur_s = cache[cache_key]
        plume_idx = int(row.plume_idx)
        if plume_idx < 0 or plume_idx >= cleaned_series.shape[0]:
            continue

        raw_series = cleaned_series[plume_idx]
        valid_raw = np.isfinite(time_ms) & np.isfinite(raw_series)
        if not np.any(valid_raw):
            continue

        color = rng.random(3)

        plt.plot(time_ms[valid_raw], raw_series[valid_raw], alpha=0.65, linewidth=1.0, color=color)
        draw_fit = (
            bool(getattr(row, "success", False))
            and np.isfinite(getattr(row, "log_k_sqrt", np.nan))
            and np.isfinite(getattr(row, "log_k_quarter", np.nan))
            and np.isfinite(getattr(row, "log_t0", np.nan))
            and np.isfinite(getattr(row, "log_s", np.nan))
        )
        if draw_fit:
            t_end = float(np.nanmax(time_s) * PLOT_EXTRAP_FACTOR)
            t_extrap_s = np.linspace(0.0, t_end, PLOT_NUM_POINTS)
            log_params = [row.log_k_sqrt, row.log_k_quarter, row.log_t0, row.log_s]
            y_extrap = spray_penetration_model_sigmoid(log_params, t_extrap_s)
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

    plt.title(f"{folder.name}: {plot_kind} raw traces (solid) vs fitted curves (dashed)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Penetration (mm)")
    plt.grid(alpha=0.25)
    plt.ylim(0, PLOT_YLIM_MM)

    out_plot_path = out_plot_dir / f"{folder.name}.png"
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=140)
    plt.close()
    return out_plot_path


def process_folder(
    folder,
    out_all_dir,
    out_clean_dir,
    out_plots_clean_dir,
    out_plots_flagged_dir,
    mm_per_px_scale,
    fps_default,
    max_hydraulic_delay_frames,
    delay_clip_half_window,
):
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"Skip {folder.name}: no csv files.")
        return

    rows = []
    for file_path in csv_files:
        df_file = pd.read_csv(file_path)
        time_s, _, cleaned_series, delays_raw, delays_used = prepare_cleaned_series(
            df_file,
            mm_per_px_scale=mm_per_px_scale,
            fps_default=fps_default,
            max_hydraulic_delay_frames=max_hydraulic_delay_frames,
            delay_clip_half_window=delay_clip_half_window,
            diff_threshold_lower=DIFF_THRESHOLD_LOWER,
            diff_threshold_upper=DIFF_THRESHOLD_UPPER,
        )

        meta = {}
        for col in META_COLS:
            meta[col] = df_file[col].iloc[0] if col in df_file.columns else np.nan

        inj_dur_s = float(meta["injection_duration_us"]) * 1e-6
        x0 = np.log([1.0, 1.0, max(2.0 * inj_dur_s, 1e-9), 1.0])

        number_of_plumes = cleaned_series.shape[0]
        t_max_s = float(np.nanmax(time_s))
        for plume_idx in range(number_of_plumes):
            series = cleaned_series[plume_idx]
            fit = fit_sigmoid(time_s, series, inj_dur_s, x0)
            log_k_sqrt, log_k_quarter, log_t0, log_s = fit["log_params"]

            valid = np.isfinite(time_s) & np.isfinite(series)
            if np.any(valid):
                y_true = series[valid]
                y_hat = spray_penetration_model_sigmoid(
                    [log_k_sqrt, log_k_quarter, log_t0, log_s], time_s[valid]
                )
                rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
                ss_res = float(np.sum((y_true - y_hat) ** 2))
                ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
            else:
                rmse = np.nan
                r2 = np.nan

            rows.append(
                {
                    "file_path": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "file_stem": file_path.stem,
                    "plume_idx": plume_idx,
                    "delay_frames": delays_used[plume_idx],
                    "delay_frames_raw": delays_raw[plume_idx],
                    "delay_frames_used": delays_used[plume_idx],
                    "k_sqrt": fit["k_sqrt"],
                    "k_quarter": fit["k_quarter"],
                    "t0": fit["t0"],
                    "s": fit["s"],
                    "cost": fit["cost"],
                    "success": fit["success"],
                    "n": fit["n"],
                    "rmse": rmse,
                    "r2": r2,
                    "log_k_sqrt": log_k_sqrt,
                    "log_k_quarter": log_k_quarter,
                    "log_t0": log_t0,
                    "log_s": log_s,
                    "t_max_s": t_max_s,
                    **meta,
                }
            )

    results_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    masked_df, clean_df, flagged_df = apply_filter_masking(results_df, group_cols=MASK_GROUP_COLS)

    out_all_path = out_all_dir / f"{folder.name}.csv"
    out_clean_path = out_clean_dir / f"{folder.name}.csv"
    out_flagged_path = out_all_dir / f"{folder.name}_flagged.csv"
    out_plot_clean_path = save_fit_plot(
        folder,
        clean_df,
        csv_files,
        out_plots_clean_dir,
        "clean",
        mm_per_px_scale=mm_per_px_scale,
        fps_default=fps_default,
        max_hydraulic_delay_frames=max_hydraulic_delay_frames,
        delay_clip_half_window=delay_clip_half_window,
    )
    out_plot_flagged_path = save_fit_plot(
        folder,
        flagged_df,
        csv_files,
        out_plots_flagged_dir,
        "flagged",
        mm_per_px_scale=mm_per_px_scale,
        fps_default=fps_default,
        max_hydraulic_delay_frames=max_hydraulic_delay_frames,
        delay_clip_half_window=delay_clip_half_window,
    )

    masked_df.to_csv(out_all_path, index=False)
    clean_df.to_csv(out_clean_path, index=False)
    flagged_df.to_csv(out_flagged_path, index=False)

    print(
        f"Saved {out_all_path.name} ({len(masked_df)} total rows), "
        f"{out_clean_path.name} ({len(clean_df)} clean), "
        f"{out_flagged_path.name} ({len(flagged_df)} flagged), "
        f"{out_plot_clean_path.name} (clean-curve plot), "
        f"{out_plot_flagged_path.name} (flagged-curve plot) from {len(csv_files)} files"
    )


def get_dataset_settings(name):
    if name == "BC20220627 - Heinzman DS300 - Mie Top view":
        return {
            "or_mm_per_px_reference": 412.0,
            "fps_default": 34000,
            "max_hydraulic_delay_frames": 25,
            "delay_clip_half_window": 2,
        }
    return {
        "or_mm_per_px_reference": 376.0,  # 90 mm reference in px
        "fps_default": 25000,
        "max_hydraulic_delay_frames": 15,
        "delay_clip_half_window": 2,
    }


def main():
    for name in names:
        settings = get_dataset_settings(name)
        mm_per_px_scale = 90.0 / settings["or_mm_per_px_reference"]
        root = data_root / name
        out_dir = data_out_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_all_dir = out_dir / "all"
        out_clean_dir = out_dir / "clean"
        out_plots_clean_dir = out_dir / "plots_clean"
        out_plots_flagged_dir = out_dir / "plots_flagged"
        out_all_dir.mkdir(parents=True, exist_ok=True)
        out_clean_dir.mkdir(parents=True, exist_ok=True)
        out_plots_clean_dir.mkdir(parents=True, exist_ok=True)
        out_plots_flagged_dir.mkdir(parents=True, exist_ok=True)

        subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if not subdirs:
            raise FileNotFoundError(f"No subdirs found in {root}")

        print(f"Found {len(subdirs)} subdirs in {root}")
        for folder in subdirs:
            process_folder(
                folder,
                out_all_dir=out_all_dir,
                out_clean_dir=out_clean_dir,
                out_plots_clean_dir=out_plots_clean_dir,
                out_plots_flagged_dir=out_plots_flagged_dir,
                mm_per_px_scale=mm_per_px_scale,
                fps_default=settings["fps_default"],
                max_hydraulic_delay_frames=settings["max_hydraulic_delay_frames"],
                delay_clip_half_window=settings["delay_clip_half_window"],
            )


if __name__ == "__main__":
    main()
