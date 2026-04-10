import re

with open(r"c:\Users\Jiang\Documents\Mie_Postprocessing_Py\MLP\fit_raw_data.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add MIN_INITIAL_VELOCITY and calculate_subframe_delay
replacement1 = """NUM_POINTS_SOI_LINEAR_REGRESSION = 3
MIN_INITIAL_VELOCITY = 1e-6

def calculate_subframe_delay(time_s, series, first_valid_idx, n_points=3, fallback_delay_s=float('nan')):
    max_idx = len(time_s)
    
    t_early = []
    y_early = []
    for i in range(first_valid_idx, max_idx):
        if len(t_early) >= n_points:
            break
        if np.isfinite(series[i]) and series[i] > 0:
            t_early.append(time_s[i])
            y_early.append(series[i])
            
    if len(t_early) < 2:
        return fallback_delay_s
        
    t_early_arr = np.array(t_early)
    y_early_arr = np.array(y_early)
    
    slope, intercept = np.polyfit(t_early_arr, y_early_arr, 1)
    if slope > MIN_INITIAL_VELOCITY:
        t_intercept = -intercept / slope
        if 0 <= t_intercept <= t_early_arr[-1]:
            return float(t_intercept)
    return fallback_delay_s

# Input/output roots"""
content = content.replace("NUM_POINTS_SOI_LINEAR_REGRESSION = 3\n\n# Input/output roots", replacement1)


# 2. Replace penetration_cleaning
old_cleaning = """def penetration_cleaning(
    arr,
    scaling_factor,
    diff_threshold_lower=1.0,
    diff_threshold_upper=np.inf,
    hd_upper_lim=15,
    forced_delay=None,
    replace_negative_with_zero=False,
):
    arr = np.asarray(arr, dtype=float).copy()
    penetration_delay = 0 if forced_delay is None else int(forced_delay)

    if forced_delay is None and ENABLE_HYDRAULIC_DELAY_SCAN:
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

    if replace_negative_with_zero and ENABLE_REPLACE_NEGATIVE_WITH_ZERO:
        arr[arr < 0] = 0.0

    positive_idx = np.flatnonzero(np.isfinite(arr) & (arr > 0))
    if positive_idx.size == 0:
        arr[:] = np.nan
        return arr * scaling_factor, penetration_delay

    first_positive_idx = int(positive_idx[0])
    if first_positive_idx > 0:
        # Pre-onset placeholders after delay alignment should not influence
        # threshold cuts or the downstream fit.
        arr[:first_positive_idx] = np.nan

    if ENABLE_DIFF_THRESHOLD_LOWER:
        arr_diff = np.diff(arr[first_positive_idx:])
        lower_cut_idx = np.where(arr_diff < diff_threshold_lower)[0]
        if lower_cut_idx.size > 0:
            arr = arr[: first_positive_idx + lower_cut_idx[0].item() + 1]

    # Remove frames onward if a sudden unrealistically large jump appears.
    if ENABLE_DIFF_THRESHOLD_UPPER:
        valid_positive_idx = np.flatnonzero(np.isfinite(arr) & (arr > 0))
        if valid_positive_idx.size == 0:
            arr[:] = np.nan
            return arr * scaling_factor, penetration_delay
        first_positive_idx = int(valid_positive_idx[0])
        arr_diff = np.diff(arr[first_positive_idx:])
        upper_cut_idx = np.where(arr_diff > diff_threshold_upper)[0]
        if upper_cut_idx.size > 0:
            arr = arr[: first_positive_idx + upper_cut_idx[0].item() + 1]

    arr *= scaling_factor
    return arr, penetration_delay"""

new_cleaning = """def penetration_cleaning(
    arr,
    scaling_factor,
    diff_threshold_lower=1.0,
    diff_threshold_upper=np.inf,
    hd_upper_lim=15,
    forced_delay=None,
    replace_negative_with_zero=False,
):
    arr = np.asarray(arr, dtype=float).copy()
    first_positive_idx = 0 if forced_delay is None else int(forced_delay)

    if forced_delay is None and ENABLE_HYDRAULIC_DELAY_SCAN:
        scan_limit = min(hd_upper_lim, arr.size - 1)
        for f in range(scan_limit):
            if arr[f + 1] == 0 or np.isnan(arr[f + 1]):
                first_positive_idx += 1
                arr[f] = np.nan

    if replace_negative_with_zero and ENABLE_REPLACE_NEGATIVE_WITH_ZERO:
        arr[arr < 0] = 0.0

    if first_positive_idx > 0 and first_positive_idx < arr.size:
        arr[:first_positive_idx] = np.nan

    positive_idx = np.flatnonzero(np.isfinite(arr) & (arr > 0))
    if positive_idx.size == 0:
        arr[:] = np.nan
        return arr * scaling_factor, first_positive_idx

    actual_first_positive_idx = int(positive_idx[0])

    if ENABLE_DIFF_THRESHOLD_LOWER:
        arr_diff = np.diff(arr[actual_first_positive_idx:])
        lower_cut_idx = np.where(arr_diff < diff_threshold_lower)[0]
        if lower_cut_idx.size > 0:
            arr = arr[: actual_first_positive_idx + lower_cut_idx[0].item() + 1]

    if ENABLE_DIFF_THRESHOLD_UPPER:
        valid_positive_idx = np.flatnonzero(np.isfinite(arr) & (arr > 0))
        if valid_positive_idx.size == 0:
            arr[:] = np.nan
            return arr * scaling_factor, first_positive_idx
        actual_first_positive_idx = int(valid_positive_idx[0])
        arr_diff = np.diff(arr[actual_first_positive_idx:])
        upper_cut_idx = np.where(arr_diff > diff_threshold_upper)[0]
        if upper_cut_idx.size > 0:
            arr = arr[: actual_first_positive_idx + upper_cut_idx[0].item() + 1]

    arr *= scaling_factor
    return arr, first_positive_idx"""
content = content.replace(old_cleaning, new_cleaning)


# 3. Replace prepare_cleaned_series core logic
old_prepare_loop = """    for plume_idx in range(number_of_plumes):
        col = f"{penetration_column_prefix}{plume_idx}"
        if col not in df_file.columns:
            continue

        area_delay, delay_source = get_area_based_delay(df_file, plume_idx)
        arr = np.asarray(df_file[col], dtype=float).copy()
        cleaned_serie, delay = penetration_cleaning(
            arr,
            pen_correction,
            diff_threshold_lower=diff_threshold_lower,
            diff_threshold_upper=diff_threshold_upper,
            hd_upper_lim=max_hydraulic_delay_frames,
            forced_delay=area_delay if np.isfinite(area_delay) else None,
            replace_negative_with_zero=replace_negative_with_zero,
        )
        delays_raw[plume_idx] = delay
        delay_sources[plume_idx] = delay_source if np.isfinite(area_delay) else "penetration_fallback"
        temp_series[plume_idx] = np.asarray(cleaned_serie, dtype=float)

    valid_delays = delays_raw[np.isfinite(delays_raw)]
    median_delay = int(np.round(np.nanmedian(valid_delays))) if valid_delays.size else 0
    delays_used = np.full(number_of_plumes, median_delay, dtype=float)
    valid_raw_mask = np.isfinite(delays_raw)
    if ENABLE_DELAY_CLIP:
        lower_bound = median_delay - int(delay_clip_half_window)
        upper_bound = median_delay + int(delay_clip_half_window)
        if np.any(valid_raw_mask):
            delays_used[valid_raw_mask] = np.clip(
                np.round(delays_raw[valid_raw_mask]),
                lower_bound,
                upper_bound,
            )
    elif np.any(valid_raw_mask):
        delays_used[valid_raw_mask] = np.round(delays_raw[valid_raw_mask])

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

    return time_s, time_ms, cleaned_series, delays_raw, delays_used, delay_sources"""

new_prepare_loop = """    for plume_idx in range(number_of_plumes):
        col = f"{penetration_column_prefix}{plume_idx}"
        if col not in df_file.columns:
            continue

        area_delay, delay_source = get_area_based_delay(df_file, plume_idx)
        arr = np.asarray(df_file[col], dtype=float).copy()
        cleaned_serie, first_pos_idx = penetration_cleaning(
            arr,
            pen_correction,
            diff_threshold_lower=diff_threshold_lower,
            diff_threshold_upper=diff_threshold_upper,
            hd_upper_lim=max_hydraulic_delay_frames,
            forced_delay=area_delay if np.isfinite(area_delay) else None,
            replace_negative_with_zero=replace_negative_with_zero,
        )
        
        fallback_delay_s = float(first_pos_idx) / fps
        delay_s = calculate_subframe_delay(
            time_s, cleaned_serie, first_pos_idx, 
            n_points=NUM_POINTS_SOI_LINEAR_REGRESSION, 
            fallback_delay_s=fallback_delay_s
        )
        delays_raw[plume_idx] = delay_s
        delay_sources[plume_idx] = delay_source if np.isfinite(area_delay) else "penetration_fallback"
        temp_series[plume_idx] = np.asarray(cleaned_serie, dtype=float)

    valid_delays = delays_raw[np.isfinite(delays_raw)]
    median_delay_s = np.nanmedian(valid_delays) if valid_delays.size else 0.0
    delays_used = np.full(number_of_plumes, median_delay_s, dtype=float)
    valid_raw_mask = np.isfinite(delays_raw)
    if ENABLE_DELAY_CLIP:
        lower_bound = median_delay_s - (float(delay_clip_half_window) / fps)
        upper_bound = median_delay_s + (float(delay_clip_half_window) / fps)
        if np.any(valid_raw_mask):
            delays_used[valid_raw_mask] = np.clip(
                delays_raw[valid_raw_mask],
                lower_bound,
                upper_bound,
            )
    elif np.any(valid_raw_mask):
        delays_used[valid_raw_mask] = delays_raw[valid_raw_mask]

    time_s_aligned = np.full((number_of_plumes, max_len), np.nan)
    time_ms_aligned = np.full((number_of_plumes, max_len), np.nan)

    for plume_idx in range(number_of_plumes):
        series = temp_series[plume_idx]
        if series is None:
            continue

        plume_delay_s = delays_used[plume_idx]
        delay_frames_int = int(np.round(plume_delay_s * fps))

        aligned = np.full_like(series, np.nan, dtype=float)
        if delay_frames_int > 0 and delay_frames_int < series.size:
            aligned[:-delay_frames_int] = series[delay_frames_int:]
        elif delay_frames_int == 0:
            aligned = series
        elif delay_frames_int < 0 and -delay_frames_int < series.size:
            aligned[-delay_frames_int:] = series[:delay_frames_int]

        n = min(aligned.size, max_len)
        if n > 0:
            cleaned_series[plume_idx, :n] = aligned[:n]
            t_exact = (np.arange(n) + delay_frames_int) / fps - plume_delay_s
            time_s_aligned[plume_idx, :n] = t_exact
            time_ms_aligned[plume_idx, :n] = t_exact * 1000.0

    return time_s_aligned, time_ms_aligned, cleaned_series, delays_raw * fps, delays_used * fps, delay_sources"""
content = content.replace(old_prepare_loop, new_prepare_loop)

# 4. Replace collect_series_rows
old_collect = """def collect_series_rows(
    file_path,
    time_s,
    time_ms,
    cleaned_series,
    delays_raw,
    delays_used,
    delay_sources,
):
    rows = []
    file_path = Path(file_path)
    for plume_idx in range(cleaned_series.shape[0]):
        series = np.asarray(cleaned_series[plume_idx], dtype=float)
        valid = np.isfinite(time_s) & np.isfinite(time_ms) & np.isfinite(series)
        if not np.any(valid):
            continue

        for idx in np.flatnonzero(valid):
            rows.append(
                {
                    "file_path": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "file_stem": file_path.stem,
                    "plume_idx": plume_idx,
                    "frame_pos": int(idx),
                    "time_s": float(time_s[idx]),
                    "time_ms": float(time_ms[idx]),
                    "penetration_mm": float(series[idx]),
                    "delay_frames_raw": delays_raw[plume_idx],
                    "delay_frames_used": delays_used[plume_idx],
                    "delay_source": delay_sources[plume_idx],
                }
            )
    return rows"""

new_collect = """def collect_series_rows(
    file_path,
    time_s,
    time_ms,
    cleaned_series,
    delays_raw,
    delays_used,
    delay_sources,
):
    rows = []
    file_path = Path(file_path)
    for plume_idx in range(cleaned_series.shape[0]):
        series = np.asarray(cleaned_series[plume_idx], dtype=float)
        ts = time_s[plume_idx]
        tms = time_ms[plume_idx]
        valid = np.isfinite(ts) & np.isfinite(tms) & np.isfinite(series)
        if not np.any(valid):
            continue

        for idx in np.flatnonzero(valid):
            rows.append(
                {
                    "file_path": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "file_stem": file_path.stem,
                    "plume_idx": plume_idx,
                    "frame_pos": int(idx),
                    "time_s": float(ts[idx]),
                    "time_ms": float(tms[idx]),
                    "penetration_mm": float(series[idx]),
                    "delay_frames_raw": delays_raw[plume_idx],
                    "delay_frames_used": delays_used[plume_idx],
                    "delay_source": delay_sources[plume_idx],
                }
            )
    return rows"""
content = content.replace(old_collect, new_collect)

# 5. Update save_fit_plot for 2D time array
old_save_fit = """        time_s, time_ms, cleaned_series, inj_dur_s = cache[cache_key]
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
            t_extrap_s = np.linspace(0.0, t_end, PLOT_NUM_POINTS)"""

new_save_fit = """        time_s, time_ms, cleaned_series, inj_dur_s = cache[cache_key]
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
        draw_fit = (
            bool(getattr(row, "success", False))
            and np.isfinite(getattr(row, "log_k_sqrt", np.nan))
            and np.isfinite(getattr(row, "log_k_quarter", np.nan))
            and np.isfinite(getattr(row, "log_t0", np.nan))
            and np.isfinite(getattr(row, "log_s", np.nan))
        )
        if draw_fit:
            t_end = float(np.nanmax(ts) * PLOT_EXTRAP_FACTOR)
            t_extrap_s = np.linspace(0.0, t_end, PLOT_NUM_POINTS)"""
content = content.replace(old_save_fit, new_save_fit)

# 6. Add save_raw_plot function below save_fit_plot
save_raw_plot_func = """def save_raw_plot(
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
    cache = {}

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

        ts = time_s[plume_idx]
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
        plt.text(1.0, 0.5, f"No traces for this folder", ha="center", va="center")

    plt.title(
        f"{folder.name}: {penetration_source['label']} aligned raw traces (0-2ms)"
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Penetration (mm)")
    plt.grid(alpha=0.25)
    plt.xlim(0, 2.0)
    plt.ylim(0, PLOT_YLIM_MM)

    out_plot_path = out_plot_dir / f"{folder.name}.png"
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=140)
    plt.close()
    return out_plot_path

def process_folder"""
content = content.replace("def process_folder", save_raw_plot_func)


# 7. Update process_folder signature and loop for 2D time_s
old_process_folder_def = """def process_folder(
    folder,
    out_all_dir,
    out_clean_dir,
    out_series_all_dir,
    out_series_clean_dir,
    out_series_wide_all_dir,
    out_series_wide_clean_dir,
    out_plots_clean_dir,
    out_plots_flagged_dir,
    penetration_source,"""
new_process_folder_def = """def process_folder(
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
    penetration_source,"""
content = content.replace(old_process_folder_def, new_process_folder_def)

old_loop = """        number_of_plumes = cleaned_series.shape[0]
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
                )"""

new_loop = """        number_of_plumes = cleaned_series.shape[0]
        for plume_idx in range(number_of_plumes):
            series = cleaned_series[plume_idx]
            ts = time_s[plume_idx]
            t_max_s = float(np.nanmax(ts)) if np.any(np.isfinite(ts)) else float('nan')
            fit = fit_sigmoid(ts, series, inj_dur_s, x0)
            log_k_sqrt, log_k_quarter, log_t0, log_s = fit["log_params"]

            valid = np.isfinite(ts) & np.isfinite(series)
            if np.any(valid):
                y_true = series[valid]
                y_hat = spray_penetration_model_sigmoid(
                    [log_k_sqrt, log_k_quarter, log_t0, log_s], ts[valid]
                )"""
content = content.replace(old_loop, new_loop)


# 8. Call save_raw_plot and print update
old_save_plot_flagged = """    out_plot_flagged_path = save_fit_plot(
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
    )"""

new_save_plot_flagged = """    out_plot_flagged_path = save_fit_plot(
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

    out_plot_raw_all_path = save_raw_plot(
        folder,
        results_df,
        csv_files,
        out_plots_raw_all_dir,
        penetration_source,
        mm_per_px_scale=mm_per_px_scale,
        fps_default=fps_default,
        max_hydraulic_delay_frames=max_hydraulic_delay_frames,
        delay_clip_half_window=delay_clip_half_window,
    )"""
content = content.replace(old_save_plot_flagged, new_save_plot_flagged)

old_print = """        f"{out_plot_clean_path.name} (clean-curve plot), "
        f"{out_plot_flagged_path.name} (flagged-curve plot) from {len(csv_files)} files"
    )"""
new_print = """        f"{out_plot_clean_path.name} (clean-curve plot), "
        f"{out_plot_flagged_path.name} (flagged-curve plot), "
        f"{out_plot_raw_all_path.name} (raw-all plot) from {len(csv_files)} files"
    )"""
content = content.replace(old_print, new_print)


# 9. Update main
old_main = """                out_plots_clean_dir = metric_out_dir / "plots_clean"
                out_plots_flagged_dir = metric_out_dir / "plots_flagged"
                out_all_dir.mkdir(parents=True, exist_ok=True)
                out_clean_dir.mkdir(parents=True, exist_ok=True)
                out_series_all_dir.mkdir(parents=True, exist_ok=True)
                out_series_clean_dir.mkdir(parents=True, exist_ok=True)
                out_series_wide_all_dir.mkdir(parents=True, exist_ok=True)
                out_series_wide_clean_dir.mkdir(parents=True, exist_ok=True)
                out_plots_clean_dir.mkdir(parents=True, exist_ok=True)
                out_plots_flagged_dir.mkdir(parents=True, exist_ok=True)

                process_folder(
                    folder,
                    out_all_dir=out_all_dir,
                    out_clean_dir=out_clean_dir,
                    out_series_all_dir=out_series_all_dir,
                    out_series_clean_dir=out_series_clean_dir,
                    out_series_wide_all_dir=out_series_wide_all_dir,
                    out_series_wide_clean_dir=out_series_wide_clean_dir,
                    out_plots_clean_dir=out_plots_clean_dir,
                    out_plots_flagged_dir=out_plots_flagged_dir,"""

new_main = """                out_plots_clean_dir = metric_out_dir / "plots_clean"
                out_plots_flagged_dir = metric_out_dir / "plots_flagged"
                out_plots_raw_all_dir = metric_out_dir / "plots_raw_all"
                out_all_dir.mkdir(parents=True, exist_ok=True)
                out_clean_dir.mkdir(parents=True, exist_ok=True)
                out_series_all_dir.mkdir(parents=True, exist_ok=True)
                out_series_clean_dir.mkdir(parents=True, exist_ok=True)
                out_series_wide_all_dir.mkdir(parents=True, exist_ok=True)
                out_series_wide_clean_dir.mkdir(parents=True, exist_ok=True)
                out_plots_clean_dir.mkdir(parents=True, exist_ok=True)
                out_plots_flagged_dir.mkdir(parents=True, exist_ok=True)
                out_plots_raw_all_dir.mkdir(parents=True, exist_ok=True)

                process_folder(
                    folder,
                    out_all_dir=out_all_dir,
                    out_clean_dir=out_clean_dir,
                    out_series_all_dir=out_series_all_dir,
                    out_series_clean_dir=out_series_clean_dir,
                    out_series_wide_all_dir=out_series_wide_all_dir,
                    out_series_wide_clean_dir=out_series_wide_clean_dir,
                    out_plots_clean_dir=out_plots_clean_dir,
                    out_plots_flagged_dir=out_plots_flagged_dir,
                    out_plots_raw_all_dir=out_plots_raw_all_dir,"""
content = content.replace(old_main, new_main)

with open(r"c:\Users\Jiang\Documents\Mie_Postprocessing_Py\MLP\fit_raw_data.py", "w", encoding="utf-8") as f:
    f.write(content)
print("done")
