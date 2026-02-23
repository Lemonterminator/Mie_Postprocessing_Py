from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit
import matplotlib.pyplot as plt



# Input/output roots
root = Path(r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241016_HZ_Nozzle8")
out_dir = Path(r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\MLP\synthetic_data\BC20241016_HZ_Nozzle8")


# Image processing settings
OR_MM_PER_PX_REFERENCE = 376.0  # 90 mm reference in px
DIFF_THRESHOLD = 2.0  # px
MM_PER_PX_SCALE = 90.0 / OR_MM_PER_PX_REFERENCE
MIN_TI = 0.0

# Filtering/masking settings (from notebook prototype defaults)
MASK_GROUP_COLS = ("file_name",)
MASK_Z_THRESH = 3.0
MASK_MIN_N = 10
MASK_S_UPPER = 1e-3  # < 1 ms
MASK_T0_UPPER = 0.8e-3 

PLOT_EXTRAP_FACTOR = 1.6
PLOT_NUM_POINTS = 300
PLOT_YLIM_MM = 200.0


out_dir.mkdir(parents=True, exist_ok=True)
out_all_dir = out_dir / "all"
out_clean_dir = out_dir / "clean"
out_plots_clean_dir = out_dir / "plots_clean"
out_all_dir.mkdir(parents=True, exist_ok=True)
out_clean_dir.mkdir(parents=True, exist_ok=True)
out_plots_clean_dir.mkdir(parents=True, exist_ok=True)



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
        out["z_t0"] = np.nan
        out["z_rmse"] = np.nan
        out["z_cost"] = np.nan
        out["mask_basic"] = False
        out["mask_outlier"] = False
        out["flag_bad_fit"] = False
        return out, out.copy(), out.copy()

    out["cost_per_point"] = 2.0 * pd.to_numeric(out["cost"], errors="coerce") / pd.to_numeric(
        out["n"], errors="coerce"
    ).clip(lower=1)

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

    out["flag_bad_fit"] = (~out["mask_basic"]) | out["mask_outlier"]

    clean_df = out.loc[~out["flag_bad_fit"]].copy()
    flagged_df = out.loc[out["flag_bad_fit"]].copy()
    return out, clean_df, flagged_df


def penetration_cleaning(arr, scaling_factor, diff_threshold=1.0, hd_upper_lim=15):
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
    valid_idx = np.where(arr_diff < diff_threshold)[0]
    if valid_idx.size > 0:
        arr = arr[: valid_idx[0].item()]

    arr *= scaling_factor
    return arr, penetration_delay


def spray_penetration_model_sigmoid(params, t, ti):
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
        y_hat = spray_penetration_model_sigmoid(params, t_fit, ti)
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


def prepare_cleaned_series(df_file, diff_threshold=2.0):
    number_of_plumes = int(df_file["plumes"].iloc[0])
    fps = float(df_file["fps"].iloc[0])
    frame_idx = np.asarray(df_file["frame_idx"]).astype(int)

    time_s = frame_idx / fps
    time_ms = time_s * 1e3

    tilt_ang = (180.0 - float(df_file["umbrella_angle_deg"].iloc[0])) / 2.0
    umbrella_angle_correction = 1.0 / np.cos(np.deg2rad(tilt_ang))
    pen_correction = MM_PER_PX_SCALE * umbrella_angle_correction

    cleaned_series = np.full((number_of_plumes, int(frame_idx.max()) + 1), np.nan)
    delays = np.zeros(number_of_plumes, dtype=float)

    for plume_idx in range(number_of_plumes):
        col = f"penetration_highpass_bw_plume_{plume_idx}"
        if col not in df_file.columns:
            continue

        arr = np.asarray(df_file[col], dtype=float).copy()
        cleaned_serie, delay = penetration_cleaning(
            arr, pen_correction, diff_threshold=diff_threshold
        )

        delays[plume_idx] = delay
        n = min(len(cleaned_serie), cleaned_series.shape[1])
        cleaned_series[plume_idx, :n] = cleaned_serie[:n]

    return time_s, time_ms, cleaned_series, delays


def save_clean_plot(folder, clean_df, csv_files):
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

    for row in clean_df.itertuples(index=False):
        csv_path = resolve_csv(row)
        if csv_path is None:
            continue

        cache_key = str(csv_path.resolve())
        if cache_key not in cache:
            df_file = pd.read_csv(csv_path)
            time_s, time_ms, cleaned_series, _ = prepare_cleaned_series(
                df_file, diff_threshold=DIFF_THRESHOLD
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

        t_end = float(np.nanmax(time_s) * PLOT_EXTRAP_FACTOR)
        t_extrap_s = np.linspace(0.0, t_end, PLOT_NUM_POINTS)
        log_params = [row.log_k_sqrt, row.log_k_quarter, row.log_t0, row.log_s]
        y_extrap = spray_penetration_model_sigmoid(log_params, t_extrap_s, inj_dur_s)

        plt.plot(time_ms[valid_raw], raw_series[valid_raw], alpha=0.65, linewidth=1.0)
        plt.plot(1e3 * t_extrap_s, y_extrap, linestyle="--", alpha=0.45, linewidth=1.0)
        has_curve = True

    if not has_curve:
        plt.text(0.5, 0.5, "No clean fits for this folder", ha="center", va="center")

    plt.title(f"{folder.name}: clean raw traces (solid) vs clean fitted curves (dashed)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Penetration (mm)")
    plt.grid(alpha=0.25)
    plt.ylim(0, PLOT_YLIM_MM)

    out_plot_path = out_plots_clean_dir / f"{folder.name}.png"
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=140)
    plt.close()
    return out_plot_path


def process_folder(folder):
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"Skip {folder.name}: no csv files.")
        return

    rows = []
    for file_path in csv_files:
        df_file = pd.read_csv(file_path)
        time_s, _, cleaned_series, delays = prepare_cleaned_series(
            df_file, diff_threshold=DIFF_THRESHOLD
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
                    [log_k_sqrt, log_k_quarter, log_t0, log_s], time_s[valid], inj_dur_s
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
                    "delay_frames": delays[plume_idx],
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
    out_plot_path = save_clean_plot(folder, clean_df, csv_files)

    masked_df.to_csv(out_all_path, index=False)
    clean_df.to_csv(out_clean_path, index=False)
    flagged_df.to_csv(out_flagged_path, index=False)

    print(
        f"Saved {out_all_path} ({len(masked_df)} total rows), "
        f"{out_clean_path} ({len(clean_df)} clean), "
        f"{out_flagged_path} ({len(flagged_df)} flagged), "
        f"{out_plot_path} (clean-curve plot) from {len(csv_files)} files"
    )


def main():
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"No subdirs found in {root}")

    print(f"Found {len(subdirs)} subdirs in {root}")
    for folder in subdirs:
        process_folder(folder)


if __name__ == "__main__":
    main()
