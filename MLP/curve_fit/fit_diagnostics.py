"""Diagnostic reports for the q1 production curve fit.

Reads the per-folder clean CSVs and series_clean CSVs produced by
``fit_raw_data.main()`` and writes a set of diagnostic plots and summary
tables under ``MLP/synthetic_data/fit_diagnostics/``.

Five report sections are produced:

* ``error_by_condition/``   -- RMSE distributions by injection / chamber
                               pressure and nozzle.
* ``param_distributions/``  -- fitted (k_quarter, t0, s) vs operating
                               conditions with an empirical log-log
                               scaling regression.
* ``residual_structure/``   -- model-vs-observation residual structure
                               and per-time-bin sigma(t) (heteroscedasticity).
* ``convergence/``          -- solver iteration count, first-order
                               optimality, and termination status.
* ``identifiability/``      -- parameter standard errors and pairwise
                               correlations recovered from the Jacobian.

Designed to be invoked at the end of ``fit_raw_data.__main__``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import fit_raw_data as frd

DEFAULT_PEN_KEY = "cdf"
DEFAULT_OUT_DIR = frd.data_out_dir / "fit_diagnostics"
DEFAULT_THESIS_IMAGE_DIR = frd._THIS_DIR.parent.parent / "Thesis" / "images"
NITROGEN_RHO_PER_BAR = 1.165  # kg/m^3 per bar for N2 at ~298 K
TIME_BIN_MS = 0.10
MAX_TIME_MS = 6.0
MIN_BIN_COUNT = 50  # minimum observations per time bin to report sigma(t)
THESIS_FIGURE_MAP = {
    "param_distributions/param_vs_conditions.png": "fig_q1_param_scaling.png",
    "residual_structure/residual_vs_time.png":     "fig_q1_residual_structure.png",
    "identifiability/identifiability.png":         "fig_q1_identifiability.png",
}


def _aggregate_clean_rows(pen_key: str = DEFAULT_PEN_KEY) -> pd.DataFrame:
    """Read every ``<dataset>/<pen_key>/clean/*.csv`` row table into a single
    DataFrame, attaching a ``nozzle`` label inferred from the dataset name.
    """
    frames = []
    for name in frd.names:
        clean_dir = frd.data_out_dir / name / pen_key / "clean"
        if not clean_dir.exists():
            continue
        for csv_path in sorted(clean_dir.glob("*.csv")):
            try:
                part = pd.read_csv(csv_path)
            except (pd.errors.EmptyDataError, OSError):
                continue
            if part.empty:
                continue
            part["dataset"] = name
            part["nozzle"] = _nozzle_label(name)
            part["folder"] = csv_path.stem
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _aggregate_series_clean(pen_key: str = DEFAULT_PEN_KEY) -> pd.DataFrame:
    frames = []
    for name in frd.names:
        series_dir = frd.data_out_dir / name / pen_key / "series_clean"
        if not series_dir.exists():
            continue
        for csv_path in sorted(series_dir.glob("*.csv")):
            try:
                part = pd.read_csv(csv_path)
            except (pd.errors.EmptyDataError, OSError):
                continue
            if part.empty:
                continue
            part["dataset"] = name
            part["nozzle"] = _nozzle_label(name)
            part["folder"] = csv_path.stem
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _nozzle_label(dataset_name: str) -> str:
    """Map raw dataset name to a thesis-style nozzle label."""
    if dataset_name == "Nozzle0":
        return "Nozzle0"
    lower = dataset_name.lower()
    for token in lower.split("_"):
        if token.startswith("nozzle"):
            return token.capitalize()
    return dataset_name


def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _save_fig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Error by condition
# ---------------------------------------------------------------------------

def diag_error_by_condition(rows: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _coerce_numeric(
        rows,
        ["rmse", "injection_pressure_bar", "chamber_pressure_bar", "diameter_mm"],
    )
    df = df.loc[df["rmse"].notna()].copy()
    if df.empty:
        return {"status": "skipped (no rows)"}

    summary = (
        df.groupby(["nozzle", "injection_pressure_bar", "chamber_pressure_bar"], dropna=False)
        ["rmse"].agg(
            count="count",
            median="median",
            q25=lambda s: float(np.nanpercentile(s, 25)),
            q75=lambda s: float(np.nanpercentile(s, 75)),
            p95=lambda s: float(np.nanpercentile(s, 95)),
        )
        .reset_index()
    )
    summary_csv = out_dir / "rmse_by_condition.csv"
    summary.to_csv(summary_csv, index=False)

    # Box plot: RMSE by nozzle
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    nozzles = sorted(df["nozzle"].dropna().unique())
    data = [df.loc[df["nozzle"] == nz, "rmse"].dropna().values for nz in nozzles]
    if data:
        ax.boxplot(data, tick_labels=nozzles, showfliers=False)
    ax.set_ylabel("Fit RMSE [mm]")
    ax.set_title("Per-nozzle q1 fit RMSE distribution (clean rows)")
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, out_dir / "rmse_by_nozzle.png")

    # Heatmap: median RMSE on (P_inj, P_ch) grid, aggregated across nozzles
    pivot = (
        df.groupby(["injection_pressure_bar", "chamber_pressure_bar"], dropna=False)
        ["rmse"].median()
        .reset_index()
        .pivot(index="injection_pressure_bar", columns="chamber_pressure_bar", values="rmse")
        .sort_index()
        .sort_index(axis=1)
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r:g}" for r in pivot.index])
        ax.set_xlabel("Chamber pressure [bar]")
        ax.set_ylabel("Injection pressure [bar]")
        ax.set_title("Median q1 RMSE [mm] by (P_inj, P_ch)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > pivot.values[np.isfinite(pivot.values)].mean() else "black",
                            fontsize=8)
        fig.colorbar(im, ax=ax, label="median RMSE [mm]")
        _save_fig(fig, out_dir / "rmse_heatmap_pinj_pch.png")

    return {
        "n_rows": int(len(df)),
        "outputs": {
            "summary_csv": str(summary_csv),
            "boxplot_png": str(out_dir / "rmse_by_nozzle.png"),
            "heatmap_png": str(out_dir / "rmse_heatmap_pinj_pch.png"),
        },
    }


# ---------------------------------------------------------------------------
# 2. Parameter distributions vs operating conditions
# ---------------------------------------------------------------------------

def diag_param_distributions(rows: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _coerce_numeric(
        rows,
        [
            "k_quarter", "t0", "s",
            "injection_pressure_bar", "chamber_pressure_bar", "diameter_mm",
        ],
    )
    df = df.loc[
        df["k_quarter"].notna()
        & df["t0"].notna()
        & df["s"].notna()
        & df["injection_pressure_bar"].notna()
        & df["chamber_pressure_bar"].notna()
        & df["diameter_mm"].notna()
    ].copy()
    if df.empty:
        return {"status": "skipped (no rows)"}

    df["delta_p_bar"] = df["injection_pressure_bar"] - df["chamber_pressure_bar"]
    df = df.loc[df["delta_p_bar"] > 0].copy()
    if df.empty:
        return {"status": "skipped (delta_p <= 0 for all rows)"}

    df["rho_air_kg_m3"] = NITROGEN_RHO_PER_BAR * df["chamber_pressure_bar"]
    df["t0_ms"] = df["t0"] * 1e3
    df["s_ms"] = df["s"] * 1e3

    # Empirical log-log regression: log y = a*log(dP) + b*log(rho_a) + c*log(d) + const
    regression_records = []
    for col in ("k_quarter", "t0_ms", "s_ms"):
        sub = df.loc[df[col] > 0, [col, "delta_p_bar", "rho_air_kg_m3", "diameter_mm"]].dropna()
        if len(sub) < 10:
            regression_records.append({"target": col, "n": len(sub), "status": "insufficient_rows"})
            continue
        X = np.column_stack([
            np.log(sub["delta_p_bar"].values),
            np.log(sub["rho_air_kg_m3"].values),
            np.log(sub["diameter_mm"].values),
            np.ones(len(sub)),
        ])
        y = np.log(sub[col].values)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ coef
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        regression_records.append({
            "target": col,
            "n": int(len(sub)),
            "exp_delta_p": float(coef[0]),
            "exp_rho_air": float(coef[1]),
            "exp_diameter": float(coef[2]),
            "intercept": float(coef[3]),
            "r2": float(r2),
        })
    reg_df = pd.DataFrame(regression_records)
    reg_csv = out_dir / "scaling_regression.csv"
    reg_df.to_csv(reg_csv, index=False)

    # Three-panel scatter: each parameter vs delta_p, colored by P_ch
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), constrained_layout=True)
    panels = [("k_quarter", "$k_{1/4}$ [mm/s$^{1/4}$]", df["k_quarter"]),
              ("t0_ms", "$t_0$ [ms]", df["t0_ms"]),
              ("s_ms", "$s$ [ms]", df["s_ms"])]
    import matplotlib.ticker as ticker
    for ax, (key, ylab, yvals) in zip(axes, panels):
        sc = ax.scatter(
            df["delta_p_bar"] / 1000.0, yvals,
            c=df["chamber_pressure_bar"], cmap="viridis",
            s=4, alpha=0.35, edgecolors="none",
        )
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 3])
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_yscale("log" if key == "k_quarter" else "linear")
        ax.set_xlabel(r"$\Delta P = P_{inj} - P_{ch}$ [$10^3$ bar]")
        ax.set_ylabel(ylab)
        ax.grid(True, which="both", alpha=0.25)
        rec = next((r for r in regression_records if r["target"] == key), None)
        if rec and "r2" in rec:
            ax.set_title(
                f"{key}: $\\Delta P^{{{rec['exp_delta_p']:.2f}}}\\,"
                f"\\rho_a^{{{rec['exp_rho_air']:.2f}}}\\,"
                f"d^{{{rec['exp_diameter']:.2f}}}$\n$R^2$={rec['r2']:.2f}"
            )
        else:
            ax.set_title(key)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("$P_{ch}$ [bar]")
    fig.suptitle("Fitted q1 parameters vs operating conditions (clean rows)", fontsize=16)
    fig.savefig(out_dir / "param_vs_conditions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # H-A reference annotation: for sqrt(t) post-breakup, k_sqrt would scale as
    # delta_p^(1/4) * rho_a^(-1/4) * d_n^(1/2). The q1 model uses t^(1/4) after
    # impingement instead, so the table reports empirical exponents alongside
    # those classical references for context only.
    ha_reference = pd.DataFrame([
        {"branch": "Hiroyasu-Arai post-breakup S(t)~sqrt(t)",
         "exp_delta_p": 0.25, "exp_rho_air": -0.25, "exp_diameter": 0.5},
    ])
    ha_reference.to_csv(out_dir / "ha_reference_exponents.csv", index=False)

    return {
        "n_rows": int(len(df)),
        "outputs": {
            "regression_csv": str(reg_csv),
            "scatter_png": str(out_dir / "param_vs_conditions.png"),
            "ha_reference_csv": str(out_dir / "ha_reference_exponents.csv"),
        },
        "regression": regression_records,
    }


# ---------------------------------------------------------------------------
# 3. Residual structure over time
# ---------------------------------------------------------------------------

def diag_residual_structure(
    rows: pd.DataFrame,
    series: pd.DataFrame,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows.empty or series.empty:
        return {"status": "skipped (no rows or series)"}

    fit_cols = ["dataset", "folder", "file_stem", "plume_idx",
                "k_quarter", "t0", "s"]
    fit = _coerce_numeric(rows[fit_cols], ["k_quarter", "t0", "s", "plume_idx"])
    fit = fit.loc[
        fit["k_quarter"].notna() & fit["t0"].notna() & fit["s"].notna()
    ].copy()
    if fit.empty:
        return {"status": "skipped (no fits)"}

    series = _coerce_numeric(
        series,
        ["plume_idx", "time_s", "time_ms", "penetration_mm"],
    )
    merged = series.merge(
        fit, on=["dataset", "folder", "file_stem", "plume_idx"], how="inner"
    )
    if merged.empty:
        return {"status": "skipped (merge empty)"}

    log_params = np.column_stack([
        np.log(np.maximum(merged["k_quarter"].values, 1e-30)),
        np.log(np.maximum(merged["t0"].values, 1e-30)),
        np.log(np.maximum(merged["s"].values, 1e-30)),
    ])
    t_arr = merged["time_s"].values
    y_hat = np.empty(len(merged), dtype=float)
    for i in range(len(merged)):
        y_hat[i] = float(
            frd.spray_penetration_model_quarter_only(log_params[i], np.array([t_arr[i]]))[0]
        )
    residual = y_hat - merged["penetration_mm"].values
    rel_residual = residual / np.where(
        merged["penetration_mm"].values != 0, merged["penetration_mm"].values, np.nan
    )
    merged = merged.assign(residual_mm=residual, rel_residual=rel_residual)

    bins = np.arange(0.0, MAX_TIME_MS + TIME_BIN_MS, TIME_BIN_MS)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.digitize(merged["time_ms"].values, bins) - 1
    keep = (bin_idx >= 0) & (bin_idx < len(centers)) & np.isfinite(residual)
    bin_idx = bin_idx[keep]
    res_keep = residual[keep]

    rows_summary = []
    for k in range(len(centers)):
        sel = bin_idx == k
        n = int(sel.sum())
        if n < MIN_BIN_COUNT:
            rows_summary.append({"time_ms": centers[k], "n": n,
                                  "median": np.nan, "mad": np.nan,
                                  "sigma_robust": np.nan})
            continue
        slice_vals = res_keep[sel]
        med = float(np.nanmedian(slice_vals))
        mad = float(np.nanmedian(np.abs(slice_vals - med)))
        sigma = 1.4826 * mad
        rows_summary.append({
            "time_ms": float(centers[k]),
            "n": n,
            "median": med,
            "mad": mad,
            "sigma_robust": sigma,
        })
    summary_df = pd.DataFrame(rows_summary)
    summary_csv = out_dir / "residual_vs_time.csv"
    summary_df.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.4), sharex=True)
    valid = summary_df["sigma_robust"].notna()
    axes[0].plot(summary_df.loc[valid, "time_ms"], summary_df.loc[valid, "median"],
                 color="#2a72c5", lw=1.4, label="median residual")
    axes[0].fill_between(
        summary_df.loc[valid, "time_ms"],
        summary_df.loc[valid, "median"] - summary_df.loc[valid, "sigma_robust"],
        summary_df.loc[valid, "median"] + summary_df.loc[valid, "sigma_robust"],
        color="#2a72c5", alpha=0.18, label=r"$\pm\sigma_{robust}$",
    )
    axes[0].axhline(0, color="black", lw=0.8, alpha=0.6)
    axes[0].set_ylabel("Residual $\\hat y - y$ [mm]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)
    axes[0].set_title("q1 residual structure across all clean trajectories")

    axes[1].plot(summary_df.loc[valid, "time_ms"], summary_df.loc[valid, "sigma_robust"],
                 color="#d04a35", lw=1.4)
    axes[1].set_xlabel("time after onset [ms]")
    axes[1].set_ylabel("$\\sigma_{robust}(t)$ [mm]")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xlim(0, MAX_TIME_MS)
    _save_fig(fig, out_dir / "residual_vs_time.png")

    return {
        "n_rows": int(len(merged)),
        "outputs": {
            "summary_csv": str(summary_csv),
            "plot_png": str(out_dir / "residual_vs_time.png"),
        },
    }


# ---------------------------------------------------------------------------
# 4. Convergence diagnostics
# ---------------------------------------------------------------------------

_STATUS_NAMES = {
    -10: "no_fit_attempted",
    -1:  "improper_input",
    0:   "max_nfev_reached",
    1:   "gtol_satisfied",
    2:   "ftol_satisfied",
    3:   "xtol_satisfied",
    4:   "ftol_and_xtol_satisfied",
}


def diag_convergence(rows: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows.empty:
        return {"status": "skipped (no rows)"}
    df = _coerce_numeric(rows, ["nfev", "optimality", "status", "rmse"])
    df = df.loc[df["status"].notna()].copy()
    if df.empty:
        return {"status": "skipped (no convergence columns)"}

    df["status_name"] = df["status"].astype(int).map(_STATUS_NAMES).fillna("unknown")

    status_summary = (
        df.groupby("status_name").agg(
            count=("status", "size"),
            median_nfev=("nfev", "median"),
            median_optimality=("optimality", "median"),
            median_rmse=("rmse", "median"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    status_csv = out_dir / "convergence_by_status.csv"
    status_summary.to_csv(status_csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    nfev = df["nfev"].dropna().values
    if nfev.size:
        axes[0].hist(nfev, bins=30, color="#4c78a8", alpha=0.85)
    axes[0].set_xlabel("nfev (function evaluations)")
    axes[0].set_ylabel("count")
    axes[0].grid(True, alpha=0.25)

    opt = df["optimality"].dropna().values
    opt = opt[opt > 0]
    if opt.size:
        axes[1].hist(np.log10(opt), bins=30, color="#54a24b", alpha=0.85)
    axes[1].set_xlabel("$\\log_{10}$(first-order optimality)")
    axes[1].grid(True, alpha=0.25)

    counts = status_summary.set_index("status_name")["count"]
    if len(counts) > 0:
        axes[2].bar(range(len(counts)), counts.values, color="#f58518", alpha=0.85)
        axes[2].set_xticks(range(len(counts)))
        axes[2].set_xticklabels(counts.index, rotation=30, ha="right")
        axes[2].set_ylabel("count")
    axes[2].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Solver convergence diagnostics across clean q1 fits")
    _save_fig(fig, out_dir / "convergence_diagnostics.png")

    return {
        "n_rows": int(len(df)),
        "outputs": {
            "status_csv": str(status_csv),
            "plot_png": str(out_dir / "convergence_diagnostics.png"),
        },
    }


# ---------------------------------------------------------------------------
# 5. Identifiability (parameter standard errors and correlations)
# ---------------------------------------------------------------------------

def diag_identifiability(rows: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "std_log_k_quarter", "std_log_t0", "std_log_s",
        "corr_logk_logt0", "corr_logk_logs", "corr_logt0_logs",
    ]
    df = _coerce_numeric(rows, cols)
    df = df.dropna(subset=cols, how="all").copy()
    if df.empty:
        return {"status": "skipped (no identifiability columns)"}

    summary_records = []
    for col in cols:
        vals = df[col].dropna().values
        if vals.size == 0:
            continue
        summary_records.append({
            "metric": col,
            "n": int(vals.size),
            "median": float(np.nanmedian(vals)),
            "p25": float(np.nanpercentile(vals, 25)),
            "p75": float(np.nanpercentile(vals, 75)),
            "p95": float(np.nanpercentile(vals, 95)),
        })
    summary_df = pd.DataFrame(summary_records)
    summary_csv = out_dir / "identifiability_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    std_cols = ["std_log_k_quarter", "std_log_t0", "std_log_s"]
    std_data = [df[c].dropna().clip(upper=df[c].quantile(0.99)) for c in std_cols]
    axes[0].boxplot(std_data, tick_labels=["log k_1/4", "log t_0", "log s"], showfliers=False)
    axes[0].set_ylabel("posterior std (asymptotic, log-space)")
    axes[0].set_title("Per-parameter standard error")
    axes[0].grid(True, axis="y", alpha=0.25)

    corr_cols = ["corr_logk_logt0", "corr_logk_logs", "corr_logt0_logs"]
    corr_medians = np.full((3, 3), np.nan)
    np.fill_diagonal(corr_medians, 1.0)
    corr_medians[0, 1] = corr_medians[1, 0] = float(np.nanmedian(df["corr_logk_logt0"]))
    corr_medians[0, 2] = corr_medians[2, 0] = float(np.nanmedian(df["corr_logk_logs"]))
    corr_medians[1, 2] = corr_medians[2, 1] = float(np.nanmedian(df["corr_logt0_logs"]))
    im = axes[1].imshow(corr_medians, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    axes[1].set_xticklabels(["log k_1/4", "log t_0", "log s"])
    axes[1].set_yticklabels(["log k_1/4", "log t_0", "log s"])
    axes[1].set_title("Median pairwise correlation")
    for i in range(3):
        for j in range(3):
            v = corr_medians[i, j]
            if np.isfinite(v):
                axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                             color="white" if abs(v) > 0.5 else "black", fontsize=10)
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    _save_fig(fig, out_dir / "identifiability.png")

    return {
        "n_rows": int(len(df)),
        "outputs": {
            "summary_csv": str(summary_csv),
            "plot_png": str(out_dir / "identifiability.png"),
        },
        "median_correlations": {
            "logk_logt0": float(np.nanmedian(df["corr_logk_logt0"])),
            "logk_logs": float(np.nanmedian(df["corr_logk_logs"])),
            "logt0_logs": float(np.nanmedian(df["corr_logt0_logs"])),
        },
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all(
    *,
    pen_key: str = DEFAULT_PEN_KEY,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fit-diagnostics] aggregating clean rows from pen_key='{pen_key}' …")
    rows = _aggregate_clean_rows(pen_key)
    print(f"[fit-diagnostics] aggregated rows: {len(rows)}")
    if rows.empty:
        msg = {"status": "no clean rows aggregated", "out_dir": str(out_dir)}
        (out_dir / "fit_diagnostics_summary.json").write_text(json.dumps(msg, indent=2))
        return msg

    print("[fit-diagnostics] aggregating per-frame series …")
    series = _aggregate_series_clean(pen_key)
    print(f"[fit-diagnostics] aggregated series rows: {len(series)}")

    summary = {
        "pen_key": pen_key,
        "out_dir": str(out_dir),
        "n_clean_rows": int(len(rows)),
        "n_series_rows": int(len(series)),
        "sections": {},
    }
    summary["sections"]["error_by_condition"] = diag_error_by_condition(
        rows, out_dir / "error_by_condition"
    )
    summary["sections"]["param_distributions"] = diag_param_distributions(
        rows, out_dir / "param_distributions"
    )
    summary["sections"]["residual_structure"] = diag_residual_structure(
        rows, series, out_dir / "residual_structure"
    )
    summary["sections"]["convergence"] = diag_convergence(
        rows, out_dir / "convergence"
    )
    summary["sections"]["identifiability"] = diag_identifiability(
        rows, out_dir / "identifiability"
    )

    if DEFAULT_THESIS_IMAGE_DIR.exists():
        copies = []
        for src_rel, dest_name in THESIS_FIGURE_MAP.items():
            src = out_dir / src_rel
            if not src.exists():
                continue
            dest = DEFAULT_THESIS_IMAGE_DIR / dest_name
            shutil.copy2(src, dest)
            copies.append(str(dest))
        summary["thesis_figure_copies"] = copies
        print(f"[fit-diagnostics] copied {len(copies)} figure(s) to {DEFAULT_THESIS_IMAGE_DIR}")

    (out_dir / "fit_diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"[fit-diagnostics] summary -> {out_dir / 'fit_diagnostics_summary.json'}")
    return summary


def main() -> None:
    summary = run_all()
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
