"""Time-windowed log-log scaling regression on raw S(t).

For each time bin t ∈ {0.25, 0.5, ..., 5.0} ms, fit
    log S(t) = a(t)·log ΔP + b(t)·log ρ_a + c(t)·log d + const
on the raw measured penetration from the clean CDF wide table.

The Hiroyasu-Arai post-breakup asymptote predicts a(t) → 0.25 (the prefactor
on √(d·t) inherits ΔP^0.25 / ρ_a^0.25). The Bernoulli orifice-velocity scale
predicts a(t) → 0.5 (S ∝ U_exit·t with U ∝ √(ΔP/ρ_l)). If a(t) drifts from
~0.5 at small t toward ~0.25 at larger t, the regime-transition story is
empirically supported.

Output: CSV of per-bin coefficients + a publication-ready figure.

Compared with ``raw_series_physics_screening.py``, this version uses the clean
CDF wide table and canonical chamber pressure/density conversion from the
Stage-3 feature code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.MLP_training.engineered_feature_common import (  # noqa: E402
    build_dataset_registry,
    canonicalize_chamber_state,
    normalize_dataset_key,
)
from MLP.MLP_training.train_stage3_distillation_plus_raw_series import (  # noqa: E402
    load_source_table,
)


OUT_DIR = (
    PROJECT_ROOT
    / "MLP"
    / "synthetic_data"
    / "fit_diagnostics"
    / "time_windowed_exponent"
)
T_BINS_MS = np.arange(0.20, 1.55, 0.10)  # 14 bins covering populated region
HALF_WIDTH_MS = 0.075  # +/- 0.075 ms around each center (overlapping bins for smoother curve)


def build_long_table() -> pd.DataFrame:
    """Load the clean CDF wide table; emit a long table (sample, t, S, ΔP, ρ_a, d)."""
    registry = build_dataset_registry()
    cdf_wide_df = load_source_table("cdf", split="clean")

    time_cols = [c for c in cdf_wide_df.columns if c.startswith("time_ms_")]
    time_cols.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    frame_ids = [int(c.rsplit("_", 1)[1]) for c in time_cols]
    pen_cols = [f"penetration_mm_{frame_id:03d}" for frame_id in frame_ids]
    pen_cols = [c for c in pen_cols if c in cdf_wide_df.columns]

    # Canonicalize chamber state (raw pressure or density) → physical ambient
    # pressure + density per row using the dataset registry.
    canonical = []
    for _, row in cdf_wide_df.iterrows():
        try:
            dkey = normalize_dataset_key(str(row["experiment_name"]))
        except Exception:
            dkey = str(row["experiment_name"])
        chamber_raw = float(row["chamber_pressure_bar"])
        try:
            p_amb, rho_amb, _ = canonicalize_chamber_state(dkey, chamber_raw, registry)
        except Exception:
            # Fallback: linear density approximation from raw pressure.
            from MLP.MLP_training.engineered_feature_common import (
                linear_density_from_pressure,
            )
            p_amb = chamber_raw
            rho_amb = linear_density_from_pressure(chamber_raw)
        canonical.append((p_amb, rho_amb))

    cdf = cdf_wide_df.copy()
    cdf["ambient_pressure_bar_phys"] = [c[0] for c in canonical]
    cdf["ambient_density_kg_m3"] = [c[1] for c in canonical]
    cdf["delta_pressure_bar_phys"] = (
        cdf["injection_pressure_bar"] - cdf["ambient_pressure_bar_phys"]
    )
    cdf = cdf.loc[cdf["delta_pressure_bar_phys"] > 0].copy()

    # Build long format: one row per (sample, frame) with valid t and S.
    n_rows = len(cdf)
    n_frames = len(frame_ids)
    print(f"  loaded {n_rows} trajectories × {n_frames} frames")

    cond_cols = ["delta_pressure_bar_phys", "ambient_density_kg_m3", "diameter_mm"]
    cond_arr = cdf[cond_cols].to_numpy(dtype=float)

    t_grid = cdf[time_cols].to_numpy(dtype=float)
    s_grid = cdf[pen_cols].to_numpy(dtype=float)

    n_total = n_rows * n_frames
    t_flat = t_grid.reshape(-1)
    s_flat = s_grid.reshape(-1)
    sample_idx = np.repeat(np.arange(n_rows), n_frames)
    cond_flat = np.repeat(cond_arr, n_frames, axis=0)

    valid = (
        np.isfinite(t_flat) & np.isfinite(s_flat)
        & (t_flat > 0.0) & (s_flat > 0.0)
        & (t_flat <= 5.0 + HALF_WIDTH_MS)
    )
    long_df = pd.DataFrame({
        "sample_idx": sample_idx[valid],
        "t_ms": t_flat[valid],
        "S_mm": s_flat[valid],
        "delta_pressure_bar_phys": cond_flat[valid, 0],
        "ambient_density_kg_m3": cond_flat[valid, 1],
        "diameter_mm": cond_flat[valid, 2],
    })
    print(f"  long table: {len(long_df):,} valid (t, S) points")
    return long_df


def regress_per_bin(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t_center in T_BINS_MS:
        lo = t_center - HALF_WIDTH_MS
        hi = t_center + HALF_WIDTH_MS
        sub = long_df.loc[(long_df["t_ms"] >= lo) & (long_df["t_ms"] < hi)]
        if len(sub) < 50:
            rows.append({
                "t_center_ms": t_center, "n": int(len(sub)),
                "exp_delta_p": np.nan, "se_delta_p": np.nan,
                "exp_rho_air": np.nan, "se_rho_air": np.nan,
                "exp_diameter": np.nan, "se_diameter": np.nan,
                "intercept": np.nan, "r2": np.nan,
            })
            continue
        X = np.column_stack([
            np.log(sub["delta_pressure_bar_phys"].to_numpy(dtype=float)),
            np.log(sub["ambient_density_kg_m3"].to_numpy(dtype=float)),
            np.log(sub["diameter_mm"].to_numpy(dtype=float)),
            np.ones(len(sub)),
        ])
        y = np.log(sub["S_mm"].to_numpy(dtype=float))
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ coef
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        # OLS standard errors via residual variance × (X^T X)^{-1}.
        n, p = X.shape
        sigma2 = ss_res / max(n - p, 1)
        try:
            cov = sigma2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.full(p, np.nan)
        rows.append({
            "t_center_ms": t_center,
            "n": int(len(sub)),
            "exp_delta_p": float(coef[0]),
            "se_delta_p": float(se[0]),
            "exp_rho_air": float(coef[1]),
            "se_rho_air": float(se[1]),
            "exp_diameter": float(coef[2]),
            "se_diameter": float(se[2]),
            "intercept": float(coef[3]),
            "r2": float(r2),
        })
    return pd.DataFrame(rows)


def plot_results(reg_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)

    panels = [
        ("exp_delta_p", "se_delta_p",
         r"$\partial \log S / \partial \log \Delta P$", "C0"),
        ("exp_rho_air", "se_rho_air",
         r"$\partial \log S / \partial \log \rho_a$", "C1"),
        ("exp_diameter", "se_diameter",
         r"$\partial \log S / \partial \log d$", "C2"),
    ]
    references = [
        [(0.50, r"Bernoulli orifice  ($\Delta P^{0.5}$)", "k", "--"),
         (0.25, r"HA post-breakup  ($\Delta P^{0.25}$)", "0.5", ":")],
        [(-0.25, r"HA post-breakup  ($\rho_a^{-0.25}$)", "0.5", ":"),
         (0.0, r"Bernoulli  ($\rho_a^0$)", "k", "--")],
        [(0.5, r"HA / Bernoulli  ($d^{0.5}$)", "0.5", ":")],
    ]

    for ax, (key, se_key, ylabel, color), refs in zip(axes, panels, references):
        t = reg_df["t_center_ms"].to_numpy()
        y = reg_df[key].to_numpy()
        se = reg_df[se_key].to_numpy()
        ci = 1.96 * se
        ax.fill_between(t, y - ci, y + ci, color=color, alpha=0.18, label="95% CI")
        ax.plot(t, y, "-o", color=color, ms=4, label="OLS estimate")
        for ref_y, label, ref_color, ls in refs:
            ax.axhline(ref_y, color=ref_color, ls=ls, lw=1.2, label=label)
        ax.set_xlabel("Time bin center [ms]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)

    axes[0].set_title("ΔP exponent vs time")
    axes[1].set_title("ρ_a exponent vs time")
    axes[2].set_title("d exponent vs time")
    fig.suptitle(
        "Time-windowed empirical scaling of raw S(t) on operating conditions\n"
        f"(per {(T_BINS_MS[1]-T_BINS_MS[0])*1e3:.0f} us bin, ±{HALF_WIDTH_MS*1e3:.0f} us; clean CDF wide table, n={int(reg_df['n'].sum()):,})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading clean CDF wide table and computing canonical conditions ...")
    long_df = build_long_table()
    print(f"\nFitting per-bin OLS ({len(T_BINS_MS)} bins, ±{HALF_WIDTH_MS} ms half-width) ...")
    reg_df = regress_per_bin(long_df)
    csv_path = OUT_DIR / "time_windowed_scaling_regression.csv"
    reg_df.to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")

    fig_path = OUT_DIR / "time_windowed_exponents.png"
    plot_results(reg_df, fig_path)
    print(f"  saved {fig_path}")

    print("\n=== Summary ===")
    cols = ["t_center_ms", "n", "exp_delta_p", "se_delta_p",
            "exp_rho_air", "exp_diameter", "r2"]
    print(reg_df[cols].to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    valid = reg_df.dropna(subset=["exp_delta_p"])
    if len(valid) >= 2:
        first = valid.iloc[0]
        last = valid.iloc[-1]
        print(
            f"\ndelta_P exponent: t={first['t_center_ms']:.2f} ms -> {first['exp_delta_p']:.3f}  ||  "
            f"t={last['t_center_ms']:.2f} ms -> {last['exp_delta_p']:.3f}"
        )
        print(
            f"rho_a   exponent: t={first['t_center_ms']:.2f} ms -> {first['exp_rho_air']:.3f}  ||  "
            f"t={last['t_center_ms']:.2f} ms -> {last['exp_rho_air']:.3f}"
        )
        print(
            f"d       exponent: t={first['t_center_ms']:.2f} ms -> {first['exp_diameter']:.3f}  ||  "
            f"t={last['t_center_ms']:.2f} ms -> {last['exp_diameter']:.3f}"
        )
        # HA reference: 0.25; Bernoulli: 0.5
        print("\nReference exponents:")
        print("  Bernoulli orifice (S ~ U_exit*t):           delta_P = +0.50")
        print("  HA post-breakup  (S ~ (dP/rho_a)^0.25*sqrt(d*t)): delta_P = +0.25")


if __name__ == "__main__":
    main()
