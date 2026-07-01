"""Thesis-quality A-scaling collapse figure: DeltaP^0.25 vs DeltaP^0.5.

Standalone, larger-font, 3-panel re-render of the comparison originally produced by
compare_delta_p_exponent_collapse.py for the defense slides (which used a cluttered,
small-font 3-panel-per-exponent layout unsuitable for print). This script instead
renders one shared physical-curve panel plus the two A-scaled panels side by side,
sized and labeled for the thesis page.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # MLP_training/
from efc.models import reconstruct_penetration_series  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROW_TABLE = (
    PROJECT_ROOT
    / "MLP"
    / "runs_mlp"
    / "old"
    / "stage1_engineered_mse_a_only_20260508_235519"
    / "row_table.csv"
)
OUT_PATH = PROJECT_ROOT / "Thesis" / "images" / "collapse_dp_exponent_comparison.png"


def make_a_scale(df: pd.DataFrame, dp_exp: float) -> np.ndarray:
    return (
        np.power(df["delta_pressure_bar_phys"].astype(float), dp_exp)
        * np.power(df["ambient_density_kg_m3"].astype(float), -0.25)
        * np.sqrt(df["diameter_mm"].astype(float))
    ).to_numpy(dtype=float)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )

    df = pd.read_csv(ROW_TABLE)
    times_ms = np.linspace(0.5, 5.0, 10)
    time_s = times_ms * 1e-3

    trajectories = np.stack(
        [
            reconstruct_penetration_series(
                df["log_k_sqrt"].to_numpy(dtype=float),
                df["log_k_quarter"].to_numpy(dtype=float),
                df["log_t0"].to_numpy(dtype=float),
                df["log_s"].to_numpy(dtype=float),
                t_value,
            )
            for t_value in time_s
        ],
        axis=1,
    )

    rng = np.random.default_rng(0)
    plot_count = min(len(trajectories), 160)
    sample_idx = rng.choice(len(trajectories), size=plot_count, replace=False) if len(trajectories) > plot_count else np.arange(len(trajectories))

    cases = [("0.25", 0.25), ("0.50", 0.50)]
    scaled = {}
    median_ratio = {}
    for label, dp_exp in cases:
        a_scale = make_a_scale(df, dp_exp)[:, None]
        scaled_traj = trajectories / a_scale
        scaled[label] = scaled_traj
        std_raw = np.nanstd(trajectories, axis=0)
        std_scaled = np.nanstd(scaled_traj, axis=0)
        post = times_ms >= 0.5
        median_ratio[label] = float(np.nanmedian((std_scaled / std_raw)[post]))

    y_max = max(np.nanpercentile(scaled["0.25"][sample_idx], 99), np.nanpercentile(scaled["0.50"][sample_idx], 99))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    axes[0].plot(times_ms, trajectories[sample_idx].T, alpha=0.12, color="#1f77b4")
    axes[0].set_title("(a) Physical curves $S(t)$")
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel("Penetration $S$ [mm]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times_ms, scaled["0.25"][sample_idx].T, alpha=0.12, color="#d62728")
    axes[1].set_title(r"(b) $A$-scaled, $\Delta P^{0.25}$")
    axes[1].set_xlabel("Time [ms]")
    axes[1].set_ylabel(r"$\hat{S}=S/A$")
    axes[1].set_ylim(0, y_max * 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(
        0.97, 0.05, f"median collapse ratio = {median_ratio['0.25']:.3f}",
        transform=axes[1].transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
    )

    axes[2].plot(times_ms, scaled["0.50"][sample_idx].T, alpha=0.12, color="#2ca02c")
    axes[2].set_title(r"(c) $A$-scaled, $\Delta P^{0.50}$ (adopted)")
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_ylabel(r"$\hat{S}=S/A$")
    axes[2].set_ylim(0, y_max * 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].text(
        0.97, 0.05, f"median collapse ratio = {median_ratio['0.50']:.3f}",
        transform=axes[2].transAxes, ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
    )

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")
    print(f"Median post-0.5ms collapse ratio: 0.25 -> {median_ratio['0.25']:.3f}, 0.50 -> {median_ratio['0.50']:.3f}")


if __name__ == "__main__":
    main()
