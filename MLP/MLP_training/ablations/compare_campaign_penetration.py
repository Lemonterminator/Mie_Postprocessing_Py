"""
Compare injector penetration across campaigns in A_scale-normalised coordinates.

Reads the quality-filtered wide series table (split='clean'), computes dimensionless
penetration S* = S_mm / A_scale and normalised time t* = t_onset_us / inj_duration_us,
then compares a reference campaign (default: N0) against one or all other campaigns.

Normalisation:
    A_scale = (delta_pressure_bar / ambient_density_kg_m3)^0.25 * sqrt(diameter_mm)
    S*      = S_mm / A_scale
    t*      = t_ms_from_onset * 1000 / injection_duration_us

After A_scale normalisation all operating-condition variation (pressure, density,
diameter) is collapsed.  Residual systematic offset between campaigns is attributable
solely to injector characteristics (Cd, aging state, needle dynamics).

Usage
-----
    python compare_campaign_penetration.py
    python compare_campaign_penetration.py --ref BC20220627_HZ_Nozzle0
    python compare_campaign_penetration.py \\
        --ref BC20220627_HZ_Nozzle0 \\
        --compare BC20241003_HZ_Nozzle1 \\
        --output-dir MLP/figures/injector_comparison
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── repo path bootstrap ────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))                       # MLP_training/
sys.path.insert(0, str(_HERE.parent.parent / "baseline" / "Naber_Siebers"))  # data_io

import data_io  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────
FOV_LIMIT_MM = 92.0          # readings at or above this are FOV-censored
MIN_PEN_MM   = 0.1           # ignore near-zero (pre-onset) points
T_STAR_BINS  = np.linspace(0.0, 2.5, 51)   # normalised time grid for stats
T_STAR_MIN_N = 5             # minimum points per bin to include in plot


# ── data loading ──────────────────────────────────────────────────────────────
def load_normalised(source: str = "cdf", dp_exp: float = 0.25) -> pd.DataFrame:
    """Return long-format table with S* and t* columns.

    ``dp_exp`` is the ΔP exponent in A_scale = ΔP^dp_exp · ρ^-0.25 · √d.
    0.25 = legacy Hiroyasu–Arai; 0.50 = production momentum/Bernoulli regime.
    """
    wide = data_io.load_source_table(source=source, split="clean")
    registry = data_io.build_dataset_registry()
    wide = data_io.add_canonical_features(wide, registry)

    # Recompute A_scale with the requested ΔP exponent (add_canonical_features
    # hardcodes 0.25). Density exponent -0.25 and diameter exponent 0.5 are
    # held fixed so only the ΔP scaling regime changes.
    wide["A_scale"] = (
        np.power(wide["delta_pressure_bar_phys"], dp_exp)
        * np.power(wide["ambient_density_kg_m3"], -0.25)
        * np.sqrt(wide["diameter_mm"].clip(lower=1e-12))
    )

    # Identify paired (time_ms_XXX, penetration_mm_XXX) columns
    time_cols = sorted(c for c in wide.columns if c.startswith("time_ms_"))
    pen_cols  = sorted(c for c in wide.columns if c.startswith("penetration_mm_"))
    assert len(time_cols) == len(pen_cols), "time/pen column mismatch"

    meta_cols = [
        "experiment_name", "file_stem", "plume_idx",
        "injection_duration_us", "injection_pressure_bar",
        "chamber_pressure_bar", "diameter_mm",
        "ambient_density_kg_m3", "delta_pressure_bar_phys", "A_scale",
    ]
    meta = wide[meta_cols].copy()

    # Melt to long format
    n_pts = len(time_cols)
    frames: list[pd.DataFrame] = []
    for i, (tc, pc) in enumerate(zip(time_cols, pen_cols)):
        chunk = meta.copy()
        chunk["t_ms"]   = wide[tc].values
        chunk["S_mm"]   = wide[pc].values
        chunk["pt_idx"] = i
        frames.append(chunk)

    long = pd.concat(frames, ignore_index=True)

    # Filter censored and sub-threshold points
    long = long[
        (long["S_mm"] >= MIN_PEN_MM) &
        (long["S_mm"] < FOV_LIMIT_MM) &
        long["t_ms"].notna() &
        long["S_mm"].notna()
    ].copy()

    # Dimensionless coordinates
    long["S_star"] = long["S_mm"] / long["A_scale"]
    long["t_us"]   = long["t_ms"] * 1000.0
    long["t_star"] = long["t_us"] / long["injection_duration_us"]

    return long


# ── per-campaign statistics ────────────────────────────────────────────────────
def bin_stats(df: pd.DataFrame, label_col: str = "campaign") -> pd.DataFrame:
    """Compute mean/std/median/count of S* in t* bins, per campaign."""
    df = df.copy()
    df["t_bin"] = pd.cut(df["t_star"], bins=T_STAR_BINS, labels=False)
    t_centers = 0.5 * (T_STAR_BINS[:-1] + T_STAR_BINS[1:])

    records = []
    for camp, grp in df.groupby(label_col):
        for bin_idx, bgrp in grp.groupby("t_bin"):
            if len(bgrp) < T_STAR_MIN_N:
                continue
            records.append({
                label_col:    camp,
                "t_star":     t_centers[int(bin_idx)],
                "S_star_mean":  bgrp["S_star"].mean(),
                "S_star_std":   bgrp["S_star"].std(),
                "S_star_median": bgrp["S_star"].median(),
                "n":            len(bgrp),
            })
    return pd.DataFrame(records)


# ── ratio table ───────────────────────────────────────────────────────────────
def build_ratio_table(
    stats: pd.DataFrame,
    ref_label: str,
    cmp_label: str,
    label_col: str = "campaign",
) -> pd.DataFrame:
    """Return per-bin ratio table: S*_ref / S*_cmp (how much more ref penetrates)."""
    ref = stats[stats[label_col] == ref_label].set_index("t_star")
    cmp = stats[stats[label_col] == cmp_label].set_index("t_star")
    common = ref.index.intersection(cmp.index)
    ratio = (ref.loc[common, "S_star_mean"] / cmp.loc[common, "S_star_mean"]).rename("ratio")
    out = pd.DataFrame({
        "t_star": common,
        "S_star_ref": ref.loc[common, "S_star_mean"].values,
        "S_star_cmp": cmp.loc[common, "S_star_mean"].values,
        "ratio":      ratio.values,
        "n_ref":      ref.loc[common, "n"].values,
        "n_cmp":      cmp.loc[common, "n"].values,
    })
    return out.reset_index(drop=True)


# ── plotting ──────────────────────────────────────────────────────────────────
_COLOURS = {
    "N0 (new)":   "#e74c3c",
    "N1–N8 (used)": "#2980b9",
}


def plot_comparison(
    long: pd.DataFrame,
    stats: pd.DataFrame,
    ratio: pd.DataFrame,
    ref_label: str,
    cmp_label: str,
    output_dir: Path,
    dp_exp: float = 0.25,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── left: mean ± std bands ────────────────────────────────────────────────
    ax = axes[0]
    for lbl, colour in _COLOURS.items():
        sub = stats[stats["campaign"] == lbl]
        if sub.empty:
            continue
        ax.plot(sub["t_star"], sub["S_star_mean"], lw=2, color=colour, label=lbl)
        ax.fill_between(
            sub["t_star"],
            sub["S_star_mean"] - sub["S_star_std"],
            sub["S_star_mean"] + sub["S_star_std"],
            alpha=0.20, color=colour,
        )
    ax.axvline(1.0, ls="--", lw=0.8, color="k", alpha=0.5, label="t* = 1 (EOI)")
    ax.set_xlabel(r"$t^* = t_{\rm onset} \;/\; t_{\rm inj}$", fontsize=12)
    ax.set_ylabel(r"$S^* = S_{\rm mm} \;/\; A_{\rm scale}$", fontsize=12)
    ax.set_title("Dimensionless penetration: mean ± 1 std")
    ax.legend(fontsize=10)
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── right: ratio S*_ref / S*_cmp ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(ratio["t_star"], ratio["ratio"], lw=2, color="#e74c3c", marker="o",
             ms=3, label=f"{ref_label[:2]} / {cmp_label[:2]}")
    ax2.axhline(1.0, ls="--", lw=1, color="k", alpha=0.6, label="ratio = 1 (identical)")
    ax2.set_xlabel(r"$t^* = t_{\rm onset} \;/\; t_{\rm inj}$", fontsize=12)
    ax2.set_ylabel(r"$S^*_{\rm N0} \;/\; S^*_{\rm N1\text{-}N8}$", fontsize=12)
    ax2.set_title(f"Penetration ratio after A_scale normalisation")
    ax2.legend(fontsize=10)
    ax2.grid(True, lw=0.4, alpha=0.5)
    ax2.set_ylim(bottom=0)

    regime = "Hiroyasu–Arai" if abs(dp_exp - 0.25) < 1e-9 else (
        "momentum/Bernoulli" if abs(dp_exp - 0.5) < 1e-9 else "custom")
    plt.suptitle(
        f"Injector comparison: {ref_label[:25]}  vs  {cmp_label[:25]}\n"
        f"$A_{{\\rm scale}} = \\Delta p^{{{dp_exp:g}}} \\cdot \\rho^{{-0.25}} \\cdot \\sqrt{{d}}$"
        f"  ({regime})   |   FOV-censored points excluded",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = output_dir / f"injector_comparison_dp{int(round(dp_exp * 100)):03d}.png"
    plt.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")
    plt.close(fig)


def plot_raw_scatter(
    long: pd.DataFrame,
    output_dir: Path,
    max_pts: int = 5000,
) -> None:
    """Scatter all individual S* vs t* points coloured by nozzle."""
    cmap = plt.get_cmap("tab10")
    experiments = sorted(long["experiment_name"].unique())
    colour_map = {e: cmap(i % 10) for i, e in enumerate(experiments)}

    fig, ax = plt.subplots(figsize=(10, 5))
    for exp in experiments:
        sub = long[long["experiment_name"] == exp]
        if len(sub) > max_pts:
            sub = sub.sample(max_pts, random_state=42)
        short = exp.split("_")[-1]   # e.g. "Nozzle0"
        ax.scatter(sub["t_star"], sub["S_star"], s=2, alpha=0.25,
                   color=colour_map[exp], label=short)

    ax.axvline(1.0, ls="--", lw=0.8, color="k", alpha=0.4)
    ax.set_xlabel(r"$t^* = t_{\rm onset} / t_{\rm inj}$", fontsize=12)
    ax.set_ylabel(r"$S^* = S_{\rm mm} / A_{\rm scale}$", fontsize=12)
    ax.set_title("All campaigns – dimensionless penetration scatter")
    ax.legend(markerscale=4, fontsize=9, ncol=3)
    ax.grid(True, lw=0.3, alpha=0.4)
    plt.tight_layout()
    out_path = output_dir / "scatter_all_nozzles.png"
    plt.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ref",
        default="Nozzle0",
        help="Reference experiment name (default: N0; sanitized dataset naming)",
    )
    ap.add_argument(
        "--compare",
        default=None,
        help="Comparison experiment name. If omitted, pool all non-ref experiments.",
    )
    ap.add_argument(
        "--source",
        default="cdf",
        choices=["cdf", "bw_polar"],
        help="Penetration source (default: cdf)",
    )
    ap.add_argument(
        "--dp-exp",
        type=float,
        default=0.25,
        help="ΔP exponent in A_scale (0.25 = legacy Hiroyasu–Arai, 0.50 = production momentum regime)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("c:/Users/Jiang/Documents/Mie_Postprocessing_Py/MLP/figures/injector_comparison"),
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    long = load_normalised(source=args.source, dp_exp=args.dp_exp)
    print(f"  {len(long):,} valid (non-censored) data points across "
          f"{long['experiment_name'].nunique()} experiments")

    # Assign campaign labels
    ref_name = args.ref
    if args.compare:
        cmp_name = args.compare
        long = long[long["experiment_name"].isin([ref_name, cmp_name])].copy()
        long["campaign"] = long["experiment_name"].map(
            {ref_name: "N0 (new)", cmp_name: "N1–N8 (used)"}
        )
    else:
        cmp_name = "N1-N8 pooled"
        long["campaign"] = long["experiment_name"].apply(
            lambda x: "N0 (new)" if x == ref_name else "N1–N8 (used)"
        )

    # Per-condition summary printout
    print("\nA_scale summary per experiment:")
    print(
        long.groupby("experiment_name")["A_scale"]
        .agg(["mean", "std", "min", "max"])
        .round(3)
        .to_string()
    )

    # Bin statistics
    stats = bin_stats(long, label_col="campaign")

    # Ratio table
    ratio = build_ratio_table(stats, "N0 (new)", "N1–N8 (used)", label_col="campaign")
    print(f"\nRatio S*(N0) / S*(N1-N8) at selected t* (after A_scale normalisation):")
    mask = ratio["t_star"].between(0.1, 2.0)
    display = ratio[mask].iloc[::5]   # every ~0.5 in t*
    print(display[["t_star", "S_star_ref", "S_star_cmp", "ratio", "n_ref", "n_cmp"]].to_string(index=False))

    # Save CSVs
    ratio_path = args.output_dir / "ratio_table.csv"
    ratio.to_csv(ratio_path, index=False)
    print(f"\n[saved] {ratio_path}")

    stats_path = args.output_dir / "binned_stats.csv"
    stats.to_csv(stats_path, index=False)
    print(f"[saved] {stats_path}")

    # Plots
    plot_comparison(long, stats, ratio, "N0 (new)", "N1–N8 (used)", args.output_dir, dp_exp=args.dp_exp)
    plot_raw_scatter(long, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
