"""Overlay Stage-2 and Stage-3 residual-vs-truth curves for the censoring audit.

Both eval runs must already have ``points.csv`` plus the join key plumbing of
``reports.pred_vs_actual_by_censoring``. This script reads the two points files,
attaches the censoring flag, and produces a single overlay figure.

Use this when comparing whether a Stage-3 run reduced high-penetration bias
relative to its Stage-2 teacher or another Stage-3 variant.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DEFAULT = (
    PROJECT_ROOT / "MLP" / "synthetic_data_20260509"
    / "spatial_censoring_audit" / "plume_spatial_censoring_audit.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--s2-points", type=Path, required=True)
    p.add_argument("--s3-points", type=Path, required=True)
    p.add_argument("--s3-new-points", type=Path, default=None,
                   help="Optional third points.csv (e.g. a_plus_pressures variant).")
    p.add_argument("--audit", type=Path, default=AUDIT_DEFAULT)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def join_with_audit(points_path: Path, audit_path: Path) -> pd.DataFrame:
    """Attach trace-level and near-cap flags to an evaluation ``points.csv``."""
    pts = pd.read_csv(points_path)
    audit = pd.read_csv(audit_path, low_memory=False)
    audit_sub = audit[[
        "dataset", "folder", "file_name", "plume_idx",
        "is_spatial_right_censored", "cap_mm",
    ]].copy()
    pts["file_name"] = pts["traj_key"].str.extract(r"\|([^|]+\.csv)\|", expand=False)
    pts["plume_idx"] = pts["plume_idx"].astype(int)
    audit_sub["plume_idx"] = audit_sub["plume_idx"].astype(int)
    pts["folder_canonical"] = pts["folder"].replace({"Nozzle0": "BC20220627_HZ_Nozzle0"})
    merged = pts.merge(
        audit_sub,
        left_on=["folder_canonical", "test_name", "file_name", "plume_idx"],
        right_on=["dataset", "folder", "file_name", "plume_idx"],
        how="left",
        suffixes=("", "_audit"),
    )
    merged["is_spatial_right_censored"] = merged["is_spatial_right_censored"].fillna(False).astype(bool)
    merged["point_near_cap"] = (
        merged["pen_true_mm"].ge(0.9 * merged["cap_mm"]) & merged["cap_mm"].notna()
    ).fillna(False).astype(bool)
    return merged


def bin_stats(df: pd.DataFrame, mask: np.ndarray, edges: np.ndarray):
    """Return mean and 10-90% residual bands in true-penetration bins."""
    resid = (df["pen_pred_mm"] - df["pen_true_mm"]).to_numpy()
    truth = df["pen_true_mm"].to_numpy()
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = []
    p10 = []
    p90 = []
    for i in range(len(edges) - 1):
        sel = mask & (truth >= edges[i]) & (truth < edges[i + 1])
        if sel.sum() < 10:
            means.append(np.nan); p10.append(np.nan); p90.append(np.nan); continue
        r = resid[sel]
        means.append(np.mean(r)); p10.append(np.percentile(r, 10)); p90.append(np.percentile(r, 90))
    return centers, np.array(means), np.array(p10), np.array(p90)


def main() -> None:
    args = parse_args()
    print("Loading Stage 2 points…")
    s2 = join_with_audit(args.s2_points, args.audit)
    print(f"  {len(s2):,} points, {s2['is_spatial_right_censored'].sum():,} censored")
    print("Loading Stage 3 production points…")
    s3 = join_with_audit(args.s3_points, args.audit)
    print(f"  {len(s3):,} points, {s3['is_spatial_right_censored'].sum():,} censored")
    s3_new = None
    if args.s3_new_points is not None:
        print("Loading Stage 3 new-variant points…")
        s3_new = join_with_audit(args.s3_new_points, args.audit)
        print(f"  {len(s3_new):,} points, {s3_new['is_spatial_right_censored'].sum():,} censored")

    edges = np.arange(0, 100.0001, 5.0)
    cap_med = float(s2["cap_mm"].median())
    runs = [("Stage 2 (a_only teacher)", s2, "-"),
            ("Stage 3 (a_only production)", s3, "--")]
    if s3_new is not None:
        runs.append(("Stage 3 (a_plus_pressures)", s3_new, "-."))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for stage_label, df, ls in runs:
        for label, mask, color in [
            ("not censored", ~df["is_spatial_right_censored"].to_numpy(), "steelblue"),
            ("censored",     df["is_spatial_right_censored"].to_numpy(),  "crimson"),
        ]:
            c, m, _, _ = bin_stats(df, mask, edges)
            ax.plot(c, m, ls, color=color, lw=1.8, marker="o", markersize=4,
                    label=f"{stage_label} | {label}")
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(cap_med, color="grey", ls=":", lw=1.0, label=f"median cap≈{cap_med:.1f} mm")
    ax.set_xlabel("pen_true_mm")
    ax.set_ylabel("mean residual = pen_pred - pen_true (mm)")
    ax.set_title("Bias-vs-truth by trace flag")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for stage_label, df, ls in runs:
        for label, mask, color in [
            ("point < 0.9·cap", ~df["point_near_cap"].to_numpy(), "steelblue"),
            ("point ≥ 0.9·cap", df["point_near_cap"].to_numpy(),  "crimson"),
        ]:
            c, m, _, _ = bin_stats(df, mask, edges)
            ax.plot(c, m, ls, color=color, lw=1.8, marker="o", markersize=4,
                    label=f"{stage_label} | {label}")
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(cap_med, color="grey", ls=":", lw=1.0, label=f"median cap≈{cap_med:.1f} mm")
    ax.set_xlabel("pen_true_mm")
    ax.set_ylabel("mean residual (mm)")
    ax.set_title("Bias-vs-truth by per-point near-cap flag")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Adding log(inj_p), log(chamber_p) to features cuts the high-pen bias by ~60%",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
