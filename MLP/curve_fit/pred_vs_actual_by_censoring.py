"""Re-plot NN pred_vs_actual coloured by spatial right-censoring flag.

Phase-1.1 Option A. Joins the existing per-plume censoring labels (already in
``MLP/synthetic_data_20260509/spatial_censoring_audit/plume_spatial_censoring_audit.csv``)
onto the most recent NN evaluation ``points.csv`` and produces:

   - pred_vs_actual_by_censoring.png   scatter coloured by flag, with metrics
   - metrics_by_censoring.csv          RMSE / MAE / bias / coverage by group
   - residual_vs_time_by_censoring.png residual structure split by flag

Originally written to test an over-prediction hypothesis. The empirical result
reversed that hypothesis: censored / near-cap points actually exhibit
*under*-prediction (pen_pred < pen_true). The script now simply slices
the residuals by trace-level FOV-censoring and by the per-point near-cap rule
and reports RMSE/MAE/bias/coverage in each slice for downstream diagnosis.

This is useful after ``audit_cdf_spatial_censoring.py`` has produced plume
labels and an NN evaluation has produced ``points.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTS = (
    PROJECT_ROOT / "MLP" / "eval"
    / "rmse_eval_clean_20260509_215303_newdp_seed99_points" / "points.csv"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT / "MLP" / "synthetic_data_20260509"
    / "spatial_censoring_audit" / "plume_spatial_censoring_audit.csv"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "MLP" / "eval"
    / "rmse_eval_clean_20260509_215303_newdp_seed99_points"
)
DEFAULT_THESIS_DIR = PROJECT_ROOT / "Thesis" / "images" / "neural_network_fit_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    p.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--thesis-dir", type=Path, default=DEFAULT_THESIS_DIR,
                   help="Also copy figures here (set empty to skip).")
    p.add_argument("--max-scatter-points", type=int, default=120_000,
                   help="Random subsample for the scatter plot.")
    return p.parse_args()


def load_and_join(points_path: Path, audit_path: Path) -> pd.DataFrame:
    print(f"Loading NN points: {points_path}")
    pts = pd.read_csv(points_path)
    print(f"  {len(pts):,} rows, columns={list(pts.columns)[:8]}…")
    print(f"Loading censoring audit: {audit_path}")
    audit = pd.read_csv(audit_path, low_memory=False)
    print(f"  {len(audit):,} rows")

    audit_subset = audit[[
        "dataset", "folder", "file_name", "plume_idx",
        "is_spatial_right_censored", "raw_near_cap", "clean_near_cap",
        "cap_mm", "clean_last_mm", "first_cap_hit_time_ms", "raw_last_positive_ms",
    ]].copy()

    pts["file_name"] = pts["traj_key"].str.extract(r"\|([^|]+\.csv)\|", expand=False)
    pts["plume_idx"] = pts["plume_idx"].astype(int)
    audit_subset["plume_idx"] = audit_subset["plume_idx"].astype(int)
    # Map evaluator-side folder aliases (e.g. "Nozzle0") to the audit's dataset
    # name (e.g. "BC20220627_HZ_Nozzle0").
    folder_alias = {"Nozzle0": "BC20220627_HZ_Nozzle0"}
    pts["folder_canonical"] = pts["folder"].replace(folder_alias)

    merged = pts.merge(
        audit_subset,
        left_on=["folder_canonical", "test_name", "file_name", "plume_idx"],
        right_on=["dataset", "folder", "file_name", "plume_idx"],
        how="left",
        suffixes=("", "_audit"),
    )
    n_matched = merged["is_spatial_right_censored"].notna().sum()
    print(f"Matched {n_matched:,}/{len(merged):,} points "
          f"({100*n_matched/len(merged):.1f}%) with audit flag.")
    unmatched_traj = merged.loc[merged["is_spatial_right_censored"].isna(), "traj_key"].nunique()
    if unmatched_traj:
        print(f"  ({unmatched_traj} distinct traj_key without a match — likely cleaning/audit mismatch.)")

    merged["is_spatial_right_censored"] = merged["is_spatial_right_censored"].fillna(False).astype(bool)
    # Per-point flag: this specific (time, pen_true) point is near or above the FOV cap.
    # Uses 0.9 * cap_mm as the near-cap threshold (matches the proposed Step-5 rule).
    cap = merged["cap_mm"]
    merged["point_near_cap"] = (
        merged["pen_true_mm"].ge(0.9 * cap)
        & cap.notna()
    ).fillna(False).astype(bool)
    return merged


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {"n": 0, "rmse_mm": np.nan, "mae_mm": np.nan, "bias_mm": np.nan,
                "p95_abs_err_mm": np.nan, "coverage_1sigma": np.nan, "coverage_2sigma": np.nan}
    resid = df["pen_pred_mm"].to_numpy() - df["pen_true_mm"].to_numpy()
    abs_err = np.abs(resid)
    sigma = df["pen_std_mm"].to_numpy()
    return {
        "n": int(len(df)),
        "rmse_mm": float(np.sqrt(np.mean(resid**2))),
        "mae_mm": float(np.mean(abs_err)),
        "bias_mm": float(np.mean(resid)),
        "p95_abs_err_mm": float(np.percentile(abs_err, 95)),
        "coverage_1sigma": float(np.mean(abs_err <= sigma)),
        "coverage_2sigma": float(np.mean(abs_err <= 2 * sigma)),
    }


def write_metrics_table(merged: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    rows = []
    cap_th = 0.9 * merged["cap_mm"]
    for label, sub in [
        ("all", merged),
        ("trace_not_censored", merged[~merged["is_spatial_right_censored"]]),
        ("trace_censored", merged[merged["is_spatial_right_censored"]]),
        ("point_below_0.9cap", merged[~merged["point_near_cap"]]),
        ("point_near_or_above_0.9cap", merged[merged["point_near_cap"]]),
        ("trace_censored_AND_point_near_cap",
         merged[merged["is_spatial_right_censored"] & merged["point_near_cap"]]),
        ("trace_censored_AND_point_below_cap",
         merged[merged["is_spatial_right_censored"] & ~merged["point_near_cap"]]),
    ]:
        m = compute_metrics(sub)
        m["group"] = label
        rows.append(m)
    # also per-folder x censoring
    for (folder, group), sub in merged.groupby(["folder", "is_spatial_right_censored"]):
        m = compute_metrics(sub)
        m["group"] = f"{folder}__{'censored' if group else 'not_censored'}"
        rows.append(m)
    table = pd.DataFrame(rows)[
        ["group", "n", "rmse_mm", "mae_mm", "bias_mm", "p95_abs_err_mm",
         "coverage_1sigma", "coverage_2sigma"]
    ]
    table.to_csv(out_csv, index=False)
    return table


def plot_pred_vs_actual(merged: pd.DataFrame, out_png: Path, max_points: int) -> None:
    rng = np.random.default_rng(2026)
    not_c = merged[~merged["is_spatial_right_censored"]]
    cen = merged[merged["is_spatial_right_censored"]]
    if len(not_c) > max_points:
        not_c = not_c.sample(n=max_points, random_state=int(rng.integers(0, 2**31 - 1)))
    if len(cen) > max_points:
        cen = cen.sample(n=max_points, random_state=int(rng.integers(0, 2**31 - 1)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    ax.scatter(not_c["pen_true_mm"], not_c["pen_pred_mm"],
               s=2, c="steelblue", alpha=0.18, label=f"not censored (n={len(not_c):,})")
    ax.scatter(cen["pen_true_mm"], cen["pen_pred_mm"],
               s=2, c="crimson", alpha=0.25, label=f"censored (n={len(cen):,})")
    lo = float(min(merged["pen_true_mm"].min(), merged["pen_pred_mm"].min()))
    hi = float(max(merged["pen_true_mm"].max(), merged["pen_pred_mm"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
    ax.set_xlabel("pen_true_mm")
    ax.set_ylabel("pen_pred_mm")
    ax.set_title("Predicted vs measured (coloured by spatial right-censoring)")
    ax.legend(loc="upper left", fontsize=9, markerscale=3, framealpha=0.85)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    bin_edges = np.linspace(-60, 60, 121)
    not_c_resid = (not_c["pen_pred_mm"] - not_c["pen_true_mm"]).to_numpy()
    cen_resid = (cen["pen_pred_mm"] - cen["pen_true_mm"]).to_numpy()
    ax.hist(not_c_resid, bins=bin_edges, color="steelblue", alpha=0.55,
            density=True, label=f"not censored  bias={not_c_resid.mean():.2f} mm")
    ax.hist(cen_resid, bins=bin_edges, color="crimson", alpha=0.55,
            density=True, label=f"censored  bias={cen_resid.mean():.2f} mm")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("residual = pen_pred - pen_true (mm)")
    ax.set_ylabel("density")
    ax.set_title("Residual distribution by group")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Predicted vs measured and residual distribution, sliced by FOV right-censoring.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_residual_vs_truth(merged: pd.DataFrame, out_png: Path) -> None:
    """Bin residual by pen_true_mm in 5mm bins, separate by trace flag."""
    bin_edges = np.arange(0, 100.0001, 5.0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    resid = (merged["pen_pred_mm"] - merged["pen_true_mm"]).to_numpy()
    truth = merged["pen_true_mm"].to_numpy()
    flag = merged["is_spatial_right_censored"].to_numpy()
    cap_med = float(merged["cap_mm"].median())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for label, mask, color in [
        ("not censored", ~flag, "steelblue"),
        ("censored",     flag,  "crimson"),
    ]:
        means = []
        p10s = []
        p90s = []
        counts = []
        for i in range(len(bin_edges) - 1):
            sel = mask & (truth >= bin_edges[i]) & (truth < bin_edges[i + 1])
            if sel.sum() == 0:
                means.append(np.nan); p10s.append(np.nan); p90s.append(np.nan); counts.append(0)
                continue
            r = resid[sel]
            means.append(np.mean(r))
            p10s.append(np.percentile(r, 10))
            p90s.append(np.percentile(r, 90))
            counts.append(int(sel.sum()))
        means = np.array(means); p10s = np.array(p10s); p90s = np.array(p90s)
        ax.fill_between(bin_centers, p10s, p90s, color=color, alpha=0.18,
                        label=f"{label}: 10-90% band")
        ax.plot(bin_centers, means, color=color, marker="o", lw=1.6,
                label=f"{label}: mean bias")
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(cap_med, color="grey", ls=":", lw=1.0, label=f"median cap≈{cap_med:.1f} mm")
    ax.set_xlabel("pen_true_mm")
    ax.set_ylabel("residual = pen_pred - pen_true (mm)")
    ax.set_title("Residual vs truth, binned (5 mm), by trace-level censoring")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Same plot but stratified by per-POINT near-cap rule
    near = merged["point_near_cap"].to_numpy()
    for label, mask, color in [
        ("point below 0.9·cap", ~near, "steelblue"),
        ("point ≥ 0.9·cap",     near,  "crimson"),
    ]:
        means = []
        p10s = []
        p90s = []
        for i in range(len(bin_edges) - 1):
            sel = mask & (truth >= bin_edges[i]) & (truth < bin_edges[i + 1])
            if sel.sum() == 0:
                means.append(np.nan); p10s.append(np.nan); p90s.append(np.nan)
                continue
            r = resid[sel]
            means.append(np.mean(r))
            p10s.append(np.percentile(r, 10))
            p90s.append(np.percentile(r, 90))
        means = np.array(means); p10s = np.array(p10s); p90s = np.array(p90s)
        ax.fill_between(bin_centers, p10s, p90s, color=color, alpha=0.18,
                        label=f"{label}: 10-90% band")
        ax.plot(bin_centers, means, color=color, marker="o", lw=1.6,
                label=f"{label}: mean bias")
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(cap_med, color="grey", ls=":", lw=1.0, label=f"median cap≈{cap_med:.1f} mm")
    ax.set_xlabel("pen_true_mm")
    ax.set_ylabel("residual (mm)")
    ax.set_title("Residual vs truth, by per-point near-cap flag")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Bias vs truth in 5 mm bins, sliced by trace-level FOV censoring and per-point near-cap flag.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def maybe_copy_to_thesis(out_dir: Path, thesis_dir: Path) -> None:
    if thesis_dir is None:
        return
    # Path("") and Path(".") both resolve to the cwd; treat empty/dot as "skip".
    raw = str(thesis_dir).strip()
    if raw in {"", "."}:
        return
    if not thesis_dir.exists():
        return
    import shutil
    for fn in ["pred_vs_actual_by_censoring.png", "residual_vs_truth_by_censoring.png"]:
        src = out_dir / fn
        if src.exists():
            shutil.copy2(src, thesis_dir / f"latest_{fn.replace('.png', '_seed99.png')}")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_and_join(args.points, args.audit)
    merged.to_parquet(out_dir / "points_with_censoring.parquet", index=False) if False else None

    table = write_metrics_table(merged, out_dir / "metrics_by_censoring.csv")
    primary_groups = [
        "all",
        "trace_not_censored",
        "trace_censored",
        "point_below_0.9cap",
        "point_near_or_above_0.9cap",
        "trace_censored_AND_point_near_cap",
        "trace_censored_AND_point_below_cap",
    ]
    primary = (
        table.loc[table["group"].isin(primary_groups)]
        .set_index("group")
        .reindex(primary_groups)
        .reset_index()
    )
    print("\nMetrics by primary slice:")
    print(primary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plot_pred_vs_actual(
        merged, out_dir / "pred_vs_actual_by_censoring.png", args.max_scatter_points)
    plot_residual_vs_truth(
        merged, out_dir / "residual_vs_truth_by_censoring.png")
    maybe_copy_to_thesis(out_dir, args.thesis_dir)

    print(f"\nWrote outputs to {out_dir}")
    if args.thesis_dir and args.thesis_dir.exists():
        print(f"Mirrored figures to {args.thesis_dir}")


if __name__ == "__main__":
    main()
