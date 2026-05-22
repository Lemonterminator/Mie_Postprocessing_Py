"""Build the thesis-sync evaluation report for the new Stage-3 production model
(``kd_mode=mse_mu_plus_sigma``, ``kd_sigma_weight=5.0``).

The script loads:
  - the eval points (``points.csv``) from the w=5.0 run,
  - the Stage 2 teacher points (for sigma calibration reference),
  - the regime labels (``cdf_plume_audit.csv``) emitted during Stage-3 training,
  - the per-plume FOV audit (``plume_spatial_censoring_audit.csv``).

It then computes per-slice metrics (RMSE/MAE/bias, sigma calibration) for:
  - all points,
  - raw_reliable points (where the model was supervised by data),
  - raw_uncertain + teacher_only (KD-only supervision),
  - point < 0.9*cap, point >= 0.9*cap,
  - trace_censored vs trace_not_censored.

Figures saved:
  - pred-vs-actual coloured by regime,
  - residual-vs-truth by regime + by censoring,
  - sigma_student vs sigma_teacher hexbin / scatter,
  - bias-vs-truth (5 mm bins) overlay.

Outputs go to:
  - Thesis/images/neural_network_fit_results/*.png
  - Thesis/generated/current/stage3_kd_mse_mu_plus_sigma_metrics.csv

This is a reporting/synchronization script, not a training script: it assumes
the named Stage-3, Stage-2, and baseline evaluation runs already exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THESIS_IMG_DIR = PROJECT_ROOT / "Thesis" / "images" / "neural_network_fit_results"
THESIS_GEN_DIR = PROJECT_ROOT / "Thesis" / "generated" / "current"

DEFAULT_RUN_DIR = (
    PROJECT_ROOT / "MLP" / "runs_mlp"
    / "stage3_diag_kd_mse_mu_plus_sigma_w5p0_20260519_153706"
)
DEFAULT_EVAL_POINTS = (
    PROJECT_ROOT / "MLP" / "eval"
    / "rmse_eval_clean_20260519_154036_stage3_diag_kd_mse_mu_plus_sigma_w5p0_20260519_153706"
    / "points.csv"
)
DEFAULT_TEACHER_POINTS = (
    PROJECT_ROOT / "MLP" / "eval"
    / "rmse_eval_clean_20260519_134117_stage2_aplus_pressures_teacher_audit"
    / "points.csv"
)
DEFAULT_BASELINE_POINTS = (
    PROJECT_ROOT / "MLP" / "eval"
    / "rmse_eval_clean_20260519_004049_distill_cdf_onset_v2_ablate_anchor_off_20260519_003814"
    / "points.csv"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT / "MLP" / "synthetic_data_20260509"
    / "spatial_censoring_audit" / "plume_spatial_censoring_audit.csv"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--eval-points", type=Path, default=DEFAULT_EVAL_POINTS)
    p.add_argument("--teacher-points", type=Path, default=DEFAULT_TEACHER_POINTS)
    p.add_argument("--baseline-points", type=Path, default=DEFAULT_BASELINE_POINTS)
    p.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--img-out", type=Path, default=THESIS_IMG_DIR)
    p.add_argument("--gen-out", type=Path, default=THESIS_GEN_DIR)
    p.add_argument("--regime-bin-ms", type=float, default=0.1)
    return p.parse_args()


def prep_points(pts_path: Path, audit_sub: pd.DataFrame) -> pd.DataFrame:
    """Load eval points and attach trace-level spatial-censoring labels."""
    pts = pd.read_csv(pts_path)
    pts["file_name"] = pts["traj_key"].str.extract(r"\|([^|]+\.csv)\|", expand=False)
    pts["plume_idx"] = pts["plume_idx"].astype(int)
    pts["folder_canonical"] = pts["folder"].replace({"Nozzle0": "BC20220627_HZ_Nozzle0"})
    m = pts.merge(
        audit_sub,
        left_on=["folder_canonical", "test_name", "file_name", "plume_idx"],
        right_on=["dataset", "folder", "file_name", "plume_idx"],
        how="left",
        suffixes=("", "_a"),
    )
    m["is_spatial_right_censored"] = m["is_spatial_right_censored"].fillna(False).astype(bool)
    m["point_near_cap"] = (
        m["pen_true_mm"].ge(0.9 * m["cap_mm"]) & m["cap_mm"].notna()
    ).fillna(False).astype(bool)
    m["resid"] = m["pen_pred_mm"] - m["pen_true_mm"]
    return m


def attach_regime(points: pd.DataFrame, plume_audit_path: Path, regime_bin_ms: float) -> pd.DataFrame:
    """Attach Stage-3 supervision regime labels by condition and time bin."""
    plume_audit = pd.read_csv(plume_audit_path, low_memory=False)
    plume_meta = (
        plume_audit[["experiment_name", "file_name", "plume_idx",
                     "plumes", "diameter_mm", "umbrella_angle_deg",
                     "fps", "chamber_pressure_bar", "injection_duration_us",
                     "injection_pressure_bar", "control_backpressure_bar"]]
        .drop_duplicates(subset=["experiment_name", "file_name", "plume_idx"])
        .copy()
    )
    plume_meta = plume_meta.rename(columns={"experiment_name": "folder_canonical"})
    plume_meta["plume_idx"] = plume_meta["plume_idx"].astype(int)
    # points.csv may already carry some condition columns (e.g. injection_duration_us);
    # drop those from the bridge so the merge keys stay unambiguous.
    overlap = [c for c in plume_meta.columns
               if c in points.columns and c not in ("folder_canonical", "file_name", "plume_idx")]
    plume_meta = plume_meta.drop(columns=overlap)
    pts = points.merge(plume_meta, on=["folder_canonical", "file_name", "plume_idx"], how="left")

    regime_bins = pd.read_csv(plume_audit_path.parent / "cdf_regime_bins.csv")
    regime_bins = regime_bins.rename(columns={"experiment_name": "folder_canonical"})
    pts["time_bin"] = np.floor(pts["time_ms"] / regime_bin_ms).astype(int)
    join_cols = [
        "folder_canonical", "plumes", "diameter_mm", "umbrella_angle_deg",
        "fps", "chamber_pressure_bar", "injection_duration_us",
        "injection_pressure_bar", "control_backpressure_bar", "time_bin",
    ]
    pts = pts.merge(regime_bins[join_cols + ["regime"]], on=join_cols, how="left")
    pts["regime"] = pts["regime"].fillna("teacher_only")  # past the labeled horizon
    return pts


def slice_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0, "rmse_mm": float("nan"), "mae_mm": float("nan"), "bias_mm": float("nan")}
    return {
        "n": int(len(df)),
        "rmse_mm": float(np.sqrt((df["resid"] ** 2).mean())),
        "mae_mm": float(df["resid"].abs().mean()),
        "bias_mm": float(df["resid"].mean()),
    }


def build_metrics_table(student: pd.DataFrame, teacher_sigma: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    slices = {
        "all": student,
        "raw_reliable": student[student["regime"] == "raw_reliable"],
        "raw_uncertain": student[student["regime"] == "raw_uncertain"],
        "teacher_only": student[student["regime"] == "teacher_only"],
        "kd_only (raw_uncertain + teacher_only)": student[student["regime"].isin(["raw_uncertain", "teacher_only"])],
        "point_below_0.9_cap": student[~student["point_near_cap"]],
        "point_near_or_above_0.9_cap": student[student["point_near_cap"]],
        "trace_not_censored": student[~student["is_spatial_right_censored"]],
        "trace_censored": student[student["is_spatial_right_censored"]],
        "trace_censored & near_cap": student[student["is_spatial_right_censored"] & student["point_near_cap"]],
    }
    for name, g in slices.items():
        m = slice_metrics(g)
        # sigma calibration on the same slice
        if len(g) > 0 and "pen_std_mm" in g.columns:
            with_t = g.merge(teacher_sigma, on=["folder_canonical", "test_name", "file_name", "plume_idx", "time_ms"], how="left")
            dsig = (with_t["pen_std_mm"] - with_t["teacher_sigma_mm"]).dropna()
            m["sigma_med_mm"] = float(g["pen_std_mm"].median())
            m["abs_dsigma_med_mm"] = float(dsig.abs().median()) if len(dsig) else float("nan")
            m["dsigma_iqr_mm"] = float(dsig.quantile(0.75) - dsig.quantile(0.25)) if len(dsig) else float("nan")
        else:
            m.update({"sigma_med_mm": float("nan"), "abs_dsigma_med_mm": float("nan"), "dsigma_iqr_mm": float("nan")})
        m["run"] = label
        m["slice"] = name
        rows.append(m)
    return pd.DataFrame(rows)[["run", "slice", "n", "rmse_mm", "mae_mm", "bias_mm",
                                "sigma_med_mm", "abs_dsigma_med_mm", "dsigma_iqr_mm"]]


def plot_pred_vs_actual_by_regime(student: pd.DataFrame, out_path: Path) -> None:
    regimes = ["raw_reliable", "raw_uncertain", "teacher_only"]
    colors = {"raw_reliable": "steelblue", "raw_uncertain": "darkorange", "teacher_only": "crimson"}
    fig, ax = plt.subplots(figsize=(7, 7))
    for r in regimes:
        sub = student[student["regime"] == r]
        if len(sub) == 0:
            continue
        ax.scatter(sub["pen_true_mm"], sub["pen_pred_mm"], s=1.2, c=colors[r], alpha=0.18, rasterized=True,
                   label=f"{r}  (n={len(sub):,})")
    lim = float(np.nanmax([student["pen_true_mm"].max(), student["pen_pred_mm"].max()])) * 1.05
    ax.plot([0, lim], [0, lim], color="k", lw=0.7)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("pen_true (mm)"); ax.set_ylabel("pen_pred (mm)")
    ax.set_title("Stage-3 kd_mse_mu_plus_sigma (w=5.0): predictions by supervision regime")
    ax.legend(markerscale=4, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_residual_vs_truth_by_slice(student: pd.DataFrame, out_path: Path) -> None:
    bins = np.arange(0, 100.01, 5.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for r, color in [("raw_reliable", "steelblue"), ("raw_uncertain", "darkorange"), ("teacher_only", "crimson")]:
        sub = student[student["regime"] == r]
        if len(sub) == 0:
            continue
        means = []; p10s = []; p90s = []
        for i in range(len(bins) - 1):
            sel = (sub["pen_true_mm"] >= bins[i]) & (sub["pen_true_mm"] < bins[i + 1])
            r_vals = sub.loc[sel, "resid"].to_numpy()
            if len(r_vals) < 10:
                means.append(np.nan); p10s.append(np.nan); p90s.append(np.nan); continue
            means.append(r_vals.mean()); p10s.append(np.percentile(r_vals, 10)); p90s.append(np.percentile(r_vals, 90))
        ax.plot(centers, means, "-", color=color, lw=1.8, marker="o", markersize=4, label=f"{r}")
        ax.fill_between(centers, p10s, p90s, color=color, alpha=0.10)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("pen_true (mm)"); ax.set_ylabel("residual = pred − true (mm)")
    ax.set_title("Bias vs truth, by Stage-3 supervision regime")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for label_, mask, color in [
        ("trace_not_censored", ~student["is_spatial_right_censored"], "steelblue"),
        ("trace_censored",     student["is_spatial_right_censored"],  "crimson"),
    ]:
        sub = student[mask]
        if len(sub) == 0:
            continue
        means = []; p10s = []; p90s = []
        for i in range(len(bins) - 1):
            sel = (sub["pen_true_mm"] >= bins[i]) & (sub["pen_true_mm"] < bins[i + 1])
            r_vals = sub.loc[sel, "resid"].to_numpy()
            if len(r_vals) < 10:
                means.append(np.nan); p10s.append(np.nan); p90s.append(np.nan); continue
            means.append(r_vals.mean()); p10s.append(np.percentile(r_vals, 10)); p90s.append(np.percentile(r_vals, 90))
        ax.plot(centers, means, "-", color=color, lw=1.8, marker="o", markersize=4, label=label_)
        ax.fill_between(centers, p10s, p90s, color=color, alpha=0.10)
    ax.axhline(0, color="k", lw=0.7)
    cap_med = float(student["cap_mm"].median())
    ax.axvline(cap_med, color="grey", ls=":", lw=1.0, label=f"median cap≈{cap_med:.1f} mm")
    ax.set_xlabel("pen_true (mm)"); ax.set_ylabel("residual (mm)")
    ax.set_title("Bias vs truth, by trace FOV-censoring flag")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle("Stage-3 kd_mse_mu_plus_sigma (w=5.0): residual structure", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sigma_calibration(student: pd.DataFrame, teacher_sigma: pd.DataFrame, out_path: Path) -> None:
    merged = student.merge(teacher_sigma, on=["folder_canonical", "test_name", "file_name", "plume_idx", "time_ms"], how="inner")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    hb = ax.hexbin(merged["teacher_sigma_mm"], merged["pen_std_mm"], gridsize=80, mincnt=1, cmap="viridis", bins="log")
    lim = float(np.nanmax([merged["teacher_sigma_mm"].max(), merged["pen_std_mm"].max()])) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("teacher σ (mm)"); ax.set_ylabel("student σ (mm)")
    ax.set_title("σ calibration: student vs teacher")
    plt.colorbar(hb, ax=ax, label="log10(count)")

    ax = axes[1]
    dsig = merged["pen_std_mm"] - merged["teacher_sigma_mm"]
    for label_, mask, color in [
        ("raw_reliable", merged["regime"] == "raw_reliable", "steelblue"),
        ("raw_uncertain", merged["regime"] == "raw_uncertain", "darkorange"),
        ("teacher_only", merged["regime"] == "teacher_only", "crimson"),
    ]:
        v = dsig[mask].dropna()
        if len(v) == 0:
            continue
        ax.hist(v.clip(-5, 5), bins=80, histtype="step", color=color, lw=1.5,
                label=f"{label_}  med={v.median():+.2f}  |med|={v.abs().median():.2f}")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("σ_student − σ_teacher (mm)"); ax.set_ylabel("count")
    ax.set_title("σ residual by regime")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Stage-3 kd_mse_mu_plus_sigma (w=5.0): σ calibration", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_overlay_with_baseline(student: pd.DataFrame, baseline: pd.DataFrame, teacher: pd.DataFrame, out_path: Path) -> None:
    bins = np.arange(0, 100.01, 5.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(8, 5))
    for label_, df, color, ls in [
        ("Stage 2 teacher", teacher, "tab:gray", ":"),
        ("Stage 3 baseline (fwd_kl)", baseline, "crimson", "--"),
        ("Stage 3 kd_mse_mu_plus_sigma (w=5.0)", student, "tab:blue", "-"),
    ]:
        means = []
        for i in range(len(bins) - 1):
            sel = (df["pen_true_mm"] >= bins[i]) & (df["pen_true_mm"] < bins[i + 1])
            r_vals = df.loc[sel, "resid"].to_numpy()
            means.append(r_vals.mean() if len(r_vals) >= 10 else np.nan)
        ax.plot(centers, means, ls, color=color, lw=2.0, marker="o", markersize=4, label=label_)
    ax.axhline(0, color="k", lw=0.7)
    cap_med = float(student["cap_mm"].median())
    ax.axvline(cap_med, color="grey", ls=":", lw=1.0, label=f"median cap≈{cap_med:.1f} mm")
    ax.set_xlabel("pen_true (mm)"); ax.set_ylabel("mean residual (mm)")
    ax.set_title("Stage 2 teacher vs Stage 3 baseline vs Stage 3 kd_mse_mu_plus_sigma")
    ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.img_out.mkdir(parents=True, exist_ok=True)
    args.gen_out.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(args.audit, low_memory=False)
    audit_sub = audit[["dataset", "folder", "file_name", "plume_idx",
                       "is_spatial_right_censored", "cap_mm"]].copy()
    audit_sub["plume_idx"] = audit_sub["plume_idx"].astype(int)

    print("Loading student points (kd_mse_mu_plus_sigma w=5.0)...")
    student = prep_points(args.eval_points, audit_sub)
    print(f"  {len(student):,} points")
    print("Attaching regime labels...")
    plume_audit_path = args.run_dir / "cdf_plume_audit.csv"
    student = attach_regime(student, plume_audit_path, regime_bin_ms=args.regime_bin_ms)
    print(student["regime"].value_counts().to_string())

    print("\nLoading teacher points (Stage 2 a_plus_pressures)...")
    teacher = prep_points(args.teacher_points, audit_sub)
    print(f"  {len(teacher):,} points")
    teacher_sigma_key = teacher[["folder_canonical", "test_name", "file_name", "plume_idx", "time_ms"]].copy()
    teacher_sigma_key["teacher_sigma_mm"] = teacher["pen_std_mm"]

    print("Loading baseline points (Stage 3 forward_kl)...")
    baseline = prep_points(args.baseline_points, audit_sub)

    print("\nComputing metrics table...")
    table = build_metrics_table(student, teacher_sigma_key, label="kd_mse_mu_plus_sigma w=5.0")
    table.to_csv(args.gen_out / "stage3_kd_mse_mu_plus_sigma_metrics.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nWriting figures...")
    plot_pred_vs_actual_by_regime(student, args.img_out / "stage3_kd_mse_mu_plus_sigma_pred_vs_actual.png")
    plot_residual_vs_truth_by_slice(student, args.img_out / "stage3_kd_mse_mu_plus_sigma_residual_vs_truth.png")
    plot_sigma_calibration(student, teacher_sigma_key, args.img_out / "stage3_kd_mse_mu_plus_sigma_sigma_calibration.png")
    plot_overlay_with_baseline(student, baseline, teacher, args.img_out / "stage3_kd_mse_mu_plus_sigma_overlay_baseline.png")

    summary = {
        "run_dir": str(args.run_dir),
        "eval_points": str(args.eval_points),
        "n_points_total": int(len(student)),
        "n_raw_reliable": int((student["regime"] == "raw_reliable").sum()),
        "n_raw_uncertain": int((student["regime"] == "raw_uncertain").sum()),
        "n_teacher_only": int((student["regime"] == "teacher_only").sum()),
        "n_trace_censored": int(student["is_spatial_right_censored"].sum()),
        "n_point_near_cap": int(student["point_near_cap"].sum()),
        "overall_rmse_mm": float(np.sqrt((student["resid"] ** 2).mean())),
        "overall_bias_mm": float(student["resid"].mean()),
    }
    with open(args.gen_out / "stage3_kd_mse_mu_plus_sigma_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.gen_out / 'stage3_kd_mse_mu_plus_sigma_summary.json'}")


if __name__ == "__main__":
    main()
