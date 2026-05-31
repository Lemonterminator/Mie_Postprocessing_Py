"""Build calibration and coverage diagnostics from Stage-3 point predictions."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = PROJECT_ROOT / "MLP" / "eval" / "rmse_eval_clean_20260429_130733_winner_full"
DEFAULT_THESIS_IMAGE_DIR = PROJECT_ROOT / "Thesis" / "images"


def _normal_central_coverage(k: float) -> float:
    return math.erf(float(k) / math.sqrt(2.0))


def build_calibration_tables(points_df: pd.DataFrame, *, n_bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"pen_true_mm", "pen_pred_mm", "pen_std_mm", "resid_mm"}
    missing = sorted(required - set(points_df.columns))
    if missing:
        raise KeyError(f"points.csv is missing required columns: {', '.join(missing)}")

    df = points_df.copy()
    df["abs_err_mm"] = df["resid_mm"].abs()
    df["std_safe_mm"] = df["pen_std_mm"].clip(lower=1e-9)
    df["abs_z"] = df["abs_err_mm"] / df["std_safe_mm"]

    try:
        df["sigma_bin"] = pd.qcut(df["std_safe_mm"], q=n_bins, duplicates="drop")
    except ValueError:
        df["sigma_bin"] = pd.cut(df["std_safe_mm"], bins=min(n_bins, 3), include_lowest=True)

    by_sigma = (
        df.groupby("sigma_bin", observed=True)
        .agg(
            n=("abs_err_mm", "size"),
            mean_sigma_mm=("std_safe_mm", "mean"),
            median_sigma_mm=("std_safe_mm", "median"),
            mae_mm=("abs_err_mm", "mean"),
            rmse_mm=("resid_mm", lambda s: float(np.sqrt(np.mean(np.asarray(s, dtype=float) ** 2)))),
            coverage_1sigma=("abs_z", lambda s: float(np.mean(np.asarray(s, dtype=float) <= 1.0))),
            coverage_2sigma=("abs_z", lambda s: float(np.mean(np.asarray(s, dtype=float) <= 2.0))),
            mean_abs_z=("abs_z", "mean"),
            p90_abs_z=("abs_z", lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.90))),
        )
        .reset_index()
    )
    by_sigma["sigma_bin"] = by_sigma["sigma_bin"].astype(str)

    k_values = np.asarray([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0], dtype=float)
    abs_z = df["abs_z"].to_numpy(dtype=float)
    coverage_curve = pd.DataFrame(
        {
            "k_sigma": k_values,
            "nominal_gaussian_coverage": [_normal_central_coverage(k) for k in k_values],
            "empirical_coverage": [float(np.mean(abs_z <= k)) for k in k_values],
        }
    )
    coverage_curve["coverage_error"] = (
        coverage_curve["empirical_coverage"] - coverage_curve["nominal_gaussian_coverage"]
    )

    return by_sigma, coverage_curve


def plot_calibration(by_sigma: pd.DataFrame, coverage_curve: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), dpi=180)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    nominal = pd.to_numeric(coverage_curve["nominal_gaussian_coverage"], errors="coerce").to_numpy(dtype=float)
    empirical = pd.to_numeric(coverage_curve["empirical_coverage"], errors="coerce").to_numpy(dtype=float)
    ax.fill_between([0.0, 1.0], [0.0, 1.0], [1.0, 1.0], color="#59a14f", alpha=0.08, label="conservative")
    ax.plot([0, 1], [0, 1], color="#222222", linestyle="--", linewidth=1.1, label="ideal")
    ax.plot(nominal, empirical, marker="o", linewidth=2.0, color="#4c78a8", label="empirical")
    ax.set_xlabel("Nominal Gaussian central coverage")
    ax.set_ylabel("Empirical central coverage")
    ax.set_title("Central-interval coverage")
    ax.set_xlim(0.35, 1.0)
    ax.set_ylim(0.35, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    ax = axes[1]
    x = np.arange(1, len(by_sigma) + 1)
    mean_sigma = pd.to_numeric(by_sigma["mean_sigma_mm"], errors="coerce").to_numpy(dtype=float)
    mae = pd.to_numeric(by_sigma["mae_mm"], errors="coerce").to_numpy(dtype=float)
    cov1 = pd.to_numeric(by_sigma["coverage_1sigma"], errors="coerce").to_numpy(dtype=float)
    cov2 = pd.to_numeric(by_sigma["coverage_2sigma"], errors="coerce").to_numpy(dtype=float)

    sigma_line = ax.plot(x, mean_sigma, marker="o", linewidth=2.0, color="#4c78a8", label="mean predicted sigma [mm]")
    mae_line = ax.plot(x, mae, marker="s", linewidth=2.0, color="#f28e2b", label="MAE [mm]")
    ax.set_xticks(x)
    ax.set_xlabel("Predicted-sigma quantile bin")
    ax.set_ylabel("Error scale [mm]")
    ax.set_title("Sharpness, error, and coverage by sigma bin")
    y_max = float(np.nanmax(np.concatenate([mean_sigma, mae]))) if len(by_sigma) else 1.0
    ax.set_ylim(0.0, max(1.0, y_max * 1.12))
    ax.grid(True, axis="y", alpha=0.22)

    ax_cov = ax.twinx()
    cov_bar = ax_cov.bar(x, cov1, width=0.52, alpha=0.26, color="#9ecae1", label="1-sigma coverage")
    cov2_line = ax_cov.plot(x, cov2, marker="^", linewidth=1.7, color="#59a14f", label="2-sigma coverage")
    nominal_line = ax_cov.axhline(0.682689, color="#525252", linestyle=":", linewidth=1.2, label="nominal 1-sigma")
    ax_cov.set_ylim(0.0, 1.05)
    ax_cov.set_ylabel("Empirical coverage")

    handles = [*sigma_line, *mae_line, cov_bar, *cov2_line, nominal_line]
    labels = [handle.get_label() for handle in handles]
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.22, 0.5),
        borderaxespad=0.0,
        fontsize=7.5,
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_audit(
    *,
    eval_dir: Path,
    output_dir: Path | None = None,
    n_bins: int = 10,
    thesis_image_dir: Path | None = None,
) -> dict:
    points_path = eval_dir / "points.csv"
    if not points_path.exists():
        raise FileNotFoundError(f"points.csv not found: {points_path}")

    out_dir = output_dir or (eval_dir / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    points_df = pd.read_csv(points_path, low_memory=False)
    by_sigma, coverage_curve = build_calibration_tables(points_df, n_bins=n_bins)
    by_sigma.to_csv(out_dir / "calibration_by_sigma_bin.csv", index=False)
    coverage_curve.to_csv(out_dir / "coverage_curve.csv", index=False)
    plot_path = out_dir / "calibration_coverage.png"
    plot_calibration(by_sigma, coverage_curve, plot_path)

    summary = {
        "points_csv": str(points_path),
        "n_points": int(len(points_df)),
        "n_sigma_bins": int(len(by_sigma)),
        "coverage_1sigma": float(np.mean(np.abs(points_df["resid_mm"]) <= np.maximum(points_df["pen_std_mm"], 1e-9))),
        "coverage_2sigma": float(np.mean(np.abs(points_df["resid_mm"]) <= 2.0 * np.maximum(points_df["pen_std_mm"], 1e-9))),
        "mean_pred_std_mm": float(points_df["pen_std_mm"].mean()),
        "mean_abs_err_mm": float(points_df["resid_mm"].abs().mean()),
        "plot": str(plot_path),
    }
    (out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if thesis_image_dir is not None:
        thesis_image_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plot_path, thesis_image_dir / "stage3_calibration_coverage.png")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--thesis-image-dir", type=Path, default=DEFAULT_THESIS_IMAGE_DIR)
    parser.add_argument("--no-thesis-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_audit(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
        thesis_image_dir=None if args.no_thesis_copy else args.thesis_image_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
