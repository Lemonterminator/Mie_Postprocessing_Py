"""Calibration diagnostics: reliability, ECE, CRPS, and PIT for HA/NS/MLP/SVGP."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

REPORT_ROOT = PROJECT_ROOT / "MLP" / "baseline" / "comparison_reports"
EVAL_ROOT = PROJECT_ROOT / "MLP" / "eval"
THESIS_CALIBRATION_DIR = PROJECT_ROOT / "Thesis" / "images" / "calibration_20260521"

HA_POINTS_CSV = (
    PROJECT_ROOT
    / "MLP" / "baseline" / "Hiroyasu_Arai" / "outputs"
    / "20260521_145756_ha_calibrated_grouped_condition_all_thesis_refresh_20260521"
    / "points.csv"
)
NS_POINTS_CSV = (
    PROJECT_ROOT
    / "MLP" / "baseline" / "Naber_Siebers" / "outputs"
    / "20260521_145724_ns_delay_grouped_condition_thesis_refresh_20260521"
    / "points.csv"
)

# Production anchor-off 5-seed eval directories (full-clean CDF, 692,942 points).
PROD_SEED_EVAL_DIRS: dict[int, Path] = {
    7:    EVAL_ROOT / "rmse_eval_clean_20260521_122646_distill_cdf_onset_v2_ablate_anchor_off_20260521_122435",
    17:   EVAL_ROOT / "rmse_eval_clean_20260521_123104_distill_cdf_onset_v2_ablate_anchor_off_20260521_122708",
    42:   EVAL_ROOT / "rmse_eval_clean_20260521_123603_distill_cdf_onset_v2_ablate_anchor_off_20260521_123126",
    99:   EVAL_ROOT / "rmse_eval_clean_20260521_123942_distill_cdf_onset_v2_ablate_anchor_off_20260521_123626",
    2024: EVAL_ROOT / "rmse_eval_clean_20260521_124229_distill_cdf_onset_v2_ablate_anchor_off_20260521_124004",
}

SVGP_STAGE3_RUN_DIR = PROJECT_ROOT / "MLP" / "runs_mlp" / "gp_baseline_stage3_20260521_112229"
SVGP_STAGE3_SEED = 42
SVGP_POINTS_SEARCH_ROOTS = (
    SVGP_STAGE3_RUN_DIR / "external_eval_full_points",
    SVGP_STAGE3_RUN_DIR / "external_eval_full",
    SVGP_STAGE3_RUN_DIR / "external_eval",
)

EXPECTED_N_POINTS = 692_942
RELIABILITY_LEVELS = np.linspace(0.025, 0.975, 19)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    points_csv: Path
    kind: str
    seed: int | None = None


def resolve_project_path(value: str | Path) -> Path:
    """Resolve repo-relative paths and Windows paths stored in CSV reports."""
    raw = str(value)
    normalized = raw.replace("\\", "/")
    project_variants = [
        str(PROJECT_ROOT).replace("\\", "/"),
        "C:/Users/Jiang/Documents/Mie_Postprocessing_Py",
        "c:/Users/Jiang/Documents/Mie_Postprocessing_Py",
    ]
    for root in project_variants:
        root_norm = root.rstrip("/")
        if normalized.lower() == root_norm.lower():
            return PROJECT_ROOT
        prefix = root_norm + "/"
        if normalized.lower().startswith(prefix.lower()):
            rel = normalized[len(prefix):]
            return PROJECT_ROOT / rel

    path = Path(raw)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return safe[:160] or "model"


def compute_pit(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return stats.norm.cdf((truth - mu) / np.maximum(sigma, 1e-12))


def reliability_curve(pit: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.array([np.mean(pit <= alpha) for alpha in levels], dtype=float)


def ece(pit: np.ndarray, levels: np.ndarray) -> float:
    rel = reliability_curve(pit, levels)
    return float(np.mean(np.abs(rel - levels)))


def crps_gaussian(truth: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (truth - mu) / sigma_safe
    return sigma_safe * (
        z * (2.0 * stats.norm.cdf(z) - 1.0)
        + 2.0 * stats.norm.pdf(z)
        - 1.0 / math.sqrt(math.pi)
    )


def _pick_column(columns: set[str], candidates: tuple[str, ...], path: Path) -> str:
    for col in candidates:
        if col in columns:
            return col
    raise KeyError(f"{path} missing any of columns {candidates}")


def load_points_arrays(points_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    header = pd.read_csv(points_csv, nrows=0)
    columns = set(header.columns)
    truth_col = _pick_column(columns, ("pen_true_mm", "truth_mm", "y_true", "target_mm"), points_csv)
    pred_col = _pick_column(columns, ("pen_pred_mm", "pred_mm", "mu_mm", "mu", "prediction_mm"), points_csv)
    std_col = _pick_column(columns, ("pen_std_mm", "std_mm", "sigma_mm", "sigma", "pred_std_mm"), points_csv)

    df = pd.read_csv(points_csv, usecols=[truth_col, pred_col, std_col])
    truth = df[truth_col].to_numpy(dtype=float)
    mu = df[pred_col].to_numpy(dtype=float)
    sigma = df[std_col].to_numpy(dtype=float)
    finite = np.isfinite(truth) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0.0)
    if not np.all(finite):
        dropped = int((~finite).sum())
        print(f"  dropping {dropped} non-finite/non-positive-sigma rows from {points_csv}")
        truth, mu, sigma = truth[finite], mu[finite], sigma[finite]
    return truth, mu, sigma


def build_prod_mlp_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for seed, eval_dir in sorted(PROD_SEED_EVAL_DIRS.items()):
        points = eval_dir / "points.csv"
        if not points.exists():
            raise FileNotFoundError(f"Missing production MLP points for seed {seed}: {points}")
        specs.append(ModelSpec(f"Production MLP anchor-off seed {seed}", points, "mlp_dp05", seed=seed))
    return specs


def find_latest_svgp_points() -> Path | None:
    candidates: list[Path] = []
    for root in SVGP_POINTS_SEARCH_ROOTS:
        candidates.extend(root.glob("*/points.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_svgp_spec(*, skip_rerun: bool, device: str | None, batch_points: int) -> ModelSpec:
    points = find_latest_svgp_points()
    if points is not None:
        return ModelSpec("SVGP stage3 seed 42", points, "svgp_stage3", seed=SVGP_STAGE3_SEED)

    if skip_rerun:
        raise FileNotFoundError(
            "No SVGP full-clean points.csv found and --skip-rerun-new-points was set. "
            f"Expected one under {SVGP_STAGE3_RUN_DIR}/external_eval_full_points."
        )

    checkpoint = SVGP_STAGE3_RUN_DIR / "per_seed" / f"seed_{SVGP_STAGE3_SEED}" / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing SVGP checkpoint: {checkpoint}")

    from MLP.MLP_training.run_gp_baseline import choose_device, run_external_rmse_evaluation

    print("\nNo SVGP full-clean points.csv found; generating it from the stored checkpoint.")
    torch_device = choose_device(device or "auto")
    out_dir, _summary = run_external_rmse_evaluation(
        checkpoint_path=checkpoint,
        split="clean",
        device=torch_device,
        output_root=SVGP_STAGE3_RUN_DIR / "external_eval_full_points",
        tag=f"seed_{SVGP_STAGE3_SEED}_full_clean_points",
        batch_points=int(batch_points),
        fast=False,
        save_points=True,
        lono_holdout=None,
    )
    points = out_dir / "points.csv"
    if not points.exists():
        raise FileNotFoundError(f"SVGP full-clean evaluation did not write {points}")
    return ModelSpec("SVGP stage3 seed 42", points, "svgp_stage3", seed=SVGP_STAGE3_SEED)


def build_model_specs(*, skip_rerun: bool, device: str | None, batch_points: int) -> list[ModelSpec]:
    specs = [
        ModelSpec("Hiroyasu-Arai calibrated", HA_POINTS_CSV, "ha"),
        ModelSpec("Naber-Siebers delay", NS_POINTS_CSV, "ns"),
    ]
    specs.extend(build_prod_mlp_specs())
    specs.append(build_svgp_spec(skip_rerun=skip_rerun, device=device, batch_points=batch_points))
    for spec in specs:
        if not spec.points_csv.exists():
            raise FileNotFoundError(f"Missing points CSV for {spec.name}: {spec.points_csv}")
    return specs


def summarize_model(spec: ModelSpec, output_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    print(f"\nProcessing {spec.name}")
    print(f"  points: {spec.points_csv}")
    truth, mu, sigma = load_points_arrays(spec.points_csv)
    if len(truth) != EXPECTED_N_POINTS:
        print(f"  warning: expected {EXPECTED_N_POINTS:,} points, got {len(truth):,}")

    pit = np.clip(compute_pit(truth, mu, sigma), 1e-9, 1.0 - 1e-9)
    rel = reliability_curve(pit, RELIABILITY_LEVELS)
    crps = crps_gaussian(truth, mu, sigma)
    abs_err = np.abs(truth - mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    _, ks_p = stats.kstest(pit, "uniform")

    row = {
        "model": spec.name,
        "kind": spec.kind,
        "seed": spec.seed,
        "points_csv": str(spec.points_csv),
        "n_points": int(len(truth)),
        "ece": ece(pit, RELIABILITY_LEVELS),
        "crps_mean": float(np.mean(crps)),
        "crps_std": float(np.std(crps, ddof=1)) if len(crps) > 1 else 0.0,
        "sharpness_mm": float(np.mean(sigma)),
        "pit_ks_pvalue": float(ks_p),
        "coverage_1sigma": float(np.mean(abs_err <= sigma_safe)),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * sigma_safe)),
    }
    print(
        "  "
        f"ECE={row['ece']:.4f}, CRPS={row['crps_mean']:.4f}, "
        f"sharpness={row['sharpness_mm']:.4f}, "
        f"cov1={row['coverage_1sigma']:.4f}, cov2={row['coverage_2sigma']:.4f}"
    )

    pit_dir = output_dir / "per_model_pit"
    pit_dir.mkdir(exist_ok=True)
    pd.DataFrame({"pit": pit}).to_csv(pit_dir / f"{safe_filename(spec.name)}.csv", index=False)
    return row, rel, pit


def add_new_mlp_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    new_rows = summary.loc[summary["kind"] == "mlp_dp05"].copy()
    if new_rows.empty:
        return summary

    metric_cols = [
        "ece", "crps_mean", "crps_std", "sharpness_mm",
        "pit_ks_pvalue", "coverage_1sigma", "coverage_2sigma",
    ]
    aggregate: dict[str, Any] = {
        "model": "Production MLP anchor-off (5-seed mean)",
        "kind": "mlp_dp05_5seed_mean",
        "seed": np.nan,
        "points_csv": "",
        "n_points": int(new_rows["n_points"].iloc[0]),
    }
    for col in metric_cols:
        values = pd.to_numeric(new_rows[col], errors="coerce")
        aggregate[col] = float(values.mean())
        aggregate[f"{col}_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return pd.concat([summary, pd.DataFrame([aggregate])], ignore_index=True)


def short_label(name: str) -> str:
    return (
        name.replace("Hiroyasu-Arai calibrated", "HA")
        .replace("Naber-Siebers delay", "NS")
        .replace("Production MLP anchor-off seed ", "MLP s")
        .replace("SVGP stage3 seed 42", "SVGP s42")
        .replace("Stage-3 MLP ", "MLP ")
        .replace("deltaP^0.25", "dP0.25")
        .replace("deltaP^0.5", "dP0.5")
        .replace("anchor_off ", "")
    )


def plot_reliability(rows: list[dict[str, Any]], rel_by_model: dict[str, np.ndarray], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=170)
    ax.plot(RELIABILITY_LEVELS, RELIABILITY_LEVELS, color="black", linewidth=1.0, label="ideal")
    ax.fill_between(
        RELIABILITY_LEVELS,
        np.clip(RELIABILITY_LEVELS - 0.02, 0.0, 1.0),
        np.clip(RELIABILITY_LEVELS + 0.02, 0.0, 1.0),
        color="0.8",
        alpha=0.25,
        linewidth=0,
        label="+/- 2%",
    )
    colors = plt.get_cmap("tab10").colors
    for idx, row in enumerate(rows):
        name = str(row["model"])
        kind = str(row["kind"])
        linestyle = "--" if kind in {"ha", "ns"} else "-."
        if kind == "mlp_dp05":
            linestyle = "-"
        if kind == "svgp_stage3":
            linestyle = ":"
        ax.plot(
            RELIABILITY_LEVELS,
            rel_by_model[name],
            label=short_label(name),
            color=colors[idx % len(colors)],
            linestyle=linestyle,
            linewidth=1.7,
            alpha=0.9,
        )
    ax.set_xlabel("Nominal lower-tail probability")
    ax.set_ylabel("Empirical lower-tail fraction")
    ax.set_title("Reliability diagram (clean diagnostic)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.2, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "reliability_overlay.png")
    plt.close(fig)


def plot_crps_sharpness(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary.loc[~summary["kind"].astype(str).str.endswith("_mean")].copy()
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=170)
    colors = plt.get_cmap("tab10").colors
    for idx, row in plot_df.iterrows():
        x = float(row["sharpness_mm"])
        y = float(row["crps_mean"])
        ax.scatter(x, y, s=46, color=colors[idx % len(colors)], edgecolor="black", linewidth=0.4)
        ax.annotate(short_label(str(row["model"])), (x, y), xytext=(5, 3),
                    textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Sharpness = mean predicted sigma [mm]")
    ax.set_ylabel("Mean Gaussian CRPS [mm]")
    ax.set_title("Calibration-sharpness trade-off")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "crps_sharpness_scatter.png")
    plt.close(fig)


def plot_pit_histograms(rows: list[dict[str, Any]], pit_by_model: dict[str, np.ndarray], output_dir: Path) -> None:
    n_models = len(rows)
    n_cols = 4
    n_rows = int(math.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.0, 3.1 * n_rows), dpi=170,
                             squeeze=False)
    bins = np.linspace(0.0, 1.0, 21)
    for ax, row in zip(axes.ravel(), rows):
        pit = pit_by_model[str(row["model"])]
        weights = np.ones_like(pit) / max(len(pit), 1)
        ax.hist(pit, bins=bins, weights=weights, color="#4c78a8", alpha=0.86)
        ax.axhline(1.0 / 20.0, color="#c44e52", linestyle="--", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"{short_label(str(row['model']))}\nKS p={float(row['pit_ks_pvalue']):.1e}", fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.22)
    for ax in axes.ravel()[n_models:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "pit_histograms.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-points", type=int, default=262144)
    parser.add_argument("--skip-rerun-new-points", action="store_true",
                        help="Fail if required cached points.csv files, including SVGP, do not already exist.")
    return parser.parse_args()


def copy_thesis_outputs(output_dir: Path) -> None:
    THESIS_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "summary.csv",
        "new_mlp_per_seed_summary.csv",
        "reliability_overlay.png",
        "crps_sharpness_scatter.png",
        "pit_histograms.png",
    ]:
        src = output_dir / name
        if src.exists():
            dst = THESIS_CALIBRATION_DIR / name
            shutil.copy2(src, dst)
            print(f"Copied: {dst}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (REPORT_ROOT / f"calibration_{pd.Timestamp.now():%Y%m%d_%H%M%S}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    specs = build_model_specs(
        skip_rerun=bool(args.skip_rerun_new_points),
        device=args.device,
        batch_points=int(args.batch_points),
    )
    print(f"Models: {len(specs)}")

    rows: list[dict[str, Any]] = []
    rel_by_model: dict[str, np.ndarray] = {}
    pit_by_model: dict[str, np.ndarray] = {}
    for spec in specs:
        row, rel, pit = summarize_model(spec, output_dir)
        rows.append(row)
        rel_by_model[spec.name] = rel
        pit_by_model[spec.name] = pit

    base_summary = pd.DataFrame(rows)
    summary = add_new_mlp_aggregate(base_summary)
    summary.to_csv(output_dir / "summary.csv", index=False)
    base_summary.loc[base_summary["kind"] == "mlp_dp05"].to_csv(
        output_dir / "new_mlp_per_seed_summary.csv", index=False
    )

    plot_reliability(rows, rel_by_model, output_dir)
    plot_crps_sharpness(summary, output_dir)
    plot_pit_histograms(rows, pit_by_model, output_dir)
    copy_thesis_outputs(output_dir)

    print("\nSummary:")
    print(summary[[
        "model", "n_points", "ece", "crps_mean", "sharpness_mm",
        "pit_ks_pvalue", "coverage_1sigma", "coverage_2sigma",
    ]].to_string(index=False))
    print(f"\nCalibration diagnostics complete: {output_dir}")


if __name__ == "__main__":
    main()
