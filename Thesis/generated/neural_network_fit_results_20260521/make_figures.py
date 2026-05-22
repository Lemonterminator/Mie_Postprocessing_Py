from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
THESIS_DIR = DATA_DIR.parents[1]
PROJECT_ROOT = THESIS_DIR.parent
IMAGE_DIR = THESIS_DIR / "images" / "neural_network_fit_results_20260521"

EVAL_DIR = (
    PROJECT_ROOT
    / "MLP"
    / "eval"
    / "rmse_eval_clean_20260521_123603_distill_cdf_onset_v2_ablate_anchor_off_20260521_123126"
)
SEED_TABLE = THESIS_DIR / "generated" / "baseline_comparison_20260521" / "production_mlp_full_clean_per_seed.csv"
FIXED_TABLE = THESIS_DIR / "generated" / "baseline_comparison_20260521" / "fixed_table_per_run_metrics.csv"


def save_pred_vs_actual(points: pd.DataFrame, out_path: Path) -> None:
    sample = points.sample(n=min(len(points), 80_000), random_state=42)
    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=180)
    ax.scatter(
        sample["pen_true_mm"],
        sample["pen_pred_mm"],
        s=2,
        alpha=0.12,
        color="#2F6FA3",
        linewidths=0,
    )
    lim_max = max(float(points["pen_true_mm"].max()), float(points["pen_pred_mm"].max()))
    ax.plot([0, lim_max], [0, lim_max], color="#222222", linewidth=1.1, linestyle="--")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("Measured penetration [mm]")
    ax.set_ylabel("Predicted penetration [mm]")
    ax.set_title("Production MLP seed 42: predicted vs. measured")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_residual_hist(points: pd.DataFrame, out_path: Path) -> None:
    resid = points["resid_mm"].clip(-30, 30)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax.hist(resid, bins=90, color="#2F6FA3", alpha=0.88, edgecolor="white", linewidth=0.35)
    ax.axvline(0.0, color="#222222", linewidth=1.1)
    ax.axvline(float(points["resid_mm"].mean()), color="#C26A2E", linewidth=1.4, label="mean bias")
    ax.set_xlabel("Residual, prediction - truth [mm]")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution on full-clean CDF")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_per_folder(per_folder: pd.DataFrame, out_rmse: Path, out_cov: Path) -> None:
    folder = per_folder.sort_values("rmse_mean_mm", ascending=False).copy()
    labels = [item.replace("BC", "BC\n").replace("_HZ_", "\n") for item in folder["folder"]]
    x = list(range(len(folder)))

    fig, ax = plt.subplots(figsize=(8.8, 4.7), dpi=180)
    bars = ax.bar(x, folder["rmse_mean_mm"], color="#2F6FA3", edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean trajectory RMSE [mm]")
    ax.set_title("Per-folder RMSE, production MLP seed 42")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_rmse)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.7), dpi=180)
    width = 0.36
    bars1 = ax.bar([pos - width / 2 for pos in x], folder["coverage_1sigma"], width=width, label="1 sigma", color="#7A5BB5")
    bars2 = ax.bar([pos + width / 2 for pos in x], folder["coverage_2sigma"], width=width, label="2 sigma", color="#3A9AA1")
    ax.axhline(0.683, color="#7A5BB5", linestyle="--", linewidth=1.0)
    ax.axhline(0.954, color="#3A9AA1", linestyle="--", linewidth=1.0)
    ax.bar_label(bars1, fmt="%.2f", fontsize=6, padding=2)
    ax.bar_label(bars2, fmt="%.2f", fontsize=6, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.45, 1.04)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Per-folder coverage, production MLP seed 42")
    ax.legend(frameon=False, ncols=2, loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_cov)
    plt.close(fig)


def save_trajectory(points: pd.DataFrame, traj_key: str, title: str, out_path: Path) -> None:
    sub = points.loc[points["traj_key"] == traj_key].sort_values("time_ms").copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax.plot(sub["time_ms"], sub["pen_true_mm"], marker="o", markersize=3, linewidth=1.4, color="#222222", label="truth")
    ax.plot(sub["time_ms"], sub["pen_pred_mm"], marker="o", markersize=3, linewidth=1.4, color="#2F6FA3", label="prediction")
    lower = sub["pen_pred_mm"] - sub["pen_std_mm"]
    upper = sub["pen_pred_mm"] + sub["pen_std_mm"]
    ax.fill_between(sub["time_ms"], lower, upper, color="#2F6FA3", alpha=0.16, label="plus/minus 1 sigma")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Penetration [mm]")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_seed_repeatability(seed_table: pd.DataFrame, fixed_table: pd.DataFrame, out_path: Path) -> None:
    full = seed_table.loc[seed_table["kind"] == "mlp_stage3_production"].copy()
    fixed = fixed_table.loc[
        (fixed_table["eval_set"] == "cdf_uncensored")
        & (fixed_table["model_group"] == "production_mlp")
        & (fixed_table["model_label"].str.startswith("seed"))
    ].copy()
    fixed["seed"] = fixed["model_label"].str.replace("seed", "", regex=False).astype(int)
    full["seed"] = full["seed"].astype(int)
    merged = full[["seed", "rmse_mm"]].merge(
        fixed[["seed", "rmse_mm"]],
        on="seed",
        suffixes=("_full_clean", "_cdf_uncensored"),
    ).sort_values("seed")
    x = list(range(len(merged)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=180)
    bars1 = ax.bar([pos - width / 2 for pos in x], merged["rmse_mm_full_clean"], width=width, color="#2F6FA3", label="full clean")
    bars2 = ax.bar([pos + width / 2 for pos in x], merged["rmse_mm_cdf_uncensored"], width=width, color="#6BA56B", label="CDF uncensored")
    ax.bar_label(bars1, fmt="%.2f", fontsize=8, padding=2)
    ax.bar_label(bars2, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(seed) for seed in merged["seed"]])
    ax.set_ylim(3.9, 5.05)
    ax.set_xlabel("Seed")
    ax.set_ylabel("RMSE [mm]")
    ax.set_title("Production MLP repeatability")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(EVAL_DIR / "points.csv")
    per_folder = pd.read_csv(EVAL_DIR / "per_folder.csv")
    per_traj = pd.read_csv(EVAL_DIR / "per_trajectory.csv")
    seed_table = pd.read_csv(SEED_TABLE)
    fixed_table = pd.read_csv(FIXED_TABLE)

    save_pred_vs_actual(points, IMAGE_DIR / "pred_vs_actual_seed42.png")
    save_residual_hist(points, IMAGE_DIR / "residual_histogram_seed42.png")
    save_per_folder(
        per_folder,
        IMAGE_DIR / "per_folder_rmse_seed42.png",
        IMAGE_DIR / "per_folder_coverage_seed42.png",
    )
    save_seed_repeatability(seed_table, fixed_table, IMAGE_DIR / "production_mlp_per_seed_rmse.png")

    candidates = per_traj.loc[per_traj["n_points"] >= 8].copy()
    best = candidates.sort_values("rmse_mm").iloc[0]
    worst = candidates.sort_values("rmse_mm", ascending=False).iloc[0]
    save_trajectory(
        points,
        str(best["traj_key"]),
        f"Low-error trajectory: {best['folder']} {best['test_name']} plume {int(best['plume_idx'])}",
        IMAGE_DIR / "best_traj_seed42.png",
    )
    save_trajectory(
        points,
        str(worst["traj_key"]),
        f"High-error trajectory: {worst['folder']} {worst['test_name']} plume {int(worst['plume_idx'])}",
        IMAGE_DIR / "worst_traj_seed42.png",
    )

    metrics = json.loads((EVAL_DIR / "metrics_summary.json").read_text(encoding="utf-8"))
    summary = {
        "source_eval_dir": str(EVAL_DIR.relative_to(PROJECT_ROOT)),
        "representative_seed": 42,
        "overall": metrics["overall"],
        "best_trajectory": best.to_dict(),
        "worst_trajectory": worst.to_dict(),
        "image_dir": str(IMAGE_DIR.relative_to(PROJECT_ROOT)),
    }
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote neural fit figures to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
