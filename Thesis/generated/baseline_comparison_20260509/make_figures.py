from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent
THESIS_DIR = DATA_DIR.parents[1]
IMAGE_DIR = THESIS_DIR / "images" / "baseline_comparison_20260509"

HEADLINE_CSV = DATA_DIR / "headline_comparison.csv"
AGGREGATE_CSV = DATA_DIR / "new_mlp_5seed_aggregate.csv"
PER_SEED_CSV = DATA_DIR / "per_seed_new_mlp_eval.csv"

MODEL_ORDER = [
    ("Hiroyasu-Arai calibrated", "H-A"),
    ("Naber-Siebers delay", "N-S"),
    ("Stage-3 MLP anchor_off", "MLP\n$\\Delta P^{0.25}$"),
    ("Stage-3 MLP ΔP^0.5 (5-seed mean)", "MLP\n$\\Delta P^{0.5}$ mean"),
    ("Sparse heteroscedastic GP mean", "Sparse\nhetero. GP"),
]

COLORS = {
    "rmse_mm": "#2F6FA3",
    "mae_mm": "#2E8B57",
    "p95_abs_err_mm": "#C26A2E",
    "coverage_1sigma": "#7A5BB5",
    "coverage_2sigma": "#3A9AA1",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def row_by_model(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["model"]: row for row in rows}


def save_metric_bars(rows: list[dict[str, str]]) -> None:
    lookup = row_by_model(rows)
    selected = [(label, lookup[model]) for model, label in MODEL_ORDER]
    metrics = [
        ("rmse_mm", "RMSE"),
        ("mae_mm", "MAE"),
        ("p95_abs_err_mm", "P95 abs. err."),
    ]
    base_x = list(range(len(selected)))
    width = 0.23

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=180)
    offsets = [-width, 0, width]
    for (metric, metric_label), offset in zip(metrics, offsets):
        values = [as_float(row, metric) for _, row in selected]
        positions = [x + offset for x in base_x]
        bars = ax.bar(
            positions,
            values,
            width=width,
            label=metric_label,
            color=COLORS[metric],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    ax.set_xticks(base_x)
    ax.set_xticklabels([label for label, _ in selected], fontsize=9)
    ax.set_ylabel("Error [mm]")
    ax.set_ylim(0, 23)
    ax.set_title("External clean evaluation: error metrics")
    ax.legend(ncols=3, loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "metric_bars_rmse_mae_p95.png")
    plt.close(fig)


def save_coverage(rows: list[dict[str, str]]) -> None:
    lookup = row_by_model(rows)
    selected = [(label, lookup[model]) for model, label in MODEL_ORDER]
    base_x = list(range(len(selected)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=180)
    cov1 = [as_float(row, "coverage_1sigma") for _, row in selected]
    cov2 = [as_float(row, "coverage_2sigma") for _, row in selected]
    bars1 = ax.bar(
        [x - width / 2 for x in base_x],
        cov1,
        width=width,
        color=COLORS["coverage_1sigma"],
        label="1 sigma coverage",
        edgecolor="white",
        linewidth=0.8,
    )
    bars2 = ax.bar(
        [x + width / 2 for x in base_x],
        cov2,
        width=width,
        color=COLORS["coverage_2sigma"],
        label="2 sigma coverage",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)
    ax.axhline(0.683, color="#7A5BB5", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.axhline(0.954, color="#3A9AA1", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.text(4.58, 0.683, "nominal 1 sigma", va="center", ha="right", fontsize=8)
    ax.text(4.58, 0.954, "nominal 2 sigma", va="center", ha="right", fontsize=8)
    ax.set_xticks(base_x)
    ax.set_xticklabels([label for label, _ in selected], fontsize=9)
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.45, 1.04)
    ax.set_title("External clean evaluation: uncertainty coverage")
    ax.legend(ncols=2, loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "coverage_comparison.png")
    plt.close(fig)


def save_per_seed_rmse(
    mlp_rows: list[dict[str, str]],
    headline_rows: list[dict[str, str]],
    aggregate: dict[str, str],
) -> None:
    mlp_by_seed = {int(row["seed"]): row for row in mlp_rows}
    gp_by_seed = {}
    for row in headline_rows:
        model = row["model"]
        if model.startswith("Sparse heteroscedastic GP (seed "):
            gp_by_seed[int(float(row["seed"]))] = row

    seeds = [7, 17, 42, 99, 2024]
    labels = [str(seed) for seed in seeds]
    mlp_rmse = [as_float(mlp_by_seed[seed], "rmse_mm") for seed in seeds]
    gp_rmse = [as_float(gp_by_seed[seed], "rmse_mm") for seed in seeds]
    mean = as_float(aggregate, "rmse_mm")
    std = as_float(aggregate, "rmse_mm_std")
    gp_mean = as_float(row_by_model(headline_rows)["Sparse heteroscedastic GP mean"], "rmse_mm")

    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=180)
    x = list(range(len(seeds)))
    width = 0.34
    bars_mlp = ax.bar(
        [pos - width / 2 for pos in x],
        mlp_rmse,
        width=width,
        color="#2F6FA3",
        label="MLP $\\Delta P^{0.5}$",
        edgecolor="white",
        linewidth=0.8,
    )
    bars_gp = ax.bar(
        [pos + width / 2 for pos in x],
        gp_rmse,
        width=width,
        color="#9A5B38",
        label="Sparse hetero. GP",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars_mlp, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(bars_gp, fmt="%.2f", padding=2, fontsize=8)
    ax.axhline(mean, color="#2F6FA3", linewidth=1.3, label=f"MLP mean = {mean:.3f} mm")
    ax.axhline(gp_mean, color="#9A5B38", linewidth=1.3, linestyle="--", label=f"GP mean = {gp_mean:.3f} mm")
    ax.fill_between(
        [-0.5, len(labels) - 0.5],
        [mean - std, mean - std],
        [mean + std, mean + std],
        color="#2F6FA3",
        alpha=0.14,
        label=f"MLP ±1 std = {std:.3f} mm",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE [mm]")
    ax.set_xlabel("seed")
    ax.set_ylim(5.6, 7.7)
    ax.set_title("External clean RMSE by seed: MLP vs sparse GP")
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        4.0,
        5.72,
        "MLP seed 2024 selected raw_reliable_no_kd;\nother MLP seeds selected anchor_off",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4A4A4A",
    )
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "per_seed_rmse.png")
    plt.close(fig)


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    headline_rows = read_csv(HEADLINE_CSV)
    aggregate_rows = read_csv(AGGREGATE_CSV)
    per_seed_rows = read_csv(PER_SEED_CSV)

    save_metric_bars(headline_rows)
    save_coverage(headline_rows)
    save_per_seed_rmse(per_seed_rows, headline_rows, aggregate_rows[0])

    print(f"Wrote slide figures to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
