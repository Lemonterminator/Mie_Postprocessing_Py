from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
THESIS_DIR = DATA_DIR.parents[1]
PROJECT_ROOT = THESIS_DIR.parent
IMAGE_DIR = THESIS_DIR / "images" / "baseline_comparison_20260521"

HA_RUN = PROJECT_ROOT / "MLP" / "baseline" / "Hiroyasu_Arai" / "outputs" / "20260521_145756_ha_calibrated_grouped_condition_all_thesis_refresh_20260521"
NS_RUN = PROJECT_ROOT / "MLP" / "baseline" / "Naber_Siebers" / "outputs" / "20260521_145724_ns_delay_grouped_condition_thesis_refresh_20260521"
PROD_REPORT = PROJECT_ROOT / "MLP" / "baseline" / "comparison_reports" / "mlp_production_stage3_vs_svgp_stage3_full_clean_20260521" / "headline_comparison.csv"
MU_REPORT = PROJECT_ROOT / "MLP" / "baseline" / "comparison_reports" / "mlp_mu_anchor_raw_uncertain_prodKD_vs_svgp_stage3_full_clean_20260521" / "headline_comparison.csv"
FIXED_REPORT = PROJECT_ROOT / "MLP" / "baseline" / "comparison_reports" / "stage3_fixed_table_eval_with_ha_ns_20260521"

MODEL_LABELS = {
    "hiroyasu_arai_calibrated": "H-A",
    "naber_siebers_delay": "N-S",
    "production_mlp": "Production\nMLP mean",
    "mu_anchor_raw_uncertain_prodKD": "mu-anchor\nMLP mean",
    "svgp_stage3": "SVGP\nstage 3",
    "q1_oracle": "Q1\noracle",
}

MODEL_COLORS = {
    "hiroyasu_arai_calibrated": "#8F6BB3",
    "naber_siebers_delay": "#C44E52",
    "production_mlp": "#2F6FA3",
    "mu_anchor_raw_uncertain_prodKD": "#6BA56B",
    "svgp_stage3": "#9A5B38",
    "q1_oracle": "#555555",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metrics_from_summary(run_dir: Path, model: str, group: str) -> dict[str, Any]:
    payload = read_json(run_dir / "metrics_summary.json")
    overall = dict(payload["overall"])
    return {
        "model": model,
        "model_group": group,
        "seed": "",
        "n_points": overall.get("n_points"),
        "n_trajectories": overall.get("n_trajectories"),
        "rmse_mm": overall.get("rmse_mm"),
        "mae_mm": overall.get("mae_mm"),
        "bias_mm": overall.get("bias_mm"),
        "p95_abs_err_mm": overall.get("p95_abs_err_mm"),
        "coverage_1sigma": overall.get("coverage_1sigma"),
        "coverage_2sigma": overall.get("coverage_2sigma"),
        "mean_pred_std_mm": overall.get("mean_pred_std_mm"),
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
    }


def select_report_row(path: Path, model: str, group: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    row = df.loc[df["model"] == model].iloc[0].to_dict()
    row["model_group"] = group
    return row


def build_full_clean_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        metrics_from_summary(HA_RUN, "Hiroyasu-Arai calibrated", "hiroyasu_arai_calibrated"),
        metrics_from_summary(NS_RUN, "Naber-Siebers delay", "naber_siebers_delay"),
        select_report_row(PROD_REPORT, "MLP production Stage-3 anchor_off mean(5)", "production_mlp"),
        select_report_row(MU_REPORT, "MLP mu_anchor raw_uncertain prodKD mean(5)", "mu_anchor_raw_uncertain_prodKD"),
        select_report_row(PROD_REPORT, "SVGP stage3 seed 42", "svgp_stage3"),
    ]
    full = pd.DataFrame(rows)
    seed_rows = pd.read_csv(PROD_REPORT)
    seed_rows = seed_rows.loc[seed_rows["kind"] == "mlp_stage3_production"].copy()
    seed_rows["model_group"] = "production_mlp"
    return full, seed_rows


def fixed_group_summary() -> pd.DataFrame:
    return pd.read_csv(FIXED_REPORT / "group_summary.csv")


def bar_plot(
    df: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    include_oracle: bool = False,
    ymax: float | None = None,
) -> None:
    order = [
        "q1_oracle",
        "hiroyasu_arai_calibrated",
        "naber_siebers_delay",
        "production_mlp",
        "mu_anchor_raw_uncertain_prodKD",
        "svgp_stage3",
    ]
    if not include_oracle:
        order = [item for item in order if item != "q1_oracle"]
    plot_df = df.loc[df["model_group"].isin(order)].copy()
    plot_df["order"] = plot_df["model_group"].map({key: idx for idx, key in enumerate(order)})
    plot_df = plot_df.sort_values("order")
    x = list(range(len(plot_df)))
    width = 0.23
    metrics = [
        ("rmse_mm", "RMSE"),
        ("mae_mm", "MAE"),
        ("p95_abs_err_mm", "P95"),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=180)
    for offset, (metric, label) in zip([-width, 0.0, width], metrics):
        values = plot_df[metric].astype(float).tolist()
        bars = ax.bar(
            [pos + offset for pos in x],
            values,
            width=width,
            label=label,
            color={"rmse_mm": "#2F6FA3", "mae_mm": "#2E8B57", "p95_abs_err_mm": "#C26A2E"}[metric],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(g, g) for g in plot_df["model_group"]], fontsize=8)
    ax.set_ylabel("Error [mm]")
    ax.set_title(title)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.legend(ncols=3, loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def coverage_plot(df: pd.DataFrame, *, title: str, out_path: Path) -> None:
    order = [
        "hiroyasu_arai_calibrated",
        "naber_siebers_delay",
        "production_mlp",
        "mu_anchor_raw_uncertain_prodKD",
        "svgp_stage3",
    ]
    plot_df = df.loc[df["model_group"].isin(order)].copy()
    plot_df["order"] = plot_df["model_group"].map({key: idx for idx, key in enumerate(order)})
    plot_df = plot_df.sort_values("order")
    x = list(range(len(plot_df)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=180)
    bars1 = ax.bar(
        [pos - width / 2 for pos in x],
        plot_df["coverage_1sigma"].astype(float),
        width=width,
        color="#7A5BB5",
        label="1 sigma",
        edgecolor="white",
        linewidth=0.8,
    )
    bars2 = ax.bar(
        [pos + width / 2 for pos in x],
        plot_df["coverage_2sigma"].astype(float),
        width=width,
        color="#3A9AA1",
        label="2 sigma",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)
    ax.axhline(0.683, color="#7A5BB5", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.axhline(0.954, color="#3A9AA1", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(g, g) for g in plot_df["model_group"]], fontsize=8)
    ax.set_ylim(0.20, 1.05)
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.legend(ncols=2, loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def per_seed_rmse_plot(seed_rows: pd.DataFrame, fixed_per_run: pd.DataFrame, out_path: Path) -> None:
    seeds = [7, 17, 42, 99, 2024]
    seed_lookup = {int(row["seed"]): row for _, row in seed_rows.iterrows()}
    rmse_full = [float(seed_lookup[seed]["rmse_mm"]) for seed in seeds]
    prod_fixed = fixed_per_run.loc[
        (fixed_per_run["eval_set"] == "cdf_uncensored")
        & (fixed_per_run["model_group"] == "production_mlp")
    ].copy()
    fixed_lookup = {
        int(str(row["model_label"]).replace("seed", "")): row
        for _, row in prod_fixed.iterrows()
        if str(row["model_label"]).startswith("seed")
    }
    rmse_fixed = [float(fixed_lookup[seed]["rmse_mm"]) for seed in seeds]

    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=180)
    x = list(range(len(seeds)))
    width = 0.34
    bars_full = ax.bar(
        [pos - width / 2 for pos in x],
        rmse_full,
        width=width,
        color="#2F6FA3",
        label="full clean",
        edgecolor="white",
    )
    bars_fixed = ax.bar(
        [pos + width / 2 for pos in x],
        rmse_fixed,
        width=width,
        color="#6BA56B",
        label="CDF uncensored",
        edgecolor="white",
    )
    ax.bar_label(bars_full, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(bars_fixed, fmt="%.2f", padding=2, fontsize=8)
    ax.axhline(sum(rmse_full) / len(rmse_full), color="#2F6FA3", linewidth=1.1)
    ax.axhline(sum(rmse_fixed) / len(rmse_fixed), color="#6BA56B", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(seed) for seed in seeds])
    ax.set_xlabel("seed")
    ax.set_ylabel("RMSE [mm]")
    ax.set_ylim(3.8, 5.1)
    ax.set_title("Production MLP repeatability across evaluation tables")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fixed_rows_for_eval(group_summary: pd.DataFrame, eval_set: str) -> pd.DataFrame:
    sub = group_summary.loc[group_summary["eval_set"] == eval_set].copy()
    rename = {
        "rmse_mm_mean": "rmse_mm",
        "mae_mm_mean": "mae_mm",
        "bias_mm_mean": "bias_mm",
        "p95_abs_err_mm_mean": "p95_abs_err_mm",
        "coverage_1sigma_mean": "coverage_1sigma",
        "coverage_2sigma_mean": "coverage_2sigma",
    }
    return sub.rename(columns=rename)


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    full, seed_rows = build_full_clean_tables()
    group_summary = fixed_group_summary()
    per_run = pd.read_csv(FIXED_REPORT / "per_run_metrics.csv")

    full.to_csv(DATA_DIR / "headline_full_clean.csv", index=False)
    seed_rows.to_csv(DATA_DIR / "production_mlp_full_clean_per_seed.csv", index=False)
    group_summary.to_csv(DATA_DIR / "fixed_table_group_summary.csv", index=False)
    per_run.to_csv(DATA_DIR / "fixed_table_per_run_metrics.csv", index=False)

    bar_plot(
        full,
        title="Full clean CDF evaluation: error metrics",
        out_path=IMAGE_DIR / "full_clean_metric_bars_rmse_mae_p95.png",
        ymax=24.0,
    )
    coverage_plot(
        full,
        title="Full clean CDF evaluation: uncertainty coverage",
        out_path=IMAGE_DIR / "full_clean_coverage_comparison.png",
    )

    cdf = fixed_rows_for_eval(group_summary, "cdf_uncensored")
    bar_plot(
        cdf,
        title="Conservatively uncensored CDF points: error metrics",
        out_path=IMAGE_DIR / "cdf_uncensored_metric_bars_rmse_mae_p95.png",
        ymax=23.5,
    )
    coverage_plot(
        cdf,
        title="Conservatively uncensored CDF points: uncertainty coverage",
        out_path=IMAGE_DIR / "cdf_uncensored_coverage_comparison.png",
    )

    p50 = fixed_rows_for_eval(group_summary, "p50_observed")
    bar_plot(
        p50,
        title="P50 observed points: error metrics",
        out_path=IMAGE_DIR / "p50_observed_metric_bars_rmse_mae_p95.png",
        include_oracle=True,
        ymax=22.0,
    )

    q1_extra = fixed_rows_for_eval(group_summary, "q1_grid_extrapolated")
    bar_plot(
        q1_extra,
        title="Q1 extrapolated grid: error metrics",
        out_path=IMAGE_DIR / "q1_extrapolated_metric_bars_rmse_mae_p95.png",
        ymax=78.0,
    )
    per_seed_rmse_plot(seed_rows, per_run, IMAGE_DIR / "production_mlp_per_seed_rmse.png")

    summary = {
        "full_clean_rows": len(full),
        "fixed_group_rows": len(group_summary),
        "image_dir": str(IMAGE_DIR.relative_to(PROJECT_ROOT)),
        "sources": {
            "ha_run": str(HA_RUN.relative_to(PROJECT_ROOT)),
            "ns_run": str(NS_RUN.relative_to(PROJECT_ROOT)),
            "production_report": str(PROD_REPORT.relative_to(PROJECT_ROOT)),
            "mu_anchor_report": str(MU_REPORT.relative_to(PROJECT_ROOT)),
            "fixed_report": str(FIXED_REPORT.relative_to(PROJECT_ROOT)),
        },
    }
    (DATA_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote thesis baseline figures to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
