"""Generate Stage-2, Stage-3, and cross-architecture LONO figures."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THESIS_DIR = PROJECT_ROOT / "Thesis"
IMAGE_DIR = THESIS_DIR / "images"
GENERATED_DIR = THESIS_DIR / "generated" / "lono_cross_arch_20260521"

STAGE2_CSV = PROJECT_ROOT / "MLP" / "runs_mlp" / "stage2_lono_ablation_20260520_235200" / "per_fold.csv"
STAGE3_CSV = PROJECT_ROOT / "MLP" / "runs_mlp" / "stage3_lono_ablation_20260521_092405" / "per_fold.csv"
GP_LONO_ROOT = PROJECT_ROOT / "MLP" / "runs_mlp"

# Sanitized fold names. Legacy run artifacts (CSV holdout columns, SVGP run-dir
# names, metrics JSON payloads) may still carry the confidential
# "BC<date>_HZ_" campaign prefix; _strip_bc() normalises both worlds.
_BC_PREFIX_RE = re.compile(r"BC\d{8}_HZ_")


def _strip_bc(value: Any) -> str:
    return _BC_PREFIX_RE.sub("", str(value))


FOLD_ORDER = ["Nozzle0", "Nozzle2", "Nozzle1", "Nozzle5", "Nozzle4"]
FOLD_LABELS = ["Nozzle0\n(311k pts)", "Nozzle2\n(60k)", "Nozzle1\n(26k)", "Nozzle5\n(33k)", "Nozzle4\n(28k)"]

STAGE2_ABLATIONS = ["no_anchor", "mu_anchor", "mu_sigma_anchor"]
STAGE2_LABELS   = ["no anchor", "μ anchor", "μσ anchor"]
STAGE2_COLORS   = ["#C05050", "#2F6FA3", "#6BA56B"]

STAGE3_ABLATIONS = ["baseline", "raw_reliable_no_kd", "raw_uncertain_raw05_kd025", "anchor_off"]
STAGE3_LABELS    = ["baseline", "raw-reliable\nno KD", "blended\nuncertain", "anchor\noff"]
STAGE3_COLORS    = ["#888888", "#C05050", "#2F6FA3", "#6BA56B"]

NOMINAL_1SIGMA = 0.683
STAGE3_LONO_WINNER = "raw_uncertain_raw05_kd025"
LEGACY_HA_RMSE = 10.713
LEGACY_NS_RMSE = 9.784
CROSS_ARCH_MODELS = ["mlp_stage3", "svgp_stage3"]
CROSS_ARCH_LABELS = {
    "mlp_stage3": "Stage-3 MLP",
    "svgp_stage3": "SVGP stage 3",
}
CROSS_ARCH_COLORS = {
    "mlp_stage3": "#2F6FA3",
    "svgp_stage3": "#9A5B38",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_lono_figure(df: pd.DataFrame,
                     ablations: list[str],
                     labels: list[str],
                     colors: list[str],
                     out_path: Path,
                     title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=180)

    n_folds = len(FOLD_ORDER)
    n_ab = len(ablations)
    width = 0.72 / n_ab
    offsets = [(i - (n_ab - 1) / 2) * width for i in range(n_ab)]
    x = list(range(n_folds))

    # ── left panel: RMSE ────────────────────────────────────────────────────
    ax = axes[0]
    for ab, lab, col, off in zip(ablations, labels, colors, offsets):
        sub = df[df["ablation"] == ab].set_index("holdout")
        vals = [float(sub.loc[fold, "rmse_mm"]) if fold in sub.index else float("nan")
                for fold in FOLD_ORDER]
        positions = [xi + off for xi in x]
        bars = ax.bar(positions, vals, width=width * 0.88, color=col, label=lab,
                      edgecolor="white", linewidth=0.4)
        ax.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(FOLD_LABELS, fontsize=8)
    ax.set_ylabel("RMSE [mm]")
    ax.set_title(f"{title_prefix}: per-fold RMSE")
    ax.legend(frameon=False, fontsize=8, ncols=n_ab)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── right panel: 1σ coverage ────────────────────────────────────────────
    ax = axes[1]
    ax.axhline(NOMINAL_1SIGMA, color="#333333", linewidth=1.1, linestyle="--",
               label=f"nominal Gaussian ({NOMINAL_1SIGMA:.3f})")
    for ab, lab, col, off in zip(ablations, labels, colors, offsets):
        sub = df[df["ablation"] == ab].set_index("holdout")
        vals = [float(sub.loc[fold, "coverage_1sigma"]) if fold in sub.index else float("nan")
                for fold in FOLD_ORDER]
        positions = [xi + off for xi in x]
        bars = ax.bar(positions, vals, width=width * 0.88, color=col, label=lab,
                      edgecolor="white", linewidth=0.4)
        ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(FOLD_LABELS, fontsize=8)
    ax.set_ylabel("1σ empirical coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{title_prefix}: per-fold 1σ coverage")
    ax.legend(frameon=False, fontsize=8, ncols=n_ab + 1)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def latest_svgp_uncensored_metrics(holdout: str) -> Path:
    candidates: list[Path] = []
    # `*{holdout}_` also matches legacy run dirs named after the unsanitized
    # campaign (gp_baseline_stage3_lono_BC..._HZ_NozzleN_<ts>).
    for run_dir in GP_LONO_ROOT.glob(f"gp_baseline_stage3_lono_*{holdout}_*"):
        candidates.extend(run_dir.glob("external_eval_uncensored/*/metrics_summary.json"))
    valid: list[Path] = []
    for path in candidates:
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if _strip_bc(payload.get("lono_holdout")) == holdout and "overall" in payload:
            valid.append(path)
    if not valid:
        raise FileNotFoundError(f"No SVGP uncensored metrics_summary.json found for {holdout}")
    return max(valid, key=lambda path: path.stat().st_mtime)


def build_cross_arch_lono_table(stage3_df: pd.DataFrame) -> pd.DataFrame:
    mlp = stage3_df.loc[stage3_df["ablation"] == STAGE3_LONO_WINNER].copy()
    if mlp.empty:
        raise ValueError(f"Missing Stage-3 LONO winner rows for {STAGE3_LONO_WINNER!r}")
    mlp = mlp.set_index("holdout")

    rows: list[dict[str, Any]] = []
    for fold_idx, holdout in enumerate(FOLD_ORDER):
        if holdout not in mlp.index:
            raise KeyError(f"Missing MLP fold {holdout} in {STAGE3_CSV}")
        mlp_row = mlp.loc[holdout]
        rows.append(
            {
                "fold_order": fold_idx,
                "holdout": holdout,
                "fold_label": FOLD_LABELS[fold_idx].replace("\n", " "),
                "model_group": "mlp_stage3",
                "model_label": CROSS_ARCH_LABELS["mlp_stage3"],
                "rmse_mm": float(mlp_row["rmse_mm"]),
                "mae_mm": float(mlp_row["mae_mm"]),
                "bias_mm": float(mlp_row["bias_mm"]),
                "coverage_1sigma": float(mlp_row["coverage_1sigma"]),
                "coverage_2sigma": float(mlp_row["coverage_2sigma"]),
                "mean_pred_std_mm": float(mlp_row.get("mean_pred_std_mm", float("nan"))),
                "n_points": int(mlp_row.get("n_points", 0)),
                "source": str(STAGE3_CSV.relative_to(PROJECT_ROOT)),
            }
        )

        svgp_path = latest_svgp_uncensored_metrics(holdout)
        payload = read_json(svgp_path)
        overall = payload["overall"]
        rows.append(
            {
                "fold_order": fold_idx,
                "holdout": holdout,
                "fold_label": FOLD_LABELS[fold_idx].replace("\n", " "),
                "model_group": "svgp_stage3",
                "model_label": CROSS_ARCH_LABELS["svgp_stage3"],
                "rmse_mm": float(overall["rmse_mm"]),
                "mae_mm": float(overall["mae_mm"]),
                "bias_mm": float(overall["bias_mm"]),
                "coverage_1sigma": float(overall["coverage_1sigma"]),
                "coverage_2sigma": float(overall["coverage_2sigma"]),
                "mean_pred_std_mm": float(overall.get("mean_pred_std_mm", float("nan"))),
                "n_points": int(overall["n_points"]),
                "source": str(svgp_path.relative_to(PROJECT_ROOT)),
            }
        )

    out = pd.DataFrame(rows)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(GENERATED_DIR / "current_lono_per_fold.csv", index=False)
    print(f"Saved: {GENERATED_DIR / 'current_lono_per_fold.csv'}")
    return out


def _cross_arch_values(table: pd.DataFrame, model_group: str, metric: str) -> list[float]:
    sub = table.loc[table["model_group"] == model_group].set_index("holdout")
    return [float(sub.loc[fold, metric]) for fold in FOLD_ORDER]


def _style_lono_axis(ax: plt.Axes) -> None:
    ax.set_xticks(list(range(len(FOLD_ORDER))))
    ax.set_xticklabels(FOLD_LABELS, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_cross_arch_lono_combined(table: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=180)
    x = list(range(len(FOLD_ORDER)))
    width = 0.34
    offsets = [-width / 2, width / 2]

    ax = axes[0]
    ax.axhline(LEGACY_HA_RMSE, color="#8F6BB3", linestyle="--", linewidth=1.0,
               label=f"legacy H-A mean ({LEGACY_HA_RMSE:.1f})")
    ax.axhline(LEGACY_NS_RMSE, color="#C44E52", linestyle="--", linewidth=1.0,
               label=f"legacy N-S mean ({LEGACY_NS_RMSE:.1f})")
    for model, off in zip(CROSS_ARCH_MODELS, offsets):
        vals = _cross_arch_values(table, model, "rmse_mm")
        bars = ax.bar(
            [pos + off for pos in x],
            vals,
            width=width * 0.9,
            color=CROSS_ARCH_COLORS[model],
            label=CROSS_ARCH_LABELS[model],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    _style_lono_axis(ax)
    ax.set_ylabel("RMSE [mm]")
    ax.set_title("Current uncensored LONO: per-fold RMSE")
    ax.legend(frameon=False, fontsize=7.2, ncols=2)

    ax = axes[1]
    ax.axhline(NOMINAL_1SIGMA, color="#333333", linewidth=1.1, linestyle="--",
               label=f"nominal Gaussian ({NOMINAL_1SIGMA:.3f})")
    for model, off in zip(CROSS_ARCH_MODELS, offsets):
        vals = _cross_arch_values(table, model, "coverage_1sigma")
        bars = ax.bar(
            [pos + off for pos in x],
            vals,
            width=width * 0.9,
            color=CROSS_ARCH_COLORS[model],
            label=CROSS_ARCH_LABELS[model],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    _style_lono_axis(ax)
    ax.set_ylabel("1σ empirical coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Current uncensored LONO: per-fold 1σ coverage")
    ax.legend(frameon=False, fontsize=7.2, ncols=3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_cross_arch_lono_single(table: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    x = list(range(len(FOLD_ORDER)))
    width = 0.34
    offsets = [-width / 2, width / 2]

    if metric == "rmse_mm":
        ax.axhline(LEGACY_HA_RMSE, color="#8F6BB3", linestyle="--", linewidth=1.0,
                   label=f"legacy H-A mean ({LEGACY_HA_RMSE:.1f})")
        ax.axhline(LEGACY_NS_RMSE, color="#C44E52", linestyle="--", linewidth=1.0,
                   label=f"legacy N-S mean ({LEGACY_NS_RMSE:.1f})")
        ylabel = "RMSE [mm]"
        title = "Current uncensored LONO: RMSE by fold"
        fmt = "%.1f"
    elif metric == "coverage_1sigma":
        ax.axhline(NOMINAL_1SIGMA, color="#333333", linewidth=1.1, linestyle="--",
                   label=f"nominal Gaussian ({NOMINAL_1SIGMA:.3f})")
        ax.set_ylim(0.0, 1.05)
        ylabel = "1σ empirical coverage"
        title = "Current uncensored LONO: 1σ coverage by fold"
        fmt = "%.2f"
    else:
        raise ValueError(f"Unsupported metric {metric!r}")

    for model, off in zip(CROSS_ARCH_MODELS, offsets):
        vals = _cross_arch_values(table, model, metric)
        bars = ax.bar(
            [pos + off for pos in x],
            vals,
            width=width * 0.9,
            color=CROSS_ARCH_COLORS[model],
            label=CROSS_ARCH_LABELS[model],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.bar_label(bars, fmt=fmt, fontsize=7, padding=2)

    _style_lono_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7.4, ncols=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_cross_arch_summary(table: pd.DataFrame) -> None:
    print("\nCurrent uncensored LONO summary:")
    for include_nozzle0, label in [(True, "all folds"), (False, "excluding Nozzle0")]:
        sub = table if include_nozzle0 else table.loc[table["holdout"] != "Nozzle0"]
        for model in CROSS_ARCH_MODELS:
            vals = sub.loc[sub["model_group"] == model, "rmse_mm"].astype(float)
            cov = sub.loc[sub["model_group"] == model, "coverage_1sigma"].astype(float)
            print(
                f"  {label:16s} {CROSS_ARCH_LABELS[model]:13s} "
                f"RMSE={vals.mean():.3f}±{vals.std(ddof=1):.3f}, cov1={cov.mean():.3f}"
            )


def main() -> None:
    df2 = pd.read_csv(STAGE2_CSV)
    df3 = pd.read_csv(STAGE3_CSV)
    df2["holdout"] = df2["holdout"].map(_strip_bc)
    df3["holdout"] = df3["holdout"].map(_strip_bc)

    make_lono_figure(
        df2, STAGE2_ABLATIONS, STAGE2_LABELS, STAGE2_COLORS,
        IMAGE_DIR / "stage2_anchor_ablation.png",
        "Stage-2 anchor ablation (LONO)"
    )
    make_lono_figure(
        df3, STAGE3_ABLATIONS, STAGE3_LABELS, STAGE3_COLORS,
        IMAGE_DIR / "neural_network_fit_results" / "ablation_comparison_best.png",
        "Stage-3 regime ablation (LONO, Stage-2 = μ anchor)"
    )

    cross_arch = build_cross_arch_lono_table(df3)
    print_cross_arch_summary(cross_arch)
    plot_cross_arch_lono_combined(cross_arch, IMAGE_DIR / "svgp_lono_comparison.png")
    plot_cross_arch_lono_single(
        cross_arch,
        "rmse_mm",
        IMAGE_DIR / "lono_20260509" / "lono_rmse_by_fold.png",
    )
    plot_cross_arch_lono_single(
        cross_arch,
        "coverage_1sigma",
        IMAGE_DIR / "lono_20260509" / "lono_coverage_by_fold.png",
    )


if __name__ == "__main__":
    main()
