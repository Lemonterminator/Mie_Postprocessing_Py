"""Summarize fit-filter survival rates from ``MLP/synthetic_data/fit_report.csv``.

The fitter emits one report row per nozzle/folder/source. This script
aggregates those counts by nozzle, folder, and source; writes CSV/JSON/TEX
summaries; and copies the CDF survival plot to the thesis image directory by
default.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = PROJECT_ROOT / "MLP" / "synthetic_data" / "fit_report.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "MLP" / "synthetic_data" / "fit_survival_report"
DEFAULT_THESIS_IMAGE_DIR = PROJECT_ROOT / "Thesis" / "images"


def _add_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Attach percentage columns using ``n_total`` as the denominator."""
    out = df.copy()
    denom = out["n_total"].replace(0, np.nan)
    out["clean_pct"] = 100.0 * out["n_clean"] / denom
    out["flagged_pct"] = 100.0 * out["n_flagged"] / denom
    out["success_pct"] = 100.0 * out["success_main"] / denom
    return out


def summarize_fit_report(report_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate the fit report into nozzle/source, folder/source, and source tables."""
    if not report_path.exists():
        raise FileNotFoundError(f"fit report not found: {report_path}")

    df = pd.read_csv(report_path)
    required = {"nozzle", "folder", "penetration_source", "n_total", "n_clean", "n_flagged", "success_main"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"fit report is missing required columns: {', '.join(missing)}")

    numeric_cols = ["n_total", "n_clean", "n_flagged", "success_main"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    by_nozzle_source = (
        df.groupby(["nozzle", "penetration_source", "fit_model"], dropna=False)[numeric_cols]
        .sum()
        .reset_index()
        .pipe(_add_rates)
    )
    by_folder_source = (
        df.groupby(["nozzle", "folder", "penetration_source", "fit_model"], dropna=False)[numeric_cols]
        .sum()
        .reset_index()
        .pipe(_add_rates)
    )
    by_source = (
        df.groupby(["penetration_source", "fit_model"], dropna=False)[numeric_cols]
        .sum()
        .reset_index()
        .pipe(_add_rates)
    )
    return by_nozzle_source, by_folder_source, by_source


def plot_cdf_survival(by_nozzle_source: pd.DataFrame, out_path: Path) -> None:
    cdf = by_nozzle_source.loc[by_nozzle_source["penetration_source"] == "cdf"].copy()
    if cdf.empty:
        return
    cdf = cdf.sort_values("clean_pct")

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    y = np.arange(len(cdf))
    ax.barh(y, cdf["clean_pct"], color="#4c78a8", alpha=0.86, label="clean survival")
    ax.scatter(cdf["success_pct"], y, color="#f58518", s=28, label="RMSE-success rate", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(cdf["nozzle"])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Rate [% of candidate CDF series]")
    ax.set_title("CDF fit-filter survival by nozzle")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_summary(
    *,
    report_path: Path = DEFAULT_REPORT,
    out_dir: Path = DEFAULT_OUT_DIR,
    thesis_image_dir: Path | None = DEFAULT_THESIS_IMAGE_DIR,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_nozzle_source, by_folder_source, by_source = summarize_fit_report(report_path)

    nozzle_csv = out_dir / "filter_survival_by_nozzle_source.csv"
    folder_csv = out_dir / "filter_survival_by_folder_source.csv"
    source_csv = out_dir / "filter_survival_by_source.csv"
    by_nozzle_source.to_csv(nozzle_csv, index=False)
    by_folder_source.to_csv(folder_csv, index=False)
    by_source.to_csv(source_csv, index=False)

    plot_path = out_dir / "filter_survival_cdf_by_nozzle.png"
    plot_cdf_survival(by_nozzle_source, plot_path)

    cdf = by_nozzle_source.loc[by_nozzle_source["penetration_source"] == "cdf"].copy()
    cdf_for_tex = cdf.loc[:, ["nozzle", "n_total", "n_clean", "clean_pct", "success_pct"]].copy()
    cdf_for_tex["clean_pct"] = cdf_for_tex["clean_pct"].map(lambda x: f"{x:.1f}")
    cdf_for_tex["success_pct"] = cdf_for_tex["success_pct"].map(lambda x: f"{x:.1f}")
    (out_dir / "filter_survival_cdf_by_nozzle.tex").write_text(
        cdf_for_tex.to_latex(index=False, escape=True),
        encoding="utf-8",
    )

    if thesis_image_dir is not None and plot_path.exists():
        thesis_image_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plot_path, thesis_image_dir / "filter_survival_cdf_by_nozzle.png")

    summary = {
        "fit_report": str(report_path),
        "outputs": {
            "by_nozzle_source": str(nozzle_csv),
            "by_folder_source": str(folder_csv),
            "by_source": str(source_csv),
            "plot": str(plot_path),
        },
        "cdf_nozzle_count": int(len(cdf)),
        "cdf_total_series": int(cdf["n_total"].sum()) if not cdf.empty else 0,
        "cdf_clean_pct_min": float(cdf["clean_pct"].min()) if not cdf.empty else None,
        "cdf_clean_pct_max": float(cdf["clean_pct"].max()) if not cdf.empty else None,
    }
    (out_dir / "filter_survival_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--thesis-image-dir", type=Path, default=DEFAULT_THESIS_IMAGE_DIR)
    parser.add_argument("--no-thesis-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_summary(
        report_path=args.fit_report,
        out_dir=args.out_dir,
        thesis_image_dir=None if args.no_thesis_copy else args.thesis_image_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
