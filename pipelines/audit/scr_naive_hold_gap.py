"""B.1 audit: quantify the naive-hold extrapolation gap for spatially right-censored traces.

Reads plume_spatial_censoring_audit.csv (produced by audit_cdf_spatial_censoring.py) and
emits the canonical median naive-hold gap number plus a distribution histogram and a
LaTeX snippet the thesis can \\input{}.

Outputs (in --out-dir):
    scr_naive_hold_gap_summary.csv    -- overall + by-nozzle summary table
    scr_naive_hold_gap_histogram.png  -- distribution of the gap for censored traces
    scr_naive_hold_gap.tex            -- \\newcommand snippets for thesis \\input{}
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.common.archive_layout import SYNTHETIC_DATA_RUNS, resolve_latest, append_manifest
from pipelines.common.latex_helpers import write_newcommands, write_table

DEFAULT_AUDIT_CSV_REL = "spatial_censoring_audit/plume_spatial_censoring_audit.csv"
DEFAULT_SUMMARY_CSV_REL = "spatial_censoring_audit/spatial_censoring_summary_overall.csv"

GAP_COL = "naive_hold_underestimate_if_censored_mm"
CENSORED_FLAG = "is_spatial_right_censored"
np = None


def _data_deps():
    global np
    if np is None:
        import numpy as _np

        np = _np
    return np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit-run-dir", type=Path, default=None,
                   help="Specific fit run directory. Defaults to latest.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory. Defaults to audit_runs/latest/b1_scr_gap/")
    return p.parse_args()


def _parse_bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "t", "yes", "y")


def _parse_float(v: str) -> float:
    try:
        return float(v.strip())
    except (ValueError, AttributeError):
        return math.nan


def load_per_trajectory(csv_path: Path) -> tuple[list[float], list[str], float, int]:
    """Load per-trajectory data. Returns (gaps_mm, nozzles, censored_rate, n_total)."""
    gaps: list[float] = []
    nozzles: list[str] = []
    n_total = 0
    n_censored = 0

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_total += 1
            if _parse_bool(row.get(CENSORED_FLAG, "0")):
                n_censored += 1
                g = _parse_float(row.get(GAP_COL, "nan"))
                if math.isfinite(g):
                    gaps.append(g)
                    # Nozzle label from dataset/folder column
                    dataset = row.get("dataset", row.get("folder", ""))
                    import re
                    m = re.search(r"Nozzle(\d+)", dataset)
                    nozzles.append(f"Nozzle{m.group(1)}" if m else "Unknown")

    censored_rate = n_censored / n_total if n_total > 0 else float("nan")
    return gaps, nozzles, censored_rate, n_total


def compute_summary(
    gaps: list[float], nozzles: list[str], censored_rate: float, n_total: int
) -> dict:
    _data_deps()
    arr = np.array(gaps)
    by_nozzle: dict[str, list[float]] = {}
    for g, n in zip(gaps, nozzles):
        by_nozzle.setdefault(n, []).append(g)

    rows = []
    for nozzle in sorted(by_nozzle):
        vals = np.array(by_nozzle[nozzle])
        rows.append({
            "nozzle": nozzle,
            "n_censored": len(vals),
            "median_mm": float(np.median(vals)),
            "q25_mm": float(np.percentile(vals, 25)),
            "q75_mm": float(np.percentile(vals, 75)),
            "mean_mm": float(np.mean(vals)),
        })

    return {
        "overall_median_mm": float(np.median(arr)) if len(arr) > 0 else float("nan"),
        "overall_q25_mm": float(np.percentile(arr, 25)) if len(arr) > 0 else float("nan"),
        "overall_q75_mm": float(np.percentile(arr, 75)) if len(arr) > 0 else float("nan"),
        "overall_mean_mm": float(np.mean(arr)) if len(arr) > 0 else float("nan"),
        "n_censored_total": len(arr),
        "n_total": int(n_total),
        "scr_rate": censored_rate,
        "by_nozzle": rows,
    }


def write_summary_csv(out_dir: Path, summary: dict) -> None:
    path = out_dir / "scr_naive_hold_gap_summary.csv"
    rows = [
        {
            "nozzle": "ALL",
            "n_censored": summary["n_censored_total"],
            "scr_rate_pct": round(summary["scr_rate"] * 100, 1),
            "median_mm": round(summary["overall_median_mm"], 1),
            "q25_mm": round(summary["overall_q25_mm"], 1),
            "q75_mm": round(summary["overall_q75_mm"], 1),
            "mean_mm": round(summary["overall_mean_mm"], 1),
        }
    ]
    for r in summary["by_nozzle"]:
        rows.append({
            "nozzle": r["nozzle"],
            "n_censored": r["n_censored"],
            "scr_rate_pct": "",
            "median_mm": round(r["median_mm"], 1),
            "q25_mm": round(r["q25_mm"], 1),
            "q75_mm": round(r["q75_mm"], 1),
            "mean_mm": round(r["mean_mm"], 1),
        })
    fieldnames = ["nozzle", "n_censored", "scr_rate_pct", "median_mm", "q25_mm", "q75_mm", "mean_mm"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}")


def write_histogram(out_dir: Path, gaps: list[float], median_mm: float) -> None:
    plt = _pyplot()
    path = out_dir / "scr_naive_hold_gap_histogram.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gaps, bins=40, color="#4878CF", edgecolor="white", linewidth=0.4)
    ax.axvline(median_mm, color="#E34A33", linewidth=1.5, linestyle="--",
               label=f"Median = {median_mm:.0f} mm")
    ax.set_xlabel("Naive-hold extrapolation gap (mm)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("SCR Naive-Hold Gap Distribution (censored traces only)", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Written: {path}")


def write_latex_snippet(out_dir: Path, summary: dict) -> None:
    path = out_dir / "scr_naive_hold_gap.tex"
    commands = {
        "scrNaiveHoldGapMedianMm": str(round(summary["overall_median_mm"])),
        "scrNaiveHoldGapMeanMm": str(round(summary["overall_mean_mm"])),
        "scrNaiveHoldGapQ25Mm": str(round(summary["overall_q25_mm"])),
        "scrNaiveHoldGapQ75Mm": str(round(summary["overall_q75_mm"])),
        "scrRatePct": str(round(summary["scr_rate"] * 100, 1)),
        "scrNCensored": str(summary["n_censored_total"]),
        "scrNTotal": str(summary["n_total"]),
    }
    write_newcommands(path, commands)
    print(f"  Written: {path}")
    print(f"  Median naive-hold gap: {summary['overall_median_mm']:.1f} mm  "
          f"(SCR rate {summary['scr_rate']*100:.1f}%)")


def run(fit_run_dir: Path, out_dir: Path) -> None:
    _data_deps()
    audit_csv = fit_run_dir / DEFAULT_AUDIT_CSV_REL
    if not audit_csv.exists():
        print(f"[B.1] Audit CSV not found: {audit_csv}")
        print("       Run audit_cdf_spatial_censoring.py first (or run the full fit pipeline).")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[B.1] Reading: {audit_csv}")

    gaps, nozzles, censored_rate, _n_total = load_per_trajectory(audit_csv)
    if not gaps:
        print("[B.1] No censored trajectories with finite gap values found. Exiting.")
        sys.exit(1)

    summary = compute_summary(gaps, nozzles, censored_rate, _n_total)
    write_summary_csv(out_dir, summary)
    write_histogram(out_dir, gaps, summary["overall_median_mm"])
    write_latex_snippet(out_dir, summary)

    append_manifest(out_dir.parent, role="scr_gap_summary",
                    filename=f"{out_dir.name}/scr_naive_hold_gap_summary.csv",
                    description="B.1: SCR naive-hold gap summary table")
    append_manifest(out_dir.parent, role="scr_gap_histogram",
                    filename=f"{out_dir.name}/scr_naive_hold_gap_histogram.png",
                    description="B.1: SCR naive-hold gap distribution histogram")
    append_manifest(out_dir.parent, role="scr_gap_tex",
                    filename=f"{out_dir.name}/scr_naive_hold_gap.tex",
                    description="B.1: LaTeX \\newcommand snippets for thesis \\input{}")


def main() -> None:
    args = parse_args()

    if args.fit_run_dir is not None:
        fit_run_dir = args.fit_run_dir
    else:
        try:
            fit_run_dir = resolve_latest(SYNTHETIC_DATA_RUNS)
        except FileNotFoundError:
            print("[B.1] No fit run found. Specify --fit-run-dir or run the fit pipeline first.")
            sys.exit(1)

    out_dir = args.out_dir or (fit_run_dir / "b1_scr_gap")
    run(fit_run_dir, out_dir)


if __name__ == "__main__":
    main()
