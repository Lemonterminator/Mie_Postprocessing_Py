"""B.5 audit: visualise raw-CDF coverage behind the Stage-3 regime thresholds.

The preferred input is a Stage-3-style ``cdf_plume_audit.csv`` or
``cdf_regime_bins.csv`` table.  If the input is a long point table with
``time_ms`` values, this script rebuilds the same per-condition time-bin
coverage ratios used by ``train_stage3_distillation_plus_raw_series.py``.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.common.archive_layout import SYNTHETIC_DATA_RUNS, append_manifest, resolve_latest
from pipelines.common.latex_helpers import write_newcommands

REGIME_GROUP_COLS = [
    "experiment_name",
    "plumes",
    "diameter_mm",
    "umbrella_angle_deg",
    "fps",
    "chamber_pressure_bar",
    "injection_duration_us",
    "injection_pressure_bar",
    "control_backpressure_bar",
]
SAMPLE_ID_COLS = ["experiment_name", "file_path", "file_name", "file_stem", "plume_idx"]
np = None
pd = None


def _data_deps():
    global np, pd
    if np is None or pd is None:
        import numpy as _np
        import pandas as _pd

        np = _np
        pd = _pd
    return np, pd


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-run-dir", type=Path, default=None)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bin-ms", type=float, default=0.1)
    parser.add_argument("--time-max-ms", type=float, default=5.0)
    parser.add_argument("--uncertain-threshold", type=float, default=0.7)
    parser.add_argument("--teacher-threshold", type=float, default=0.2)
    return parser.parse_args()


def find_input_csv(explicit: Path | None, fit_run_dir: Path | None) -> Path:
    if explicit is not None:
        return explicit
    roots: list[Path] = []
    if fit_run_dir is not None:
        roots.append(fit_run_dir)
    else:
        try:
            roots.append(resolve_latest(SYNTHETIC_DATA_RUNS))
        except FileNotFoundError:
            pass
    roots.append(REPO_ROOT / "MLP" / "figures" / "fit_bias_audit_cdf")

    names = ("cdf_regime_bins.csv", "cdf_plume_audit.csv")
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
            nested = sorted(root.rglob(name)) if root.exists() else []
            if nested:
                return nested[0]
    raise FileNotFoundError(
        "Could not find cdf_regime_bins.csv or cdf_plume_audit.csv. "
        "Pass --input-csv explicitly after running Stage 3 or the CDF bias audit."
    )


def _safe_condition_label(row: pd.Series, group_cols: list[str]) -> str:
    _data_deps()
    parts = []
    for col in group_cols:
        value = row.get(col)
        if pd.isna(value):
            continue
        if col == "experiment_name":
            parts.append(str(value).replace("BC2024", "").replace("_HZ_", "_"))
        elif col.endswith("_bar"):
            parts.append(f"{col.replace('_bar', '')}={float(value):g}")
        elif col.endswith("_us"):
            parts.append(f"dur={float(value):g}")
    return " | ".join(parts) or "condition"


def _add_time_bin(df: pd.DataFrame, *, bin_ms: float, time_max_ms: float) -> pd.DataFrame:
    out = df.copy()
    if "time_bin" in out.columns:
        out["time_bin"] = pd.to_numeric(out["time_bin"], errors="coerce").astype("Int64")
        return out.dropna(subset=["time_bin"]).copy()
    if "time_ms" not in out.columns:
        raise KeyError("Input table must contain either time_bin or time_ms.")
    time_ms = pd.to_numeric(out["time_ms"], errors="coerce")
    n_bins = max(1, int(math.ceil(time_max_ms / bin_ms)))
    out["time_bin"] = np.floor(time_ms / bin_ms).clip(0, n_bins - 1).astype("Int64")
    return out.dropna(subset=["time_bin"]).copy()


def _from_existing_coverage(df: pd.DataFrame, *, bin_ms: float) -> pd.DataFrame:
    group_cols = [c for c in REGIME_GROUP_COLS if c in df.columns]
    if not group_cols:
        group_cols = [c for c in ("condition_group", "experiment_name") if c in df.columns]
    if not group_cols:
        raise KeyError("Existing coverage table has no recognisable condition columns.")
    out = df.copy()
    out["coverage_ratio"] = pd.to_numeric(out["coverage_ratio"], errors="coerce")
    out["time_bin"] = pd.to_numeric(out["time_bin"], errors="coerce").astype("Int64")
    if "time_bin_start_ms" not in out.columns:
        out["time_bin_start_ms"] = out["time_bin"].astype(float) * bin_ms
    if "time_bin_end_ms" not in out.columns:
        out["time_bin_end_ms"] = (out["time_bin"].astype(float) + 1.0) * bin_ms
    labels = out[group_cols].drop_duplicates().copy()
    labels["condition_group"] = labels.apply(lambda r: _safe_condition_label(r, group_cols), axis=1)
    return out.merge(labels, on=group_cols, how="left")


def build_coverage_table(df: pd.DataFrame, *, bin_ms: float, time_max_ms: float) -> pd.DataFrame:
    if {"coverage_ratio", "time_bin"}.issubset(df.columns):
        return _from_existing_coverage(df, bin_ms=bin_ms)

    long_df = _add_time_bin(df, bin_ms=bin_ms, time_max_ms=time_max_ms)
    group_cols = [c for c in REGIME_GROUP_COLS if c in long_df.columns]
    if not group_cols:
        group_cols = [c for c in ("experiment_name", "folder") if c in long_df.columns]
    if not group_cols:
        raise KeyError("Cannot infer condition grouping columns from input table.")

    sample_cols = [c for c in SAMPLE_ID_COLS if c in long_df.columns and c not in group_cols]
    dedup_cols = group_cols + sample_cols + ["time_bin"]
    dedup = long_df.drop_duplicates(subset=dedup_cols)
    counts = (
        dedup.groupby(group_cols + ["time_bin"], dropna=False)
        .size()
        .rename("n_raw")
        .reset_index()
    )

    n_bins = max(1, int(math.ceil(time_max_ms / bin_ms)))
    all_rows = []
    for group_key, group in counts.groupby(group_cols, dropna=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        base = {col: value for col, value in zip(group_cols, group_key)}
        series = group.set_index("time_bin")["n_raw"].reindex(range(n_bins), fill_value=0).astype(float)
        smooth = series.rolling(window=3, center=True, min_periods=1).mean()
        ref = float(max(smooth.max(), 1.0))
        for time_bin, n_raw in series.items():
            row = dict(base)
            row.update(
                {
                    "time_bin": int(time_bin),
                    "time_bin_start_ms": float(time_bin) * bin_ms,
                    "time_bin_end_ms": (float(time_bin) + 1.0) * bin_ms,
                    "n_raw": int(n_raw),
                    "n_raw_smooth": float(smooth.loc[time_bin]),
                    "coverage_ratio": float(smooth.loc[time_bin] / ref),
                }
            )
            all_rows.append(row)
    out = pd.DataFrame(all_rows)
    labels = out[group_cols].drop_duplicates().copy()
    labels["condition_group"] = labels.apply(lambda r: _safe_condition_label(r, group_cols), axis=1)
    return out.merge(labels, on=group_cols, how="left")


def write_plot(table: pd.DataFrame, out_path: Path, *, uncertain: float, teacher: float) -> None:
    plt = _pyplot()
    pivot = (
        table.pivot_table(index="condition_group", columns="time_bin_start_ms", values="coverage_ratio", aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    values = pivot.to_numpy(dtype=float)
    fig_h = min(12.0, max(4.0, 0.22 * max(len(pivot.index), 1)))
    fig, ax = plt.subplots(figsize=(9.0, fig_h), dpi=170)
    im = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Raw CDF coverage ratio")
    ax.set_xlabel("Time bin start (ms)")
    ax.set_ylabel("Condition group")
    x_labels = [f"{x:g}" for x in pivot.columns]
    step = max(1, len(x_labels) // 12)
    ax.set_xticks(np.arange(len(x_labels))[::step])
    ax.set_xticklabels(x_labels[::step], rotation=45, ha="right")
    if len(pivot.index) <= 45:
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=6)
    else:
        ax.set_yticks([])
    if values.shape[0] >= 2 and values.shape[1] >= 2 and np.isfinite(values).any():
        x = np.arange(values.shape[1])
        y = np.arange(values.shape[0])
        ax.contour(x, y, values, levels=[teacher], colors=["white"], linewidths=0.9, linestyles=":")
        ax.contour(x, y, values, levels=[uncertain], colors=["#ffdf5d"], linewidths=1.0)
    ax.set_title(f"Raw CDF coverage; contours at {uncertain:.0%} and {teacher:.0%}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(input_csv: Path, out_dir: Path, *, bin_ms: float, time_max_ms: float, uncertain: float, teacher: float) -> None:
    _data_deps()
    if not input_csv.exists():
        raise FileNotFoundError(input_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv, low_memory=False)
    table = build_coverage_table(df, bin_ms=bin_ms, time_max_ms=time_max_ms)
    table.to_csv(out_dir / "raw_coverage_by_bin.csv", index=False)
    write_plot(table, out_dir / "raw_coverage_heatmap_with_thresholds.png", uncertain=uncertain, teacher=teacher)
    write_newcommands(
        out_dir / "raw_coverage_thresholds.tex",
        {
            "rawCoverageUncertainThresholdPct": f"{100.0 * uncertain:.0f}",
            "rawCoverageTeacherThresholdPct": f"{100.0 * teacher:.0f}",
            "rawCoverageConditionCount": str(int(table["condition_group"].nunique())),
            "rawCoverageBinCount": str(int(table["time_bin"].nunique())),
        },
    )
    append_manifest(out_dir.parent, "raw_coverage_csv", f"{out_dir.name}/raw_coverage_by_bin.csv", "B.5 raw coverage by time bin")
    append_manifest(out_dir.parent, "raw_coverage_heatmap", f"{out_dir.name}/raw_coverage_heatmap_with_thresholds.png", "B.5 raw coverage heatmap")
    append_manifest(out_dir.parent, "raw_coverage_tex", f"{out_dir.name}/raw_coverage_thresholds.tex", "B.5 raw coverage LaTeX thresholds")


def main() -> None:
    args = parse_args()
    input_csv = find_input_csv(args.input_csv, args.fit_run_dir)
    run(
        input_csv,
        args.out_dir,
        bin_ms=args.bin_ms,
        time_max_ms=args.time_max_ms,
        uncertain=args.uncertain_threshold,
        teacher=args.teacher_threshold,
    )


if __name__ == "__main__":
    main()
