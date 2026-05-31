"""Pre-training plume quality gate for synthetic_data exports.

The gate is intentionally independent of any trained surrogate.  It audits CDF
wide-series traces, identifies condition/plume pairs with repeated shape
pathologies or systematically low penetration relative to the other holes in
the same operating condition, and can materialize a filtered synthetic-data
root with those whole plume_idx rows removed.

Typical audit:

    python -m MLP.curve_fit.workflows.qc_gate_plumes \
        --input-root MLP/synthetic_data_clean_lv2

Materialize a training-ready subset:

    python -m MLP.curve_fit.workflows.qc_gate_plumes \
        --input-root MLP/synthetic_data_clean_lv2 \
        --output-root MLP/synthetic_data_clean_lv3_qc_gated
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_GRID_MS = (0.6, 0.8, 1.0, 1.2)
DEFAULT_SOURCE = "cdf"
DEFAULT_APPLY_SOURCES = ("cdf", "bw_x", "bw_polar")
DEFAULT_MATERIALIZE_SUBDIRS = (
    "clean",
    "series_clean",
    "series_wide_clean",
    "series_wide_truncated",
)


@dataclass(frozen=True)
class QCGateConfig:
    input_root: str
    decision_source: str = DEFAULT_SOURCE
    grid_ms: tuple[float, ...] = DEFAULT_GRID_MS
    min_points: int = 4
    negative_step_mm: float = -2.0
    positive_step_spike_mm: float = 14.0
    d1_spike_mm_per_ms: float = 450.0
    d2_spike_mm_per_ms2: float = 10000.0
    plateau_step_mm: float = 1.3
    plateau_jump_mm: float = 8.0
    low_time_ratio: float = 0.80
    low_ratio: float = 0.75
    very_low_ratio: float = 0.70
    severe_low_ratio: float = 0.65
    min_low_timepoints: int = 2
    min_comparison_plumes: int = 4
    repeat_min_conditions: int = 3
    repeat_min_fraction: float = 0.25
    shape_min_traces_loose: int = 5
    shape_trace_fraction_loose: float = 0.20
    shape_min_traces_strict: int = 3
    shape_trace_fraction_strict: float = 0.40


def _prefixed_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    def suffix_int(name: str) -> int:
        suffix = name[len(prefix):]
        return int(suffix) if suffix.isdigit() else 10**9

    return sorted([c for c in df.columns if c.startswith(prefix)], key=suffix_int)


def _grid_tag(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _parse_float_list(text: str) -> tuple[float, ...]:
    vals = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not vals:
        raise ValueError("Expected at least one comma-separated float.")
    return vals


def _bool_series(values: pd.Series, default: bool = False) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(default)
    text = values.astype(str).str.strip().str.lower()
    out = text.isin({"1", "true", "yes", "y", "drop"})
    out = out.where(~text.isin({"0", "false", "no", "n", "keep"}), False)
    return out.fillna(default)


def _nanmedian_abs_dev(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def _trace_metrics(row: pd.Series, time_cols: list[str], pen_cols: list[str], config: QCGateConfig) -> dict:
    time = row[time_cols].to_numpy(dtype=float)
    pen = row[pen_cols].to_numpy(dtype=float)
    finite = np.isfinite(time) & np.isfinite(pen)
    time = time[finite]
    pen = pen[finite]
    if time.size < config.min_points:
        return {
            "valid_trace": False,
            "n_points": int(time.size),
        }

    order = np.argsort(time)
    time = time[order]
    pen = pen[order]
    keep_unique = np.r_[True, np.diff(time) > 1e-12]
    time = time[keep_unique]
    pen = pen[keep_unique]
    if time.size < config.min_points:
        return {
            "valid_trace": False,
            "n_points": int(time.size),
        }

    dt = np.diff(time)
    dy = np.diff(pen)
    dt_ok = dt > 1e-12
    d1 = dy[dt_ok] / dt[dt_ok] if np.any(dt_ok) else np.array([], dtype=float)
    d2 = (
        np.diff(d1) / np.maximum((dt[dt_ok][1:] + dt[dt_ok][:-1]) * 0.5, 1e-12)
        if d1.size >= 2
        else np.array([], dtype=float)
    )

    negative_flag = bool(dy.size and np.nanmin(dy) <= config.negative_step_mm)
    spike_flag = bool(
        (dy.size and np.nanmax(dy) >= config.positive_step_spike_mm)
        or (d1.size and np.nanmax(d1) >= config.d1_spike_mm_per_ms)
        or (d2.size and np.nanmax(np.abs(d2)) >= config.d2_spike_mm_per_ms2)
    )

    plateau_jump_flag = False
    plateau_jump_mm = 0.0
    if dy.size >= 4:
        for idx in range(2, dy.size):
            if np.nanmax(dy[idx - 2: idx]) <= config.plateau_step_mm and dy[idx] >= config.plateau_jump_mm:
                plateau_jump_flag = True
                plateau_jump_mm = max(plateau_jump_mm, float(dy[idx]))

    out = {
        "valid_trace": True,
        "n_points": int(time.size),
        "t_min_ms": float(time[0]),
        "t_max_ms": float(time[-1]),
        "last_penetration_mm": float(pen[-1]),
        "p90_penetration_mm": float(np.nanpercentile(pen, 90)),
        "min_step_mm": float(np.nanmin(dy)) if dy.size else np.nan,
        "max_step_mm": float(np.nanmax(dy)) if dy.size else np.nan,
        "max_d1_mm_per_ms": float(np.nanmax(d1)) if d1.size else np.nan,
        "p95_d1_mm_per_ms": float(np.nanpercentile(d1, 95)) if d1.size else np.nan,
        "max_abs_d2_mm_per_ms2": float(np.nanmax(np.abs(d2))) if d2.size else np.nan,
        "negative_step_flag": negative_flag,
        "trace_spike_flag": spike_flag,
        "plateau_then_jump_flag": plateau_jump_flag,
        "plateau_then_jump_mm": plateau_jump_mm,
    }
    for tm in config.grid_ms:
        col = f"pen_at_{_grid_tag(tm)}ms"
        out[col] = float(np.interp(tm, time, pen)) if time[0] <= tm <= time[-1] else np.nan
    return out


def load_trace_metrics(input_root: Path, config: QCGateConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for csv_path in sorted(input_root.glob(f"*/{config.decision_source}/series_wide_clean/*.csv")):
        experiment_name = csv_path.parents[2].name
        test_name = csv_path.stem
        df = pd.read_csv(csv_path, low_memory=False)
        time_cols = _prefixed_columns(df, "time_ms_")
        pen_cols = _prefixed_columns(df, "penetration_mm_")
        if not time_cols or len(time_cols) != len(pen_cols):
            continue
        for wide_row_idx, row in df.iterrows():
            plume_idx = int(pd.to_numeric(row.get("plume_idx"), errors="coerce"))
            metrics = _trace_metrics(row, time_cols, pen_cols, config)
            metrics.update({
                "experiment_name": experiment_name,
                "test_name": test_name,
                "file_path": row.get("file_path", ""),
                "file_name": row.get("file_name", ""),
                "file_stem": row.get("file_stem", ""),
                "plume_idx": plume_idx,
                "wide_row_idx": int(wide_row_idx),
            })
            rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _condition_plume_aggregate(trace_df: pd.DataFrame, config: QCGateConfig) -> pd.DataFrame:
    valid = trace_df.loc[trace_df["valid_trace"].fillna(False)].copy()
    if valid.empty:
        return pd.DataFrame()

    group_cols = ["experiment_name", "test_name", "plume_idx"]
    agg = valid.groupby(group_cols, dropna=False).agg(
        n_traces=("file_name", "count"),
        median_n_points=("n_points", "median"),
        median_t_max_ms=("t_max_ms", "median"),
        median_last_penetration_mm=("last_penetration_mm", "median"),
        median_p90_penetration_mm=("p90_penetration_mm", "median"),
        max_step_mm=("max_step_mm", "max"),
        median_max_step_mm=("max_step_mm", "median"),
        max_d1_mm_per_ms=("max_d1_mm_per_ms", "max"),
        median_max_d1_mm_per_ms=("max_d1_mm_per_ms", "median"),
        max_abs_d2_mm_per_ms2=("max_abs_d2_mm_per_ms2", "max"),
        median_max_abs_d2_mm_per_ms2=("max_abs_d2_mm_per_ms2", "median"),
        trace_spike_fraction=("trace_spike_flag", "mean"),
        trace_spike_count=("trace_spike_flag", "sum"),
        negative_step_fraction=("negative_step_flag", "mean"),
        negative_step_count=("negative_step_flag", "sum"),
        plateau_jump_fraction=("plateau_then_jump_flag", "mean"),
        plateau_jump_count=("plateau_then_jump_flag", "sum"),
        max_plateau_jump_mm=("plateau_then_jump_mm", "max"),
    ).reset_index()

    ratio_cols: list[str] = []
    z_cols: list[str] = []
    for tm in config.grid_ms:
        pen_col = f"pen_at_{_grid_tag(tm)}ms"
        if pen_col not in valid.columns:
            continue
        cp = valid.groupby(group_cols, dropna=False)[pen_col].median().reset_index()
        cond_cols = ["experiment_name", "test_name"]
        cond_median = cp.groupby(cond_cols, dropna=False)[pen_col].transform("median")
        cond_mad = cp.groupby(cond_cols, dropna=False)[pen_col].transform(_nanmedian_abs_dev)
        ratio_col = f"{pen_col}_ratio_to_condition_median"
        z_col = f"{pen_col}_robust_z_in_condition"
        cp[ratio_col] = cp[pen_col] / cond_median.replace(0.0, np.nan)
        cp[z_col] = 0.6745 * (cp[pen_col] - cond_median) / cond_mad.replace(0.0, np.nan)
        ratio_cols.append(ratio_col)
        z_cols.append(z_col)
        agg = agg.merge(cp[group_cols + [pen_col, ratio_col, z_col]], on=group_cols, how="left")

    if ratio_cols:
        agg["min_low_ratio"] = agg[ratio_cols].min(axis=1)
        agg["median_low_ratio"] = agg[ratio_cols].median(axis=1)
        agg["n_low_timepoints"] = (agg[ratio_cols] < config.low_time_ratio).sum(axis=1)
        agg["n_very_low_timepoints"] = (agg[ratio_cols] < config.very_low_ratio).sum(axis=1)
        agg["n_available_comparison_timepoints"] = agg[ratio_cols].notna().sum(axis=1)
    else:
        agg["min_low_ratio"] = np.nan
        agg["median_low_ratio"] = np.nan
        agg["n_low_timepoints"] = 0
        agg["n_very_low_timepoints"] = 0
        agg["n_available_comparison_timepoints"] = 0

    n_plumes = (
        agg.groupby(["experiment_name", "test_name"], dropna=False)["plume_idx"]
        .transform("nunique")
    )
    agg["n_comparison_plumes"] = n_plumes.astype(int)

    enough_plumes = agg["n_comparison_plumes"] >= config.min_comparison_plumes
    low_a = (agg["min_low_ratio"] <= config.low_ratio) & (agg["n_low_timepoints"] >= config.min_low_timepoints)
    low_b = (agg["min_low_ratio"] <= config.very_low_ratio) & (agg["n_low_timepoints"] >= 1)
    agg["systematic_low_condition_flag"] = (enough_plumes & (low_a | low_b)).fillna(False)
    agg["severe_low_condition_flag"] = (
        enough_plumes
        & (agg["min_low_ratio"] <= config.severe_low_ratio)
        & (agg["n_low_timepoints"] >= config.min_low_timepoints)
    ).fillna(False)

    shape_strict = (
        (agg["n_traces"] >= config.shape_min_traces_strict)
        & (
            (agg["trace_spike_fraction"] >= config.shape_trace_fraction_strict)
            | (agg["plateau_jump_fraction"] >= config.shape_trace_fraction_strict)
            | (agg["negative_step_fraction"] >= config.shape_trace_fraction_strict)
        )
    )
    shape_loose = (
        (agg["n_traces"] >= config.shape_min_traces_loose)
        & (
            (agg["trace_spike_fraction"] >= config.shape_trace_fraction_loose)
            | (agg["plateau_jump_fraction"] >= config.shape_trace_fraction_loose)
            | (agg["negative_step_fraction"] >= config.shape_trace_fraction_loose)
        )
    )
    agg["recurrent_shape_flag"] = (shape_strict | shape_loose).fillna(False)
    return agg


def _attach_repeated_plume_support(manifest: pd.DataFrame, config: QCGateConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if manifest.empty:
        return manifest, pd.DataFrame()

    rep = manifest.groupby(["experiment_name", "plume_idx"], dropna=False).agg(
        n_conditions=("test_name", "count"),
        systematic_low_conditions=("systematic_low_condition_flag", "sum"),
        severe_low_conditions=("severe_low_condition_flag", "sum"),
        recurrent_shape_conditions=("recurrent_shape_flag", "sum"),
        min_low_ratio=("min_low_ratio", "min"),
        median_min_low_ratio=("min_low_ratio", "median"),
    ).reset_index()
    rep["systematic_low_condition_fraction"] = (
        rep["systematic_low_conditions"] / rep["n_conditions"].clip(lower=1)
    )
    rep["repeat_low_support_flag"] = (
        (rep["systematic_low_conditions"] >= config.repeat_min_conditions)
        & (rep["systematic_low_condition_fraction"] >= config.repeat_min_fraction)
    )
    manifest = manifest.merge(
        rep[
            [
                "experiment_name",
                "plume_idx",
                "systematic_low_conditions",
                "systematic_low_condition_fraction",
                "repeat_low_support_flag",
            ]
        ],
        on=["experiment_name", "plume_idx"],
        how="left",
    )
    return manifest, rep


def _load_manual_flags(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["experiment_name", "test_name", "plume_idx", "manual_drop", "manual_reason"])
    df = pd.read_csv(path, low_memory=False)
    required = {"experiment_name", "test_name", "plume_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manual flag file is missing columns: {sorted(missing)}")
    out = df.copy()
    out["plume_idx"] = pd.to_numeric(out["plume_idx"], errors="raise").astype(int)
    out["manual_drop"] = _bool_series(out["drop"], default=True) if "drop" in out.columns else True
    if "reason" not in out.columns:
        out["reason"] = "manual_image_evidence"
    out = out.rename(columns={"reason": "manual_reason"})
    return out[["experiment_name", "test_name", "plume_idx", "manual_drop", "manual_reason"]]


def build_qc_manifest(
    trace_df: pd.DataFrame,
    config: QCGateConfig,
    *,
    manual_flags: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = _condition_plume_aggregate(trace_df, config)
    manifest, repeated = _attach_repeated_plume_support(manifest, config)
    if manifest.empty:
        return manifest, repeated

    manual_flags = manual_flags if manual_flags is not None else pd.DataFrame()
    if not manual_flags.empty:
        manifest = manifest.merge(
            manual_flags,
            on=["experiment_name", "test_name", "plume_idx"],
            how="left",
        )
    else:
        manifest["manual_drop"] = False
        manifest["manual_reason"] = ""

    manifest["manual_drop"] = manifest["manual_drop"].fillna(False).astype(bool)
    manifest["manual_reason"] = manifest["manual_reason"].fillna("")
    manifest["drop_systematic_low"] = (
        manifest["systematic_low_condition_flag"]
        & (
            manifest["repeat_low_support_flag"].fillna(False)
            | manifest["severe_low_condition_flag"].fillna(False)
        )
    )
    manifest["drop_recurrent_shape"] = manifest["recurrent_shape_flag"].fillna(False)
    manifest["hard_drop"] = (
        manifest["manual_drop"]
        | manifest["drop_systematic_low"]
        | manifest["drop_recurrent_shape"]
    )

    def reason(row: pd.Series) -> str:
        parts: list[str] = []
        if bool(row.get("manual_drop", False)):
            parts.append(str(row.get("manual_reason") or "manual_image_evidence"))
        if bool(row.get("drop_systematic_low", False)):
            parts.append("systematic_low_penetration_with_repeat_or_severity")
        if bool(row.get("drop_recurrent_shape", False)):
            parts.append("recurrent_shape_derivative_pathology")
        return ";".join(parts)

    manifest["drop_reason"] = manifest.apply(reason, axis=1)
    return manifest.sort_values(["experiment_name", "test_name", "plume_idx"]).reset_index(drop=True), repeated


def _drop_lookup(manifest: pd.DataFrame) -> dict[tuple[str, str], set[int]]:
    lookup: dict[tuple[str, str], set[int]] = {}
    drops = manifest.loc[manifest["hard_drop"].fillna(False), ["experiment_name", "test_name", "plume_idx"]]
    for (experiment_name, test_name), group in drops.groupby(["experiment_name", "test_name"], dropna=False):
        lookup[(str(experiment_name), str(test_name))] = set(group["plume_idx"].astype(int).tolist())
    return lookup


def _filter_by_lookup(df: pd.DataFrame, *, experiment_name: str, test_name: str, lookup: dict[tuple[str, str], set[int]]) -> pd.DataFrame:
    drops = lookup.get((experiment_name, test_name), set())
    if not drops or "plume_idx" not in df.columns:
        return df
    plume = pd.to_numeric(df["plume_idx"], errors="coerce")
    return df.loc[~plume.isin(drops)].copy()


def _condition_from_table_path(path: Path) -> str:
    stem = path.stem
    return stem[:-8] if stem.endswith("_flagged") else stem


def materialize_filtered_root(
    input_root: Path,
    output_root: Path,
    manifest: pd.DataFrame,
    *,
    apply_sources: Iterable[str],
    materialize_subdirs: Iterable[str],
    overwrite: bool = False,
) -> dict:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    lookup = _drop_lookup(manifest)
    stats = {
        "csv_files_written": 0,
        "rows_before": 0,
        "rows_after": 0,
        "rows_removed": 0,
        "tables_filtered": 0,
    }
    apply_sources = tuple(apply_sources)
    materialize_subdirs = tuple(materialize_subdirs)

    for exp_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        experiment_name = exp_dir.name
        for source in apply_sources:
            source_dir = exp_dir / source
            if not source_dir.exists():
                continue
            for subdir_name in materialize_subdirs:
                src_subdir = source_dir / subdir_name
                if not src_subdir.exists():
                    continue
                dst_subdir = output_root / experiment_name / source / subdir_name
                dst_subdir.mkdir(parents=True, exist_ok=True)
                for csv_path in sorted(src_subdir.glob("*.csv")):
                    test_name = _condition_from_table_path(csv_path)
                    df = pd.read_csv(csv_path, low_memory=False)
                    before = len(df)
                    df_out = _filter_by_lookup(df, experiment_name=experiment_name, test_name=test_name, lookup=lookup)
                    after = len(df_out)
                    df_out.to_csv(dst_subdir / csv_path.name, index=False)
                    stats["csv_files_written"] += 1
                    stats["rows_before"] += int(before)
                    stats["rows_after"] += int(after)
                    if before != after:
                        stats["tables_filtered"] += 1
                        stats["rows_removed"] += int(before - after)

    # Fixed CDF point tables are useful for evaluation.  They are filtered when
    # row keys are present, but downstream censoring summaries should be
    # regenerated before using them as headline evaluation artifacts.
    cdf_points_dir = input_root / "cdf_right_censoring_points"
    if cdf_points_dir.exists():
        dst_dir = output_root / "cdf_right_censoring_points"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in sorted(cdf_points_dir.glob("*.csv")):
            df = pd.read_csv(csv_path, low_memory=False)
            before = len(df)
            df_out = df
            if {"experiment_name", "test_name", "plume_idx"}.issubset(df.columns):
                parts = []
                for (experiment_name, test_name), group in df.groupby(["experiment_name", "test_name"], dropna=False):
                    parts.append(_filter_by_lookup(group, experiment_name=str(experiment_name), test_name=str(test_name), lookup=lookup))
                df_out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
            after = len(df_out)
            df_out.to_csv(dst_dir / csv_path.name, index=False)
            stats["csv_files_written"] += 1
            stats["rows_before"] += int(before)
            stats["rows_after"] += int(after)
            if before != after:
                stats["tables_filtered"] += 1
                stats["rows_removed"] += int(before - after)

    return stats


def write_reports(
    report_dir: Path,
    *,
    trace_df: pd.DataFrame,
    manifest: pd.DataFrame,
    repeated: pd.DataFrame,
    config: QCGateConfig,
    materialize_stats: dict | None = None,
) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    trace_path = report_dir / "trace_shape_metrics.csv"
    manifest_path = report_dir / "condition_plume_qc_manifest.csv"
    repeated_path = report_dir / "repeated_plume_summary.csv"
    trace_df.to_csv(trace_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    repeated.to_csv(repeated_path, index=False)

    hard = manifest.loc[manifest["hard_drop"].fillna(False)].copy() if not manifest.empty else manifest
    summary = {
        "config": asdict(config),
        "n_trace_rows": int(len(trace_df)),
        "n_condition_plumes": int(len(manifest)),
        "n_hard_drop_condition_plumes": int(len(hard)),
        "n_experiment_plume_repeats": int(len(repeated)),
        "hard_drop_by_experiment": (
            hard.groupby("experiment_name").size().astype(int).to_dict() if not hard.empty else {}
        ),
        "hard_drop_by_reason": (
            hard["drop_reason"].value_counts().astype(int).to_dict() if not hard.empty else {}
        ),
        "trace_metrics_csv": str(trace_path),
        "condition_plume_manifest_csv": str(manifest_path),
        "repeated_plume_summary_csv": str(repeated_path),
        "materialize_stats": materialize_stats or {},
        "note": (
            "Rules use only pre-training trajectory shape, same-condition inter-hole "
            "comparisons, repeated same-plume support, and optional manual image evidence."
        ),
    }
    with (report_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, default=Path("MLP/synthetic_data_clean_lv2"))
    p.add_argument("--output-root", type=Path, default=None,
                   help="Optional filtered synthetic-data root. If omitted, only reports are written.")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow replacing an existing --output-root.")
    p.add_argument("--report-dir", type=Path, default=None,
                   help="Report directory. Defaults to output_root/qc_gate_report or input_root/qc_gate_report.")
    p.add_argument("--decision-source", default=DEFAULT_SOURCE)
    p.add_argument("--apply-sources", default=",".join(DEFAULT_APPLY_SOURCES),
                   help="Comma-separated sources to materialize when --output-root is set.")
    p.add_argument("--materialize-subdirs", default=",".join(DEFAULT_MATERIALIZE_SUBDIRS),
                   help="Comma-separated source subdirectories to materialize.")
    p.add_argument("--grid-ms", default=",".join(str(v) for v in DEFAULT_GRID_MS))
    p.add_argument("--manual-flags", type=Path, default=None,
                   help="Optional CSV with experiment_name,test_name,plume_idx,drop,reason.")
    p.add_argument("--low-time-ratio", type=float, default=0.80)
    p.add_argument("--low-ratio", type=float, default=0.75)
    p.add_argument("--very-low-ratio", type=float, default=0.70)
    p.add_argument("--severe-low-ratio", type=float, default=0.65)
    p.add_argument("--repeat-min-conditions", type=int, default=3)
    p.add_argument("--repeat-min-fraction", type=float, default=0.25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root is not None else None
    report_dir = args.report_dir
    if report_dir is None:
        report_dir = (output_root if output_root is not None else input_root) / "qc_gate_report"
    else:
        report_dir = report_dir.expanduser().resolve()

    config = QCGateConfig(
        input_root=str(input_root),
        decision_source=str(args.decision_source),
        grid_ms=_parse_float_list(args.grid_ms),
        low_time_ratio=float(args.low_time_ratio),
        low_ratio=float(args.low_ratio),
        very_low_ratio=float(args.very_low_ratio),
        severe_low_ratio=float(args.severe_low_ratio),
        repeat_min_conditions=int(args.repeat_min_conditions),
        repeat_min_fraction=float(args.repeat_min_fraction),
    )

    trace_df = load_trace_metrics(input_root, config)
    manual = _load_manual_flags(args.manual_flags)
    manifest, repeated = build_qc_manifest(trace_df, config, manual_flags=manual)

    materialize_stats = None
    if output_root is not None:
        materialize_stats = materialize_filtered_root(
            input_root,
            output_root,
            manifest,
            apply_sources=[s.strip() for s in str(args.apply_sources).split(",") if s.strip()],
            materialize_subdirs=[s.strip() for s in str(args.materialize_subdirs).split(",") if s.strip()],
            overwrite=bool(args.overwrite),
        )

    summary = write_reports(
        report_dir,
        trace_df=trace_df,
        manifest=manifest,
        repeated=repeated,
        config=config,
        materialize_stats=materialize_stats,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
