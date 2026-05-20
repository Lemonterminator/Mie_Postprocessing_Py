"""D.2 audit: build a denominator-consistent trajectory attrition report.

The canonical population is enumerated from the raw Mie output directory using
the same traversal as ``audit_cdf_spatial_censoring.py``:
``dataset/T*/recording.csv`` with ``penetration_cdf(mm)_plume_*`` columns.
All downstream gates are left-joined to that raw population.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.common.archive_layout import SYNTHETIC_DATA_RUNS, append_manifest, resolve_latest

DEFAULT_RAW_ROOT = REPO_ROOT / "Mie_scattering_top_view_results"
SCR_REL = "spatial_censoring_audit/plume_spatial_censoring_audit.csv"
KEY_COLS = ["dataset", "folder", "file_stem", "plume_idx"]
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


def _collect_raw_inventory(*args, **kwargs):
    from MLP.curve_fit.audit_cdf_spatial_censoring import collect_raw_inventory

    return collect_raw_inventory(*args, **kwargs)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-run-dir", type=Path, default=None)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--stage2-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-synthetic-population", action="store_true",
                        help="Fallback to SCR/fit rows when raw-root is unavailable; for dry structural checks only.")
    return parser.parse_args()


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])


def _resolve_fit_run(fit_run_dir: Path | None) -> Path:
    return fit_run_dir or resolve_latest(SYNTHETIC_DATA_RUNS)


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["dataset", "folder", "file_stem"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    if "plume_idx" in out.columns:
        out["plume_idx"] = pd.to_numeric(out["plume_idx"], errors="coerce").astype("Int64")
    return out


def enumerate_raw_population(raw_root: Path) -> pd.DataFrame:
    _data_deps()
    raw_metrics, _, _ = _collect_raw_inventory(
        raw_root,
        datasets=None,
        folders=None,
        penetration_prefix="penetration_cdf(mm)_plume_",
    )
    rows = [dict(value) for value in raw_metrics.values()]
    if not rows:
        raise FileNotFoundError(f"No raw CDF plume rows found under {raw_root}")
    return _normalize_keys(pd.DataFrame(rows).loc[:, KEY_COLS + ["file_name", "raw_n_finite", "raw_n_positive"]])


def read_scr(fit_run_dir: Path) -> pd.DataFrame:
    _data_deps()
    path = fit_run_dir / SCR_REL
    if not path.exists():
        nested = sorted(fit_run_dir.rglob("plume_spatial_censoring_audit.csv"))
        if not nested:
            return pd.DataFrame(columns=KEY_COLS)
        path = nested[0]
    df = pd.read_csv(path, low_memory=False)
    keep = KEY_COLS + [c for c in ["is_spatial_right_censored", "naive_hold_underestimate_if_censored_mm"] if c in df.columns]
    return _normalize_keys(df.loc[:, [c for c in keep if c in df.columns]]).drop_duplicates(KEY_COLS)


def read_fit_rows(fit_run_dir: Path) -> pd.DataFrame:
    _data_deps()
    frames = []
    for path in sorted(fit_run_dir.glob("*/cdf/all/*.csv")):
        if path.stem.endswith("_flagged"):
            continue
        df = pd.read_csv(path, low_memory=False)
        df.insert(0, "folder", path.stem)
        df.insert(0, "dataset", path.parents[2].name)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=KEY_COLS)
    df = pd.concat(frames, ignore_index=True, sort=False)
    cols = KEY_COLS + [
        c
        for c in [
            "mask_basic",
            "mask_penetration_far",
            "mask_outlier",
            "flag_bad_fit",
            "success",
            "penetration_source",
            "fit_model",
        ]
        if c in df.columns
    ]
    return _normalize_keys(df.loc[:, cols]).drop_duplicates(KEY_COLS)


def read_stage2_manifest(path: Path | None) -> pd.DataFrame:
    _data_deps()
    if path is None or not path.exists():
        return pd.DataFrame(columns=KEY_COLS)
    df = pd.read_csv(path, low_memory=False)
    if "penetration_source" in df.columns:
        df = df.loc[df["penetration_source"].astype(str).eq("cdf")].copy()
    rename = {}
    if "source_dataset_name" in df.columns and "dataset" not in df.columns:
        rename["source_dataset_name"] = "dataset"
    if "source_file" in df.columns and "folder" not in df.columns:
        # source_file is a full CSV path; folder is best-effort from the stem.
        df["folder"] = df["source_file"].map(lambda x: Path(str(x)).stem)
    df = df.rename(columns=rename)
    keep = [c for c in KEY_COLS if c in df.columns]
    if len(keep) < len(KEY_COLS):
        return pd.DataFrame(columns=KEY_COLS)
    return _normalize_keys(df.loc[:, KEY_COLS]).drop_duplicates()


def synthetic_population(scr_df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
    _data_deps()
    frames = [df.loc[:, KEY_COLS] for df in (scr_df, fit_df) if all(c in df.columns for c in KEY_COLS) and not df.empty]
    if not frames:
        raise FileNotFoundError("Cannot build synthetic fallback population; SCR and fit rows are empty.")
    return _normalize_keys(pd.concat(frames, ignore_index=True)).drop_duplicates(KEY_COLS)


def build_wide(pop: pd.DataFrame, scr_df: pd.DataFrame, fit_df: pd.DataFrame, stage2_df: pd.DataFrame) -> pd.DataFrame:
    wide = pop.drop_duplicates(KEY_COLS).copy()
    wide = wide.merge(scr_df, on=KEY_COLS, how="left")
    wide = wide.merge(fit_df, on=KEY_COLS, how="left", suffixes=("", "_fit"))
    if not stage2_df.empty:
        stage2_df = stage2_df.copy()
        stage2_df["included_in_stage2_training"] = True
        wide = wide.merge(stage2_df, on=KEY_COLS, how="left")
    else:
        wide["included_in_stage2_training"] = np.nan

    wide["is_spatial_right_censored"] = _bool_series(wide.get("is_spatial_right_censored", pd.Series(False, index=wide.index)))
    wide["pass_spatial_cap"] = ~wide["is_spatial_right_censored"]
    for col in ["mask_basic", "mask_penetration_far"]:
        if col in wide.columns:
            known = wide[col].notna()
            wide[f"pass_{col}"] = _bool_series(wide[col]).where(known, np.nan)
        else:
            wide[f"pass_{col}"] = np.nan
    if "mask_outlier" in wide.columns:
        known = wide["mask_outlier"].notna()
        wide["pass_robust_outlier"] = (~_bool_series(wide["mask_outlier"])).where(known, np.nan)
    else:
        wide["pass_robust_outlier"] = np.nan
    if "flag_bad_fit" in wide.columns:
        known = wide["flag_bad_fit"].notna()
        wide["pass_fit_filter"] = (~_bool_series(wide["flag_bad_fit"])).where(known, np.nan)
    else:
        wide["pass_fit_filter"] = np.nan
    wide["included_in_stage2_training"] = wide["included_in_stage2_training"].fillna(False).astype(bool)
    return wide


def build_long(wide: pd.DataFrame) -> pd.DataFrame:
    gates = [
        ("raw_population", pd.Series(True, index=wide.index), "Raw CDF plume trajectory"),
        ("pass_spatial_cap", wide["pass_spatial_cap"], "Not spatially right-censored"),
        ("pass_basic_mask", wide["pass_mask_basic"], "Pass basic fit sanity mask"),
        ("pass_far_penetration_mask", wide["pass_mask_penetration_far"], "Pass far-penetration range mask"),
        ("pass_robust_outlier", wide["pass_robust_outlier"], "Not a robust grouped outlier"),
        ("pass_fit_filter", wide["pass_fit_filter"], "Pass combined fit filter"),
        ("included_in_stage2_training", wide["included_in_stage2_training"], "Included in Stage-2 row table"),
    ]
    rows = []
    for order, (gate, series, desc) in enumerate(gates):
        values = pd.Series(series, index=wide.index)
        for idx, value in values.items():
            rows.append({**wide.loc[idx, KEY_COLS].to_dict(), "gate_order": order, "gate": gate, "description": desc, "passed": bool(value) if pd.notna(value) else False, "known": bool(pd.notna(value))})
    return pd.DataFrame(rows)


def build_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    n_raw = int(long_df.loc[long_df["gate"] == "raw_population", KEY_COLS].drop_duplicates().shape[0])
    summary = (
        long_df.groupby(["gate_order", "gate", "description"], dropna=False)
        .agg(n_known=("known", "sum"), n_pass=("passed", "sum"))
        .reset_index()
        .sort_values("gate_order")
    )
    summary["n_raw_denominator"] = n_raw
    summary["pct_of_raw"] = 100.0 * summary["n_pass"] / max(n_raw, 1)
    summary["pct_of_known"] = 100.0 * summary["n_pass"] / summary["n_known"].replace(0, np.nan)
    return summary


def write_tex(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    str(row["description"]).replace("&", r"\&"),
                    f"{int(row['n_pass']):,}",
                    f"{int(row['n_raw_denominator']):,}",
                    f"{float(row['pct_of_raw']):.1f}\\%",
                ]
            )
            + r" \\"
        )
    text = "\n".join(
        [
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Gate & Kept & Raw denominator & Kept / raw \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def write_funnel(summary: pd.DataFrame, path: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=170)
    labels = summary["gate"].astype(str).tolist()
    vals = summary["n_pass"].astype(float).to_numpy()
    y = np.arange(len(vals))
    ax.barh(y, vals, color="#4c78a8")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Trajectories")
    ax.set_title("Trajectory attrition, common raw denominator")
    for yi, val in zip(y, vals):
        ax.text(val, yi, f" {int(val):,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(fit_run_dir: Path, raw_root: Path, out_dir: Path, *, stage2_manifest: Path | None, allow_synthetic_population: bool) -> None:
    _data_deps()
    out_dir.mkdir(parents=True, exist_ok=True)
    scr_df = read_scr(fit_run_dir)
    fit_df = read_fit_rows(fit_run_dir)
    try:
        pop = enumerate_raw_population(raw_root)
    except FileNotFoundError:
        if not allow_synthetic_population:
            raise
        pop = synthetic_population(scr_df, fit_df)
    stage2_df = read_stage2_manifest(stage2_manifest)
    wide = build_wide(pop, scr_df, fit_df, stage2_df)
    long_df = build_long(wide)
    summary = build_summary(long_df)

    wide.to_csv(out_dir / "_attrition_wide.csv", index=False)
    long_df.to_csv(out_dir / "_attrition_long.csv", index=False)
    summary.to_csv(out_dir / "data_attrition_summary.csv", index=False)
    write_tex(summary, out_dir / "data_attrition.tex")
    write_funnel(summary, out_dir / "data_attrition_sankey.png")
    append_manifest(out_dir.parent, "data_attrition_long", f"{out_dir.name}/_attrition_long.csv", "D.2 per-trajectory gate table")
    append_manifest(out_dir.parent, "data_attrition_summary", f"{out_dir.name}/data_attrition_summary.csv", "D.2 attrition summary")
    append_manifest(out_dir.parent, "data_attrition_tex", f"{out_dir.name}/data_attrition.tex", "D.2 attrition LaTeX table")
    append_manifest(out_dir.parent, "data_attrition_sankey", f"{out_dir.name}/data_attrition_sankey.png", "D.2 attrition funnel plot")


def main() -> None:
    args = parse_args()
    run(
        _resolve_fit_run(args.fit_run_dir),
        args.raw_root,
        args.out_dir,
        stage2_manifest=args.stage2_manifest,
        allow_synthetic_population=args.allow_synthetic_population,
    )


if __name__ == "__main__":
    main()
