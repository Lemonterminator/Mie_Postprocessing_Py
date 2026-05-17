"""E.1 audit: join spatial right-censoring rates with empirical OOD status.

The join is intentionally condition-level.  SCR is measured per trajectory, but
the empirical-support check is defined on operating-condition feature vectors;
therefore this script aggregates SCR first and only then evaluates support.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.common.archive_layout import SYNTHETIC_DATA_RUNS, append_manifest, resolve_latest

SCR_REL = "spatial_censoring_audit/plume_spatial_censoring_audit.csv"
CONDITION_COLS = [
    "nozzle",
    "chamber_pressure_bar",
    "injection_pressure_bar",
    "injection_duration_us",
    "control_backpressure_bar",
]
SUPPORT_COLS = [
    "tilt_angle_radian",
    "umbrella_angle_deg",
    "diameter_mm",
    "plumes",
    "injection_duration_us",
    "injection_pressure_bar",
    "chamber_pressure_bar",
    "control_backpressure_bar",
]
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


def _ood_helpers():
    from MLP.ood_sanity import build_empirical_support, check_input_sanity

    return build_empirical_support, check_input_sanity


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-run-dir", type=Path, default=None)
    parser.add_argument("--scr-csv", type=Path, default=None)
    parser.add_argument("--support-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _bool_series(series: pd.Series) -> pd.Series:
    _data_deps()
    return series.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])


def find_scr_csv(explicit: Path | None, fit_run_dir: Path | None) -> Path:
    if explicit is not None:
        return explicit
    root = fit_run_dir or resolve_latest(SYNTHETIC_DATA_RUNS)
    candidate = root / SCR_REL
    if candidate.exists():
        return candidate
    nested = sorted(root.rglob("plume_spatial_censoring_audit.csv")) if root.exists() else []
    if nested:
        return nested[0]
    raise FileNotFoundError(f"Could not find {SCR_REL} under {root}")


def _read_support_from_fit_run(fit_run_dir: Path | None) -> pd.DataFrame:
    _data_deps()
    if fit_run_dir is None:
        try:
            fit_run_dir = resolve_latest(SYNTHETIC_DATA_RUNS)
        except FileNotFoundError:
            return pd.DataFrame()
    frames = []
    for path in sorted(fit_run_dir.glob("*/cdf/clean/*.csv")):
        df = pd.read_csv(path, low_memory=False)
        df.insert(0, "experiment_name", path.parents[2].name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def find_support_df(explicit: Path | None, fit_run_dir: Path | None) -> pd.DataFrame:
    _data_deps()
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return pd.read_csv(explicit, low_memory=False)
    candidates = [
        REPO_ROOT / "MLP" / "figures" / "fit_bias_audit_cdf" / "cdf_plume_audit.csv",
    ]
    if fit_run_dir is not None:
        candidates.extend(sorted(fit_run_dir.rglob("cdf_plume_audit.csv")))
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    df = _read_support_from_fit_run(fit_run_dir)
    if df.empty:
        raise FileNotFoundError("Could not find support CSV or cdf/clean fit rows for OOD support.")
    return df


def _normalize_support_df(df: pd.DataFrame) -> pd.DataFrame:
    _data_deps()
    out = df.copy()
    if "tilt_angle_radian" not in out.columns and "umbrella_angle_deg" in out.columns:
        out["tilt_angle_radian"] = np.deg2rad((180.0 - pd.to_numeric(out["umbrella_angle_deg"], errors="coerce")) / 2.0)
    if "sample_split" not in out.columns and "flag_bad_fit" in out.columns:
        bad = out["flag_bad_fit"].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
        out["sample_split"] = np.where(bad, "flagged", "clean")
    return out


def _condition_summary(scr_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CONDITION_COLS if c not in scr_df.columns]
    if missing:
        raise KeyError(f"SCR CSV is missing condition columns: {', '.join(missing)}")
    work = scr_df.copy()
    work["is_spatial_right_censored"] = _bool_series(work["is_spatial_right_censored"])
    grouped = (
        work.groupby(CONDITION_COLS, dropna=False)
        .agg(
            n_trajectories=("is_spatial_right_censored", "size"),
            n_spatial_right_censored=("is_spatial_right_censored", "sum"),
        )
        .reset_index()
    )
    grouped["scr_rate"] = grouped["n_spatial_right_censored"] / grouped["n_trajectories"].clip(lower=1)
    return grouped


def _sample_for_condition(condition: pd.Series, support_df: pd.DataFrame) -> dict:
    combo_cols = ["injection_pressure_bar", "chamber_pressure_bar", "control_backpressure_bar"]
    mask = np.ones(len(support_df), dtype=bool)
    for col in combo_cols + ["injection_duration_us"]:
        if col not in support_df.columns or col not in condition.index:
            continue
        target = float(condition[col]) if pd.notna(condition[col]) else np.nan
        values = pd.to_numeric(support_df[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(target):
            mask &= np.isclose(values, target, atol=1e-9, rtol=0.0)
    if "nozzle" in condition.index and "experiment_name" in support_df.columns:
        nozzle_text = str(condition["nozzle"]).replace(" ", "")
        mask &= support_df["experiment_name"].astype(str).str.replace(" ", "", regex=False).str.contains(nozzle_text, case=False, regex=False)

    selected = support_df.loc[mask]
    if selected.empty:
        selected = support_df
    sample = {}
    for col in SUPPORT_COLS:
        if col in selected.columns:
            values = pd.to_numeric(selected[col], errors="coerce").dropna()
            if not values.empty:
                sample[col] = float(values.iloc[0] if col in {"plumes"} else values.median())
    for col in CONDITION_COLS:
        if col in condition.index and col != "nozzle":
            sample[col] = float(condition[col]) if pd.notna(condition[col]) else np.nan
    return sample


def attach_ood_status(condition_df: pd.DataFrame, support_df: pd.DataFrame) -> pd.DataFrame:
    _data_deps()
    build_empirical_support, check_input_sanity = _ood_helpers()
    support_norm = _normalize_support_df(support_df)
    support = build_empirical_support(support_norm, split_filter="clean" if "sample_split" in support_norm.columns else None)
    rows = []
    for _, condition in condition_df.iterrows():
        sample = _sample_for_condition(condition, support_norm)
        report = check_input_sanity(sample, support)
        row = condition.to_dict()
        row.update(
            {
                "ood_status": "ood" if report["is_ood"] else "in_support",
                "ood_severity": report["severity"],
                "combo_exact_match": bool(report["combo_exact_match"]),
                "ood_warning_count": len(report.get("warnings", [])),
                "ood_warnings": " | ".join(report.get("warnings", [])[:3]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_plot(condition_df: pd.DataFrame, out_path: Path) -> None:
    plt = _pyplot()
    df = condition_df.copy()
    if df.empty:
        return
    try:
        df["scr_decile"] = pd.qcut(df["scr_rate"], q=min(10, max(1, len(df))), duplicates="drop")
    except ValueError:
        df["scr_decile"] = pd.cut(df["scr_rate"], bins=1)
    summary = (
        df.groupby(["scr_decile", "ood_status"], observed=True)
        .size()
        .rename("n_conditions")
        .reset_index()
    )
    pivot = summary.pivot_table(index="scr_decile", columns="ood_status", values="n_conditions", fill_value=0)
    frac = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    ax = frac.plot(kind="bar", stacked=True, figsize=(8.5, 4.5), color=["#4c78a8", "#e45756"], rot=45)
    ax.set_ylabel("Fraction of conditions")
    ax.set_xlabel("SCR-rate bin")
    ax.set_title("Condition-level SCR rate vs empirical OOD status")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="OOD status")
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=170)
    plt.close(ax.figure)


def run(scr_csv: Path, support_df: pd.DataFrame, out_dir: Path) -> None:
    _data_deps()
    out_dir.mkdir(parents=True, exist_ok=True)
    scr_df = pd.read_csv(scr_csv, low_memory=False)
    condition_df = attach_ood_status(_condition_summary(scr_df), support_df)
    condition_df.to_csv(out_dir / "condition_scr_vs_ood.csv", index=False)

    tag_cols = CONDITION_COLS + ["ood_status", "ood_severity", "combo_exact_match"]
    traj = scr_df.merge(condition_df[tag_cols], on=CONDITION_COLS, how="left")
    traj.to_csv(out_dir / "trajectory_scr_ood_tags.csv", index=False)
    write_plot(condition_df, out_dir / "scr_ood_decile_summary.png")
    append_manifest(out_dir.parent, "scr_ood_condition_csv", f"{out_dir.name}/condition_scr_vs_ood.csv", "E.1 condition-level SCR vs OOD table")
    append_manifest(out_dir.parent, "scr_ood_trajectory_tags", f"{out_dir.name}/trajectory_scr_ood_tags.csv", "E.1 trajectory rows tagged with condition OOD status")
    append_manifest(out_dir.parent, "scr_ood_plot", f"{out_dir.name}/scr_ood_decile_summary.png", "E.1 SCR/OOD summary plot")


def main() -> None:
    args = parse_args()
    fit_run_dir = args.fit_run_dir
    scr_csv = find_scr_csv(args.scr_csv, fit_run_dir)
    support_df = find_support_df(args.support_csv, fit_run_dir)
    run(scr_csv, support_df, args.out_dir)


if __name__ == "__main__":
    main()
