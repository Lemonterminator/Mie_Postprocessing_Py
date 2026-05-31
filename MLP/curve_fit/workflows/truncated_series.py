"""Build ``series_wide_truncated`` from ``series_wide_clean`` using the
per-condition right-censoring thresholds emitted by
``MLP.curve_fit.workflows.cdf_censoring_points``.

For every trajectory in ``cdf/series_wide_clean/T*.csv``, look up its operating
condition in ``cdf_right_censoring_points/cdf_condition_censoring_summary.csv``
and NaN every ``time_ms_NNN`` / ``penetration_mm_NNN`` pair whose ``time_ms`` is
at or beyond ``censor_start_time_ms`` (joint density-drop + FOV-saturation
criterion). The output preserves the wide-format schema exactly so it can be
consumed by ``load_source_table(..., split="truncated")``.

Run once after each ``fit_raw_data.py`` + ``cdf_censoring_points`` refresh.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CONDITION_COLS = [
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

CENSOR_EPS_MS = 1e-9


def _stable_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Build a join key that is robust to int-vs-float dtype drift across CSVs.

    The same condition can appear as ``int64`` in one CSV (e.g. ``5``) and
    ``float64`` in another (``5.0``) depending on whether any other row in
    that CSV's column had a fractional part. ``astype("string")`` would then
    produce ``"5"`` vs ``"5.0"`` and silently fail to join. Normalising every
    numeric column to a fixed-precision float string avoids that hazard.
    """
    parts: list[pd.Series] = []
    for col in cols:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            f = pd.to_numeric(s, errors="coerce").astype("float64").round(6)
            parts.append(f.map(lambda x: "<NA>" if pd.isna(x) else f"{x:.6f}"))
        else:
            parts.append(s.astype("string").fillna("<NA>"))
    return pd.concat(parts, axis=1).agg("|".join, axis=1)


def build_censor_lookup(summary_csv: Path) -> tuple[dict[str, float], dict[str, str]]:
    summary = pd.read_csv(summary_csv)
    missing = [c for c in CONDITION_COLS if c not in summary.columns]
    if missing:
        raise KeyError(f"{summary_csv} missing condition columns: {missing}")
    key = _stable_key(summary, CONDITION_COLS)
    censor_t = pd.to_numeric(summary["censor_start_time_ms"], errors="coerce")
    finite = censor_t.notna()
    censor_lookup = dict(zip(key[finite].tolist(), censor_t[finite].astype(float).tolist()))
    reason_lookup = dict(zip(key[finite].tolist(), summary.loc[finite, "censor_start_reason"].astype(str).tolist()))
    return censor_lookup, reason_lookup


def truncate_wide_table(
    df: pd.DataFrame,
    *,
    experiment_name: str,
    censor_lookup: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy().reset_index(drop=True)
    if "experiment_name" not in out.columns:
        out.insert(0, "experiment_name", experiment_name)
        injected_experiment_name = True
    else:
        injected_experiment_name = False

    cond_key = _stable_key(out, CONDITION_COLS)
    censor_t = cond_key.map(censor_lookup).astype(float).to_numpy()
    matched_rows = np.isfinite(censor_t)

    time_cols = sorted(
        (c for c in out.columns if c.startswith("time_ms_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    pairs: list[tuple[str, str]] = []
    for tcol in time_cols:
        suffix = tcol.split("_")[-1]
        pcol = f"penetration_mm_{suffix}"
        if pcol in out.columns:
            pairs.append((tcol, pcol))

    n_truncated_frames = 0
    n_rows_truncated_at_least_once = 0
    row_has_any_truncated = np.zeros(len(out), dtype=bool)

    for tcol, pcol in pairs:
        tvals = pd.to_numeric(out[tcol], errors="coerce").to_numpy(dtype=float)
        truncate_mask = matched_rows & np.isfinite(tvals) & (tvals >= censor_t - CENSOR_EPS_MS)
        if not truncate_mask.any():
            continue
        n_truncated_frames += int(truncate_mask.sum())
        row_has_any_truncated |= truncate_mask
        out.loc[truncate_mask, tcol] = np.nan
        out.loc[truncate_mask, pcol] = np.nan

    n_rows_truncated_at_least_once = int(row_has_any_truncated.sum())

    if injected_experiment_name:
        out = out.drop(columns=["experiment_name"])

    stats = {
        "n_rows_total": int(len(out)),
        "n_rows_matched_to_condition": int(matched_rows.sum()),
        "n_rows_truncated_at_least_once": n_rows_truncated_at_least_once,
        "n_truncated_frames": n_truncated_frames,
    }
    return out, stats


def process_nozzle(nozzle_dir: Path, censor_lookup: dict[str, float]) -> dict[str, int]:
    src_dir = nozzle_dir / "cdf" / "series_wide_clean"
    if not src_dir.exists():
        return {}
    dst_dir = nozzle_dir / "cdf" / "series_wide_truncated"
    dst_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "n_files": 0,
        "n_rows_total": 0,
        "n_rows_matched_to_condition": 0,
        "n_rows_truncated_at_least_once": 0,
        "n_truncated_frames": 0,
    }
    for src_csv in sorted(src_dir.glob("*.csv")):
        df = pd.read_csv(src_csv)
        truncated_df, stats = truncate_wide_table(
            df, experiment_name=nozzle_dir.name, censor_lookup=censor_lookup,
        )
        truncated_df.to_csv(dst_dir / src_csv.name, index=False)
        totals["n_files"] += 1
        for key in ("n_rows_total", "n_rows_matched_to_condition",
                    "n_rows_truncated_at_least_once", "n_truncated_frames"):
            totals[key] += int(stats[key])
    return totals


def main() -> None:
    _THIS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = _THIS_DIR.parents[2]
    synthetic_root_env = os.environ.get("FIT_OUTPUT_ROOT")
    synthetic_root = Path(synthetic_root_env) if synthetic_root_env else PROJECT_ROOT / "MLP" / "synthetic_data"
    summary_csv = synthetic_root / "cdf_right_censoring_points" / "cdf_condition_censoring_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(
            f"Missing condition censoring summary: {summary_csv}. "
            "Run MLP.curve_fit.workflows.cdf_censoring_points first."
        )

    censor_lookup, reason_lookup = build_censor_lookup(summary_csv)
    reason_counts: dict[str, int] = {}
    for reason in reason_lookup.values():
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    print(f"Loaded censoring thresholds for {len(censor_lookup)} conditions ({reason_counts}).")

    nozzle_dirs = sorted(p for p in synthetic_root.iterdir() if p.is_dir() and p.name.startswith("BC"))
    if not nozzle_dirs:
        raise RuntimeError(f"No BC* nozzle directories found in {synthetic_root}.")

    manifest = {
        "source_summary": str(summary_csv),
        "synthetic_root": str(synthetic_root),
        "n_conditions_in_lookup": len(censor_lookup),
        "reason_counts": reason_counts,
        "nozzles": {},
    }
    for nozzle_dir in nozzle_dirs:
        stats = process_nozzle(nozzle_dir, censor_lookup)
        if not stats:
            continue
        manifest["nozzles"][nozzle_dir.name] = stats
        print(f"  {nozzle_dir.name}: {stats}")

    manifest_path = synthetic_root / "cdf_right_censoring_points" / "series_wide_truncated_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
