from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "Mie_scattering_top_view_results"

PENETRATION_COLUMN_SPECS = [
    {
        "legacy_prefix": "penetration_cdf_plume_",
        "mm_prefix": "penetration_cdf(mm)_plume_",
        "needs_umbrella_correction": True,
    },
    {
        "legacy_prefix": "penetration_bw_x_plume_",
        "mm_prefix": "penetration_bw_x(mm)_plume_",
        "needs_umbrella_correction": True,
    },
    {
        "legacy_prefix": "penetration_bw_polar_plume_",
        "mm_prefix": "penetration_bw_polar(mm)_plume_",
        "needs_umbrella_correction": False,
    },
]


def get_dataset_settings(name: str) -> dict:
    if name == "BC20220627 - Heinzman DS300 - Mie Top view":
        return {
            "or_mm_per_px_reference": 412.0,
        }
    return {
        "or_mm_per_px_reference": 377.0,
    }


def get_mm_per_px_scale(dataset_name: str) -> float:
    settings = get_dataset_settings(dataset_name)
    return 90.0 / float(settings["or_mm_per_px_reference"])


def read_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_umbrella_angle_deg(df: pd.DataFrame, meta: dict) -> float:
    if "umbrella_angle_deg" in df.columns:
        value = pd.to_numeric(df["umbrella_angle_deg"], errors="coerce")
        if len(value) and np.isfinite(value.iloc[0]):
            return float(value.iloc[0])
    value = meta.get("umbrella_angle_deg")
    if value is not None and np.isfinite(pd.to_numeric(value, errors="coerce")):
        return float(value)
    return 180.0


def iter_metric_csvs(results_root: Path):
    for csv_path in sorted(results_root.rglob("*.csv")):
        if "boundary_points" in csv_path.parts:
            continue
        if csv_path.name == "processing.log":
            continue
        yield csv_path


def backfill_csv(csv_path: Path) -> tuple[bool, int]:
    relative_parts = csv_path.relative_to(RESULTS_ROOT).parts
    if len(relative_parts) < 2:
        return False, 0
    dataset_name = relative_parts[0]
    df = pd.read_csv(csv_path)

    needed_legacy_cols = any(
        any(col.startswith(spec["legacy_prefix"]) for col in df.columns)
        for spec in PENETRATION_COLUMN_SPECS
    )
    if not needed_legacy_cols:
        return False, 0

    meta = read_meta(csv_path.with_suffix(".meta.json"))
    umbrella_angle_deg = resolve_umbrella_angle_deg(df, meta)
    tilt_ang = (180.0 - umbrella_angle_deg) / 2.0
    umbrella_angle_correction = 1.0 / np.cos(np.deg2rad(tilt_ang))
    mm_per_px_scale = get_mm_per_px_scale(dataset_name)

    added_columns = 0
    for spec in PENETRATION_COLUMN_SPECS:
        correction = mm_per_px_scale
        if spec["needs_umbrella_correction"]:
            correction *= umbrella_angle_correction

        matching_legacy_cols = [col for col in df.columns if col.startswith(spec["legacy_prefix"])]
        for legacy_col in matching_legacy_cols:
            suffix = legacy_col[len(spec["legacy_prefix"]):]
            mm_col = f"{spec['mm_prefix']}{suffix}"
            if mm_col in df.columns:
                continue
            df[mm_col] = pd.to_numeric(df[legacy_col], errors="coerce") * correction
            added_columns += 1

    if added_columns == 0:
        return False, 0

    df.to_csv(csv_path, index=False)
    return True, added_columns


def main():
    if not RESULTS_ROOT.exists():
        raise FileNotFoundError(f"Results root not found: {RESULTS_ROOT}")

    updated_files = 0
    added_columns_total = 0
    for csv_path in iter_metric_csvs(RESULTS_ROOT):
        updated, added_columns = backfill_csv(csv_path)
        if not updated:
            continue
        updated_files += 1
        added_columns_total += added_columns
        print(f"Updated {csv_path} (+{added_columns} mm columns)")

    print(
        f"Done. Updated {updated_files} file(s), added {added_columns_total} penetration(mm) columns."
    )


if __name__ == "__main__":
    main()
