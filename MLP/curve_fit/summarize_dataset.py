"""
Scan Mie_scattering_top_view_results and the test_matrix_json configs to emit
a concise campaign summary for the thesis dataset section.

Writes:
  MLP/dataset_summary.csv  -- per-dataset row
  MLP/dataset_summary.txt  -- human-readable block for the thesis
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = REPO_ROOT / "Mie_scattering_top_view_results"
JSON_DIR = REPO_ROOT / "test_matrix_json"
OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "dataset_summary.csv"
OUT_TXT = OUT_DIR / "dataset_summary.txt"

DATASET_NOZZLE_MAP = {
    "BC20241003_HZ_Nozzle1": "Nozzle1",
    "BC20241017_HZ_Nozzle2": "Nozzle2",
    "BC20241014_HZ_Nozzle3": "Nozzle3",
    "BC20241007_HZ_Nozzle4": "Nozzle4",
    "BC20241010_HZ_Nozzle5": "Nozzle5",
    "BC20241011_HZ_Nozzle6": "Nozzle6",
    "BC20241015_HZ_Nozzle7": "Nozzle7",
    "BC20241016_HZ_Nozzle8": "Nozzle8",
    "BC20220627 - Heinzman DS300 - Mie Top view": "Nozzle0",
}


def _load_json(nozzle_name: str) -> dict | None:
    p = JSON_DIR / f"{nozzle_name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _nozzle0_durations_from_lookup(cfg: dict) -> set:
    """Expand the Nozzle0 injection-duration formula over cine numbers 1..145."""
    lookup = cfg.get("injection_duration_lookup", {})
    if not lookup:
        return set()
    durations = set()
    for cine_num in range(1, 146):
        block = (cine_num - 1) // 5
        if block <= 18:
            dur = 340 + 20 * block
        else:
            dur = 750 + 50 * (block - 19)
        durations.add(dur)
    return durations


def _extract_config_ranges(cfg: dict) -> dict:
    """Return scalar summary fields parsed from a nozzle JSON config."""
    props = cfg.get("nozzle_properties", {})
    tm = cfg.get("test_matrix", {})

    inj_pressures: set = set()
    inj_durations: set = set()
    chamber_pressures: set = set()
    backpressures: set = set()

    groups = tm.get("groups", [])

    if groups and isinstance(groups[0], dict) and "id" in groups[0]:
        # Nozzle0-style: each group is a single operating point
        for g in groups:
            if "injection_pressure_bar" in g:
                inj_pressures.add(g["injection_pressure_bar"])
            if "chamber_pressure_bar" in g:
                chamber_pressures.add(g["chamber_pressure_bar"])
            if "control_backpressure" in g:
                backpressures.add(g["control_backpressure"])
        inj_durations = _nozzle0_durations_from_lookup(cfg)
        num_design_conditions = len(groups)

    elif groups:
        # Nozzle2-style: each group item is its own cartesian sub-matrix
        sub_counts = []
        for g in groups:
            inj_pressures.add(g.get("injection_pressure_bar", 2000))
            densities = g.get("chamber_density_kg_per_m3", [])
            pressures = g.get("chamber_pressures_bar", [])
            for p in pressures:
                chamber_pressures.add(p)
            durs = g.get("injection_durations_us", [])
            for d in durs:
                inj_durations.add(d)
            sub_counts.append(len(densities) * len(durs))
        num_design_conditions = sum(sub_counts)

    elif tm.get("expansion") == "cartesian":
        # Nozzle1,3-8 style: single cartesian block
        inj_pressure = tm.get("injection_pressure_bar", 2000)
        inj_pressures.add(inj_pressure)
        for p in tm.get("chamber_pressures_bar", []):
            chamber_pressures.add(p)
        for d in tm.get("injection_durations_us", []):
            inj_durations.add(d)
        n_densities = len(tm.get("chamber_density_kg_per_m3", []))
        n_durations = len(tm.get("injection_durations_us", []))
        num_design_conditions = n_densities * n_durations

    else:
        num_design_conditions = 0

    return {
        "plumes": props.get("plumes", "?"),
        "diameter_mm": props.get("diameter_mm", "?"),
        "umbrella_angle_deg": props.get("umbrella_angle_deg", "?"),
        "fps": props.get("fps", "?"),
        "inj_pressure_min_bar": min(inj_pressures) if inj_pressures else float("nan"),
        "inj_pressure_max_bar": max(inj_pressures) if inj_pressures else float("nan"),
        "inj_duration_min_us": min(inj_durations) if inj_durations else float("nan"),
        "inj_duration_max_us": max(inj_durations) if inj_durations else float("nan"),
        "chamber_pressure_min_bar": min(chamber_pressures) if chamber_pressures else float("nan"),
        "chamber_pressure_max_bar": max(chamber_pressures) if chamber_pressures else float("nan"),
        "backpressures": sorted(backpressures),
        "num_design_conditions": num_design_conditions,
    }


def _scan_results_dir(dataset_dir: Path) -> dict:
    """Count processed recordings and conditions from the results directory."""
    if not dataset_dir.exists():
        return {
            "exists": False,
            "num_condition_dirs": 0,
            "num_csv_files": 0,
            "num_meta_files": 0,
            "total_size_mb": 0.0,
        }

    condition_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    csv_files = list(dataset_dir.rglob("*.csv"))
    # Exclude any csv files inside subdirectories named "boundary_points"
    csv_files = [f for f in csv_files if "boundary_points" not in f.parts]
    meta_files = list(dataset_dir.rglob("*.meta.json"))
    total_bytes = sum(f.stat().st_size for f in csv_files + meta_files if f.exists())

    return {
        "exists": True,
        "num_condition_dirs": len(condition_dirs),
        "num_csv_files": len(csv_files),
        "num_meta_files": len(meta_files),
        "total_size_mb": total_bytes / 1e6,
    }


def _scan_meta_conditions(dataset_dir: Path) -> dict:
    """Read *.meta.json files and aggregate unique injection conditions."""
    inj_pressures: set = set()
    inj_durations: set = set()
    chamber_pressures: set = set()
    backpressures: set = set()

    for meta_path in dataset_dir.rglob("*.meta.json"):
        try:
            with open(meta_path, encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            continue
        if v := m.get("injection_pressure_bar"):
            inj_pressures.add(v)
        if v := m.get("injection_duration_us"):
            inj_durations.add(v)
        if v := m.get("chamber_pressure_bar"):
            chamber_pressures.add(v)
        if v := m.get("control_backpressure_bar"):
            backpressures.add(v)

    return {
        "unique_inj_pressures": sorted(inj_pressures),
        "unique_chamber_pressures": sorted(chamber_pressures),
        "unique_backpressures": sorted(backpressures),
        "inj_duration_min_us": min(inj_durations) if inj_durations else float("nan"),
        "inj_duration_max_us": max(inj_durations) if inj_durations else float("nan"),
    }


def build_summary() -> pd.DataFrame:
    rows = []
    for dataset_name, nozzle_name in DATASET_NOZZLE_MAP.items():
        cfg = _load_json(nozzle_name)
        cfg_ranges = _extract_config_ranges(cfg) if cfg else {}

        dataset_dir = RESULTS_ROOT / dataset_name
        dir_stats = _scan_results_dir(dataset_dir)
        meta_conds = _scan_meta_conditions(dataset_dir) if dir_stats["exists"] else {}

        row = {
            "dataset": dataset_name,
            "nozzle": nozzle_name,
            **cfg_ranges,
            **{f"dir_{k}": v for k, v in dir_stats.items() if k != "exists"},
            "dir_exists": dir_stats["exists"],
        }
        if meta_conds:
            row["meta_inj_pressures"] = str(meta_conds.get("unique_inj_pressures", []))
            row["meta_chamber_pressures"] = str(meta_conds.get("unique_chamber_pressures", []))
            row["meta_backpressures"] = str(meta_conds.get("unique_backpressures", []))
            row["meta_inj_dur_min_us"] = meta_conds.get("inj_duration_min_us", float("nan"))
            row["meta_inj_dur_max_us"] = meta_conds.get("inj_duration_max_us", float("nan"))

        rows.append(row)

    return pd.DataFrame(rows)


def _format_range(lo, hi, unit=""):
    lo_s = f"{lo:g}" if isinstance(lo, float) and np.isfinite(lo) else str(lo)
    hi_s = f"{hi:g}" if isinstance(hi, float) and np.isfinite(hi) else str(hi)
    if lo == hi:
        return f"{lo_s}{unit}"
    return f"{lo_s}--{hi_s}{unit}"


def write_human_summary(df: pd.DataFrame, path: Path):
    lines = []
    lines.append("=" * 72)
    lines.append("OSCC Mie top-view campaign — dataset summary")
    lines.append("=" * 72)
    lines.append("")

    # Global aggregate
    present = df[df["dir_exists"]]
    total_csv = int(df["dir_num_csv_files"].sum())
    total_cond = int(df["dir_num_condition_dirs"].sum())
    total_meta = int(df["dir_num_meta_files"].sum())
    total_mb = float(df["dir_total_size_mb"].sum())

    all_plumes = sorted(df["plumes"].dropna().unique())
    all_diams = sorted(df["diameter_mm"].dropna().unique())
    all_fps = sorted(df["fps"].dropna().unique())

    inj_lo = df["inj_pressure_min_bar"].min()
    inj_hi = df["inj_pressure_max_bar"].max()
    dur_lo = df["inj_duration_min_us"].min()
    dur_hi = df["inj_duration_max_us"].max()
    ch_lo = df["chamber_pressure_min_bar"].min()
    ch_hi = df["chamber_pressure_max_bar"].max()

    lines.append(f"Nozzle families   : {len(df)}  (Nozzle0--Nozzle8)")
    lines.append(f"Nozzle diameters  : {', '.join(str(d) for d in all_diams)} mm")
    lines.append(f"Plume counts (N)  : {', '.join(str(p) for p in all_plumes)}")
    lines.append(f"Frame rate        : {', '.join(str(f) for f in all_fps)} fps")
    lines.append(f"Injection pressure: {_format_range(inj_lo, inj_hi, ' bar')}")
    lines.append(f"Injection duration: {_format_range(dur_lo, dur_hi, ' µs')}")
    lines.append(f"Chamber pressure  : {_format_range(ch_lo, ch_hi, ' bar')}")
    lines.append("")
    lines.append("Results directory statistics (processed files):")
    lines.append(f"  Condition folders: {total_cond}")
    lines.append(f"  Recording CSVs   : {total_csv}  (one per .cine file)")
    lines.append(f"  Metadata JSONs   : {total_meta}")
    lines.append(f"  CSV+meta size    : {total_mb:.1f} MB")
    lines.append("")
    lines.append("Per-nozzle breakdown:")
    lines.append(
        f"  {'Nozzle':<10} {'d_mm':>6} {'N':>4} {'angle':>7} {'fps':>7} "
        f"{'P_inj(bar)':>14} {'t_inj(µs)':>14} {'conditions':>11} {'csvs':>6}"
    )
    lines.append("  " + "-" * 82)
    for _, row in df.iterrows():
        p_range = _format_range(row["inj_pressure_min_bar"], row["inj_pressure_max_bar"])
        d_range = _format_range(row["inj_duration_min_us"], row["inj_duration_max_us"])
        lines.append(
            f"  {row['nozzle']:<10} {row['diameter_mm']:>6.3f} {int(row['plumes']):>4} "
            f"{int(row['umbrella_angle_deg']):>6}° {int(row['fps']):>7} "
            f"{p_range:>14} {d_range:>14} {int(row['num_design_conditions']):>11} "
            f"{int(row['dir_num_csv_files']):>6}"
        )

    lines.append("")
    lines.append("Suggested thesis sentence:")
    lines.append(
        f"  The campaign covers {len(df)} nozzle families (Nozzle0--Nozzle8) "
        f"with hole diameters from {min(all_diams):.3f} to {max(all_diams):.3f} mm and "
        f"N = {min(all_plumes)}--{max(all_plumes)} plumes per nozzle, "
        f"spanning injection pressures from {int(inj_lo)} to {int(inj_hi)} bar, "
        f"injection durations from {int(dur_lo)} to {int(dur_hi)} µs, "
        f"and chamber pressures from {ch_lo:.2f} to {int(ch_hi)} bar, "
        f"with {total_cond} test-condition folders and {total_csv} processed recordings."
    )
    lines.append("=" * 72)

    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    print(f"Scanning results root: {RESULTS_ROOT}")
    df = build_summary()
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    write_human_summary(df, OUT_TXT)
    print(f"Saved: {OUT_TXT}")


if __name__ == "__main__":
    main()
