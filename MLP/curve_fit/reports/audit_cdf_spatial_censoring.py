"""Build plume-level spatial right-censoring labels for CDF trajectories.

The audit estimates a per-nozzle field-of-view cap from raw positive CDF
penetration samples, then compares three views of each plume: raw Mie CSV
trace, cleaned synthetic ``series_wide_all`` trace, and stored fit row. A plume
is marked spatially right-censored when either raw or cleaned penetration
reaches the estimated cap within the configured tolerance.

Outputs include per-plume labels, grouped summaries, cap tables, and optional
diagnostic plots under ``MLP/synthetic_data/spatial_censoring_audit`` by
default. Downstream evaluation scripts join these labels onto NN point
predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "Mie_scattering_top_view_results"
DEFAULT_SYNTHETIC_ROOT = PROJECT_ROOT / "MLP" / "synthetic_data"
DEFAULT_OUT_DIR = DEFAULT_SYNTHETIC_ROOT / "spatial_censoring_audit"
DEFAULT_HORIZON_MS = 5.0
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit spatial right-censoring by comparing raw Mie multi-hole CDF "
            "penetration traces against MLP/synthetic_data fit rows."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--penetration-prefix",
        default="penetration_cdf(mm)_plume_",
        help="Raw CSV penetration columns used to estimate spatial caps.",
    )
    parser.add_argument(
        "--cap-quantile",
        type=float,
        default=0.999,
        help="Per-nozzle quantile used as the field-of-view penetration cap.",
    )
    parser.add_argument(
        "--cap-abs-tol-mm",
        type=float,
        default=1.0,
        help="Absolute tolerance for declaring a trace close to the spatial cap.",
    )
    parser.add_argument(
        "--cap-rel-tol",
        type=float,
        default=0.01,
        help="Relative tolerance for declaring a trace close to the spatial cap.",
    )
    parser.add_argument(
        "--horizon-ms",
        type=float,
        default=DEFAULT_HORIZON_MS,
        help="Fit horizon used only for quantifying naive saturation bias.",
    )
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--folder", action="append", dest="folders")
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


def infer_nozzle(dataset_name: str) -> str:
    match = re.search(r"Nozzle(\d+)", dataset_name)
    return f"Nozzle {match.group(1)}" if match else "Other"


def parse_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text == "":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_int(value: object) -> int | None:
    number = parse_float(value)
    if not math.isfinite(number):
        return None
    return int(number)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def quantile(values: list[float], q: float) -> float:
    clean = sorted(finite_values(values))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def safe_mean(values: list[float]) -> float:
    clean = finite_values(values)
    return mean(clean) if clean else math.nan


def safe_median(values: list[float]) -> float:
    clean = finite_values(values)
    return median(clean) if clean else math.nan


def format_value(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    return value


def expit(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sigmoid_fit_mm(row: dict[str, str], horizon_ms: float) -> float:
    if abs(horizon_ms - DEFAULT_HORIZON_MS) < 1e-9:
        stored_far = parse_float(row.get("penetration_far_mm"))
        if math.isfinite(stored_far):
            return stored_far

    log_k_sqrt = parse_float(row.get("log_k_sqrt"))
    log_k_quarter = parse_float(row.get("log_k_quarter"))
    log_t0 = parse_float(row.get("log_t0"))
    log_s = parse_float(row.get("log_s"))
    if not all(math.isfinite(x) for x in (log_k_sqrt, log_k_quarter, log_t0, log_s)):
        return math.nan

    t_s = max(horizon_ms * 1e-3, 1e-9)
    k_sqrt = math.exp(log_k_sqrt)
    k_quarter = math.exp(log_k_quarter)
    t0 = math.exp(log_t0)
    s = math.exp(log_s)
    w = expit((t_s - t0) / max(s, EPS))
    return (1.0 - w) * k_sqrt * math.sqrt(t_s) + w * k_quarter * (t_s ** 0.25)


def row_key(dataset: str, folder: str, file_stem: object, plume_idx: object) -> tuple[str, str, str, int | None]:
    return dataset, folder, str(file_stem), parse_int(plume_idx)


def read_meta(meta_path: Path) -> dict[str, object]:
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def iter_dataset_dirs(root: Path, datasets: list[str] | None) -> list[Path]:
    selected = set(datasets or [])
    return [path for path in sorted(root.iterdir()) if path.is_dir() and (not selected or path.name in selected)]


def plume_columns(fieldnames: list[str], prefix: str) -> list[tuple[int, str]]:
    cols: list[tuple[int, str]] = []
    pattern = re.compile(re.escape(prefix) + r"(\d+)$")
    for col in fieldnames:
        match = pattern.match(col)
        if match:
            cols.append((int(match.group(1)), col))
    return sorted(cols)


def collect_raw_inventory(
    raw_root: Path,
    *,
    datasets: list[str] | None,
    folders: list[str] | None,
    penetration_prefix: str,
) -> tuple[dict[tuple[str, str, str, int | None], dict[str, object]], dict[tuple[str, str, str, int | None], list[tuple[float, float]]], dict[str, list[float]]]:
    """Scan raw CSVs and return per-plume endpoints, full raw series, and cap samples."""
    raw_metrics: dict[tuple[str, str, str, int | None], dict[str, object]] = {}
    raw_series: dict[tuple[str, str, str, int | None], list[tuple[float, float]]] = defaultdict(list)
    cap_values: dict[str, list[float]] = defaultdict(list)
    folder_filter = set(folders or [])

    for dataset_dir in iter_dataset_dirs(raw_root, datasets):
        dataset = dataset_dir.name
        nozzle = infer_nozzle(dataset)
        for csv_path in sorted(dataset_dir.glob("T*/*.csv")):
            folder = csv_path.parent.name
            if folder_filter and folder not in folder_filter:
                continue
            meta = read_meta(csv_path.with_suffix(".meta.json"))
            fps = parse_float(meta.get("fps"))
            if not math.isfinite(fps) or fps <= 0.0:
                fps = math.nan

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                cols = plume_columns(reader.fieldnames, penetration_prefix)
                states: dict[int, dict[str, object]] = {
                    plume_idx: {
                        "values": [],
                        "max_mm": math.nan,
                        "t_at_max_ms": math.nan,
                        "first_positive_ms": math.nan,
                        "last_positive_ms": math.nan,
                        "last_positive_mm": math.nan,
                        "n_finite": 0,
                        "n_positive": 0,
                    }
                    for plume_idx, _ in cols
                }
                for row in reader:
                    frame_idx = parse_float(row.get("frame_idx"))
                    t_ms = frame_idx / fps * 1e3 if math.isfinite(frame_idx) and math.isfinite(fps) else math.nan
                    for plume_idx, col in cols:
                        value = parse_float(row.get(col))
                        if not math.isfinite(value):
                            continue
                        state = states[plume_idx]
                        state["n_finite"] = int(state["n_finite"]) + 1
                        if value <= 0.0:
                            continue
                        state["n_positive"] = int(state["n_positive"]) + 1
                        state["last_positive_mm"] = value
                        state["last_positive_ms"] = t_ms
                        if not math.isfinite(state["first_positive_ms"]):
                            state["first_positive_ms"] = t_ms
                        state["values"].append(value)
                        cap_values[nozzle].append(value)
                        if not math.isfinite(state["max_mm"]) or value > state["max_mm"]:
                            state["max_mm"] = value
                            state["t_at_max_ms"] = t_ms
                        if math.isfinite(t_ms):
                            raw_series[row_key(dataset, folder, csv_path.stem, plume_idx)].append((t_ms, value))

                for plume_idx, state in states.items():
                    key = row_key(dataset, folder, csv_path.stem, plume_idx)
                    first_ms = state["first_positive_ms"]
                    last_ms = state["last_positive_ms"]
                    raw_metrics[key] = {
                        "dataset": dataset,
                        "nozzle": nozzle,
                        "folder": folder,
                        "file_stem": csv_path.stem,
                        "file_name": csv_path.name,
                        "plume_idx": plume_idx,
                        "raw_fps": fps,
                        "raw_n_finite": state["n_finite"],
                        "raw_n_positive": state["n_positive"],
                        "raw_first_positive_ms": first_ms,
                        "raw_last_positive_ms": last_ms,
                        "raw_positive_span_ms": last_ms - first_ms if math.isfinite(first_ms) and math.isfinite(last_ms) else math.nan,
                        "raw_max_mm": state["max_mm"],
                        "raw_t_at_max_ms": state["t_at_max_ms"],
                        "raw_last_positive_mm": state["last_positive_mm"],
                    }

    return raw_metrics, raw_series, cap_values


def build_nozzle_caps(cap_values: dict[str, list[float]], args: argparse.Namespace) -> dict[str, dict[str, object]]:
    """Estimate one FOV cap per nozzle from high-quantile raw penetration values."""
    caps: dict[str, dict[str, object]] = {}
    for nozzle, values in sorted(cap_values.items()):
        selected = quantile(values, args.cap_quantile)
        tol = max(args.cap_abs_tol_mm, abs(selected) * args.cap_rel_tol) if math.isfinite(selected) else math.nan
        caps[nozzle] = {
            "nozzle": nozzle,
            "n_raw_positive_values": len(finite_values(values)),
            "cap_method": f"q{args.cap_quantile:g}",
            "cap_mm": selected,
            "cap_tolerance_mm": tol,
            "raw_max_mm": quantile(values, 1.0),
            "raw_q990_mm": quantile(values, 0.990),
            "raw_q995_mm": quantile(values, 0.995),
            "raw_q999_mm": quantile(values, 0.999),
            "raw_q9995_mm": quantile(values, 0.9995),
        }
    return caps


def time_penetration_columns(fieldnames: list[str]) -> tuple[list[str], list[str]]:
    time_cols = sorted([col for col in fieldnames if col.startswith("time_ms_")])
    pen_cols = sorted([col for col in fieldnames if col.startswith("penetration_mm_")])
    if len(time_cols) != len(pen_cols):
        raise ValueError("Mismatched time_ms_* and penetration_mm_* columns.")
    return time_cols, pen_cols


def synthetic_series_metrics(row: dict[str, str], time_cols: list[str], pen_cols: list[str]) -> dict[str, float]:
    """Summarize one cleaned wide-series row into endpoints and maxima."""
    pairs: list[tuple[float, float]] = []
    for time_col, pen_col in zip(time_cols, pen_cols):
        t_ms = parse_float(row.get(time_col))
        y_mm = parse_float(row.get(pen_col))
        if math.isfinite(t_ms) and math.isfinite(y_mm):
            pairs.append((t_ms, y_mm))
    if not pairs:
        return {
            "clean_t_first_ms": math.nan,
            "clean_t_last_ms": math.nan,
            "clean_last_mm": math.nan,
            "clean_max_mm": math.nan,
            "clean_t_at_max_ms": math.nan,
            "clean_n": 0.0,
        }
    pairs.sort(key=lambda item: item[0])
    max_t, max_y = max(pairs, key=lambda item: item[1])
    return {
        "clean_t_first_ms": pairs[0][0],
        "clean_t_last_ms": pairs[-1][0],
        "clean_last_mm": pairs[-1][1],
        "clean_max_mm": max_y,
        "clean_t_at_max_ms": max_t,
        "clean_n": float(len(pairs)),
    }


def collect_synthetic_series(
    synthetic_root: Path,
    *,
    datasets: list[str] | None,
    folders: list[str] | None,
) -> dict[tuple[str, str, str, int | None], dict[str, float]]:
    """Build a lookup from plume key to cleaned CDF series summary metrics."""
    lookup: dict[tuple[str, str, str, int | None], dict[str, float]] = {}
    folder_filter = set(folders or [])
    for dataset_dir in iter_dataset_dirs(synthetic_root, datasets):
        wide_dir = dataset_dir / "cdf" / "series_wide_all"
        if not wide_dir.exists():
            continue
        for wide_path in sorted(wide_dir.glob("*.csv")):
            folder = wide_path.stem
            if folder_filter and folder not in folder_filter:
                continue
            with wide_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                time_cols, pen_cols = time_penetration_columns(reader.fieldnames)
                for row in reader:
                    key = row_key(dataset_dir.name, folder, row.get("file_stem"), row.get("plume_idx"))
                    lookup[key] = synthetic_series_metrics(row, time_cols, pen_cols)
    return lookup


def first_cap_hit(series: list[tuple[float, float]], cap_mm: float, tol_mm: float) -> tuple[float, float]:
    """Return the first raw point that enters the cap tolerance band."""
    threshold = cap_mm - tol_mm
    for t_ms, value in sorted(series):
        if value >= threshold:
            return t_ms, value
    return math.nan, math.nan


def collect_comparison_rows(
    synthetic_root: Path,
    raw_metrics: dict[tuple[str, str, str, int | None], dict[str, object]],
    raw_series: dict[tuple[str, str, str, int | None], list[tuple[float, float]]],
    synthetic_series: dict[tuple[str, str, str, int | None], dict[str, float]],
    caps: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Join raw, cleaned, and fit views into the final per-plume audit table."""
    rows: list[dict[str, object]] = []
    folder_filter = set(args.folders or [])

    for dataset_dir in iter_dataset_dirs(synthetic_root, args.datasets):
        fit_dir = dataset_dir / "cdf" / "all"
        if not fit_dir.exists():
            continue
        dataset = dataset_dir.name
        nozzle = infer_nozzle(dataset)
        cap = caps.get(nozzle)
        if not cap:
            continue
        cap_mm = float(cap["cap_mm"])
        tol_mm = float(cap["cap_tolerance_mm"])

        for fit_path in sorted(fit_dir.glob("*.csv")):
            if fit_path.name.endswith("_flagged.csv"):
                continue
            folder = fit_path.stem
            if folder_filter and folder not in folder_filter:
                continue
            with fit_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for fit_row in reader:
                    key = row_key(dataset, folder, fit_row.get("file_stem"), fit_row.get("plume_idx"))
                    raw = raw_metrics.get(key)
                    clean = synthetic_series.get(key)
                    if raw is None or clean is None:
                        continue

                    hit_t_ms, hit_value_mm = first_cap_hit(raw_series.get(key, []), cap_mm, tol_mm)
                    raw_max = float(raw["raw_max_mm"])
                    clean_max = clean["clean_max_mm"]
                    clean_last = clean["clean_last_mm"]
                    # The censoring label is intentionally conservative: a cap
                    # hit in either the raw or cleaned trace is enough to flag
                    # the plume as spatially truncated.
                    raw_near_cap = math.isfinite(raw_max) and raw_max >= cap_mm - tol_mm
                    clean_near_cap = (
                        math.isfinite(clean_max)
                        and clean_max >= cap_mm - tol_mm
                        or math.isfinite(clean_last)
                        and clean_last >= cap_mm - tol_mm
                    )
                    spatial_censored = bool(raw_near_cap or clean_near_cap)
                    raw_span = float(raw["raw_positive_span_ms"])
                    hit_fraction = (
                        (hit_t_ms - float(raw["raw_first_positive_ms"])) / raw_span
                        if spatial_censored and math.isfinite(hit_t_ms) and math.isfinite(raw_span) and raw_span > 0.0
                        else math.nan
                    )
                    fit_horizon = sigmoid_fit_mm(fit_row, args.horizon_ms)
                    fit_minus_clean_last = fit_horizon - clean_last if math.isfinite(fit_horizon) and math.isfinite(clean_last) else math.nan
                    fit_minus_cap = fit_horizon - cap_mm if math.isfinite(fit_horizon) and math.isfinite(cap_mm) else math.nan

                    rows.append(
                        {
                            "dataset": dataset,
                            "nozzle": nozzle,
                            "folder": folder,
                            "file_stem": fit_row.get("file_stem", ""),
                            "file_name": fit_row.get("file_name", ""),
                            "plume_idx": fit_row.get("plume_idx", ""),
                            "sample_split": "flagged" if parse_bool(fit_row.get("flag_bad_fit")) else "clean",
                            "success": parse_bool(fit_row.get("success")),
                            "cap_mm": cap_mm,
                            "cap_tolerance_mm": tol_mm,
                            "raw_max_mm": raw_max,
                            "raw_gap_to_cap_mm": cap_mm - raw_max if math.isfinite(raw_max) else math.nan,
                            "raw_last_positive_mm": raw["raw_last_positive_mm"],
                            "raw_first_positive_ms": raw["raw_first_positive_ms"],
                            "raw_last_positive_ms": raw["raw_last_positive_ms"],
                            "raw_positive_span_ms": raw_span,
                            "first_cap_hit_time_ms": hit_t_ms,
                            "first_cap_hit_value_mm": hit_value_mm,
                            "cap_hit_fraction_of_raw_span": hit_fraction,
                            "clean_t_first_ms": clean["clean_t_first_ms"],
                            "clean_t_last_ms": clean["clean_t_last_ms"],
                            "clean_last_mm": clean_last,
                            "clean_max_mm": clean_max,
                            "clean_gap_to_cap_mm": cap_mm - clean_max if math.isfinite(clean_max) else math.nan,
                            "raw_near_cap": raw_near_cap,
                            "clean_near_cap": clean_near_cap,
                            "is_spatial_right_censored": spatial_censored,
                            "fit_at_horizon_mm": fit_horizon,
                            "fit_minus_clean_last_mm": fit_minus_clean_last,
                            "fit_minus_cap_mm": fit_minus_cap,
                            "naive_hold_underestimate_if_censored_mm": (
                                max(0.0, fit_minus_clean_last)
                                if spatial_censored and math.isfinite(fit_minus_clean_last)
                                else math.nan
                            ),
                            "cap_saturation_underestimate_if_censored_mm": (
                                max(0.0, fit_minus_cap)
                                if spatial_censored and math.isfinite(fit_minus_cap)
                                else math.nan
                            ),
                            "fit_n": parse_float(fit_row.get("n")),
                            "fit_rmse_mm": parse_float(fit_row.get("rmse")),
                            "chamber_pressure_bar": parse_float(fit_row.get("chamber_pressure_bar")),
                            "injection_duration_us": parse_float(fit_row.get("injection_duration_us")),
                            "injection_pressure_bar": parse_float(fit_row.get("injection_pressure_bar")),
                            "control_backpressure_bar": parse_float(fit_row.get("control_backpressure_bar")),
                        }
                    )
    if not rows:
        raise FileNotFoundError("No comparable raw/synthetic CDF rows were found.")
    return rows


def summarize_group(rows: list[dict[str, object]], labels: dict[str, object]) -> dict[str, object]:
    n = len(rows)
    censored = [row for row in rows if row["is_spatial_right_censored"]]
    return {
        **labels,
        "n_trajectories": n,
        "n_spatial_right_censored": len(censored),
        "spatial_right_censored_fraction": len(censored) / n if n else math.nan,
        "cap_mm_median": safe_median([row["cap_mm"] for row in rows]),
        "raw_max_mm_median": safe_median([row["raw_max_mm"] for row in rows]),
        "raw_gap_to_cap_mm_median": safe_median([row["raw_gap_to_cap_mm"] for row in rows]),
        "raw_positive_span_ms_median": safe_median([row["raw_positive_span_ms"] for row in rows]),
        "first_cap_hit_time_ms_median": safe_median([row["first_cap_hit_time_ms"] for row in censored]),
        "cap_hit_fraction_of_raw_span_median": safe_median([row["cap_hit_fraction_of_raw_span"] for row in censored]),
        "clean_t_last_ms_median": safe_median([row["clean_t_last_ms"] for row in rows]),
        "clean_max_mm_median": safe_median([row["clean_max_mm"] for row in rows]),
        "clean_gap_to_cap_mm_median": safe_median([row["clean_gap_to_cap_mm"] for row in rows]),
        "fit_minus_clean_last_censored_mm_mean": safe_mean([row["naive_hold_underestimate_if_censored_mm"] for row in censored]),
        "fit_minus_clean_last_censored_mm_median": safe_median([row["naive_hold_underestimate_if_censored_mm"] for row in censored]),
        "fit_minus_cap_censored_mm_mean": safe_mean([row["cap_saturation_underestimate_if_censored_mm"] for row in censored]),
        "fit_minus_cap_censored_mm_median": safe_median([row["cap_saturation_underestimate_if_censored_mm"] for row in censored]),
    }


def grouped_summary(rows: list[dict[str, object]], group_cols: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[col] for col in group_cols)].append(row)
    out: list[dict[str, object]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        out.append(summarize_group(group_rows, {col: value for col, value in zip(group_cols, key)}))
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def save_plots(out_dir: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: matplotlib is unavailable ({exc}).")
        return

    clean_rows = [row for row in rows if row["sample_split"] == "clean"]
    by_nozzle: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in clean_rows:
        by_nozzle[str(row["nozzle"])].append(row)

    labels = sorted(by_nozzle)
    fractions = [
        sum(1 for row in by_nozzle[label] if row["is_spatial_right_censored"]) / len(by_nozzle[label])
        for label in labels
    ]
    plt.figure(figsize=(7.2, 3.6))
    plt.bar(labels, fractions, color="#4C78A8")
    plt.ylabel("spatial right-censored fraction")
    plt.ylim(0.0, 1.02)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "spatial_censored_fraction_by_nozzle_clean.png", dpi=200)
    plt.close()

    under = finite_values(
        [
            row["naive_hold_underestimate_if_censored_mm"]
            for row in clean_rows
            if row["is_spatial_right_censored"]
        ]
    )
    if under:
        plt.figure(figsize=(6.0, 3.6))
        plt.hist(under, bins=40, color="#59A14F", edgecolor="white")
        plt.xlabel("fit horizon - clean last observation (mm)")
        plt.ylabel("clean censored plume count")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "spatial_censored_naive_underestimate_clean_hist.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_metrics, raw_series, cap_values = collect_raw_inventory(
        args.raw_root.resolve(),
        datasets=args.datasets,
        folders=args.folders,
        penetration_prefix=args.penetration_prefix,
    )
    caps = build_nozzle_caps(cap_values, args)
    synthetic_series = collect_synthetic_series(
        args.synthetic_root.resolve(),
        datasets=args.datasets,
        folders=args.folders,
    )
    comparison_rows = collect_comparison_rows(
        args.synthetic_root.resolve(),
        raw_metrics,
        raw_series,
        synthetic_series,
        caps,
        args,
    )

    overall = [summarize_group(comparison_rows, {"scope": "overall"})]
    by_split = grouped_summary(comparison_rows, ["sample_split"])
    by_nozzle = grouped_summary(comparison_rows, ["nozzle", "sample_split"])
    by_condition = grouped_summary(
        comparison_rows,
        [
            "nozzle",
            "chamber_pressure_bar",
            "injection_pressure_bar",
            "injection_duration_us",
            "control_backpressure_bar",
            "sample_split",
        ],
    )

    write_csv(out_dir / "nozzle_spatial_limits.csv", list(caps.values()))
    write_csv(out_dir / "plume_spatial_censoring_audit.csv", comparison_rows)
    write_csv(out_dir / "spatial_censoring_summary_overall.csv", overall + by_split)
    write_csv(out_dir / "spatial_censoring_summary_by_nozzle.csv", by_nozzle)
    write_csv(out_dir / "spatial_censoring_summary_by_condition.csv", by_condition)

    if args.plots:
        save_plots(out_dir, comparison_rows)

    clean_summary = next((row for row in by_split if row["sample_split"] == "clean"), None)
    flagged_summary = next((row for row in by_split if row["sample_split"] == "flagged"), None)
    summary = {
        "raw_root": str(args.raw_root.resolve()),
        "synthetic_root": str(args.synthetic_root.resolve()),
        "out_dir": str(out_dir),
        "penetration_prefix": args.penetration_prefix,
        "cap_quantile": args.cap_quantile,
        "cap_abs_tol_mm": args.cap_abs_tol_mm,
        "cap_rel_tol": args.cap_rel_tol,
        "horizon_ms_for_bias_only": args.horizon_ms,
        "overall": overall[0],
        "clean": clean_summary,
        "flagged": flagged_summary,
        "outputs": [
            "nozzle_spatial_limits.csv",
            "plume_spatial_censoring_audit.csv",
            "spatial_censoring_summary_overall.csv",
            "spatial_censoring_summary_by_nozzle.csv",
            "spatial_censoring_summary_by_condition.csv",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    headline = overall[0]
    print(f"Wrote spatial CDF censoring audit to: {out_dir}")
    print(
        "Overall: "
        f"{headline['n_spatial_right_censored']}/{headline['n_trajectories']} spatially censored "
        f"({100.0 * headline['spatial_right_censored_fraction']:.1f}%), "
        f"median first cap hit {headline['first_cap_hit_time_ms_median']:.3f} ms."
    )
    if clean_summary:
        print(
            "Clean fits: "
            f"{clean_summary['n_spatial_right_censored']}/{clean_summary['n_trajectories']} spatially censored "
            f"({100.0 * clean_summary['spatial_right_censored_fraction']:.1f}%), "
            f"median fit-last underestimation {clean_summary['fit_minus_clean_last_censored_mm_median']:.2f} mm."
        )
    if flagged_summary:
        print(
            "Flagged fits: "
            f"{flagged_summary['n_spatial_right_censored']}/{flagged_summary['n_trajectories']} spatially censored "
            f"({100.0 * flagged_summary['spatial_right_censored_fraction']:.1f}%)."
        )


if __name__ == "__main__":
    main()
