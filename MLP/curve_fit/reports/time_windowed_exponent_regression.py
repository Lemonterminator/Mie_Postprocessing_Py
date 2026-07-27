"""Traceable time-windowed log-log scaling regressions for CDF penetration.

The primary analysis uses the Level-3 QC-gated archive explicitly.  It compares
    all censoring-inclusive observations with the condition-level
    censor-truncated point table
and repeats the same protocols on the Level-2 archive as a lineage-robustness
check.  No input is read through the ``MLP/synthetic_data`` junction.

For each time-window centre, the fitted model is

    log S = a log(delta P) + b log(P_ch,phys) + c log(d_n) + constant.

The point-level OLS estimate is accompanied by condition-cluster bootstrap
intervals, a condition-median sensitivity fit, and leave-one-nozzle-out
coefficient ranges.  The output manifest records resolved input paths, hashes,
row counts, effective positive-penetration counts, condition counts, and all
regression settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.MLP_training.engineered_feature_common import (  # noqa: E402
    build_dataset_registry,
    canonicalize_chamber_state,
    normalize_dataset_key,
)


DEFAULT_PRIMARY_ROOT = PROJECT_ROOT / "MLP" / "synthetic_data_clean_lv3_qc_gated"
DEFAULT_ROBUSTNESS_ROOT = PROJECT_ROOT / "MLP" / "synthetic_data_clean_lv2"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PRIMARY_ROOT / "fit_diagnostics" / "time_windowed_exponent"
)
DEFAULT_THESIS_FIGURE = (
    PROJECT_ROOT
    / "Thesis"
    / "images"
    / "time_windowed_exponents_protocol_comparison.png"
)

PROTOCOL_FILES = {
    "censoring-inclusive": "cdf_points_all.csv",
    "censor-truncated": "cdf_points_uncensored.csv",
}
READ_COLUMNS = [
    "condition_key",
    "experiment_name",
    "time_ms",
    "penetration_mm",
    "diameter_mm",
    "chamber_pressure_bar",
    "injection_pressure_bar",
]
COEFFICIENT_KEYS = ("exp_delta_p", "exp_ambient_pressure", "exp_diameter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=DEFAULT_PRIMARY_ROOT,
        help="Explicit primary synthetic-data root (default: Level-3 QC-gated).",
    )
    parser.add_argument(
        "--robustness-root",
        type=Path,
        default=DEFAULT_ROBUSTNESS_ROOT,
        help="Explicit secondary lineage root used only for robustness.",
    )
    parser.add_argument(
        "--protocol",
        choices=("both", "censoring-inclusive", "censor-truncated"),
        default="both",
        help="Input protocol(s) to analyse from each explicit lineage root.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--thesis-figure",
        type=Path,
        default=DEFAULT_THESIS_FIGURE,
        help="Copy the primary protocol-comparison figure to this path.",
    )
    parser.add_argument("--time-start-ms", type=float, default=0.20)
    parser.add_argument("--time-stop-ms", type=float, default=1.10)
    parser.add_argument("--time-step-ms", type=float, default=0.10)
    parser.add_argument("--half-width-ms", type=float, default=0.05)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument(
        "--fixed-cohort-reference-ms",
        type=float,
        default=1.0,
        help="Reference window used to define the changing-support sensitivity cohort.",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_state_lookup(
    experiments: pd.Series,
    chamber_raw: pd.Series,
    registry: dict[str, Any],
    cache: dict[tuple[str, float], tuple[float, float, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = pd.DataFrame(
        {
            "experiment_name": experiments.astype(str),
            "chamber_raw": pd.to_numeric(chamber_raw, errors="coerce"),
        }
    )
    for experiment_name, raw_value in pairs.drop_duplicates().itertuples(index=False):
        if not np.isfinite(raw_value):
            continue
        key = (str(experiment_name), float(raw_value))
        if key in cache:
            continue
        dataset_key = normalize_dataset_key(str(experiment_name))
        cache[key] = canonicalize_chamber_state(
            dataset_key, float(raw_value), registry
        )
    pressure = np.array(
        [
            cache.get((str(name), float(raw)), (np.nan, np.nan, "invalid"))[0]
            if np.isfinite(raw)
            else np.nan
            for name, raw in pairs.itertuples(index=False)
        ],
        dtype=float,
    )
    density = np.array(
        [
            cache.get((str(name), float(raw)), (np.nan, np.nan, "invalid"))[1]
            if np.isfinite(raw)
            else np.nan
            for name, raw in pairs.itertuples(index=False)
        ],
        dtype=float,
    )
    mode = np.array(
        [
            cache.get((str(name), float(raw)), (np.nan, np.nan, "invalid"))[2]
            if np.isfinite(raw)
            else "invalid"
            for name, raw in pairs.itertuples(index=False)
        ],
        dtype=object,
    )
    return pressure, density, mode


def read_point_table(
    path: Path,
    *,
    chunksize: int,
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Point table not found: {path}")

    frames: list[pd.DataFrame] = []
    condition_keys_all: set[str] = set()
    condition_keys_positive: set[str] = set()
    condition_keys_effective: set[str] = set()
    n_rows_total = 0
    n_rows_positive_penetration = 0
    state_cache: dict[tuple[str, float], tuple[float, float, str]] = {}
    canonicalization_mode_counts: dict[str, int] = {}

    for chunk in pd.read_csv(path, usecols=READ_COLUMNS, chunksize=chunksize):
        n_rows_total += len(chunk)
        chunk["condition_key"] = chunk["condition_key"].astype(str)
        condition_keys_all.update(chunk["condition_key"].unique().tolist())

        for col in (
            "time_ms",
            "penetration_mm",
            "diameter_mm",
            "chamber_pressure_bar",
            "injection_pressure_bar",
        ):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        positive_penetration = np.isfinite(chunk["penetration_mm"]) & (
            chunk["penetration_mm"] > 0.0
        )
        n_rows_positive_penetration += int(positive_penetration.sum())
        condition_keys_positive.update(
            chunk.loc[positive_penetration, "condition_key"].unique().tolist()
        )

        ambient_pressure, ambient_density, canonicalization_modes = canonical_state_lookup(
            chunk["experiment_name"],
            chunk["chamber_pressure_bar"],
            registry,
            state_cache,
        )
        for mode, count in pd.Series(canonicalization_modes).value_counts().items():
            canonicalization_mode_counts[str(mode)] = (
                canonicalization_mode_counts.get(str(mode), 0) + int(count)
            )
        chunk["ambient_pressure_bar_phys"] = ambient_pressure
        chunk["ambient_density_kg_m3"] = ambient_density
        chunk["delta_pressure_bar_phys"] = (
            chunk["injection_pressure_bar"] - chunk["ambient_pressure_bar_phys"]
        )
        chunk["nozzle"] = chunk["experiment_name"].map(normalize_dataset_key)

        finite = np.isfinite(
            chunk[
                [
                    "time_ms",
                    "penetration_mm",
                    "diameter_mm",
                    "ambient_pressure_bar_phys",
                    "ambient_density_kg_m3",
                    "delta_pressure_bar_phys",
                ]
            ]
        ).all(axis=1)
        valid = (
            finite
            & (chunk["time_ms"] > 0.0)
            & (chunk["penetration_mm"] > 0.0)
            & (chunk["diameter_mm"] > 0.0)
            & (chunk["ambient_pressure_bar_phys"] > 0.0)
            & (chunk["ambient_density_kg_m3"] > 0.0)
            & (chunk["delta_pressure_bar_phys"] > 0.0)
        )
        retained = chunk.loc[
            valid,
            [
                "condition_key",
                "nozzle",
                "time_ms",
                "penetration_mm",
                "delta_pressure_bar_phys",
                "ambient_pressure_bar_phys",
                "ambient_density_kg_m3",
                "diameter_mm",
            ],
        ].rename(columns={"penetration_mm": "S_mm"})
        condition_keys_effective.update(retained["condition_key"].unique().tolist())
        frames.append(retained)

    long_df = pd.concat(frames, ignore_index=True)
    metadata = {
        "resolved_path": str(path.resolve()),
        "sha256": sha256_file(path),
        "n_rows_total": int(n_rows_total),
        "n_rows_positive_penetration": int(n_rows_positive_penetration),
        "n_rows_log_ols_effective": int(len(long_df)),
        "n_conditions_total": int(len(condition_keys_all)),
        "n_conditions_positive_penetration": int(len(condition_keys_positive)),
        "n_conditions_log_ols_effective": int(len(condition_keys_effective)),
        "canonicalization_mode_counts": canonicalization_mode_counts,
    }
    return long_df, metadata


def design_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack(
        [
            np.log(frame["delta_pressure_bar_phys"].to_numpy(dtype=float)),
            np.log(frame["ambient_pressure_bar_phys"].to_numpy(dtype=float)),
            np.log(frame["diameter_mm"].to_numpy(dtype=float)),
            np.ones(len(frame), dtype=float),
        ]
    )
    y = np.log(frame["S_mm"].to_numpy(dtype=float))
    return x, y


def solve_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coef
    ss_res = float(residual @ residual)
    centered = y - y.mean()
    ss_tot = float(centered @ centered)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    sigma2 = ss_res / max(len(y) - x.shape[1], 1)
    try:
        covariance = sigma2 * np.linalg.pinv(x.T @ x)
        standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    except np.linalg.LinAlgError:
        standard_error = np.full(x.shape[1], np.nan)
    return coef, standard_error, r2


def condition_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    reps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    x, y = design_matrix(frame)
    condition_codes, conditions = pd.factorize(frame["condition_key"], sort=True)
    n_conditions = len(conditions)
    p = x.shape[1]

    group_xx = np.zeros((n_conditions, p, p), dtype=float)
    group_xy = np.zeros((n_conditions, p), dtype=float)
    np.add.at(group_xx, condition_codes, np.einsum("ni,nj->nij", x, x))
    np.add.at(group_xy, condition_codes, x * y[:, None])

    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    for _ in range(reps):
        sampled = rng.integers(0, n_conditions, size=n_conditions)
        xx = group_xx[sampled].sum(axis=0)
        xy = group_xy[sampled].sum(axis=0)
        try:
            estimates.append(np.linalg.solve(xx, xy))
        except np.linalg.LinAlgError:
            continue
    if not estimates:
        return np.full(p, np.nan), np.full(p, np.nan), 0
    boot = np.vstack(estimates)
    return (
        np.percentile(boot, 2.5, axis=0),
        np.percentile(boot, 97.5, axis=0),
        int(len(boot)),
    )


def condition_median_fit(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    condition_frame = (
        frame.groupby("condition_key", sort=False, as_index=False)
        .agg(
            S_mm=("S_mm", "median"),
            delta_pressure_bar_phys=("delta_pressure_bar_phys", "first"),
            ambient_pressure_bar_phys=("ambient_pressure_bar_phys", "first"),
            diameter_mm=("diameter_mm", "first"),
        )
    )
    x, y = design_matrix(condition_frame)
    coef, _, r2 = solve_ols(x, y)
    return coef, r2


def leave_one_nozzle_out_range(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:
    estimates: list[np.ndarray] = []
    nozzles = sorted(frame["nozzle"].unique())
    for nozzle in nozzles:
        subset = frame.loc[frame["nozzle"] != nozzle]
        if len(subset) < 50:
            continue
        x, y = design_matrix(subset)
        coef, _, _ = solve_ols(x, y)
        estimates.append(coef)
    if not estimates:
        return np.full(4, np.nan), np.full(4, np.nan), 0
    values = np.vstack(estimates)
    return values.min(axis=0), values.max(axis=0), int(len(values))


def regress_per_window(
    long_df: pd.DataFrame,
    *,
    centers_ms: np.ndarray,
    half_width_ms: float,
    bootstrap_reps: int,
    bootstrap_seed: int,
    fixed_cohort_reference_ms: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference_lo = round(fixed_cohort_reference_ms - half_width_ms, 10)
    reference_hi = round(fixed_cohort_reference_ms + half_width_ms, 10)
    fixed_cohort = set(
        long_df.loc[
            (long_df["time_ms"] >= reference_lo)
            & (long_df["time_ms"] < reference_hi),
            "condition_key",
        ].unique()
    )
    for bin_index, center in enumerate(centers_ms):
        window_start = round(float(center - half_width_ms), 10)
        window_end = round(float(center + half_width_ms), 10)
        subset = long_df.loc[
            (long_df["time_ms"] >= window_start)
            & (long_df["time_ms"] < window_end)
        ]
        if len(subset) < 50 or subset["condition_key"].nunique() < 5:
            rows.append(
                {
                    "t_center_ms": float(center),
                    "window_start_ms": window_start,
                    "window_end_ms": window_end,
                    "n": int(len(subset)),
                    "n_conditions": int(subset["condition_key"].nunique()),
                    "n_nozzles": int(subset["nozzle"].nunique()),
                }
            )
            continue

        x, y = design_matrix(subset)
        coef, standard_error, r2 = solve_ols(x, y)
        ci_low, ci_high, bootstrap_success = condition_cluster_bootstrap(
            subset,
            reps=bootstrap_reps,
            seed=bootstrap_seed + bin_index,
        )
        median_coef, median_r2 = condition_median_fit(subset)
        lono_min, lono_max, n_lono = leave_one_nozzle_out_range(subset)
        fixed_subset = subset.loc[subset["condition_key"].isin(fixed_cohort)]
        fixed_coef = np.full(4, np.nan)
        fixed_r2 = np.nan
        if len(fixed_subset) >= 50 and fixed_subset["condition_key"].nunique() >= 5:
            fixed_x, fixed_y = design_matrix(fixed_subset)
            fixed_coef, _, fixed_r2 = solve_ols(fixed_x, fixed_y)

        row: dict[str, Any] = {
            "t_center_ms": float(center),
            "window_start_ms": window_start,
            "window_end_ms": window_end,
            "n": int(len(subset)),
            "n_conditions": int(subset["condition_key"].nunique()),
            "n_nozzles": int(subset["nozzle"].nunique()),
            "exp_delta_p": float(coef[0]),
            "se_delta_p_naive": float(standard_error[0]),
            "exp_ambient_pressure": float(coef[1]),
            "se_ambient_pressure_naive": float(standard_error[1]),
            "exp_diameter": float(coef[2]),
            "se_diameter_naive": float(standard_error[2]),
            "intercept": float(coef[3]),
            "r2": float(r2),
            "bootstrap_success": bootstrap_success,
            "condition_median_r2": float(median_r2),
            "n_lono_fits": n_lono,
            "fixed_cohort_reference_ms": float(fixed_cohort_reference_ms),
            "fixed_cohort_n": int(len(fixed_subset)),
            "fixed_cohort_n_conditions": int(
                fixed_subset["condition_key"].nunique()
            ),
            "fixed_cohort_r2": float(fixed_r2),
        }
        for index, key in enumerate(COEFFICIENT_KEYS):
            row[f"{key}_cluster_ci95_low"] = float(ci_low[index])
            row[f"{key}_cluster_ci95_high"] = float(ci_high[index])
            row[f"{key}_condition_median"] = float(median_coef[index])
            row[f"{key}_lono_min"] = float(lono_min[index])
            row[f"{key}_lono_max"] = float(lono_max[index])
            row[f"{key}_fixed_cohort"] = float(fixed_coef[index])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_primary_protocols(
    clean_wide: pd.DataFrame,
    censor_truncated: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), sharex=True)
    protocol_frames = [
        ("Censor-truncated", censor_truncated, "#0066A6", "o"),
        ("All observations (censoring-inclusive)", clean_wide, "#D95F02", "s"),
    ]
    coefficient_panels = [
        (
            "exp_delta_p",
            r"$\partial\log S/\partial\log\Delta P$",
            [(0.50, "Bernoulli reference", "--"), (0.25, "H--A reference", ":")],
        ),
        (
            "exp_ambient_pressure",
            r"$\partial\log S/\partial\log P_{\mathrm{ch}}$",
            [(-0.25, "H--A ambient reference", ":")],
        ),
    ]

    for axis, (key, ylabel, references) in zip(axes[:2], coefficient_panels):
        for label, frame, color, marker in protocol_frames:
            t = frame["t_center_ms"].to_numpy(dtype=float)
            y = frame[key].to_numpy(dtype=float)
            low = frame[f"{key}_cluster_ci95_low"].to_numpy(dtype=float)
            high = frame[f"{key}_cluster_ci95_high"].to_numpy(dtype=float)
            axis.fill_between(t, low, high, color=color, alpha=0.14)
            axis.plot(t, y, marker=marker, color=color, lw=1.8, ms=4, label=label)
        for reference, label, style in references:
            axis.axhline(reference, color="0.35", ls=style, lw=1.1, label=label)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, framealpha=0.9)

    for label, frame, color, marker in protocol_frames:
        axes[2].plot(
            frame["t_center_ms"],
            frame["r2"],
            marker=marker,
            color=color,
            lw=1.8,
            ms=4,
            label=label,
        )
    axes[2].set_ylabel(r"Point-level $R^2$")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8, framealpha=0.9)

    for axis in axes:
        axis.set_xlabel("Window centre [ms]")
    axes[0].set_title(r"Effective $\Delta P$ exponent")
    axes[1].set_title(r"Ambient-pressure partial exponent")
    axes[2].set_title("Regression support")
    fig.suptitle(
        "Time-resolved empirical pressure scaling in the Level-3 QC-gated archive\n"
        "(point-level OLS; shaded bands are condition-cluster bootstrap 95% intervals)",
        y=1.03,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def requested_protocols(value: str) -> list[str]:
    if value == "both":
        return ["censor-truncated", "censoring-inclusive"]
    return [value]


def main() -> None:
    args = parse_args()
    primary_root = args.primary_root.resolve()
    robustness_root = args.robustness_root.resolve()
    output_dir = args.output_dir.resolve()
    thesis_figure = args.thesis_figure.resolve()
    protocols = requested_protocols(args.protocol)
    centers_ms = np.round(
        np.arange(
            args.time_start_ms,
            args.time_stop_ms + args.time_step_ms / 2.0,
            args.time_step_ms,
        ),
        decimals=10,
    )

    if primary_root == robustness_root:
        raise ValueError("Primary and robustness roots resolve to the same directory.")
    junction = (PROJECT_ROOT / "MLP" / "synthetic_data").resolve()
    if args.primary_root.absolute() == (PROJECT_ROOT / "MLP" / "synthetic_data").absolute():
        raise ValueError("The primary root must not be the MLP/synthetic_data junction.")
    if primary_root != DEFAULT_PRIMARY_ROOT.resolve():
        print(f"WARNING: non-default primary lineage selected: {primary_root}")
    if primary_root == junction:
        print(
            "NOTE: the explicit primary path resolves to the same target as the legacy "
            "junction, but the junction itself was not used."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    registry = build_dataset_registry()
    all_results: list[pd.DataFrame] = []
    input_manifest: dict[str, Any] = {}
    source_censoring_manifests: dict[str, Any] = {}
    result_frames: dict[tuple[str, str], pd.DataFrame] = {}

    lineages = [
        ("lv3_qc_gated", primary_root, "primary"),
        ("lv2", robustness_root, "robustness"),
    ]
    for lineage_label, root, role in lineages:
        upstream_manifest_path = root / "cdf_right_censoring_points" / "manifest.json"
        upstream_payload = json.loads(upstream_manifest_path.read_text(encoding="utf-8"))
        source_censoring_manifests[lineage_label] = {
            "resolved_path": str(upstream_manifest_path.resolve()),
            "sha256": sha256_file(upstream_manifest_path),
            "n_points_all": upstream_payload.get("n_points_all"),
            "n_points_uncensored": upstream_payload.get("n_points_uncensored"),
            "n_conditions_source": upstream_payload.get("n_conditions"),
            "estimate_fov_cap": upstream_payload.get("estimate_fov_cap"),
            "fov_cap_mm": upstream_payload.get("fov_cap_mm"),
            "fov_cap_fraction": upstream_payload.get("fov_cap_fraction"),
            "density_ratio": upstream_payload.get("density_ratio"),
            "density_min_count": upstream_payload.get("density_min_count"),
            "density_consecutive_bins": upstream_payload.get(
                "density_consecutive_bins"
            ),
            "density_smooth_window": upstream_payload.get("density_smooth_window"),
        }
        for protocol in protocols:
            input_path = (
                root / "cdf_right_censoring_points" / PROTOCOL_FILES[protocol]
            )
            print(f"Loading {lineage_label}/{protocol}: {input_path}")
            long_df, metadata = read_point_table(
                input_path,
                chunksize=args.chunksize,
                registry=registry,
            )
            result = regress_per_window(
                long_df,
                centers_ms=centers_ms,
                half_width_ms=args.half_width_ms,
                bootstrap_reps=args.bootstrap_reps,
                bootstrap_seed=args.bootstrap_seed
                + (0 if lineage_label == "lv3_qc_gated" else 10_000)
                + (0 if protocol == "censor-truncated" else 20_000),
                fixed_cohort_reference_ms=args.fixed_cohort_reference_ms,
            )
            result.insert(0, "protocol", protocol)
            result.insert(0, "lineage_role", role)
            result.insert(0, "lineage", lineage_label)
            result_frames[(lineage_label, protocol)] = result
            all_results.append(result)

            csv_name = f"{lineage_label}_{protocol.replace('-', '_')}.csv"
            result.to_csv(output_dir / csv_name, index=False)
            metadata.update(
                {
                    "lineage_role": role,
                    "protocol": protocol,
                    "output_csv": str((output_dir / csv_name).resolve()),
                    "n_rows_in_analysis_horizon": int(
                        long_df.loc[
                            (
                                long_df["time_ms"]
                                >= round(
                                    args.time_start_ms - args.half_width_ms, 10
                                )
                            )
                            & (
                                long_df["time_ms"]
                                < round(args.time_stop_ms + args.half_width_ms, 10)
                            )
                        ].shape[0]
                    ),
                }
            )
            input_manifest[f"{lineage_label}/{protocol}"] = metadata
            print(
                f"  rows={metadata['n_rows_total']:,}; "
                f"S>0={metadata['n_rows_positive_penetration']:,}; "
                f"log-effective={metadata['n_rows_log_ols_effective']:,}; "
                f"conditions={metadata['n_conditions_log_ols_effective']}"
            )
            del long_df

    combined = pd.concat(all_results, ignore_index=True)
    combined_path = output_dir / "lineage_protocol_comparison.csv"
    combined.to_csv(combined_path, index=False)

    figure_path = output_dir / "time_windowed_exponents_protocol_comparison.png"
    figure_generated = False
    thesis_figure_copied = False
    if {
        ("lv3_qc_gated", "censoring-inclusive"),
        ("lv3_qc_gated", "censor-truncated"),
    }.issubset(result_frames):
        plot_primary_protocols(
            result_frames[("lv3_qc_gated", "censoring-inclusive")],
            result_frames[("lv3_qc_gated", "censor-truncated")],
            figure_path,
        )
        figure_generated = True
        thesis_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure_path, thesis_figure)
        thesis_figure_copied = True

    manifest = {
        "script": str(Path(__file__).resolve()),
        "analysis_role": {
            "primary": "lv3_qc_gated",
            "robustness_only": "lv2",
        },
        "primary_root": str(primary_root),
        "robustness_root": str(robustness_root),
        "legacy_junction_used": False,
        "protocols": protocols,
        "regression": {
            "formula": (
                "log(S_mm) ~ log(delta_pressure_bar_phys) + "
                "log(ambient_pressure_bar_phys) + log(diameter_mm)"
            ),
            "time_centers_ms": [float(value) for value in centers_ms],
            "half_width_ms": float(args.half_width_ms),
            "interval": "[center-half_width, center+half_width)",
            "bootstrap_unit": "condition_key",
            "bootstrap_reps": int(args.bootstrap_reps),
            "bootstrap_seed": int(args.bootstrap_seed),
            "condition_median_sensitivity": True,
            "leave_one_nozzle_out_sensitivity": True,
            "fixed_cohort_reference_ms": float(args.fixed_cohort_reference_ms),
        },
        "source_censoring_manifests": source_censoring_manifests,
        "inputs": input_manifest,
        "outputs": {
            "combined_csv": str(combined_path.resolve()),
            "comparison_figure": str(figure_path.resolve())
            if figure_generated
            else None,
            "thesis_figure": str(thesis_figure)
            if thesis_figure_copied
            else None,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote combined results: {combined_path}")
    print(f"Wrote manifest: {manifest_path}")
    if figure_generated:
        print(f"Wrote comparison figure: {figure_path}")
        print(f"Updated thesis figure: {thesis_figure}")


if __name__ == "__main__":
    main()
