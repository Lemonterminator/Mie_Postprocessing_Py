from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from engineered_feature_common import (
    build_all_stage_tables,
    build_dataset_registry,
    evaluate_physical_point_with_derivatives,
    infer_feature_family,
    load_run_artifacts,
)


DEFAULT_TIME_GRID_MS = [0.5, 0.85, 1.0, 2.0, 3.5]
DEFAULT_ANCHORS = [
    {
        "label": "diameter_nozzle_family",
        "axis": "diameter_mm",
        "raw": {
            "dataset_key": "nozzle4",
            "tilt_angle_radian": float(np.deg2rad(20.0)),
            "plumes": 10.0,
            "diameter_mm": 0.355,
            "injection_duration_us": 800.0,
            "injection_pressure_bar": 2000.0,
            "control_backpressure_bar": 4.0,
            "chamber_state_raw": 15.0,
            "chamber_pressure_bar": 15.0,
        },
        "sweep": np.linspace(0.333, 0.384, 180),
        "support_levels": np.array([0.333, 0.348, 0.355, 0.365, 0.375, 0.384], dtype=float),
    },
    {
        "label": "injection_ds300",
        "axis": "injection_pressure_bar",
        "raw": {
            "dataset_key": "ds300",
            "tilt_angle_radian": float(np.deg2rad(20.0)),
            "plumes": 10.0,
            "diameter_mm": 0.384,
            "injection_duration_us": 800.0,
            "injection_pressure_bar": 2200.0,
            "control_backpressure_bar": 1.0,
            "chamber_state_raw": 15.0,
            "chamber_pressure_bar": 15.0,
        },
        "sweep": np.linspace(1400.0, 2200.0, 180),
        "support_levels": np.array([1400.0, 2200.0], dtype=float),
    },
    {
        "label": "injection_nozzle2",
        "axis": "injection_pressure_bar",
        "raw": {
            "dataset_key": "nozzle2",
            "tilt_angle_radian": float(np.deg2rad(20.0)),
            "plumes": 10.0,
            "diameter_mm": 0.375,
            "injection_duration_us": 800.0,
            "injection_pressure_bar": 2000.0,
            "control_backpressure_bar": 4.0,
            "chamber_state_raw": 10.0,
            "chamber_pressure_bar": 10.0,
        },
        "sweep": np.linspace(1400.0, 2000.0, 180),
        "support_levels": np.array([1400.0, 1600.0, 2000.0], dtype=float),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sparse-feature stability across legacy and v2 runs.")
    parser.add_argument("--run", action="append", required=True, help="Run spec in the form label=PATH.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save-dir", type=str, default=None, help="Optional output directory.")
    return parser.parse_args()


def sign_change_count(values: np.ndarray) -> int:
    grad = np.asarray(values, dtype=float)
    return int(np.sum(np.signbit(grad[1:]) != np.signbit(grad[:-1]))) if len(grad) > 1 else 0


def compute_run_metrics(
    run_label: str,
    run_path: str,
    registry: dict[str, object],
    *,
    device: str | None,
) -> pd.DataFrame:
    artifacts = load_run_artifacts(run_path, device=device)
    metrics_rows: list[dict[str, object]] = []

    for anchor in DEFAULT_ANCHORS:
        axis = str(anchor["axis"])
        raw = dict(anchor["raw"])
        sweep = np.asarray(anchor["sweep"], dtype=float)
        support_levels = np.asarray(anchor["support_levels"], dtype=float)
        for time_ms in DEFAULT_TIME_GRID_MS:
            dense_rows = []
            for x_value in sweep:
                raw_x = dict(raw)
                raw_x[axis] = float(x_value)
                dense_rows.append(
                    evaluate_physical_point_with_derivatives(
                        artifacts,
                        raw_x,
                        time_ms_value=float(time_ms),
                        axis_name=axis,
                        registry=registry,
                    )
                )
            dense_df = pd.DataFrame(dense_rows)

            support_mu = []
            for x_value in support_levels:
                raw_x = dict(raw)
                raw_x[axis] = float(x_value)
                support_mu.append(
                    evaluate_physical_point_with_derivatives(
                        artifacts,
                        raw_x,
                        time_ms_value=float(time_ms),
                        axis_name=axis,
                        registry=registry,
                    )["mu"]
                )
            support_mu = np.asarray(support_mu, dtype=float)
            secant_reference = np.interp(sweep, support_levels, support_mu)
            support_span = float(np.nanmax(support_mu) - np.nanmin(support_mu))
            max_secant_deviation = float(np.nanmax(np.abs(dense_df["mu"].to_numpy(dtype=float) - secant_reference)))
            metrics_rows.append(
                {
                    "run_label": run_label,
                    "feature_family": infer_feature_family(list(artifacts.train_config["feature_columns"])),
                    "anchor_label": anchor["label"],
                    "axis": axis,
                    "time_ms": float(time_ms),
                    "sign_changes": sign_change_count(dense_df["grad"].to_numpy(dtype=float)),
                    "curvature_abs_p95": float(np.percentile(np.abs(dense_df["curvature"].to_numpy(dtype=float)), 95)),
                    "grad_abs_p95": float(np.percentile(np.abs(dense_df["grad"].to_numpy(dtype=float)), 95)),
                    "max_abs_secant_deviation_ratio": (
                        float(max_secant_deviation / support_span) if support_span > 1e-12 else np.nan
                    ),
                    "support_levels_on_slice": len(support_levels),
                }
            )
    return pd.DataFrame(metrics_rows)


def plot_metric_comparison(metrics_df: pd.DataFrame, save_dir: Path) -> None:
    plots = [
        ("sign_changes", "Gradient sign changes"),
        ("max_abs_secant_deviation_ratio", "Max secant deviation / support span"),
        ("curvature_abs_p95", "p95 |d2 mu / dx2|"),
    ]
    for metric_key, title in plots:
        fig, axes = plt.subplots(1, len(DEFAULT_ANCHORS), figsize=(6 * len(DEFAULT_ANCHORS), 4), sharey=False)
        if len(DEFAULT_ANCHORS) == 1:
            axes = [axes]
        for axis_obj, anchor in zip(axes, DEFAULT_ANCHORS):
            group = metrics_df.loc[metrics_df["anchor_label"] == anchor["label"]]
            for run_label, run_group in group.groupby("run_label"):
                axis_obj.plot(run_group["time_ms"], run_group[metric_key], marker="o", label=run_label)
            axis_obj.set_title(anchor["label"])
            axis_obj.set_xlabel("Time [ms]")
            axis_obj.grid(True, alpha=0.3)
            axis_obj.legend(fontsize=8)
        axes[0].set_ylabel(title)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_dir / f"{metric_key}_comparison.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    registry = build_dataset_registry()
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else (Path.cwd() / "v2_sparse_feature_comparison")
    save_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"Run spec must be label=PATH, got: {item}")
        label, path = item.split("=", 1)
        print("Evaluating run:", label, path)
        all_metrics.append(compute_run_metrics(label, path, registry, device=None if args.device == "auto" else args.device))

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv(save_dir / "sparse_feature_stability_comparison.csv", index=False)
    summary_df = (
        metrics_df.groupby(["run_label", "axis"])[["sign_changes", "max_abs_secant_deviation_ratio", "curvature_abs_p95"]]
        .mean()
        .reset_index()
    )
    summary_df.to_csv(save_dir / "sparse_feature_stability_summary.csv", index=False)
    plot_metric_comparison(metrics_df, save_dir)
    print("Saved comparison CSVs and plots to:", save_dir)


if __name__ == "__main__":
    main()
