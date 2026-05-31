from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # MLP_training/

from engineered_feature_common import build_dataset_registry, load_run_artifacts, predict_physical_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy inference for engineered-feature penetration runs.")
    parser.add_argument("run_dir", help="Run directory or parent directory containing engineered runs.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dataset-key", type=str, default=None, help="Optional dataset key for raw chamber-state disambiguation, e.g. nozzle1 or ds300.")
    parser.add_argument("--tilt-angle-deg", type=float, default=20.0)
    parser.add_argument("--plumes", type=float, default=10.0)
    parser.add_argument("--diameter-mm", type=float, default=0.355)
    parser.add_argument("--injection-duration-us", type=float, default=800.0)
    parser.add_argument("--injection-pressure-bar", type=float, default=2000.0)
    parser.add_argument("--chamber-state-raw", type=float, default=None, help="Legacy/raw chamber-state label, e.g. 5/10/15 for nozzle datasets.")
    parser.add_argument("--ambient-pressure-bar-phys", type=float, default=None, help="Physical ambient pressure in bar.")
    parser.add_argument("--ambient-density-kg-m3", type=float, default=None, help="Ambient density in kg/m3.")
    parser.add_argument("--control-backpressure-bar", type=float, default=4.0)
    parser.add_argument("--time-start-ms", type=float, default=0.0)
    parser.add_argument("--time-end-ms", type=float, default=5.0)
    parser.add_argument("--n-points", type=int, default=300)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def choose_device(device_arg: str) -> str | None:
    if device_arg == "auto":
        return None
    return device_arg


def resolve_default_ambient_pressure_bar(artifacts) -> float:
    canonical = artifacts.scaler_state.get("canonicalization", {})
    return float(canonical.get("reference_pressure_bar", 4.42))


def main() -> None:
    args = parse_args()
    registry = build_dataset_registry()
    artifacts = load_run_artifacts(args.run_dir, device=choose_device(args.device))
    time_ms = np.linspace(float(args.time_start_ms), float(args.time_end_ms), int(args.n_points), dtype=np.float32)
    assumed_ambient_pressure_bar = None

    raw = {
        "dataset_key": args.dataset_key,
        "tilt_angle_radian": float(np.deg2rad(args.tilt_angle_deg)),
        "plumes": float(args.plumes),
        "diameter_mm": float(args.diameter_mm),
        "injection_duration_us": float(args.injection_duration_us),
        "injection_pressure_bar": float(args.injection_pressure_bar),
        "control_backpressure_bar": float(args.control_backpressure_bar),
    }
    if args.ambient_pressure_bar_phys is not None:
        raw["ambient_pressure_bar_phys"] = float(args.ambient_pressure_bar_phys)
    if args.ambient_density_kg_m3 is not None:
        raw["ambient_density_kg_m3"] = float(args.ambient_density_kg_m3)
    if args.chamber_state_raw is not None:
        raw["chamber_state_raw"] = float(args.chamber_state_raw)
        raw["chamber_pressure_bar"] = float(args.chamber_state_raw)
    elif args.ambient_pressure_bar_phys is not None:
        raw["chamber_pressure_bar"] = float(args.ambient_pressure_bar_phys)
    elif args.ambient_density_kg_m3 is None:
        assumed_ambient_pressure_bar = resolve_default_ambient_pressure_bar(artifacts)
        raw["ambient_pressure_bar_phys"] = assumed_ambient_pressure_bar
        raw["chamber_pressure_bar"] = assumed_ambient_pressure_bar

    prediction = predict_physical_sweep(artifacts, raw, time_ms, registry)
    mu = prediction["mu_physical"]
    std = prediction["std_physical"]

    plt.figure(figsize=(8, 5))
    plt.plot(prediction["time_ms"], mu, linewidth=2, label="Physical mean")
    plt.fill_between(prediction["time_ms"], mu - std, mu + std, alpha=0.25, label="Physical +/- 1 std")
    plt.xlabel("Time [ms]")
    plt.ylabel("Penetration")
    plt.title("Engineered-feature toy inference")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.save_path:
        save_path = Path(args.save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180)
        print("Saved figure:", save_path)
    if not args.no_show:
        plt.show()
    plt.close()

    print("Resolved run_dir:", artifacts.run_dir)
    print("Loaded checkpoint:", artifacts.model_path)
    print("Feature matrix shape:", prediction["feature_matrix_shape"])
    print("Canonical raw state:", prediction["canonical"])
    if assumed_ambient_pressure_bar is not None:
        print("Assumed ambient pressure [bar]:", assumed_ambient_pressure_bar)
    print("Predicted std range:", float(np.min(std)), float(np.max(std)))


if __name__ == "__main__":
    main()
