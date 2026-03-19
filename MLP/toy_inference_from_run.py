from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def make_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'")


class PenetrationMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    make_activation(activation),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_mu_logvar(model_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, log_var = model_output.chunk(2, dim=-1)
    return mu, log_var


def resolve_run_dir(path_str: str) -> Path:
    base = Path(path_str).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")

    if base.is_file():
        base = base.parent

    if has_run_artifacts(base):
        return base

    candidates: list[tuple[float, Path]] = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if not has_run_artifacts(child):
            continue
        model_path = resolve_model_path(child)
        mtime = model_path.stat().st_mtime
        candidates.append((mtime, child))

    if not candidates:
        raise FileNotFoundError(
            "Could not find a valid run directory. Expected a folder containing "
            "'train_config_used.json', 'scaler_state.json', and a best model checkpoint."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def resolve_model_path(run_dir: Path) -> Path:
    for name in ("best_model_stage2.pt", "best_model_stage1.pt"):
        model_path = run_dir / name
        if model_path.exists():
            return model_path
    raise FileNotFoundError(f"No supported model checkpoint found under: {run_dir}")


def has_run_artifacts(run_dir: Path) -> bool:
    return (
        run_dir.is_dir()
        and (run_dir / "train_config_used.json").exists()
        and (run_dir / "scaler_state.json").exists()
        and any((run_dir / name).exists() for name in ("best_model_stage2.pt", "best_model_stage1.pt"))
    )


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_run_artifacts(run_dir: Path, device: torch.device) -> tuple[PenetrationMLP, dict[str, Any], dict[str, Any], Path]:
    config_path = run_dir / "train_config_used.json"
    scaler_path = run_dir / "scaler_state.json"
    model_path = resolve_model_path(run_dir)

    with config_path.open("r", encoding="utf-8") as f:
        train_config = json.load(f)
    with scaler_path.open("r", encoding="utf-8") as f:
        scaler_state = json.load(f)

    model = PenetrationMLP(
        input_dim=int(train_config["input_dim"]),
        hidden_dims=[int(x) for x in train_config["hidden_dims"]],
        output_dim=int(train_config["output_dim"]),
        activation=str(train_config.get("activation", "relu")),
        dropout=float(train_config.get("dropout", 0.0)),
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, train_config, scaler_state, model_path


def zscore_from_state(value: float, z_col: str, scaler_state: dict[str, Any]) -> float:
    stats = scaler_state["zscore"][z_col]
    return (float(value) - float(stats["mean"])) / (float(stats["std"]) + 1e-12)


def build_toy_feature_matrix(
    raw: dict[str, float],
    time_ms: np.ndarray,
    scaler_state: dict[str, Any],
    feature_columns: list[str],
    time_feature: str,
) -> np.ndarray:
    p_inj = float(raw["injection_pressure_bar"])
    p_ch = float(raw["chamber_pressure_bar"])
    delta_p = max(p_inj - p_ch, 1e-6)

    time_min_ms = float(scaler_state["time"]["min_ms"])
    time_max_ms = float(scaler_state["time"]["max_ms"])
    time_span_ms = max(time_max_ms - time_min_ms, 1e-12)
    time_norm = np.clip((time_ms - time_min_ms) / time_span_ms, 0.0, 1.0).astype(np.float32)

    feature_series: dict[str, np.ndarray] = {
        time_feature: time_norm,
        "tilt_angle_radian_z": np.full_like(
            time_norm,
            zscore_from_state(raw["tilt_angle_radian"], "tilt_angle_radian_z", scaler_state),
            dtype=np.float32,
        ),
        "plumes_z": np.full_like(
            time_norm,
            zscore_from_state(raw["plumes"], "plumes_z", scaler_state),
            dtype=np.float32,
        ),
        "diameter_mm_z": np.full_like(
            time_norm,
            zscore_from_state(raw["diameter_mm"], "diameter_mm_z", scaler_state),
            dtype=np.float32,
        ),
        "injection_duration_us_z": np.full_like(
            time_norm,
            zscore_from_state(raw["injection_duration_us"], "injection_duration_us_z", scaler_state),
            dtype=np.float32,
        ),
        "log_injection_pressure_bar_z": np.full_like(
            time_norm,
            zscore_from_state(np.log(p_inj), "log_injection_pressure_bar_z", scaler_state),
            dtype=np.float32,
        ),
        "log_chamber_pressure_bar_z": np.full_like(
            time_norm,
            zscore_from_state(np.log(max(p_ch, 1e-6)), "log_chamber_pressure_bar_z", scaler_state),
            dtype=np.float32,
        ),
        "log_delta_pressure_bar_z": np.full_like(
            time_norm,
            zscore_from_state(np.log(delta_p), "log_delta_pressure_bar_z", scaler_state),
            dtype=np.float32,
        ),
        "control_backpressure_bar_z": np.full_like(
            time_norm,
            zscore_from_state(raw["control_backpressure_bar"], "control_backpressure_bar_z", scaler_state),
            dtype=np.float32,
        ),
    }

    columns: list[np.ndarray] = []
    for name in feature_columns:
        if name not in feature_series:
            raise KeyError(f"Unsupported feature column in config: {name}")
        columns.append(feature_series[name])
    return np.column_stack(columns).astype(np.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run toy penetration inference from an MLP run directory and plot the 0-5 ms sweep."
    )
    parser.add_argument("run_dir", help="Run directory or parent directory containing trained run folders.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tilt-angle-deg", type=float, default=20.0)
    parser.add_argument("--plumes", type=float, default=10.0)
    parser.add_argument("--diameter-mm", type=float, default=0.355)
    parser.add_argument("--injection-duration-us", type=float, default=800.0)
    parser.add_argument("--injection-pressure-bar", type=float, default=2000.0)
    parser.add_argument("--chamber-pressure-bar", type=float, default=5.0)
    parser.add_argument("--control-backpressure-bar", type=float, default=4.0)
    parser.add_argument("--time-start-ms", type=float, default=0.0)
    parser.add_argument("--time-end-ms", type=float, default=5.0)
    parser.add_argument("--n-points", type=int, default=300)
    parser.add_argument("--save-path", type=str, default=None, help="Optional output path for the plotted figure.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive matplotlib window.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    device = choose_device(args.device)
    run_dir = resolve_run_dir(args.run_dir)
    model, train_config, scaler_state, model_path = load_run_artifacts(run_dir, device)

    feature_columns = list(train_config["feature_columns"])
    time_feature = str(train_config.get("time_feature", "time_norm_0_5ms"))

    toy_raw = {
        "tilt_angle_radian": float(np.deg2rad(args.tilt_angle_deg)),
        "plumes": float(args.plumes),
        "diameter_mm": float(args.diameter_mm),
        "injection_duration_us": float(args.injection_duration_us),
        "injection_pressure_bar": float(args.injection_pressure_bar),
        "chamber_pressure_bar": float(args.chamber_pressure_bar),
        "control_backpressure_bar": float(args.control_backpressure_bar),
    }

    if args.n_points < 2:
        raise ValueError("--n-points must be at least 2.")
    if args.time_end_ms < args.time_start_ms:
        raise ValueError("--time-end-ms must be greater than or equal to --time-start-ms.")

    toy_time_ms = np.linspace(args.time_start_ms, args.time_end_ms, args.n_points, dtype=np.float32)
    toy_features_np = build_toy_feature_matrix(
        raw=toy_raw,
        time_ms=toy_time_ms,
        scaler_state=scaler_state,
        feature_columns=feature_columns,
        time_feature=time_feature,
    )
    toy_features = torch.as_tensor(toy_features_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        toy_out = model(toy_features)
        toy_mu, toy_log_var = split_mu_logvar(toy_out)

    toy_mu_np = toy_mu.detach().cpu().numpy().reshape(-1)
    toy_log_var_np = toy_log_var.detach().cpu().numpy().reshape(-1)
    std_floor = float(train_config.get("std_clamp_min", 0.0))
    toy_std_np = np.maximum(np.sqrt(np.exp(toy_log_var_np)), std_floor)
    toy_upper_np = toy_mu_np + toy_std_np
    toy_lower_np = toy_mu_np - toy_std_np

    plt.figure(figsize=(8, 5))
    plt.plot(toy_time_ms, toy_mu_np, linewidth=2, label="Predicted mean")
    plt.fill_between(toy_time_ms, toy_lower_np, toy_upper_np, alpha=0.25, label="Predicted +/- 1 std")
    plt.plot(toy_time_ms, 90 * np.ones_like(toy_time_ms), linestyle="--", label="90 mm reference")
    plt.xlabel("Time (ms)")
    plt.ylabel("Predicted penetration")
    plt.title("Toy Inference: 0-5 ms Penetration Sweep with Uncertainty")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 250)
    plt.xlim(args.time_start_ms, args.time_end_ms)

    if args.save_path:
        save_path = Path(args.save_path).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to: {save_path}")

    if not args.no_show:
        plt.show()
    plt.close()

    print(f"Resolved run_dir: {run_dir}")
    print(f"Loaded model checkpoint: {model_path}")
    print(f"Using device: {device}")
    print(f"Toy inference completed with feature shape: {toy_features_np.shape}")
    print(f"Expected feature dimension: {len(feature_columns)}")
    print(f"Predicted std range: {float(np.min(toy_std_np))} {float(np.max(toy_std_np))}")


if __name__ == "__main__":
    main()
