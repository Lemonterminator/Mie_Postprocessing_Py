from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import pandas as pd
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


def resolve_model_path(run_dir: Path) -> Path:
    for name in ("best_model_refinement.pt", "best_model_stage2.pt", "best_model_stage1.pt"):
        model_path = run_dir / name
        if model_path.exists():
            return model_path
    raise FileNotFoundError(f"No supported model checkpoint found under: {run_dir}")


def unwrap_state_dict(state: Any) -> dict[str, torch.Tensor]:
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def infer_output_dim_from_state(state_dict: dict[str, torch.Tensor]) -> int:
    weight_keys = [key for key, value in state_dict.items() if key.endswith("weight") and getattr(value, "ndim", None) == 2]
    if not weight_keys:
        raise KeyError("Could not infer output dimension from checkpoint state dict.")
    return int(state_dict[weight_keys[-1]].shape[0])


def resolve_run_artifacts(run_dir: Path, base_run_dir: Path | None = None) -> dict[str, Path | None]:
    model_path = resolve_model_path(run_dir)
    config_path = run_dir / "train_config_used.json"
    scaler_path = run_dir / "scaler_state.json"
    refine_config_path = run_dir / "refine_config.json"

    if config_path.exists() and scaler_path.exists():
        metadata_run_dir = run_dir
    elif refine_config_path.exists():
        if base_run_dir is None:
            raise FileNotFoundError("Refinement run requires BASE_RUN_DIR for config/scaler metadata.")
        metadata_run_dir = base_run_dir
        config_path = metadata_run_dir / "train_config_used.json"
        scaler_path = metadata_run_dir / "scaler_state.json"
        if not config_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"BASE_RUN_DIR missing metadata files: {metadata_run_dir}")
    else:
        raise FileNotFoundError(f"Could not resolve metadata files under {run_dir}")

    return {
        "model_path": model_path,
        "config_path": config_path,
        "scaler_path": scaler_path,
        "metadata_run_dir": metadata_run_dir,
        "refine_config_path": refine_config_path if refine_config_path.exists() else None,
    }


def build_context(project_root: Path):
    mlp_dir = project_root / "MLP"
    if str(mlp_dir) not in sys.path:
        sys.path.insert(0, str(mlp_dir))

    try:
        from v1_direct_feature_training.ood_sanity import load_cdf_empirical_support
    except ModuleNotFoundError:
        from ood_sanity import load_cdf_empirical_support

    from gradient_stability_diagnostics import (
        GradientStabilityContext,
        collect_sparse_family_time_metrics,
        load_analysis_support_df,
        plot_sparse_family_interpolants,
        plot_sparse_family_time_metrics,
        plot_sparse_feature_support_topology,
    )

    base_run_dir = mlp_dir / "runs_mlp" / "stage2_NLL_penetration_20260317_194155"
    run_dir = mlp_dir / "runs_mlp" / "distill_cdf_onset_20260331_194213"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifacts = resolve_run_artifacts(run_dir, base_run_dir=base_run_dir)
    with artifacts["config_path"].open("r", encoding="utf-8") as f:
        train_config = json.load(f)
    with artifacts["scaler_path"].open("r", encoding="utf-8") as f:
        scaler_state = json.load(f)

    state = torch.load(artifacts["model_path"], map_location=device)
    state_dict = unwrap_state_dict(state)
    output_dim = infer_output_dim_from_state(state_dict)

    model = PenetrationMLP(
        input_dim=int(train_config["input_dim"]),
        hidden_dims=[int(x) for x in train_config["hidden_dims"]],
        output_dim=output_dim,
        activation=str(train_config.get("activation", "relu")),
        dropout=float(train_config.get("dropout", 0.0)),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    support = load_cdf_empirical_support(project_root, split_filter="clean")
    support_df = load_analysis_support_df(support)

    ctx = GradientStabilityContext(
        model=model,
        scaler_state=scaler_state,
        feature_columns=list(train_config["feature_columns"]),
        time_feature=str(train_config.get("time_feature", "time_norm_0_5ms")),
        train_config=train_config,
        support=support,
        support_df=support_df,
        device=device,
    )
    return (
        ctx,
        collect_sparse_family_time_metrics,
        plot_sparse_family_interpolants,
        plot_sparse_family_time_metrics,
        plot_sparse_feature_support_topology,
    )


def main() -> None:
    slides_dir = Path(__file__).resolve().parent
    # slides_dir = <repo>/Thesis/slides/legacy_notebook_sources/slides_sparse_feature_instability
    project_root = slides_dir.parents[3]
    figures_dir = slides_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    (
        ctx,
        collect_sparse_family_time_metrics,
        plot_sparse_family_interpolants,
        plot_sparse_family_time_metrics,
        plot_sparse_feature_support_topology,
    ) = build_context(project_root)

    analysis_time_ms = 0.85
    analysis_times_ms = [0.5, 0.85, 1.0, 2.0, 3.5]

    sparse_base_raw = {
        "tilt_angle_radian": float(torch.deg2rad(torch.tensor(20.0)).item()),
        "plumes": 10.0,
        "diameter_mm": 0.384,
        "injection_duration_us": 800.0,
        "injection_pressure_bar": 2000.0,
        "chamber_pressure_bar": 15.0,
        "control_backpressure_bar": 4.0,
    }

    diameter_family_specs = [
        {
            "label": "Pinj=2000, Pch=5, CB=4",
            "overrides": {"injection_pressure_bar": 2000.0, "chamber_pressure_bar": 5.0, "control_backpressure_bar": 4.0},
        },
        {
            "label": "Pinj=2000, Pch=10, CB=4",
            "overrides": {"injection_pressure_bar": 2000.0, "chamber_pressure_bar": 10.0, "control_backpressure_bar": 4.0},
        },
        {
            "label": "Pinj=2000, Pch=15, CB=4",
            "overrides": {"injection_pressure_bar": 2000.0, "chamber_pressure_bar": 15.0, "control_backpressure_bar": 4.0},
        },
    ]

    injection_family_specs = [
        {
            "label": "Pch=5, CB=1, d=0.384",
            "overrides": {"chamber_pressure_bar": 5.0, "control_backpressure_bar": 1.0, "diameter_mm": 0.384},
        },
        {
            "label": "Pch=15, CB=1, d=0.384",
            "overrides": {"chamber_pressure_bar": 15.0, "control_backpressure_bar": 1.0, "diameter_mm": 0.384},
        },
        {
            "label": "Pch=35, CB=1, d=0.384",
            "overrides": {"chamber_pressure_bar": 35.0, "control_backpressure_bar": 1.0, "diameter_mm": 0.384},
        },
    ]

    fig = plot_sparse_feature_support_topology(ctx)
    fig.savefig(figures_dir / "support_topology.png", dpi=220, bbox_inches="tight")

    fig, diameter_metrics_df = plot_sparse_family_interpolants(
        ctx,
        "diameter_mm",
        diameter_family_specs,
        sparse_base_raw,
        time_ms_value=analysis_time_ms,
        expected_sign=+1.0,
        n_points=201,
        fig_title=f"Nozzle diameter: dense MLP interpolation vs observed-level secants at t={analysis_time_ms:.2f} ms",
    )
    fig.savefig(figures_dir / "diameter_interpolants.png", dpi=220, bbox_inches="tight")

    fig, injection_metrics_df = plot_sparse_family_interpolants(
        ctx,
        "injection_pressure_bar",
        injection_family_specs,
        sparse_base_raw,
        time_ms_value=analysis_time_ms,
        expected_sign=+1.0,
        n_points=201,
        fig_title=f"Injection pressure: dense MLP interpolation vs observed-level secants at t={analysis_time_ms:.2f} ms",
    )
    fig.savefig(figures_dir / "injection_interpolants.png", dpi=220, bbox_inches="tight")

    diameter_time_metrics_df = collect_sparse_family_time_metrics(
        ctx,
        "diameter_mm",
        diameter_family_specs,
        sparse_base_raw,
        analysis_times_ms,
        expected_sign=+1.0,
        n_points=121,
    )
    injection_time_metrics_df = collect_sparse_family_time_metrics(
        ctx,
        "injection_pressure_bar",
        injection_family_specs,
        sparse_base_raw,
        analysis_times_ms,
        expected_sign=+1.0,
        n_points=121,
    )

    fig = plot_sparse_family_time_metrics(
        diameter_time_metrics_df,
        "diameter_mm",
        fig_title="Nozzle diameter instability metrics across time",
    )
    fig.savefig(figures_dir / "diameter_time_metrics.png", dpi=220, bbox_inches="tight")

    fig = plot_sparse_family_time_metrics(
        injection_time_metrics_df,
        "injection_pressure_bar",
        fig_title="Injection-pressure instability metrics across time",
    )
    fig.savefig(figures_dir / "injection_time_metrics.png", dpi=220, bbox_inches="tight")

    summary_rows = [
        {
            "feature_family": "diameter",
            "time_ms": analysis_time_ms,
            "worst_max_abs_secant_deviation_ratio": float(diameter_metrics_df["max_abs_secant_deviation_ratio"].max()),
            "worst_sign_changes": int(diameter_metrics_df["sign_changes"].max()),
            "worst_curvature_abs_p95": float(diameter_metrics_df["curvature_abs_p95"].max()),
        },
        {
            "feature_family": "injection_pressure",
            "time_ms": analysis_time_ms,
            "worst_max_abs_secant_deviation_ratio": float(injection_metrics_df["max_abs_secant_deviation_ratio"].max()),
            "worst_sign_changes": int(injection_metrics_df["sign_changes"].max()),
            "worst_curvature_abs_p95": float(injection_metrics_df["curvature_abs_p95"].max()),
        },
    ]
    pd.DataFrame(summary_rows).to_csv(figures_dir / "summary_metrics.csv", index=False)
    diameter_metrics_df.to_csv(figures_dir / "diameter_interpolant_metrics.csv", index=False)
    injection_metrics_df.to_csv(figures_dir / "injection_interpolant_metrics.csv", index=False)
    diameter_time_metrics_df.to_csv(figures_dir / "diameter_time_metrics.csv", index=False)
    injection_time_metrics_df.to_csv(figures_dir / "injection_time_metrics.csv", index=False)

    print("Exported figures to", figures_dir)


if __name__ == "__main__":
    main()
