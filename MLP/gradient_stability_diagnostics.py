from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import torch


AXIS_LABELS = {
    "injection_pressure_bar": "Injection pressure [bar]",
    "chamber_pressure_bar": "Chamber pressure [bar]",
    "diameter_mm": "Nozzle diameter [mm]",
}


@dataclass(frozen=True)
class GradientStabilityContext:
    model: torch.nn.Module
    scaler_state: Mapping[str, Any]
    feature_columns: Sequence[str]
    time_feature: str
    train_config: Mapping[str, Any]
    support: Mapping[str, Any]
    support_df: pd.DataFrame
    device: torch.device


def load_analysis_support_df(support: Mapping[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(str(support["audit_csv_path"]), low_memory=False)
    split_filter = support.get("split_filter")
    if split_filter is not None:
        if "sample_split" in df.columns:
            df = df.loc[df["sample_split"] == split_filter].copy()
        elif split_filter == "clean" and "flag_bad_fit" in df.columns:
            df = df.loc[~df["flag_bad_fit"].fillna(False)].copy()

    numeric_cols = (
        "diameter_mm",
        "injection_pressure_bar",
        "chamber_pressure_bar",
        "control_backpressure_bar",
        "injection_duration_us",
        "plumes",
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "tilt_angle_radian" not in df.columns and "umbrella_angle_deg" in df.columns:
        umbrella = pd.to_numeric(df["umbrella_angle_deg"], errors="coerce")
        df["tilt_angle_radian"] = np.deg2rad((180.0 - umbrella) / 2.0)

    return df


def _scalar_tensor(value: float | torch.Tensor, *, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        if requires_grad and not tensor.requires_grad:
            tensor = tensor.clone().detach().requires_grad_(True)
        return tensor
    return torch.tensor([float(value)], dtype=torch.float32, device=device, requires_grad=requires_grad)


def _torch_zscore(ctx: GradientStabilityContext, value: torch.Tensor, z_col: str) -> torch.Tensor:
    stats = ctx.scaler_state["zscore"][z_col]
    mean = torch.tensor(float(stats["mean"]), dtype=torch.float32, device=ctx.device)
    std = torch.tensor(float(stats["std"]), dtype=torch.float32, device=ctx.device)
    return (value - mean) / (std + 1e-12)


def build_toy_feature_tensor_from_raw(
    ctx: GradientStabilityContext,
    raw_values: Mapping[str, float | torch.Tensor],
    time_ms_value: float | torch.Tensor,
) -> torch.Tensor:
    time_ms_tensor = _scalar_tensor(time_ms_value, device=ctx.device)
    p_inj = _scalar_tensor(raw_values["injection_pressure_bar"], device=ctx.device)
    p_ch = _scalar_tensor(raw_values["chamber_pressure_bar"], device=ctx.device)
    delta_p = torch.clamp(p_inj - p_ch, min=1e-6)

    time_min_ms = float(ctx.scaler_state["time"]["min_ms"])
    time_max_ms = float(ctx.scaler_state["time"]["max_ms"])
    time_span_ms = max(time_max_ms - time_min_ms, 1e-12)
    time_norm = torch.clamp((time_ms_tensor - time_min_ms) / time_span_ms, 0.0, 1.0)

    feature_series = {
        ctx.time_feature: time_norm,
        "tilt_angle_radian_z": _torch_zscore(
            ctx,
            _scalar_tensor(raw_values["tilt_angle_radian"], device=ctx.device),
            "tilt_angle_radian_z",
        ),
        "plumes_z": _torch_zscore(
            ctx,
            _scalar_tensor(raw_values["plumes"], device=ctx.device),
            "plumes_z",
        ),
        "diameter_mm_z": _torch_zscore(
            ctx,
            _scalar_tensor(raw_values["diameter_mm"], device=ctx.device),
            "diameter_mm_z",
        ),
        "injection_duration_us_z": _torch_zscore(
            ctx,
            _scalar_tensor(raw_values["injection_duration_us"], device=ctx.device),
            "injection_duration_us_z",
        ),
        "log_injection_pressure_bar_z": _torch_zscore(
            ctx,
            torch.log(torch.clamp(p_inj, min=1e-6)),
            "log_injection_pressure_bar_z",
        ),
        "log_chamber_pressure_bar_z": _torch_zscore(
            ctx,
            torch.log(torch.clamp(p_ch, min=1e-6)),
            "log_chamber_pressure_bar_z",
        ),
        "log_delta_pressure_bar_z": _torch_zscore(
            ctx,
            torch.log(delta_p),
            "log_delta_pressure_bar_z",
        ),
        "control_backpressure_bar_z": _torch_zscore(
            ctx,
            _scalar_tensor(raw_values["control_backpressure_bar"], device=ctx.device),
            "control_backpressure_bar_z",
        ),
    }
    return torch.column_stack([feature_series[name] for name in ctx.feature_columns])


def evaluate_point_with_derivatives(
    ctx: GradientStabilityContext,
    raw_point: Mapping[str, float],
    *,
    time_ms_value: float,
    axis_name: str | None = None,
    second_order: bool = False,
    coupled_axis_name: str | None = None,
) -> dict[str, float]:
    raw_tensors: dict[str, torch.Tensor] = {}
    for key, value in raw_point.items():
        needs_grad = key == axis_name or key == coupled_axis_name
        raw_tensors[key] = _scalar_tensor(value, device=ctx.device, requires_grad=needs_grad)

    features = build_toy_feature_tensor_from_raw(ctx, raw_tensors, time_ms_value)
    model_output = ctx.model(features)
    mu_tensor = model_output[..., :1]
    log_var_tensor = model_output[..., 1:2]
    onset_logit_tensor = model_output[..., 2:3] if model_output.shape[-1] >= 3 else None

    mu_value = mu_tensor.reshape(-1)[0]
    log_var_value = log_var_tensor.reshape(-1)[0]
    std_value = torch.clamp(
        torch.exp(0.5 * log_var_value),
        min=float(ctx.train_config.get("std_clamp_min", 0.0)),
    )

    result = {
        "mu": float(mu_value.detach().cpu().item()),
        "std": float(std_value.detach().cpu().item()),
    }
    if onset_logit_tensor is not None:
        onset_prob = torch.sigmoid(onset_logit_tensor.reshape(-1)[0])
        result["onset_prob"] = float(onset_prob.detach().cpu().item())

    if axis_name is None:
        return result

    grad_value = torch.autograd.grad(mu_value, raw_tensors[axis_name], create_graph=second_order)[0].reshape(-1)[0]
    result["grad"] = float(grad_value.detach().cpu().item())

    if second_order:
        if coupled_axis_name is None:
            curvature_value = torch.autograd.grad(grad_value, raw_tensors[axis_name])[0].reshape(-1)[0]
            result["curvature"] = float(curvature_value.detach().cpu().item())
        else:
            mixed_value = torch.autograd.grad(grad_value, raw_tensors[coupled_axis_name])[0].reshape(-1)[0]
            result["mixed_partial"] = float(mixed_value.detach().cpu().item())

    return result


def has_exact_pressure_combo(ctx: GradientStabilityContext, raw_point: Mapping[str, float]) -> bool:
    combo_df = ctx.support["combo_df"]
    mask = np.logical_and.reduce(
        [np.isclose(combo_df[col], raw_point[col]) for col in ctx.support["combo_cols"]]
    )
    return bool(mask.any())


def nearest_observed_pressure_combo(
    ctx: GradientStabilityContext,
    raw_point: Mapping[str, float],
) -> tuple[dict[str, float], float]:
    combo_cols = list(ctx.support["combo_cols"])
    combo_df = ctx.support["combo_df"].copy()
    combo_values = combo_df.loc[:, combo_cols].to_numpy(dtype=float)
    sample = np.array([raw_point[col] for col in combo_cols], dtype=float)
    scale = combo_df.loc[:, combo_cols].std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=float)
    distances = np.sqrt(np.sum(np.square((combo_values - sample) / scale), axis=1))
    best_idx = int(np.argmin(distances))
    return combo_df.iloc[best_idx].to_dict(), float(distances[best_idx])


def observed_pressure_points_for_backpressure(
    ctx: GradientStabilityContext,
    control_backpressure_bar: float,
) -> np.ndarray:
    combo_df = ctx.support["combo_df"]
    mask = np.isclose(combo_df["control_backpressure_bar"], control_backpressure_bar)
    return combo_df.loc[mask, ["injection_pressure_bar", "chamber_pressure_bar"]].to_numpy(dtype=float)


def get_axis_support_levels(
    ctx: GradientStabilityContext,
    axis_name: str,
    raw_point: Mapping[str, float],
) -> np.ndarray:
    df = ctx.support_df
    if axis_name == "injection_pressure_bar":
        mask = (
            np.isclose(df["chamber_pressure_bar"], raw_point["chamber_pressure_bar"])
            & np.isclose(df["control_backpressure_bar"], raw_point["control_backpressure_bar"])
        )
    elif axis_name == "chamber_pressure_bar":
        mask = (
            np.isclose(df["injection_pressure_bar"], raw_point["injection_pressure_bar"])
            & np.isclose(df["control_backpressure_bar"], raw_point["control_backpressure_bar"])
        )
    elif axis_name == "diameter_mm":
        mask = (
            np.isclose(df["injection_pressure_bar"], raw_point["injection_pressure_bar"])
            & np.isclose(df["chamber_pressure_bar"], raw_point["chamber_pressure_bar"])
            & np.isclose(df["control_backpressure_bar"], raw_point["control_backpressure_bar"])
        )
    else:
        raise KeyError(f"Unsupported axis: {axis_name}")
    return np.sort(df.loc[mask, axis_name].dropna().unique())


def diameter_support_crosstab(ctx: GradientStabilityContext) -> pd.DataFrame:
    return pd.crosstab(
        ctx.support_df["diameter_mm"].round(3),
        [
            ctx.support_df["injection_pressure_bar"],
            ctx.support_df["chamber_pressure_bar"],
            ctx.support_df["control_backpressure_bar"],
        ],
    )


def _format_level_summary(values: Sequence[float], *, precision: int = 3) -> str:
    if len(values) == 0:
        return "-"
    rounded = [round(float(v), precision) for v in values]
    if len(rounded) <= 3:
        return ", ".join(f"{v:.{precision}f}" for v in rounded)
    return f"{rounded[0]:.{precision}f}..{rounded[-1]:.{precision}f}"


def _annotate_support_heatmap(
    ax: plt.Axes,
    x_values: Sequence[float],
    y_values: Sequence[float],
    counts: np.ndarray,
    labels: np.ndarray,
) -> None:
    for row_idx, y_value in enumerate(y_values):
        for col_idx, x_value in enumerate(x_values):
            label = labels[row_idx, col_idx]
            if not label:
                continue
            text_color = "white" if counts[row_idx, col_idx] >= max(2.0, float(np.nanmax(counts)) * 0.55) else "black"
            ax.text(
                float(x_value),
                float(y_value),
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )


def plot_sparse_feature_support_topology(ctx: GradientStabilityContext) -> plt.Figure:
    df = ctx.support_df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    pch_values = np.sort(df["chamber_pressure_bar"].dropna().unique())
    cb_values = np.sort(df["control_backpressure_bar"].dropna().unique())
    inj_count_grid = np.zeros((len(cb_values), len(pch_values)), dtype=float)
    inj_label_grid = np.empty_like(inj_count_grid, dtype=object)
    inj_label_grid[:] = ""
    inj_group = (
        df.groupby(["chamber_pressure_bar", "control_backpressure_bar"])["injection_pressure_bar"]
        .apply(lambda s: tuple(np.sort(pd.Series(s).dropna().unique())))
    )
    for row_idx, cb in enumerate(cb_values):
        for col_idx, pch in enumerate(pch_values):
            if (pch, cb) not in inj_group.index:
                continue
            values = np.asarray(inj_group.loc[(pch, cb)], dtype=float)
            inj_count_grid[row_idx, col_idx] = len(values)
            inj_label_grid[row_idx, col_idx] = f"{len(values)}\n{_format_level_summary(values, precision=0)}"

    inj_ax = axes[0]
    inj_image = inj_ax.imshow(
        inj_count_grid,
        origin="lower",
        aspect="auto",
        cmap="YlGnBu",
        extent=[pch_values.min() - 2.5, pch_values.max() + 2.5, cb_values.min() - 0.5, cb_values.max() + 0.5],
    )
    inj_ax.set_xticks(pch_values)
    inj_ax.set_yticks(cb_values)
    inj_ax.set_xlabel("Chamber pressure [bar]")
    inj_ax.set_ylabel("Control backpressure [bar]")
    inj_ax.set_title("Observed injection-pressure support count")
    _annotate_support_heatmap(inj_ax, pch_values, cb_values, inj_count_grid, inj_label_grid)
    fig.colorbar(inj_image, ax=inj_ax, shrink=0.88)

    pinj_values = np.sort(df["injection_pressure_bar"].dropna().unique())
    diam_group = (
        df.groupby(["injection_pressure_bar", "chamber_pressure_bar", "control_backpressure_bar"])["diameter_mm"]
        .apply(lambda s: tuple(np.sort(pd.Series(s).dropna().unique())))
    )
    for ax, cb in zip(axes[1:], cb_values):
        diam_count_grid = np.zeros((len(pinj_values), len(pch_values)), dtype=float)
        diam_label_grid = np.empty_like(diam_count_grid, dtype=object)
        diam_label_grid[:] = ""
        for row_idx, pinj in enumerate(pinj_values):
            for col_idx, pch in enumerate(pch_values):
                key = (pinj, pch, cb)
                if key not in diam_group.index:
                    continue
                values = np.asarray(diam_group.loc[key], dtype=float)
                diam_count_grid[row_idx, col_idx] = len(values)
                diam_label_grid[row_idx, col_idx] = f"{len(values)}\n{_format_level_summary(values, precision=3)}"

        image = ax.imshow(
            diam_count_grid,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[pch_values.min() - 2.5, pch_values.max() + 2.5, pinj_values.min() - 100.0, pinj_values.max() + 100.0],
        )
        ax.set_xticks(pch_values)
        ax.set_yticks(pinj_values)
        ax.set_xlabel("Chamber pressure [bar]")
        ax.set_ylabel("Injection pressure [bar]")
        ax.set_title(f"Observed diameter support count at CB={cb:.0f} bar")
        _annotate_support_heatmap(ax, pch_values, pinj_values, diam_count_grid, diam_label_grid)
        fig.colorbar(image, ax=ax, shrink=0.88)

    fig.suptitle("Sparse-feature support topology", fontsize=14)
    return fig


def evaluate_observed_vs_continuous_interpolant(
    ctx: GradientStabilityContext,
    axis_name: str,
    raw_point: Mapping[str, float],
    *,
    time_ms_value: float,
    expected_sign: float | None = None,
    n_points: int = 201,
) -> dict[str, Any]:
    dense = sweep_axis_diagnostics(
        ctx,
        axis_name,
        raw_point,
        time_ms_value=time_ms_value,
        expected_sign=expected_sign,
        n_points=n_points,
    )
    support_levels = np.asarray(dense["support_levels"], dtype=float)

    mu_support = np.asarray(
        [
            evaluate_point_with_derivatives(
                ctx,
                {**raw_point, axis_name: float(level)},
                time_ms_value=time_ms_value,
            )["mu"]
            for level in support_levels
        ],
        dtype=float,
    )

    secant_reference = np.full_like(dense["x"], np.nan, dtype=float)
    inside_hull_mask = np.zeros_like(dense["x"], dtype=bool)
    secant_slopes = np.asarray([], dtype=float)
    if len(support_levels) >= 2:
        inside_hull_mask = (dense["x"] >= support_levels[0]) & (dense["x"] <= support_levels[-1])
        secant_reference[inside_hull_mask] = np.interp(
            dense["x"][inside_hull_mask],
            support_levels,
            mu_support,
        )
        secant_slopes = np.diff(mu_support) / np.diff(support_levels)

    max_abs_secant_deviation = float(np.nanmax(np.abs(dense["mu"] - secant_reference))) if np.any(inside_hull_mask) else np.nan
    mean_abs_secant_deviation = float(np.nanmean(np.abs(dense["mu"] - secant_reference))) if np.any(inside_hull_mask) else np.nan
    support_mu_span = float(np.max(mu_support) - np.min(mu_support)) if len(mu_support) > 0 else np.nan
    support_mu_span = support_mu_span if np.isfinite(support_mu_span) and support_mu_span > 1e-12 else np.nan

    metrics = dict(dense["metrics"])
    metrics.update(
        {
            "support_hull_width": float(support_levels[-1] - support_levels[0]) if len(support_levels) >= 2 else np.nan,
            "support_mu_span": support_mu_span,
            "max_abs_secant_deviation": max_abs_secant_deviation,
            "mean_abs_secant_deviation": mean_abs_secant_deviation,
            "max_abs_secant_deviation_ratio": (
                float(max_abs_secant_deviation / support_mu_span) if np.isfinite(support_mu_span) else np.nan
            ),
            "mean_abs_secant_deviation_ratio": (
                float(mean_abs_secant_deviation / support_mu_span) if np.isfinite(support_mu_span) else np.nan
            ),
        }
    )

    return {
        **dense,
        "support_levels": support_levels,
        "mu_support": mu_support,
        "secant_reference": secant_reference,
        "inside_hull_mask": inside_hull_mask,
        "secant_slopes": secant_slopes,
        "metrics": metrics,
    }


def plot_sparse_family_interpolants(
    ctx: GradientStabilityContext,
    axis_name: str,
    slice_specs: Sequence[Mapping[str, Any]],
    raw_template: Mapping[str, float],
    *,
    time_ms_value: float,
    expected_sign: float | None = None,
    n_points: int = 201,
    fig_title: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    analyses: list[tuple[str, dict[str, Any], dict[str, float]]] = []
    metrics_rows = []
    for spec in slice_specs:
        raw_point = dict(raw_template)
        raw_point.update(spec.get("overrides", {}))
        label = str(spec.get("label", spec.get("overrides", {})))
        result = evaluate_observed_vs_continuous_interpolant(
            ctx,
            axis_name,
            raw_point,
            time_ms_value=time_ms_value,
            expected_sign=expected_sign,
            n_points=n_points,
        )
        analyses.append((label, result, raw_point))
        metrics_row = {"slice_label": label, "time_ms": float(time_ms_value)}
        metrics_row.update(result["metrics"])
        metrics_rows.append(metrics_row)

    fig, axes = plt.subplots(2, len(analyses), figsize=(5.2 * len(analyses), 8.0), sharex="col")
    if len(analyses) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col_idx, (label, result, raw_point) in enumerate(analyses):
        top_ax = axes[0, col_idx]
        bot_ax = axes[1, col_idx]

        top_ax.plot(result["x"], result["mu"], color="tab:blue", linewidth=2.0, label="Dense MLP sweep")
        if np.any(result["inside_hull_mask"]):
            top_ax.plot(
                result["x"][result["inside_hull_mask"]],
                result["secant_reference"][result["inside_hull_mask"]],
                color="black",
                linestyle="--",
                linewidth=1.8,
                label="Observed-level secant",
            )
        if len(result["support_levels"]) > 0:
            top_ax.scatter(
                result["support_levels"],
                result["mu_support"],
                color="tab:orange",
                edgecolor="black",
                s=42,
                zorder=3,
                label="Predictions at observed levels",
            )
            for level in result["support_levels"]:
                top_ax.axvline(level, color="tab:orange", alpha=0.18, linewidth=3.0)
        top_ax.grid(True, alpha=0.25)
        top_ax.set_title(label)
        top_ax.set_ylabel("Predicted mean")
        top_ax.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"levels={len(result['support_levels'])}",
                    f"sign_changes={result['metrics']['sign_changes']}",
                    f"secant_dev={result['metrics']['max_abs_secant_deviation_ratio']:.3f}",
                ]
            ),
            transform=top_ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )

        bot_ax.plot(result["x"], result["grad"], color="tab:blue", linewidth=2.0, label="Autograd gradient")
        bot_ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        for left, right, slope in zip(result["support_levels"][:-1], result["support_levels"][1:], result["secant_slopes"]):
            bot_ax.hlines(
                slope,
                left,
                right,
                color="black",
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
                label="Observed secant slope" if left == result["support_levels"][0] else None,
            )
        for level in result["support_levels"]:
            bot_ax.axvline(level, color="tab:orange", alpha=0.18, linewidth=3.0)
        bot_ax.grid(True, alpha=0.25)
        bot_ax.set_ylabel("d mu / d x")
        bot_ax.set_xlabel(AXIS_LABELS[axis_name])

        if col_idx == 0:
            top_ax.legend(loc="best")
            bot_ax.legend(loc="best")

    fig.suptitle(
        fig_title or f"{AXIS_LABELS[axis_name]}: dense MLP interpolation vs observed-level secants at t={time_ms_value:.2f} ms",
        fontsize=14,
    )
    return fig, pd.DataFrame(metrics_rows)


def collect_sparse_family_time_metrics(
    ctx: GradientStabilityContext,
    axis_name: str,
    slice_specs: Sequence[Mapping[str, Any]],
    raw_template: Mapping[str, float],
    analysis_times_ms: Sequence[float],
    *,
    expected_sign: float | None = None,
    n_points: int = 201,
) -> pd.DataFrame:
    rows = []
    for time_ms_value in analysis_times_ms:
        for spec in slice_specs:
            raw_point = dict(raw_template)
            raw_point.update(spec.get("overrides", {}))
            label = str(spec.get("label", spec.get("overrides", {})))
            result = evaluate_observed_vs_continuous_interpolant(
                ctx,
                axis_name,
                raw_point,
                time_ms_value=float(time_ms_value),
                expected_sign=expected_sign,
                n_points=n_points,
            )
            row = {"slice_label": label, "time_ms": float(time_ms_value)}
            row.update(result["metrics"])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_sparse_family_time_metrics(
    metrics_df: pd.DataFrame,
    axis_name: str,
    *,
    fig_title: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    metric_specs = [
        ("max_abs_secant_deviation_ratio", "Max secant deviation / support span", False),
        ("sign_changes", "Gradient sign changes", False),
        ("curvature_abs_p95", "p95 |d2 mu / d x2|", True),
    ]

    for label, group_df in metrics_df.groupby("slice_label"):
        group_df = group_df.sort_values("time_ms")
        for ax, (metric_col, ylabel, log_y) in zip(axes, metric_specs):
            ax.plot(group_df["time_ms"], group_df[metric_col], marker="o", linewidth=2.0, label=label)
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            if log_y:
                positive = group_df[metric_col].to_numpy(dtype=float)
                if np.all(positive > 0.0):
                    ax.set_yscale("log")

    axes[0].legend(loc="best")
    fig.suptitle(fig_title or f"{AXIS_LABELS[axis_name]} instability metrics across time", fontsize=14)
    return fig


def sweep_axis_diagnostics(
    ctx: GradientStabilityContext,
    axis_name: str,
    raw_point: Mapping[str, float],
    *,
    time_ms_value: float,
    expected_sign: float | None = None,
    n_points: int = 61,
) -> dict[str, Any]:
    stats = ctx.support["continuous"][axis_name]
    sweep_values = np.linspace(float(stats["min"]), float(stats["max"]), int(n_points), dtype=np.float64)

    mu_values = []
    std_values = []
    onset_values = []
    grad_values = []
    curvature_values = []
    has_onset = False

    for value in sweep_values:
        sample = dict(raw_point)
        sample[axis_name] = float(value)
        out = evaluate_point_with_derivatives(
            ctx,
            sample,
            time_ms_value=time_ms_value,
            axis_name=axis_name,
            second_order=True,
        )
        mu_values.append(out["mu"])
        std_values.append(out["std"])
        grad_values.append(out["grad"])
        curvature_values.append(out["curvature"])
        if "onset_prob" in out:
            has_onset = True
            onset_values.append(out["onset_prob"])

    mu_values = np.asarray(mu_values, dtype=float)
    std_values = np.asarray(std_values, dtype=float)
    grad_values = np.asarray(grad_values, dtype=float)
    curvature_values = np.asarray(curvature_values, dtype=float)
    onset_values_arr = np.asarray(onset_values, dtype=float) if has_onset else None

    support_levels = get_axis_support_levels(ctx, axis_name, raw_point)
    sign_changes = int(np.sum(np.signbit(grad_values[1:]) != np.signbit(grad_values[:-1])))
    grad_tv = float(np.sum(np.abs(np.diff(grad_values))))
    monotonicity_violation_rate = np.nan
    if expected_sign is not None:
        monotonicity_violation_rate = float(np.mean(expected_sign * grad_values < 0.0))

    return {
        "axis": axis_name,
        "x": sweep_values,
        "mu": mu_values,
        "std": std_values,
        "onset_prob": onset_values_arr,
        "grad": grad_values,
        "curvature": curvature_values,
        "support_levels": support_levels,
        "metrics": {
            "anchor_value": float(raw_point[axis_name]),
            "support_levels_on_slice": int(len(support_levels)),
            "sign_changes": sign_changes,
            "monotonicity_violation_rate": monotonicity_violation_rate,
            "grad_total_variation": grad_tv,
            "grad_total_variation_rel": float(grad_tv / (np.sum(np.abs(grad_values)) + 1e-12)),
            "grad_abs_p95": float(np.percentile(np.abs(grad_values), 95)),
            "curvature_abs_p95": float(np.percentile(np.abs(curvature_values), 95)),
            "curvature_abs_max": float(np.max(np.abs(curvature_values))),
            "mu_range": float(mu_values.max() - mu_values.min()),
            "std_range": float(std_values.max() - std_values.min()),
        },
    }


def current_time_metrics_dataframe(
    current_time_results: Mapping[str, Mapping[str, Any]],
    *,
    analysis_time_ms: float,
) -> pd.DataFrame:
    rows = []
    for axis_name, result in current_time_results.items():
        row = {"axis": axis_name, "time_ms": float(analysis_time_ms)}
        row.update(result["metrics"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("axis")


def collect_multi_time_metrics(
    ctx: GradientStabilityContext,
    analysis_axes: Sequence[str],
    raw_point: Mapping[str, float],
    analysis_times_ms: Sequence[float],
    expected_gradient_sign: Mapping[str, float | None],
    *,
    n_points: int = 61,
) -> pd.DataFrame:
    rows = []
    for time_ms_value in analysis_times_ms:
        for axis_name in analysis_axes:
            result = sweep_axis_diagnostics(
                ctx,
                axis_name,
                raw_point,
                time_ms_value=float(time_ms_value),
                expected_sign=expected_gradient_sign.get(axis_name),
                n_points=n_points,
            )
            row = {"axis": axis_name, "time_ms": float(time_ms_value)}
            row.update(result["metrics"])
            rows.append(row)
    return pd.DataFrame(rows)


def plot_1d_sweep_diagnostics(
    current_time_results: Mapping[str, Mapping[str, Any]],
    analysis_axes: Sequence[str],
    *,
    analysis_time_ms: float,
) -> plt.Figure:
    fig, axes = plt.subplots(3, len(analysis_axes), figsize=(5.6 * len(analysis_axes), 11.0), sharex="col")
    if len(analysis_axes) == 1:
        axes = np.asarray(axes).reshape(3, 1)

    row_specs = [
        ("mu", "Predicted mean", None),
        ("grad", "d mu / d x", 0.0),
        ("curvature", "d2 mu / d x2", 0.0),
    ]

    for col_idx, axis_name in enumerate(analysis_axes):
        result = current_time_results[axis_name]
        support_levels = result["support_levels"]
        title_color = "tab:red" if len(support_levels) == 0 else "black"

        for row_idx, (key, ylabel, zero_line) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            ax.plot(result["x"], result[key], color="tab:blue", linewidth=2.0)
            ax.axvline(result["metrics"]["anchor_value"], color="gray", linestyle=":", linewidth=1.2)
            if zero_line is not None:
                ax.axhline(zero_line, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.grid(True, alpha=0.25)

            if row_idx == 0:
                for level in support_levels:
                    ax.axvline(level, color="tab:orange", alpha=0.18, linewidth=3.0)
                ax.set_title(
                    f"{AXIS_LABELS[axis_name]}\n"
                    f"support={len(support_levels)}, sign_changes={result['metrics']['sign_changes']}",
                    color=title_color,
                )
                if len(support_levels) == 0:
                    ax.text(
                        0.02,
                        0.98,
                        "No observed support levels on this slice",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="#fff0f0",
                            edgecolor="#cc4444",
                            alpha=0.92,
                        ),
                    )

            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == len(row_specs) - 1:
                ax.set_xlabel(AXIS_LABELS[axis_name])

    fig.suptitle(f"1D sweep stability diagnostics at t={analysis_time_ms:.2f} ms", y=1.01, fontsize=14)
    fig.tight_layout()
    return fig


def pressure_coupling_heatmap(
    ctx: GradientStabilityContext,
    raw_point: Mapping[str, float],
    *,
    time_ms_value: float,
    n_points: int = 25,
) -> dict[str, np.ndarray]:
    pinj_stats = ctx.support["continuous"]["injection_pressure_bar"]
    pch_stats = ctx.support["continuous"]["chamber_pressure_bar"]
    pinj_values = np.linspace(float(pinj_stats["min"]), float(pinj_stats["max"]), int(n_points), dtype=np.float64)
    pch_values = np.linspace(float(pch_stats["min"]), float(pch_stats["max"]), int(n_points), dtype=np.float64)

    mu_grid = np.empty((len(pch_values), len(pinj_values)), dtype=float)
    grad_pinj_grid = np.empty_like(mu_grid)
    grad_pch_grid = np.empty_like(mu_grid)
    mixed_grid = np.empty_like(mu_grid)

    for row_idx, pch in enumerate(pch_values):
        for col_idx, pinj in enumerate(pinj_values):
            sample = dict(raw_point)
            sample["injection_pressure_bar"] = float(pinj)
            sample["chamber_pressure_bar"] = float(pch)

            out_pinj = evaluate_point_with_derivatives(
                ctx,
                sample,
                time_ms_value=time_ms_value,
                axis_name="injection_pressure_bar",
                coupled_axis_name="chamber_pressure_bar",
                second_order=True,
            )
            out_pch = evaluate_point_with_derivatives(
                ctx,
                sample,
                time_ms_value=time_ms_value,
                axis_name="chamber_pressure_bar",
                second_order=False,
            )

            mu_grid[row_idx, col_idx] = out_pinj["mu"]
            grad_pinj_grid[row_idx, col_idx] = out_pinj["grad"]
            grad_pch_grid[row_idx, col_idx] = out_pch["grad"]
            mixed_grid[row_idx, col_idx] = out_pinj["mixed_partial"]

    return {
        "pinj_values": pinj_values,
        "pch_values": pch_values,
        "mu": mu_grid,
        "grad_pinj": grad_pinj_grid,
        "grad_pch": grad_pch_grid,
        "mixed": mixed_grid,
    }


def _centered_norm(values: np.ndarray) -> mcolors.Normalize | None:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin < 0.0 < vmax:
        return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    return None


def plot_pressure_heatmap(
    pressure_heatmap: Mapping[str, np.ndarray],
    observed_pressure_points: np.ndarray,
    analysis_anchor_raw: Mapping[str, float],
    *,
    analysis_time_ms: float,
) -> plt.Figure:
    pinj_values = pressure_heatmap["pinj_values"]
    pch_values = pressure_heatmap["pch_values"]
    extent = [pinj_values.min(), pinj_values.max(), pch_values.min(), pch_values.max()]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    plot_specs = [
        ("mu", "Predicted mean penetration", "viridis", None),
        ("grad_pinj", "d mu / d Pinj", "coolwarm", _centered_norm(pressure_heatmap["grad_pinj"])),
        ("grad_pch", "d mu / d Pch", "coolwarm", _centered_norm(pressure_heatmap["grad_pch"])),
        ("mixed", "d2 mu / d Pinj d Pch", "coolwarm", _centered_norm(pressure_heatmap["mixed"])),
    ]

    for ax, (key, title, cmap, norm) in zip(axes.flat, plot_specs):
        image = ax.imshow(
            pressure_heatmap[key],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            norm=norm,
        )
        if len(observed_pressure_points) > 0:
            ax.scatter(
                observed_pressure_points[:, 0],
                observed_pressure_points[:, 1],
                s=28,
                facecolors="none",
                edgecolors="black",
                linewidths=1.0,
                label="Observed pressure triplets",
            )
        ax.scatter(
            [analysis_anchor_raw["injection_pressure_bar"]],
            [analysis_anchor_raw["chamber_pressure_bar"]],
            s=80,
            color="gold",
            edgecolors="black",
            linewidths=1.0,
            marker="*",
            label="Current anchor",
        )
        ax.set_title(title)
        ax.set_xlabel("Injection pressure [bar]")
        ax.set_ylabel("Chamber pressure [bar]")
        ax.grid(False)
        fig.colorbar(image, ax=ax, shrink=0.88)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="upper right")

    fig.suptitle(
        "Pressure-coupling stability map at "
        f"t={analysis_time_ms:.2f} ms, diameter={analysis_anchor_raw['diameter_mm']:.3f} mm",
        fontsize=14,
    )
    return fig
