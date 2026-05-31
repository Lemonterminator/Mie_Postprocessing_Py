"""Additive residual multi-task SVGP experiment.

Search terms for future archaeology:
RESIDUAL_SVGP, family routing, unknown family fallback, hot-start shared base.

Model form:
    y_scaled(x, family) = shared_svgp(x) + delta_svgp[family](x)

The family id is intentionally kept out of the kernel input vector.  It only
routes each point to a per-family residual GP.  If a family has no trained
delta model, prediction silently falls back to the shared SVGP, which is the
new-injector extensibility behavior this experiment is testing.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import gpytorch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.GP_training.run_gp_baseline import (  # noqa: E402
    DEFAULT_RUNS_ROOT,
    PointTensorDataset,
    SVGPModel,
    TrainResult,
    build_feature_specs,
    build_real_cdf_point_datasets,
    choose_device,
    concat_point_datasets,
    evaluate_dataset,
    finite_metrics,
    gaussian_nll_scaled,
    init_inducing_points,
    load_gp_artifacts,
    load_stage1_config,
    make_loader,
    make_point_dataset,
    maybe_limit_curves,
    predict_scaled,
    read_json,
    resolve_feature_spec,
    resolve_path,
    set_seed,
    to_jsonable,
    train_svgp,
    validate_a_scale,
    write_json,
)
from MLP.MLP_training.engineered_feature_common import (  # noqa: E402
    FEATURE_COLUMNS_BY_VARIANT,
    TIME_FEATURE,
    assign_splits_leave_one_out,
    build_dataset_registry,
)
from MLP.MLP_training import train_stage3_distillation_plus_raw_series as stage3_series  # noqa: E402


DEFAULT_MLP_BOOTSTRAP = (
    PROJECT_ROOT
    / "MLP"
    / "runs_mlp"
    / "full_pipeline_A_20260519_161129"
    / "bootstrap_summary.json"
)
DEFAULT_SHARED_CHECKPOINT = (
    PROJECT_ROOT
    / "MLP"
    / "runs_mlp"
    / "gp_baseline_stage3_20260521_112229"
    / "per_seed"
    / "seed_42"
    / "model.pt"
)


@dataclass
class ResidualSVGPArtifacts:
    """Runtime bundle for residual SVGP inference/checkpoint loading."""

    shared_model: SVGPModel
    shared_likelihood: gpytorch.likelihoods.GaussianLikelihood
    delta_models: dict[int, SVGPModel]
    delta_likelihoods: dict[int, gpytorch.likelihoods.GaussianLikelihood]
    var_model: SVGPModel | None
    var_likelihood: gpytorch.likelihoods.GaussianLikelihood | None
    config: dict[str, Any]
    scaler_state: dict[str, Any]
    device: torch.device


@dataclass
class SeedContext:
    seed: int
    stage1_run: Path
    scaler_state: dict[str, Any]
    feature_columns: list[str]
    feature_spec: dict[str, Any]
    datasets: dict[str, PointTensorDataset]
    synth_datasets: dict[str, PointTensorDataset]
    split_counts: dict[str, int]
    a_scale_check: dict[str, Any]
    real_cdf_info: dict[str, Any] | None


def _weight_tag(value: float) -> str:
    if float(value) == 0.0:
        return "0"
    text = f"{float(value):.0e}".replace("-", "m").replace("+", "")
    return text


def _freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def subset_point_dataset(dataset: PointTensorDataset, family_id: int | None = None) -> PointTensorDataset:
    if family_id is None:
        return dataset
    mask = dataset.family_id == int(family_id)
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    return PointTensorDataset(
        dataset.features[idx],
        dataset.target_scaled[idx],
        dataset.target_physical[idx],
        dataset.a_scale[idx],
        dataset.family_id[idx],
    )


def dataset_with_residual_target(
    *,
    shared_model: SVGPModel,
    dataset: PointTensorDataset,
    device: torch.device,
    batch_points: int,
) -> PointTensorDataset:
    """Build delta training targets: residual = y_scaled - shared_mu_scaled."""

    shared_pred = predict_scaled(shared_model, dataset, device=device, batch_points=batch_points)
    target = dataset.target_scaled.numpy() - shared_pred.astype(np.float32)
    return dataset.with_target_scaled(torch.as_tensor(target, dtype=torch.float32))


def _delta_l2_from_output(output: gpytorch.distributions.MultivariateNormal) -> torch.Tensor:
    return output.mean.reshape(-1).pow(2).mean()


def run_delta_epoch(
    *,
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    loader: DataLoader,
    mll: gpytorch.mlls.VariationalELBO,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    delta_l2_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    likelihood.train(mode=is_train)
    total_loss = 0.0
    total_elbo = 0.0
    total_l2 = 0.0
    total_points = 0
    for batch in loader:
        x = batch["features"].to(device, non_blocking=True)
        y = batch["target_scaled"].to(device, non_blocking=True).reshape(-1)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        output = model(x)
        elbo_loss = -mll(output, y)
        delta_l2 = _delta_l2_from_output(output)
        loss = elbo_loss + float(delta_l2_weight) * delta_l2
        if is_train:
            loss.backward()
            optimizer.step()
        batch_n = int(y.numel())
        total_loss += float(loss.detach().cpu()) * batch_n
        total_elbo += float(elbo_loss.detach().cpu()) * batch_n
        total_l2 += float(delta_l2.detach().cpu()) * batch_n
        total_points += batch_n
    denom = max(total_points, 1)
    return {
        "loss": total_loss / denom,
        "elbo_loss": total_elbo / denom,
        "delta_l2": total_l2 / denom,
    }


def train_delta_svgp(
    *,
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_data: int,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    min_delta: float,
    log_interval: int,
    label: str,
    delta_l2_weight: float,
) -> TrainResult:
    model.to(device)
    likelihood.to(device)
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=float(lr),
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=int(num_data))
    history: list[dict[str, float | int | str]] = []
    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, int(epochs) + 1):
        train_metrics = run_delta_epoch(
            model=model,
            likelihood=likelihood,
            loader=train_loader,
            mll=mll,
            device=device,
            optimizer=optimizer,
            delta_l2_weight=delta_l2_weight,
        )
        with torch.no_grad():
            val_metrics = run_delta_epoch(
                model=model,
                likelihood=likelihood,
                loader=val_loader,
                mll=mll,
                device=device,
                optimizer=None,
                delta_l2_weight=delta_l2_weight,
            )
        history.append({"model": label, "epoch": epoch, "split": "train", **train_metrics})
        history.append({"model": label, "epoch": epoch, "split": "val", **val_metrics})
        val_loss = float(val_metrics["loss"])
        if val_loss < best_val - float(min_delta):
            best_val = val_loss
            best_epoch = int(epoch)
            no_improve = 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "likelihood": copy.deepcopy(likelihood.state_dict()),
            }
        else:
            no_improve += 1

        if epoch == 1 or epoch % int(log_interval) == 0 or no_improve >= int(patience):
            print(
                f"[{label}] epoch {epoch:03d}/{int(epochs)} "
                f"train_loss={train_metrics['loss']:.6f} val_loss={val_loss:.6f} "
                f"val_delta_l2={val_metrics['delta_l2']:.6f} "
                f"best={best_val:.6f}@{best_epoch} no_improve={no_improve}/{int(patience)}"
            )
        if no_improve >= int(patience):
            print(f"[{label}] early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        likelihood.load_state_dict(best_state["likelihood"])

    return TrainResult(
        model=model,
        likelihood=likelihood,
        history=pd.DataFrame(history),
        best_val_loss=float(best_val),
        best_epoch=int(best_epoch),
    )


def predict_residual_scaled(
    artifacts: ResidualSVGPArtifacts,
    features: np.ndarray | torch.Tensor,
    family_id: np.ndarray | torch.Tensor | None,
    *,
    batch_points: int,
    include_mean_posterior_var: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict shared + routed family residual in scaled penetration space.

    Unknown or untrained families are deliberately absent from ``delta_models``;
    those samples keep ``mean_scaled = shared_mu``.  This is the key deployment
    fallback for a future injector family before collecting calibration data.
    """

    artifacts.shared_model.eval()
    for model in artifacts.delta_models.values():
        model.eval()
    features_t = torch.as_tensor(features, dtype=torch.float32).detach().cpu()
    n = len(features_t)
    if family_id is None:
        family_t = torch.zeros(n, dtype=torch.long)
    else:
        family_t = torch.as_tensor(family_id, dtype=torch.long).reshape(-1).detach().cpu()
    if len(family_t) != n:
        raise ValueError(f"family_id length {len(family_t)} does not match feature length {n}.")

    mean_chunks: list[np.ndarray] = []
    var_chunks: list[np.ndarray] = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, n, int(batch_points)):
            stop = min(start + int(batch_points), n)
            x = features_t[start:stop].to(artifacts.device, non_blocking=True)
            fam = family_t[start:stop].to(artifacts.device, non_blocking=True)
            shared_dist = artifacts.shared_model(x)
            mean_scaled = shared_dist.mean.reshape(-1)
            var_scaled = torch.clamp(shared_dist.variance.reshape(-1), min=0.0) if include_mean_posterior_var else None
            for fid, delta_model in artifacts.delta_models.items():
                mask = fam == int(fid)
                if not bool(mask.any()):
                    continue
                delta_dist = delta_model(x[mask])
                mean_scaled = mean_scaled.clone()
                mean_scaled[mask] = mean_scaled[mask] + delta_dist.mean.reshape(-1)
                if include_mean_posterior_var and var_scaled is not None:
                    var_scaled = var_scaled.clone()
                    var_scaled[mask] = var_scaled[mask] + torch.clamp(delta_dist.variance.reshape(-1), min=0.0)
            mean_chunks.append(mean_scaled.detach().cpu().numpy())
            if include_mean_posterior_var and var_scaled is not None:
                var_chunks.append(var_scaled.detach().cpu().numpy())
    mean_np = np.concatenate(mean_chunks) if mean_chunks else np.asarray([], dtype=np.float32)
    var_np = np.concatenate(var_chunks) if var_chunks else None
    return mean_np, var_np


def predict_residual_physical(
    artifacts: ResidualSVGPArtifacts,
    features: np.ndarray | torch.Tensor,
    a_scale: np.ndarray | torch.Tensor,
    family_id: np.ndarray | torch.Tensor | None,
    *,
    batch_points: int,
    include_mean_posterior_var: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict physical penetration and aleatoric uncertainty.

    Mean uncertainty defaults to the learned two-stage log-variance model so it
    stays comparable with the existing single-output SVGP artifact.  Posterior
    mean variance is opt-in via ``include_mean_posterior_var``.
    """

    if artifacts.var_model is not None:
        artifacts.var_model.eval()
    features_t = torch.as_tensor(features, dtype=torch.float32).detach().cpu()
    a_scale_np = torch.as_tensor(a_scale, dtype=torch.float32).reshape(-1).detach().cpu().numpy()
    mean_scaled, posterior_var_scaled = predict_residual_scaled(
        artifacts,
        features_t,
        family_id,
        batch_points=batch_points,
        include_mean_posterior_var=include_mean_posterior_var,
    )

    logvar_min = float(artifacts.config.get("logvar_clamp_min", -20.0))
    logvar_max = float(artifacts.config.get("logvar_clamp_max", 10.0))
    std_floor_scaled = float(artifacts.config.get("std_floor_scaled", 1e-6))
    var_chunks: list[np.ndarray] = []
    if artifacts.var_model is not None:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for start in range(0, len(features_t), int(batch_points)):
                stop = min(start + int(batch_points), len(features_t))
                x = features_t[start:stop].to(artifacts.device, non_blocking=True)
                log_var = artifacts.var_model(x).mean.reshape(-1)
                log_var = log_var + float(artifacts.config.get("logvar_bias_correction", 0.0))
                var = torch.exp(torch.clamp(log_var, min=logvar_min, max=logvar_max))
                var_chunks.append(var.detach().cpu().numpy())
        var_scaled = np.concatenate(var_chunks) if var_chunks else np.asarray([], dtype=np.float32)
        if include_mean_posterior_var and posterior_var_scaled is not None:
            var_scaled = var_scaled + np.maximum(posterior_var_scaled, 0.0)
    else:
        var_scaled = (
            np.maximum(posterior_var_scaled, 0.0)
            if posterior_var_scaled is not None
            else np.full_like(mean_scaled, std_floor_scaled**2)
        )
        noise = getattr(artifacts.shared_likelihood, "noise", None)
        if noise is not None:
            var_scaled = var_scaled + float(torch.clamp(noise.reshape(-1)[0].detach().cpu(), min=0.0))

    std_scaled = np.sqrt(np.maximum(var_scaled, std_floor_scaled**2))
    mean_physical = mean_scaled * a_scale_np
    std_physical = std_scaled * a_scale_np
    return mean_physical, std_physical, mean_scaled, std_scaled


def residual_log_targets(
    *,
    artifacts: ResidualSVGPArtifacts,
    dataset: PointTensorDataset,
    batch_points: int,
    residual_eps: float,
) -> torch.Tensor:
    pred_scaled, _ = predict_residual_scaled(
        artifacts,
        dataset.features,
        dataset.family_id,
        batch_points=batch_points,
        include_mean_posterior_var=False,
    )
    residual_sq = (dataset.target_scaled.numpy() - pred_scaled) ** 2
    return torch.as_tensor(np.log(residual_sq + float(residual_eps)), dtype=torch.float32)


def evaluate_residual_dataset(
    artifacts: ResidualSVGPArtifacts,
    dataset: PointTensorDataset,
    *,
    batch_points: int,
    include_mean_posterior_var: bool,
) -> dict[str, float | int]:
    pred, std, pred_scaled, std_scaled = predict_residual_physical(
        artifacts,
        dataset.features,
        dataset.a_scale,
        dataset.family_id,
        batch_points=batch_points,
        include_mean_posterior_var=include_mean_posterior_var,
    )
    truth = dataset.target_physical.numpy()
    metrics = finite_metrics(truth, pred, std)
    metrics["nll_scaled"] = gaussian_nll_scaled(dataset.target_scaled.numpy(), pred_scaled, std_scaled)
    return metrics


def estimate_residual_logvar_bias_correction(
    *,
    artifacts: ResidualSVGPArtifacts,
    var_dataset: PointTensorDataset,
    batch_points: int,
) -> float:
    if artifacts.var_model is None:
        return 0.0
    pred_logvar = predict_scaled(
        artifacts.var_model,
        var_dataset,
        device=artifacts.device,
        batch_points=batch_points,
    )
    target_logvar = var_dataset.target_scaled.numpy()
    ratio = np.exp(np.clip(target_logvar - pred_logvar, -30.0, 30.0))
    finite = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    if finite.size == 0:
        return 0.0
    return float(np.log(np.mean(finite)))


def save_residual_checkpoint(
    path: Path,
    *,
    artifacts: ResidualSVGPArtifacts,
    seed: int,
    stage1_run: Path,
    train_history: pd.DataFrame,
) -> None:
    """Persist all pieces needed for deployed residual SVGP inference."""

    payload = {
        "method": artifacts.config["method"],
        "seed": int(seed),
        "stage1_run": str(stage1_run),
        "config": artifacts.config,
        "scaler_state": artifacts.scaler_state,
        "shared_model_state": artifacts.shared_model.state_dict(),
        "shared_likelihood_state": artifacts.shared_likelihood.state_dict(),
        "shared_inducing_points": artifacts.shared_model.variational_strategy.inducing_points.detach().cpu(),
        "delta_model_states": {str(k): v.state_dict() for k, v in artifacts.delta_models.items()},
        "delta_likelihood_states": {str(k): v.state_dict() for k, v in artifacts.delta_likelihoods.items()},
        "delta_inducing_points": {
            str(k): v.variational_strategy.inducing_points.detach().cpu()
            for k, v in artifacts.delta_models.items()
        },
        "var_model_state": artifacts.var_model.state_dict() if artifacts.var_model is not None else None,
        "var_likelihood_state": artifacts.var_likelihood.state_dict() if artifacts.var_likelihood is not None else None,
        "var_inducing_points": (
            artifacts.var_model.variational_strategy.inducing_points.detach().cpu()
            if artifacts.var_model is not None
            else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    train_history.to_csv(path.parent / "train_log.csv", index=False)


def load_residual_svgp_artifacts(checkpoint_path: Path | str, *, device: torch.device) -> ResidualSVGPArtifacts:
    """Load a residual SVGP checkpoint produced by this experiment."""

    payload = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    config = dict(payload["config"])
    feature_columns = list(config["feature_columns"])
    shared_model = SVGPModel(payload["shared_inducing_points"].float(), num_dims=len(feature_columns))
    shared_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    shared_model.load_state_dict(payload["shared_model_state"])
    shared_likelihood.load_state_dict(payload["shared_likelihood_state"])

    delta_models: dict[int, SVGPModel] = {}
    delta_likelihoods: dict[int, gpytorch.likelihoods.GaussianLikelihood] = {}
    for key, state in dict(payload.get("delta_model_states", {})).items():
        fid = int(key)
        inducing = payload["delta_inducing_points"][str(fid)].float()
        model = SVGPModel(inducing, num_dims=len(feature_columns), mean_type="zero")
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.load_state_dict(state)
        likelihood.load_state_dict(payload["delta_likelihood_states"][str(fid)])
        delta_models[fid] = model
        delta_likelihoods[fid] = likelihood

    var_model = None
    var_likelihood = None
    if payload.get("var_model_state") is not None and payload.get("var_inducing_points") is not None:
        var_model = SVGPModel(payload["var_inducing_points"].float(), num_dims=len(feature_columns))
        var_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        var_model.load_state_dict(payload["var_model_state"])
        var_likelihood.load_state_dict(payload["var_likelihood_state"])

    shared_model.to(device).eval()
    shared_likelihood.to(device).eval()
    for model in delta_models.values():
        model.to(device).eval()
    for likelihood in delta_likelihoods.values():
        likelihood.to(device).eval()
    if var_model is not None:
        var_model.to(device).eval()
    if var_likelihood is not None:
        var_likelihood.to(device).eval()

    return ResidualSVGPArtifacts(
        shared_model=shared_model,
        shared_likelihood=shared_likelihood,
        delta_models=delta_models,
        delta_likelihoods=delta_likelihoods,
        var_model=var_model,
        var_likelihood=var_likelihood,
        config=config,
        scaler_state=dict(payload["scaler_state"]),
        device=device,
    )


def expand_residual_svgp_family(
    checkpoint_path: Path | str,
    *,
    new_family_id: int,
    output_path: Path | str | None = None,
) -> Path:
    """Register a new untrained family id without creating a delta GP.

    Because ``predict_residual_scaled`` only applies residuals for keys present
    in ``delta_models``, expanded families naturally predict with shared_mu.
    """

    path = resolve_path(checkpoint_path)
    payload = torch.load(path, map_location="cpu")
    config = dict(payload["config"])
    family_ids = {int(fid) for fid in config.get("family_ids", [])}
    family_ids.add(int(new_family_id))
    config["family_ids"] = sorted(family_ids)
    config["expanded_untrained_family_ids"] = sorted(
        {int(fid) for fid in config.get("expanded_untrained_family_ids", [])} | {int(new_family_id)}
    )
    payload["config"] = config
    out_path = resolve_path(output_path) if output_path is not None else path
    torch.save(payload, out_path)
    return out_path


def prepare_seed_context(
    *,
    seed_info: Mapping[str, Any],
    base_config: Mapping[str, Any],
    device: torch.device,
) -> SeedContext:
    seed = int(seed_info["seed"])
    set_seed(seed)
    stage1_run = resolve_path(seed_info["stage1_run"])
    row_table_path = stage1_run / "row_table.csv"
    if not row_table_path.exists():
        raise FileNotFoundError(f"Missing row_table.csv for seed {seed}: {row_table_path}")
    scaler_state_path = stage1_run / "scaler_state.json"
    if not scaler_state_path.exists():
        raise FileNotFoundError(f"Missing scaler_state.json: {scaler_state_path}")
    stage1_config = load_stage1_config(stage1_run)
    scaler_state = read_json(scaler_state_path)
    a_scale_delta_pressure_exp = float(
        (scaler_state.get("target", {}) if isinstance(scaler_state, Mapping) else {}).get(
            "a_scale_delta_pressure_exp",
            stage1_config.get("a_scale_delta_pressure_exp", 0.5),
        )
    )

    df = pd.read_csv(row_table_path)
    df = maybe_limit_curves(df, max_curves_per_split=base_config.get("max_curves_per_split"), seed=seed)
    a_scale_check = validate_a_scale(df, delta_pressure_exp=a_scale_delta_pressure_exp)
    if a_scale_check.get("checked") and not a_scale_check.get("passed") and not base_config.get("allow_a_scale_mismatch", False):
        raise ValueError(
            f"A_scale mismatch in {row_table_path}; max relative error "
            f"{a_scale_check['max_relative_error']:.3e}."
        )

    mode = str(base_config.get("mode", "stage3"))
    lono_holdout = base_config.get("lono_holdout")
    lono_holdout = str(lono_holdout) if lono_holdout else None
    val_ratio = float(base_config.get("val_ratio", 0.15))
    if lono_holdout is not None:
        df = assign_splits_leave_one_out(df, holdout_value=lono_holdout, val_ratio=val_ratio, seed=seed)

    feature_spec = resolve_feature_spec(
        stage1_config=stage1_config,
        requested_variant=base_config.get("requested_variant"),
        stage1_run=stage1_run,
        seed=seed,
    )
    feature_columns = list(feature_spec["feature_columns"])
    n_points = int(base_config["n_points"])
    time_min_ms = float(base_config["time_min_ms"])
    time_max_ms = float(base_config["time_max_ms"])
    split_counts = {str(k): int(v) for k, v in df["sample_split"].value_counts().to_dict().items()}

    synth_datasets = {
        split: make_point_dataset(
            df.loc[df["sample_split"] == split],
            feature_columns=feature_columns,
            n_points=n_points,
            time_min_ms=time_min_ms,
            time_max_ms=time_max_ms,
        )
        for split in ("train", "val", "test")
    }

    real_cdf_info: dict[str, Any] | None = None
    if mode == "stage3":
        real_train_ds, real_val_ds, real_cdf_info = build_real_cdf_point_datasets(
            feature_columns=feature_columns,
            scaler_state=scaler_state,
            registry=build_dataset_registry(),
            time_min_ms=time_min_ms,
            time_max_ms=time_max_ms,
            val_ratio=val_ratio,
            seed=seed,
            lono_holdout=lono_holdout,
            cdf_split=str(base_config.get("cdf_split", "clean")),
            max_rows=base_config.get("max_real_cdf_rows"),
        )
        datasets = {
            "train": concat_point_datasets(synth_datasets["train"], real_train_ds),
            "val": concat_point_datasets(synth_datasets["val"], real_val_ds),
            "test": synth_datasets["test"],
        }
    else:
        datasets = synth_datasets

    return SeedContext(
        seed=seed,
        stage1_run=stage1_run,
        scaler_state=dict(scaler_state),
        feature_columns=feature_columns,
        feature_spec=feature_spec,
        datasets=datasets,
        synth_datasets=synth_datasets,
        split_counts=split_counts,
        a_scale_check=a_scale_check,
        real_cdf_info=real_cdf_info,
    )


def _load_existing_shared(
    *,
    checkpoint_path: Path,
    device: torch.device,
    feature_columns: Sequence[str],
) -> tuple[SVGPModel, gpytorch.likelihoods.GaussianLikelihood, dict[str, Any]]:
    source = load_gp_artifacts(checkpoint_path, device=device)
    source_columns = list(source.config["feature_columns"])
    if source_columns != list(feature_columns):
        raise ValueError(
            f"Shared checkpoint feature columns {source_columns} do not match requested {list(feature_columns)}."
        )
    _freeze_module(source.mean_model)
    _freeze_module(source.mean_likelihood)
    return source.mean_model, source.mean_likelihood, dict(source.config)


def train_one_seed(
    *,
    seed_info: Mapping[str, Any],
    run_dir: Path,
    base_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Train one hot-started residual SVGP seed.

    Default policy uses the existing full Stage-3 SVGP as a frozen shared base.
    Only the per-family delta GPs and the final log-variance GP are trained.
    """

    ctx = prepare_seed_context(seed_info=seed_info, base_config=base_config, device=device)
    seed_dir = run_dir / "per_seed" / f"seed_{ctx.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Residual SVGP seed {ctx.seed} ---")
    print("Stage-1 run:", ctx.stage1_run)
    print("Feature columns:", ctx.feature_columns)
    print("Synthetic curve split counts:", ctx.split_counts)
    print("Combined point counts:", {split: len(ds) for split, ds in ctx.datasets.items()})
    print("Family counts train:", pd.Series(ctx.datasets["train"].family_id.numpy()).value_counts().sort_index().to_dict())

    histories: list[pd.DataFrame] = []
    source_config: dict[str, Any] | None = None
    shared_best_epoch: int | None = None
    shared_best_val_loss: float | None = None
    shared_base = str(base_config.get("shared_base", "existing_full_svgp"))
    if shared_base == "existing_full_svgp":
        shared_checkpoint = resolve_path(base_config["shared_checkpoint"])
        shared_model, shared_likelihood, source_config = _load_existing_shared(
            checkpoint_path=shared_checkpoint,
            device=device,
            feature_columns=ctx.feature_columns,
        )
    elif shared_base == "modified_only_shared":
        shared_train = subset_point_dataset(ctx.datasets["train"], 1)
        shared_val = subset_point_dataset(ctx.datasets["val"], 1)
        inducing = init_inducing_points(
            shared_train,
            num_inducing=int(base_config["num_inducing"]),
            kmeans_samples=int(base_config["kmeans_samples"]),
            seed=ctx.seed,
        ).to(device)
        shared_model = SVGPModel(inducing.clone(), num_dims=len(ctx.feature_columns))
        shared_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        shared_result = train_svgp(
            model=shared_model,
            likelihood=shared_likelihood,
            train_loader=make_loader(
                shared_train,
                batch_points=int(base_config["batch_points"]),
                shuffle=True,
                seed=ctx.seed,
                num_workers=int(base_config["num_workers"]),
                device=device,
            ),
            val_loader=make_loader(
                shared_val if len(shared_val) else shared_train,
                batch_points=int(base_config["eval_batch_points"]),
                shuffle=False,
                seed=ctx.seed,
                num_workers=int(base_config["num_workers"]),
                device=device,
            ),
            num_data=len(shared_train),
            device=device,
            epochs=int(base_config["epochs"]),
            lr=float(base_config["learning_rate"]),
            patience=int(base_config["early_stopping_patience"]),
            min_delta=float(base_config["early_stopping_min_delta"]),
            log_interval=int(base_config["log_interval"]),
            label="shared_family1",
        )
        shared_model = shared_result.model
        shared_likelihood = shared_result.likelihood
        shared_best_epoch = int(shared_result.best_epoch)
        shared_best_val_loss = float(shared_result.best_val_loss)
        histories.append(shared_result.history)
        _freeze_module(shared_model)
        _freeze_module(shared_likelihood)
    else:
        raise ValueError(f"Unsupported shared_base={shared_base!r}.")

    delta_models: dict[int, SVGPModel] = {}
    delta_likelihoods: dict[int, gpytorch.likelihoods.GaussianLikelihood] = {}
    best_epochs: dict[str, int | None] = {"shared": shared_best_epoch}
    best_val_losses: dict[str, float | None] = {"shared": shared_best_val_loss}
    trained_family_ids: list[int] = []
    for family_id in [int(fid) for fid in base_config.get("family_ids", [0, 1])]:
        family_train = subset_point_dataset(ctx.datasets["train"], family_id)
        if len(family_train) == 0:
            print(f"[delta_{family_id}] no training rows; unknown-family fallback will use shared only.")
            continue
        family_val = subset_point_dataset(ctx.datasets["val"], family_id)
        family_train_resid = dataset_with_residual_target(
            shared_model=shared_model,
            dataset=family_train,
            device=device,
            batch_points=int(base_config["eval_batch_points"]),
        )
        family_val_resid = dataset_with_residual_target(
            shared_model=shared_model,
            dataset=family_val if len(family_val) else family_train,
            device=device,
            batch_points=int(base_config["eval_batch_points"]),
        )
        inducing = init_inducing_points(
            family_train_resid,
            num_inducing=int(base_config["num_inducing"]),
            kmeans_samples=int(base_config["kmeans_samples"]),
            seed=ctx.seed + 17 * (family_id + 1),
        ).to(device)
        delta_model = SVGPModel(inducing.clone(), num_dims=len(ctx.feature_columns), mean_type="zero")
        delta_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        result = train_delta_svgp(
            model=delta_model,
            likelihood=delta_likelihood,
            train_loader=make_loader(
                family_train_resid,
                batch_points=int(base_config["batch_points"]),
                shuffle=True,
                seed=ctx.seed + 97 * (family_id + 1),
                num_workers=int(base_config["num_workers"]),
                device=device,
            ),
            val_loader=make_loader(
                family_val_resid,
                batch_points=int(base_config["eval_batch_points"]),
                shuffle=False,
                seed=ctx.seed + 97 * (family_id + 1),
                num_workers=int(base_config["num_workers"]),
                device=device,
            ),
            num_data=len(family_train_resid),
            device=device,
            epochs=int(base_config["delta_epochs"]),
            lr=float(base_config["delta_learning_rate"]),
            patience=int(base_config["early_stopping_patience"]),
            min_delta=float(base_config["early_stopping_min_delta"]),
            log_interval=int(base_config["log_interval"]),
            label=f"delta_{family_id}",
            delta_l2_weight=float(base_config["delta_l2_weight"]),
        )
        delta_models[family_id] = result.model
        delta_likelihoods[family_id] = result.likelihood
        histories.append(result.history)
        best_epochs[f"delta_{family_id}"] = result.best_epoch
        best_val_losses[f"delta_{family_id}"] = result.best_val_loss
        trained_family_ids.append(family_id)

    seed_config = {
        **dict(base_config),
        "seed": ctx.seed,
        "stage1_run": str(ctx.stage1_run),
        "variant": ctx.feature_spec["variant"],
        "feature_columns": ctx.feature_columns,
        "feature_source": ctx.feature_spec["feature_source"],
        "input_dim": len(ctx.feature_columns),
        "stage1_variant": ctx.feature_spec["stage1_variant"],
        "split_curve_counts": ctx.split_counts,
        "a_scale_check": ctx.a_scale_check,
        "mode": str(base_config.get("mode", "stage3")),
        "lono_holdout": base_config.get("lono_holdout"),
        "real_cdf_info": ctx.real_cdf_info,
        "synth_point_counts": {split: int(len(ds)) for split, ds in ctx.synth_datasets.items()},
        "combined_point_counts": {split: int(len(ds)) for split, ds in ctx.datasets.items()},
        "trained_family_ids": trained_family_ids,
        "family_mapping": {"0": "Nozzle0", "1": "Nozzle1-8"},
        "source_shared_config": source_config,
        "method": "residual_multitask_svgp",
    }
    artifacts = ResidualSVGPArtifacts(
        shared_model=shared_model,
        shared_likelihood=shared_likelihood,
        delta_models=delta_models,
        delta_likelihoods=delta_likelihoods,
        var_model=None,
        var_likelihood=None,
        config=seed_config,
        scaler_state=ctx.scaler_state,
        device=device,
    )

    var_result: TrainResult | None = None
    logvar_bias_correction = 0.0
    if not bool(base_config.get("mean_only", False)):
        print("Computing residual log-variance targets...")
        train_log_resid = residual_log_targets(
            artifacts=artifacts,
            dataset=ctx.datasets["train"],
            batch_points=int(base_config["eval_batch_points"]),
            residual_eps=float(base_config["residual_eps"]),
        )
        val_log_resid = residual_log_targets(
            artifacts=artifacts,
            dataset=ctx.datasets["val"],
            batch_points=int(base_config["eval_batch_points"]),
            residual_eps=float(base_config["residual_eps"]),
        )
        var_train_ds = ctx.datasets["train"].with_target_scaled(train_log_resid)
        var_val_ds = ctx.datasets["val"].with_target_scaled(val_log_resid)
        inducing = init_inducing_points(
            var_train_ds,
            num_inducing=int(base_config["num_inducing"]),
            kmeans_samples=int(base_config["kmeans_samples"]),
            seed=ctx.seed + 1009,
        ).to(device)
        var_model = SVGPModel(inducing.clone(), num_dims=len(ctx.feature_columns))
        var_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        var_result = train_svgp(
            model=var_model,
            likelihood=var_likelihood,
            train_loader=make_loader(
                var_train_ds,
                batch_points=int(base_config["batch_points"]),
                shuffle=True,
                seed=ctx.seed + 1009,
                num_workers=int(base_config["num_workers"]),
                device=device,
            ),
            val_loader=make_loader(
                var_val_ds,
                batch_points=int(base_config["eval_batch_points"]),
                shuffle=False,
                seed=ctx.seed + 1009,
                num_workers=int(base_config["num_workers"]),
                device=device,
            ),
            num_data=len(var_train_ds),
            device=device,
            epochs=int(base_config["var_epochs"]),
            lr=float(base_config["var_learning_rate"]),
            patience=int(base_config["early_stopping_patience"]),
            min_delta=float(base_config["early_stopping_min_delta"]),
            log_interval=int(base_config["log_interval"]),
            label="logvar",
        )
        artifacts.var_model = var_result.model
        artifacts.var_likelihood = var_result.likelihood
        if bool(base_config.get("calibrate_logvar_bias", True)):
            logvar_bias_correction = estimate_residual_logvar_bias_correction(
                artifacts=artifacts,
                var_dataset=var_val_ds,
                batch_points=int(base_config["eval_batch_points"]),
            )
            print(f"Validation log-variance bias correction: {logvar_bias_correction:.6f}")
            artifacts.config["logvar_bias_correction"] = float(logvar_bias_correction)
        histories.append(var_result.history)
        best_epochs["logvar"] = var_result.best_epoch
        best_val_losses["logvar"] = var_result.best_val_loss
    else:
        artifacts.config["logvar_bias_correction"] = 0.0

    print("Evaluating residual SVGP on held-out in-pipeline test split...")
    test_metrics = evaluate_residual_dataset(
        artifacts,
        ctx.datasets["test"],
        batch_points=int(base_config["eval_batch_points"]),
        include_mean_posterior_var=bool(base_config["include_mean_posterior_var"]),
    )
    val_metrics = evaluate_residual_dataset(
        artifacts,
        ctx.datasets["val"],
        batch_points=int(base_config["eval_batch_points"]),
        include_mean_posterior_var=bool(base_config["include_mean_posterior_var"]),
    )
    train_history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    checkpoint_path = seed_dir / "model.pt"
    save_residual_checkpoint(
        checkpoint_path,
        artifacts=artifacts,
        seed=ctx.seed,
        stage1_run=ctx.stage1_run,
        train_history=train_history,
    )
    write_json(seed_dir / "test_metrics.json", test_metrics)
    write_json(seed_dir / "val_metrics.json", val_metrics)
    write_json(seed_dir / "seed_config.json", artifacts.config)
    write_json(seed_dir / "scaler_state.json", ctx.scaler_state)
    result = {
        "seed": ctx.seed,
        "stage1_run": str(ctx.stage1_run),
        "checkpoint": str(checkpoint_path),
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "best_epochs": best_epochs,
        "best_val_losses": best_val_losses,
        "trained_family_ids": trained_family_ids,
    }
    print("Test metrics:", json.dumps(to_jsonable(test_metrics), indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stage2", "stage3"), default="stage3")
    parser.add_argument("--mlp-bootstrap", type=Path, default=DEFAULT_MLP_BOOTSTRAP)
    parser.add_argument("--synthetic-root", type=Path, default=None,
                        help="Synthetic-data root used for real CDF Stage-3 training rows.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=tuple(FEATURE_COLUMNS_BY_VARIANT.keys()), default=None)
    parser.add_argument("--shared-base", choices=("existing_full_svgp", "modified_only_shared"), default="existing_full_svgp")
    parser.add_argument("--shared-checkpoint", type=Path, default=DEFAULT_SHARED_CHECKPOINT)
    parser.add_argument("--family-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--delta-l2-weight", type=float, default=1e-4)
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--batch-points", type=int, default=1024)
    parser.add_argument("--eval-batch-points", type=int, default=65536)
    parser.add_argument("--num-inducing", type=int, default=256)
    parser.add_argument("--kmeans-samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--delta-epochs", type=int, default=60)
    parser.add_argument("--var-epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--delta-learning-rate", type=float, default=1e-2)
    parser.add_argument("--var-learning-rate", type=float, default=1e-2)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--time-min-ms", type=float, default=0.0)
    parser.add_argument("--time-max-ms", type=float, default=5.0)
    parser.add_argument("--residual-eps", type=float, default=1e-6)
    parser.add_argument("--std-floor-scaled", type=float, default=1e-6)
    parser.add_argument("--logvar-clamp-min", type=float, default=-20.0)
    parser.add_argument("--logvar-clamp-max", type=float, default=10.0)
    parser.add_argument("--mean-only", action="store_true")
    parser.add_argument("--include-mean-posterior-var", action="store_true")
    parser.add_argument("--no-logvar-bias-calibration", action="store_true")
    parser.add_argument("--max-curves-per-split", type=int, default=None)
    parser.add_argument("--max-real-cdf-rows", type=int, default=None)
    parser.add_argument("--allow-a-scale-mismatch", action="store_true")
    parser.add_argument("--cdf-split", choices=("clean", "all", "truncated"), default="clean")
    parser.add_argument("--lono-holdout", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--run-name-prefix", default="residual_multitask_svgp")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic_root = resolve_path(args.synthetic_root) if args.synthetic_root is not None else None
    if synthetic_root is not None:
        stage3_series.SYNTHETIC_ROOT = synthetic_root
    device = choose_device(args.device)
    mlp_bootstrap = resolve_path(args.mlp_bootstrap)
    bootstrap_data = read_json(mlp_bootstrap)
    per_seed_manifest = [rec for rec in list(bootstrap_data["per_seed"]) if int(rec["seed"]) == int(args.seed)]
    if not per_seed_manifest:
        raise ValueError(f"Seed {args.seed} not found in {mlp_bootstrap}.")
    feature_specs = build_feature_specs(per_seed_manifest, requested_variant=args.variant)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = resolve_path(args.runs_root)
    l2_tag = _weight_tag(float(args.delta_l2_weight))
    out_dir = runs_root / f"{args.run_name_prefix}_{args.shared_base}_l2_{l2_tag}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = {
        "method": "residual_multitask_svgp",
        "mode": str(args.mode),
        "lono_holdout": args.lono_holdout,
        "val_ratio": float(args.val_ratio),
        "cdf_split": str(args.cdf_split),
        "mlp_bootstrap": str(mlp_bootstrap),
        "synthetic_root": None if synthetic_root is None else str(synthetic_root),
        "requested_variant": args.variant,
        "feature_source": "explicit_variant_validated_against_stage1" if args.variant is not None else "stage1_train_config",
        "feature_spec_by_seed": feature_specs,
        "time_feature": TIME_FEATURE,
        "shared_base": str(args.shared_base),
        "shared_checkpoint": str(resolve_path(args.shared_checkpoint)),
        "family_split": "family_id",
        "family_ids": [int(fid) for fid in args.family_ids],
        "delta_l2_weight": float(args.delta_l2_weight),
        "n_points": int(args.n_points),
        "batch_points": int(args.batch_points),
        "eval_batch_points": int(args.eval_batch_points),
        "num_inducing": int(args.num_inducing),
        "kmeans_samples": int(args.kmeans_samples),
        "epochs": int(args.epochs),
        "delta_epochs": int(args.delta_epochs),
        "var_epochs": int(args.var_epochs),
        "learning_rate": float(args.learning_rate),
        "delta_learning_rate": float(args.delta_learning_rate),
        "var_learning_rate": float(args.var_learning_rate),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "log_interval": int(args.log_interval),
        "num_workers": int(args.num_workers),
        "time_min_ms": float(args.time_min_ms),
        "time_max_ms": float(args.time_max_ms),
        "residual_eps": float(args.residual_eps),
        "std_floor_scaled": float(args.std_floor_scaled),
        "logvar_clamp_min": float(args.logvar_clamp_min),
        "logvar_clamp_max": float(args.logvar_clamp_max),
        "mean_only": bool(args.mean_only),
        "include_mean_posterior_var": bool(args.include_mean_posterior_var),
        "calibrate_logvar_bias": not bool(args.no_logvar_bias_calibration),
        "max_curves_per_split": args.max_curves_per_split,
        "max_real_cdf_rows": args.max_real_cdf_rows,
        "allow_a_scale_mismatch": bool(args.allow_a_scale_mismatch),
        "device": str(device),
    }
    write_json(out_dir / "gp_config_resolved.json", base_config)
    print("Output dir:", out_dir)
    print("Device:", device)
    print("Seed:", args.seed)
    print("Feature specs:", feature_specs)
    if args.dry_run:
        return

    result = train_one_seed(
        seed_info=per_seed_manifest[0],
        run_dir=out_dir,
        base_config=base_config,
        device=device,
    )
    write_json(out_dir / "residual_svgp_result.json", result)
    summary = pd.DataFrame([
        {
            "seed": int(result["seed"]),
            **{f"test_{k}": v for k, v in result["test_metrics"].items()},
            **{f"val_{k}": v for k, v in result["val_metrics"].items()},
            "checkpoint": result["checkpoint"],
        }
    ])
    summary.to_csv(out_dir / "residual_svgp_per_seed.csv", index=False)


if __name__ == "__main__":
    main()
