from __future__ import annotations

"""Physics-informed loss functions for Stage-1 and Stage-2 MLP training.

All losses operate on the n_points-length time grid in [0, time_max_ms] ms
passed in each batch.  Physical quantities (mu, std in mm) are obtained by
multiplying the dimensionless MLP output by A_scale from the batch dict.

    stage1_objective           — MSE in A-scaled space + log-var prior + shape penalties.
    stage2_objective           — Gaussian NLL in A-scaled space + shape + anchor penalties.
    derivative_physics_penalty — d1 (monotone) and gated d2 (concave) shape terms.
    early_time_anchor_penalties — penalise non-zero mean and excess std near t=0.
    split_mu_logvar            — unpack (mu, log_var) from 2- or 3-channel model output.
"""

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_d2_sample_weights(
    model: nn.Module | None,
    batch: Mapping[str, torch.Tensor],
    nozzle_id_channel_idx: int | None,
) -> torch.Tensor | None:
    """Build per-sample d2 weights from model.log_lambda_d2[nozzle_id_per_sample].

    Returns None if the model does not have a learnable log_lambda_d2 buffer
    or the batch lacks a nozzle_id channel - in that case derivative_physics_penalty
    falls back to its scalar-mean default.
    """
    if model is None or getattr(model, "log_lambda_d2", None) is None:
        return None
    if nozzle_id_channel_idx is None:
        return None
    features = batch.get("features")
    if features is None or features.ndim < 3:
        return None
    # nozzle_id is constant per trajectory; sample from the first time step.
    nozzle_ids = features[:, 0, int(nozzle_id_channel_idx)].long()
    n_families = int(model.log_lambda_d2.shape[0])
    nozzle_ids = torch.clamp(nozzle_ids, min=0, max=n_families - 1)
    floor = float(getattr(model, "learnable_d2_floor", 1e-5))
    lambdas = F.softplus(model.log_lambda_d2) + floor
    return lambdas[nozzle_ids]


def split_mu_logvar(model_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack (mu, log_var) from 2- or 3-channel model output.

    The third channel (onset logit / aux mu) is present in models that have an
    auxiliary onset head; it is silently dropped here so the primary Stage-1/-2
    losses are independent of which architecture produced the output.
    """
    if model_output.shape[-1] == 3:
        mu, log_var, _onset = torch.split(model_output, [1, 1, 1], dim=-1)
        return mu, log_var
    mu, log_var = model_output.chunk(2, dim=-1)
    return mu, log_var


def onset_aux_penalty(
    model_output: torch.Tensor,
    target_scaled: torch.Tensor,
    n_points: int,
    *,
    time_max_ms: float,
    onset_t_ms_max: float,
) -> torch.Tensor:
    """MSE between the aux onset channel and target_scaled, masked to t < onset_t_ms_max.

    Tier-2C auxiliary regression head. The aux mu is the 3rd output channel of
    PenetrationMLP when onset_aux_head=True. The penalty is restricted to the
    onset window (t < onset_t_ms_max) so it acts as a localized regularizer on
    the injection-onset region. Returns 0 if the model output has fewer than
    3 channels (the aux head is disabled).
    """
    if model_output.shape[-1] < 3 or n_points <= 0:
        return model_output.new_tensor(0.0)
    mu_onset = model_output[..., 2:3]
    t_ms = torch.linspace(0.0, float(time_max_ms), int(n_points), device=model_output.device)
    onset_mask = (t_ms < float(onset_t_ms_max)).to(model_output.dtype)
    if float(onset_mask.sum().item()) == 0.0:
        return model_output.new_tensor(0.0)
    # Reshape to [B, T, 1] using the time grid, broadcast the mask.
    diff = (mu_onset - target_scaled).pow(2).reshape(-1, n_points)
    weighted = (diff * onset_mask.unsqueeze(0)).sum() / torch.clamp(
        onset_mask.sum() * diff.shape[0], min=1.0
    )
    return weighted


def derivative_physics_penalty(
    mu_physical: torch.Tensor,
    n_points: int,
    *,
    time_max_ms: float,
    d2_start_ms: float,
    d2_transition_ms: float,
    d2_sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalise spray trajectories that violate monotone / concave-after-transition physics.

    d1_negative_penalty = mean(relu(-d_mu)^2)
        Any decrease in predicted penetration over time is physically forbidden
        for a spray front and is penalised quadratically.

    d2_positive_penalty = weighted mean over batch of  mean_t( relu(d2_mu)^2 * gate(t) )
        Positive second difference (concavity violation) is penalised only after
        a soft gate centred on d2_start_ms ms.  The early convex ballistic phase
        (t < d2_start_ms) passes through without penalty.

        When d2_sample_weights is None (default), all samples weight equally
        (reproduces the legacy scalar mean). When supplied as a 1-D tensor of
        shape [B], the per-sample d2 score is multiplied element-wise before
        averaging - Tier-3B uses this to apply a learnable per-nozzle lambda
        on the d2 term.

    Returns (d1_negative_penalty, d2_positive_penalty) as zero-dimensional tensors.
    """
    if n_points <= 1:
        zero = mu_physical.new_tensor(0.0)
        return zero, zero
    mu_seq = mu_physical.reshape(-1, n_points)
    d1 = mu_seq[:, 1:] - mu_seq[:, :-1]
    d1_negative_penalty = torch.relu(-d1).pow(2).mean()
    if n_points <= 2:
        return d1_negative_penalty, mu_physical.new_tensor(0.0)

    d2 = mu_seq[:, 2:] - 2.0 * mu_seq[:, 1:-1] + mu_seq[:, :-2]
    t_ms = torch.linspace(0.0, float(time_max_ms), int(n_points), device=mu_physical.device)
    t_center_ms = t_ms[1:-1]
    gate = torch.sigmoid((t_center_ms - float(d2_start_ms)) / max(float(d2_transition_ms), 1e-6)).unsqueeze(0)
    if d2_sample_weights is None:
        d2_positive_penalty = (torch.relu(d2).pow(2) * gate).mean()
    else:
        per_sample = (torch.relu(d2).pow(2) * gate).mean(dim=1)  # [B]
        weights = d2_sample_weights.to(per_sample)
        if weights.shape != per_sample.shape:
            raise ValueError(
                f"d2_sample_weights shape {tuple(weights.shape)} does not match batch {tuple(per_sample.shape)}"
            )
        d2_positive_penalty = (weights * per_sample).mean()
    return d1_negative_penalty, d2_positive_penalty


def early_time_anchor_penalties(
    mu_physical: torch.Tensor,
    std_physical: torch.Tensor,
    n_points: int,
    *,
    time_max_ms: float,
    anchor_window_ms: float,
    sigma_anchor_floor_mm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalise large mean and excess std in the early injection window.

    The linearly decaying weight (1 at t=0, 0 at t=anchor_window_ms) focuses
    the penalty on the first ~0.15 ms, where the spray front has not developed
    and raw measurements are sparse.  Without this penalty, Stage-2 NLL can
    inflate sigma in the data-sparse onset region to reduce total NLL cheaply
    (sigma-arbitrage), masking variance collapse in the main spray window.

    mu_anchor    = weighted_mean(mu_phys²)                     — anchors mean near 0 at t≈0.
    sigma_anchor = weighted_mean(relu(std_phys − floor_mm)²)  — caps early-time sigma.

    Returns (mu_anchor_scalar, sigma_anchor_scalar).
    """
    if n_points <= 0:
        zero = mu_physical.new_tensor(0.0)
        return zero, zero
    mu_seq = mu_physical.reshape(-1, n_points)
    std_seq = std_physical.reshape(-1, n_points)
    t_ms = torch.linspace(0.0, float(time_max_ms), int(n_points), device=mu_physical.device)
    weights = torch.clamp(1.0 - t_ms / max(float(anchor_window_ms), 1e-6), min=0.0)
    denom = torch.clamp(weights.sum() * mu_seq.shape[0], min=1e-12)
    mu_anchor = (mu_seq.pow(2) * weights.unsqueeze(0)).sum() / denom
    sigma_excess = torch.relu(std_seq - float(sigma_anchor_floor_mm))
    sigma_anchor = (sigma_excess.pow(2) * weights.unsqueeze(0)).sum() / denom
    return mu_anchor, sigma_anchor


def stage1_objective(
    model_output: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    n_points: int,
    time_max_ms: float,
    var_reg_weight: float,
    log_var_prior: float,
    log_var_bounds: tuple[float, float],
    d1_positive_weight: float,
    d2_concave_weight: float,
    d2_start_ms: float,
    d2_transition_ms: float,
    onset_aux_weight: float = 0.0,
    onset_t_ms_max: float = 0.3,
    model: nn.Module | None = None,
    nozzle_id_channel_idx: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stage-1 loss: MSE in A-scaled space + log-var regulariser + shape penalties.

        loss = MSE(mu_hat, S/A_scale)
             + var_reg_weight · MSE(log_var_hat, log_var_prior)
             + d1_positive_weight · d1_penalty
             + d2_concave_weight  · d2_penalty

    The log-var regulariser keeps the variance head well-initialised for Stage-2
    NLL fine-tuning.  Physical MAE is tracked for diagnostics but is not
    back-propagated.  Returns (loss_tensor, metrics_dict).
    """
    mu_hat, log_var_hat = split_mu_logvar(model_output)
    log_var_hat = torch.clamp(log_var_hat, min=float(log_var_bounds[0]), max=float(log_var_bounds[1]))
    target_scaled = batch["target_scaled"]
    a_scale = batch["a_scale"]
    mu_physical = a_scale * mu_hat

    mse_scaled = torch.mean((mu_hat - target_scaled) ** 2)
    prior = torch.full_like(log_var_hat, float(log_var_prior))
    var_reg = torch.mean((log_var_hat - prior) ** 2)
    d2_sample_weights = _compute_d2_sample_weights(model, batch, nozzle_id_channel_idx)
    d1_penalty, d2_penalty = derivative_physics_penalty(
        mu_physical,
        n_points=n_points,
        time_max_ms=time_max_ms,
        d2_start_ms=d2_start_ms,
        d2_transition_ms=d2_transition_ms,
        d2_sample_weights=d2_sample_weights,
    )
    aux_loss = onset_aux_penalty(
        model_output, target_scaled, n_points=n_points,
        time_max_ms=time_max_ms, onset_t_ms_max=onset_t_ms_max,
    )
    # When per-sample d2 weights come from a learnable lambda, the weight is
    # already inside d2_penalty so the outer d2_concave_weight collapses to 1.
    d2_outer_weight = 1.0 if d2_sample_weights is not None else float(d2_concave_weight)
    loss = (
        mse_scaled
        + float(var_reg_weight) * var_reg
        + float(d1_positive_weight) * d1_penalty
        + d2_outer_weight * d2_penalty
        + float(onset_aux_weight) * aux_loss
    )
    physical_mae = torch.mean(torch.abs(mu_physical - batch["target_physical"]))
    metrics = {
        "loss": float(loss.detach().cpu()),
        "mse_scaled": float(mse_scaled.detach().cpu()),
        "physical_mae": float(physical_mae.detach().cpu()),
        "var_reg": float(var_reg.detach().cpu()),
        "d1_penalty": float(d1_penalty.detach().cpu()),
        "d2_penalty": float(d2_penalty.detach().cpu()),
        "onset_aux": float(aux_loss.detach().cpu()),
    }
    return loss, metrics


def stage2_objective(
    model_output: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    n_points: int,
    time_max_ms: float,
    log_var_bounds: tuple[float, float],
    nll_eps: float,
    d1_positive_weight: float,
    d2_concave_weight: float,
    d2_start_ms: float,
    d2_transition_ms: float,
    lambda_mu_anchor: float = 0.0,
    lambda_sigma_anchor: float = 0.0,
    anchor_window_ms: float = 0.15,
    sigma_anchor_floor_mm: float = 0.0,
    onset_aux_weight: float = 0.0,
    onset_t_ms_max: float = 0.3,
    model: nn.Module | None = None,
    nozzle_id_channel_idx: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stage-2 loss: Gaussian NLL in A-scaled space + shape + anchor penalties.

        loss = NLL_scaled
             + d1_positive_weight  · d1_penalty
             + d2_concave_weight   · d2_penalty
             + lambda_mu_anchor    · mu_anchor
             + lambda_sigma_anchor · sigma_anchor

    NLL_scaled = 0.5 · mean(log_var_hat + (mu_hat − S/A_scale)² / var_hat).
    Warm-starting from Stage-1 means the mean trajectory is already reasonable;
    Stage-2 jointly refines the mean and calibrates the uncertainty.  Anchor
    penalties are active only when lambda_mu_anchor > 0 or lambda_sigma_anchor > 0.
    Returns (loss_tensor, metrics_dict).
    """
    mu_hat, log_var_hat = split_mu_logvar(model_output)
    log_var_hat = torch.clamp(log_var_hat, min=float(log_var_bounds[0]), max=float(log_var_bounds[1]))
    target_scaled = batch["target_scaled"]
    a_scale = batch["a_scale"]
    var_hat = torch.exp(log_var_hat) + float(nll_eps)
    scaled_nll = torch.mean(0.5 * (log_var_hat + (mu_hat - target_scaled) ** 2 / var_hat))

    mu_physical = a_scale * mu_hat
    d2_sample_weights = _compute_d2_sample_weights(model, batch, nozzle_id_channel_idx)
    d1_penalty, d2_penalty = derivative_physics_penalty(
        mu_physical,
        n_points=n_points,
        time_max_ms=time_max_ms,
        d2_start_ms=d2_start_ms,
        d2_transition_ms=d2_transition_ms,
        d2_sample_weights=d2_sample_weights,
    )
    std_physical = a_scale * torch.exp(0.5 * log_var_hat)
    mu_anchor, sigma_anchor = early_time_anchor_penalties(
        mu_physical,
        std_physical,
        n_points=n_points,
        time_max_ms=time_max_ms,
        anchor_window_ms=anchor_window_ms,
        sigma_anchor_floor_mm=sigma_anchor_floor_mm,
    )
    aux_loss = onset_aux_penalty(
        model_output, target_scaled, n_points=n_points,
        time_max_ms=time_max_ms, onset_t_ms_max=onset_t_ms_max,
    )
    d2_outer_weight = 1.0 if d2_sample_weights is not None else float(d2_concave_weight)
    loss = (
        scaled_nll
        + float(d1_positive_weight) * d1_penalty
        + d2_outer_weight * d2_penalty
        + float(lambda_mu_anchor) * mu_anchor
        + float(lambda_sigma_anchor) * sigma_anchor
        + float(onset_aux_weight) * aux_loss
    )

    physical_mae = torch.mean(torch.abs(mu_physical - batch["target_physical"]))
    metrics = {
        "loss": float(loss.detach().cpu()),
        "nll_scaled": float(scaled_nll.detach().cpu()),
        "physical_mae": float(physical_mae.detach().cpu()),
        "std_physical_mean": float(std_physical.mean().detach().cpu()),
        "d1_penalty": float(d1_penalty.detach().cpu()),
        "d2_penalty": float(d2_penalty.detach().cpu()),
        "mu_anchor": float(mu_anchor.detach().cpu()),
        "sigma_anchor": float(sigma_anchor.detach().cpu()),
        "onset_aux": float(aux_loss.detach().cpu()),
    }
    return loss, metrics
