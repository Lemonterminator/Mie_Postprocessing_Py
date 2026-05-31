"""KD strategy plugin system for Stage-3 distillation.

Each strategy takes student/teacher statistics in *physical* space (mm, mm²) and
returns a scalar KD loss plus a per-point confidence-weight tensor used only for
diagnostic logging (not back-propagated separately).

Production default: ``mse_mu_plus_sigma`` with kd_sigma_weight=5.0.
    Chosen over ``forward_kl`` because forward-KL allowed the student to reduce
    NLL by inflating sigma near the field-of-view boundary (sigma-arbitrage),
    producing systematic under-prediction of small penetration depths near the
    nozzle.  The MSE formulation imposes a harder coupling between student and
    teacher sigma without that degree of freedom.

To add a new KD form:
    1. Define a subclass of KDStrategy.
    2. Add one entry to KD_REGISTRY.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

# Bounds for sigma-based confidence reweighting in KL strategies:
#   conf_weight = clamp(sigma_conf_ref_mm / teacher_std, MIN, MAX).
# Low teacher sigma → high confidence weight → spray-tip frames are upweighted.
TEACHER_CONF_WEIGHT_MIN = 0.25
TEACHER_CONF_WEIGHT_MAX = 1.0


def weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Weight-normalised mean; returns 0 when total weight is below eps."""
    denom = weights.sum()
    if float(denom.detach().cpu()) <= eps:
        return values.new_tensor(0.0)
    return (values * weights).sum() / denom


class KDStrategy(ABC):
    """Abstract base for knowledge-distillation loss strategies.

    Each strategy receives student/teacher statistics in physical space and
    returns ``(kd_loss_scalar, conf_weight)`` where ``conf_weight`` [B, T] is
    used only for diagnostic logging.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def __call__(
        self,
        *,
        mu_phys: torch.Tensor,
        log_var_phys: torch.Tensor,
        var_phys: torch.Tensor,
        teacher_mu_phys: torch.Tensor,
        teacher_var_phys: torch.Tensor,
        kd_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(kd_loss_scalar, conf_weight [B, T])``."""
        ...


class MseMuKD(KDStrategy):
    """MSE on means only; ignores variance."""

    def __call__(self, *, mu_phys, log_var_phys, var_phys, teacher_mu_phys, teacher_var_phys, kd_weight):
        kd_point = (mu_phys - teacher_mu_phys).pow(2)
        conf_weight = torch.ones_like(teacher_var_phys.sqrt()).squeeze(-1)
        kd_loss = weighted_mean(kd_point.squeeze(-1), kd_weight)
        return kd_loss, conf_weight


class MseMuPlusSigmaKD(KDStrategy):
    """MSE on means + λ · MSE on log-var (production default for engineered-v2).

    ``kd_sigma_weight`` (config key, default 1.0) controls λ.
    """

    def __call__(self, *, mu_phys, log_var_phys, var_phys, teacher_mu_phys, teacher_var_phys, kd_weight):
        teacher_log_var_phys = torch.log(teacher_var_phys)
        mu_term = (mu_phys - teacher_mu_phys).pow(2)
        sigma_term = (log_var_phys - teacher_log_var_phys).pow(2)
        lambda_sigma = float(self.config.get("kd_sigma_weight", 1.0))
        kd_point = mu_term + lambda_sigma * sigma_term
        conf_weight = torch.ones_like(teacher_var_phys.sqrt()).squeeze(-1)
        kd_loss = weighted_mean(kd_point.squeeze(-1), kd_weight)
        return kd_loss, conf_weight


class ForwardKLKD(KDStrategy):
    """KL(teacher ‖ student) with sigma-based confidence reweighting."""

    def __call__(self, *, mu_phys, log_var_phys, var_phys, teacher_mu_phys, teacher_var_phys, kd_weight):
        kl_point = 0.5 * (
            torch.log(var_phys / teacher_var_phys)
            + (teacher_var_phys + (teacher_mu_phys - mu_phys).pow(2)) / var_phys
            - 1.0
        )
        conf_weight = torch.clamp(
            float(self.config["sigma_conf_ref_mm"]) / teacher_var_phys.sqrt(),
            min=float(self.config.get("teacher_conf_weight_min", TEACHER_CONF_WEIGHT_MIN)),
            max=float(self.config.get("teacher_conf_weight_max", TEACHER_CONF_WEIGHT_MAX)),
        ).squeeze(-1)
        kd_loss = weighted_mean(kl_point.squeeze(-1), kd_weight * conf_weight)
        return kd_loss, conf_weight


class ReverseKLKD(KDStrategy):
    """KL(student ‖ teacher) with sigma-based confidence reweighting."""

    def __call__(self, *, mu_phys, log_var_phys, var_phys, teacher_mu_phys, teacher_var_phys, kd_weight):
        kl_point = 0.5 * (
            torch.log(teacher_var_phys / var_phys)
            + (var_phys + (mu_phys - teacher_mu_phys).pow(2)) / teacher_var_phys
            - 1.0
        )
        conf_weight = torch.clamp(
            float(self.config["sigma_conf_ref_mm"]) / teacher_var_phys.sqrt(),
            min=float(self.config.get("teacher_conf_weight_min", TEACHER_CONF_WEIGHT_MIN)),
            max=float(self.config.get("teacher_conf_weight_max", TEACHER_CONF_WEIGHT_MAX)),
        ).squeeze(-1)
        kd_loss = weighted_mean(kl_point.squeeze(-1), kd_weight * conf_weight)
        return kd_loss, conf_weight


class ForwardKLUniformConfKD(KDStrategy):
    """KL(teacher ‖ student) with uniform confidence weights (disables sigma-based reweighting)."""

    def __call__(self, *, mu_phys, log_var_phys, var_phys, teacher_mu_phys, teacher_var_phys, kd_weight):
        kl_point = 0.5 * (
            torch.log(var_phys / teacher_var_phys)
            + (teacher_var_phys + (teacher_mu_phys - mu_phys).pow(2)) / var_phys
            - 1.0
        )
        conf_weight = torch.ones_like(teacher_var_phys.sqrt()).squeeze(-1)
        kd_loss = weighted_mean(kl_point.squeeze(-1), kd_weight * conf_weight)
        return kd_loss, conf_weight


# Dispatch table: kd_mode config string → strategy class.
# refinement_loss() in train_stage3_*.py calls KD_REGISTRY[kd_mode](config)(...).
KD_REGISTRY: dict[str, type[KDStrategy]] = {
    "mse_mu": MseMuKD,
    "mse_mu_plus_sigma": MseMuPlusSigmaKD,
    "forward_kl": ForwardKLKD,
    "reverse_kl": ReverseKLKD,
    "forward_kl_uniform_conf": ForwardKLUniformConfKD,
}
