from __future__ import annotations

from .kd_losses import (
    KD_REGISTRY,
    KDStrategy,
    MseMuKD,
    MseMuPlusSigmaKD,
    ForwardKLKD,
    ReverseKLKD,
    ForwardKLUniformConfKD,
    TEACHER_CONF_WEIGHT_MIN,
    TEACHER_CONF_WEIGHT_MAX,
    weighted_mean,
)

__all__ = [
    "KD_REGISTRY",
    "KDStrategy",
    "MseMuKD",
    "MseMuPlusSigmaKD",
    "ForwardKLKD",
    "ReverseKLKD",
    "ForwardKLUniformConfKD",
    "TEACHER_CONF_WEIGHT_MIN",
    "TEACHER_CONF_WEIGHT_MAX",
    "weighted_mean",
]
