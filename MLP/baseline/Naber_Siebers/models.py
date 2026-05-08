from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NSParams:
    k: float
    delay_s: float
    variant: str
    use_angle_factor: bool = False


def angle_factor(
    angle_deg: np.ndarray,
    *,
    floor: float = 0.25,
    ceiling: float = 4.0,
) -> np.ndarray:
    """Return the optional spreading-angle multiplier.

    The constant of proportionality is still fitted, so this factor only
    contributes relative variation. Treating the cone angle as a hard
    predictor is optional because test-video cone angles can leak information.
    """
    angle = np.asarray(angle_deg, dtype=float)
    half_rad = np.deg2rad(np.clip(angle, 1.0, 89.0) / 2.0)
    factor = 1.0 / np.sqrt(np.maximum(np.tan(half_rad), 1e-9))
    return np.clip(factor, float(floor), float(ceiling))


def design_vector(
    df: pd.DataFrame,
    *,
    delay_s: float,
    use_angle_factor: bool = False,
    angle_column: str = "angle_for_prediction_deg",
    angle_factor_floor: float = 0.25,
    angle_factor_ceiling: float = 4.0,
) -> np.ndarray:
    t = np.maximum(pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float) - float(delay_s), 0.0)
    x = pd.to_numeric(df["A_scale"], errors="coerce").to_numpy(dtype=float) * np.sqrt(t)
    if use_angle_factor:
        if angle_column not in df.columns:
            raise KeyError(f"Angle factor requested but {angle_column!r} is missing.")
        x = x * angle_factor(
            pd.to_numeric(df[angle_column], errors="coerce").to_numpy(dtype=float),
            floor=angle_factor_floor,
            ceiling=angle_factor_ceiling,
        )
    return x


def predict(
    df: pd.DataFrame,
    params: NSParams,
    *,
    angle_column: str = "angle_for_prediction_deg",
    angle_factor_floor: float = 0.25,
    angle_factor_ceiling: float = 4.0,
) -> np.ndarray:
    x = design_vector(
        df,
        delay_s=params.delay_s,
        use_angle_factor=params.use_angle_factor,
        angle_column=angle_column,
        angle_factor_floor=angle_factor_floor,
        angle_factor_ceiling=angle_factor_ceiling,
    )
    return float(params.k) * x


def params_to_dict(params: NSParams) -> dict[str, Any]:
    return {
        "k": float(params.k),
        "delay_s": float(params.delay_s),
        "delay_ms": float(params.delay_s * 1e3),
        "variant": params.variant,
        "use_angle_factor": bool(params.use_angle_factor),
    }

