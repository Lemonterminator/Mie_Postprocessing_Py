from __future__ import annotations

from OSCC_postprocessing.utils.backend import cp, get_array_module


def min_max_scale(arr):
    xp = get_array_module(arr)
    mn = arr.min()
    mx = arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return xp.zeros_like(arr)


def robust_scale(arr, q_min=5, q_max=95, clip=True, eps=1e-8):
    """Percentile-based robust min-max scaling for NumPy or CuPy arrays."""
    xp = get_array_module(arr)
    p_low, p_high = xp.percentile(arr, [q_min, q_max])

    denominator = p_high - p_low
    if denominator < eps:
        denominator = eps

    scaled = (arr - p_low) / denominator
    if clip:
        scaled = xp.clip(scaled, 0, 1)
    return scaled


__all__ = ["min_max_scale", "robust_scale", "cp"]
