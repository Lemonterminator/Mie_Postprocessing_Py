from __future__ import annotations

import numpy as np

from OSCC_postprocessing.utils.backend import cp, get_array_module, is_cupy_array, to_numpy


def _as_numpy(arr):
    return to_numpy(arr)


def to_py_scalar(x):
    # x can be a CuPy scalar / CuPy 0-d array / NumPy scalar / Python number
    if is_cupy_array(x):
        return float(x.get().item())
    try:
        return float(x)
    except TypeError:
        return float(cp.asarray(x).get().item())


def mad(x):
    xp = get_array_module(x)
    med = xp.median(x)
    return xp.median(xp.abs(x - med)) + 1e-12


def hysteresis_threshold(y, th_lo, th_hi):
    """
    Apply hysteresis thresholding to a 1D signal.

    Values above ``th_hi`` enter the high state; values below ``th_lo`` leave
    the high state.
    """
    xp = get_array_module(y)
    high = y > th_hi
    low = y < th_lo

    mask = xp.zeros_like(y, dtype=bool)
    state = False
    for i in range(len(y)):
        if not state:
            if high[i]:
                state = True
        else:
            if low[i]:
                state = False
        mask[i] = state
    return mask


def fill_short_false_runs(mask, max_len=3):
    """
    Fill short False runs inside a True region.

    This is a simple hole-filling step on a 1D boolean mask.
    """
    xp = get_array_module(mask)
    m = mask.copy()
    diff = xp.diff(m.astype(xp.int8))
    starts = xp.where(diff == -1)[0] + 1
    ends = xp.where(diff == 1)[0] + 1
    if not bool(m[0]):
        starts = xp.r_[0, starts]
    if not bool(m[-1]):
        ends = xp.r_[ends, len(m)]
    for s, e in zip(starts, ends):
        if (e - s) <= max_len:
            m[s:e] = True
    return m


def remove_short_true_runs(mask, min_len=5):
    """Remove short True runs from a 1D boolean mask."""
    xp = get_array_module(mask)
    m = mask.copy()
    diff = xp.diff(m.astype(xp.int8))
    starts = xp.where(diff == 1)[0] + 1
    ends = xp.where(diff == -1)[0] + 1
    if bool(m[0]):
        starts = xp.r_[0, starts]
    if bool(m[-1]):
        ends = xp.r_[ends, len(m)]
    for s, e in zip(starts, ends):
        if (e - s) < min_len:
            m[s:e] = False
    return m


def longest_true_run(mask):
    """Return the longest True run as ``(start, end_exclusive)`` or ``None``."""
    xp = get_array_module(mask)
    m = mask.astype(bool)
    if not m.any():
        return None
    diff = xp.diff(m.astype(xp.int8))
    starts = xp.where(diff == 1)[0] + 1
    ends = xp.where(diff == -1)[0] + 1
    if m[0]:
        starts = xp.r_[0, starts]
    if m[-1]:
        ends = xp.r_[ends, len(m)]
    lengths = ends - starts
    k = xp.argmax(lengths)
    return int(starts[k]), int(ends[k])


def detect_single_high_interval(
    y,
    x=None,
    base_quantile=0.10,
    k_hi=0.9,
    k_lo=0.1,
    th_lo=None,
    th_hi=None,
    fill_hole_len=3,
    min_island_len=5,
):
    """
    y: 1D signal
    x: optional real coordinates; defaults to index coordinates
    thresholds:
        th_hi = base + k_hi * sigma
        th_lo = base + k_lo * sigma
    sigma estimated from MAD.
    """
    xp = get_array_module(y)
    y = xp.asarray(y)
    if x is None:
        x = xp.arange(len(y))
    else:
        x = xp.asarray(x)

    base = xp.quantile(y, base_quantile)
    sigma = 1.4826 * mad(y - base)
    if th_hi is None:
        th_hi = base + k_hi * sigma
    if th_lo is None:
        th_lo = base + k_lo * sigma

    mask = hysteresis_threshold(y, th_lo=th_lo, th_hi=th_hi)
    mask = fill_short_false_runs(mask, max_len=fill_hole_len)
    mask = remove_short_true_runs(mask, min_len=min_island_len)

    run = longest_true_run(mask)
    if run is None:
        return None, mask, (th_lo, th_hi)

    s, e = run
    return (x[s], x[e - 1], s, e - 1), mask, (th_lo, th_hi)


__all__ = [
    "_as_numpy",
    "detect_single_high_interval",
    "fill_short_false_runs",
    "hysteresis_threshold",
    "longest_true_run",
    "mad",
    "remove_short_true_runs",
    "to_py_scalar",
]
