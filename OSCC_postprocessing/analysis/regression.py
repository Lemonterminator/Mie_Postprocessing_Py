from __future__ import annotations

import numpy as np


def ransac_line_1d(
    y,
    x=None,
    max_trials=1000,
    residual_threshold=None,
    min_inliers=2,
    random_state=None,
):
    """RANSAC line fit for an indexed 1D array robust to NaNs."""
    y = np.asarray(y, dtype=float)
    n = y.size

    if x is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float)

    valid_mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if x_valid.size < 2:
        raise ValueError("Not enough valid (non-NaN) points for line fitting.")

    rng = np.random.default_rng(random_state)

    if residual_threshold is None:
        med = np.median(y_valid)
        mad = np.median(np.abs(y_valid - med)) + 1e-9
        residual_threshold = 2.5 * mad

    best_inlier_count = 0
    best_a = None
    best_b = None
    best_inlier_mask_valid = None

    for _ in range(max_trials):
        idx = rng.choice(x_valid.size, size=2, replace=False)
        x_s = x_valid[idx]
        y_s = y_valid[idx]

        if x_s[1] == x_s[0]:
            continue

        a = (y_s[1] - y_s[0]) / (x_s[1] - x_s[0])
        b = y_s[0] - a * x_s[0]

        residuals = np.abs(y_valid - (a * x_valid + b))
        inliers = residuals <= residual_threshold
        inlier_count = int(inliers.sum())

        if inlier_count > best_inlier_count and inlier_count >= min_inliers:
            best_inlier_count = inlier_count
            best_a, best_b = a, b
            best_inlier_mask_valid = inliers

    if best_inlier_mask_valid is None:
        a_mat = np.vstack([x_valid, np.ones_like(x_valid)]).T
        best_a, best_b = np.linalg.lstsq(a_mat, y_valid, rcond=None)[0]
        best_inlier_mask_valid = np.ones_like(y_valid, dtype=bool)

    x_in = x_valid[best_inlier_mask_valid]
    y_in = y_valid[best_inlier_mask_valid]
    a_mat = np.vstack([x_in, np.ones_like(x_in)]).T
    best_a, best_b = np.linalg.lstsq(a_mat, y_in, rcond=None)[0]

    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[valid_mask] = best_inlier_mask_valid
    return best_a, best_b, inlier_mask


def ransac_fixed_intercept(x, y, b0, max_iter=2000, residual_thresh=1.0, min_inliers=10):
    x = np.asarray(x)
    y = np.asarray(y)

    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n == 0:
        raise ValueError("No valid points")

    best_a = None
    best_inliers = []

    for _ in range(max_iter):
        idx = np.random.randint(0, n)
        xi, yi = x[idx], y[idx]
        if xi == 0:
            continue

        a_candidate = (yi - b0) / xi
        residuals = np.abs((a_candidate * x + b0) - y)
        inliers = np.where(residuals < residual_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_a = a_candidate

        if len(best_inliers) >= min_inliers:
            break

    if best_a is None:
        raise RuntimeError("RANSAC failed.")

    xi = x[best_inliers]
    yi = y[best_inliers]
    refined_a = np.sum(xi * (yi - b0)) / np.sum(xi**2)
    return refined_a, best_inliers


def linear_regression_fixed_intercept(x, y, b0):
    x = np.asarray(x)
    y = np.asarray(y)

    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    numerator = np.sum(x * (y - b0))
    denominator = np.sum(x**2)
    if denominator == 0:
        raise ValueError("Cannot estimate slope: x has no variation or all x=0.")
    return numerator / denominator


__all__ = [
    "linear_regression_fixed_intercept",
    "ransac_fixed_intercept",
    "ransac_line_1d",
]
