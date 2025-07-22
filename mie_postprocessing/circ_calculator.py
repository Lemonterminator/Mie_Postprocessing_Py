import numpy as np

def calc_circle(*pts):
    """
    Return best‐fit circle center and radius to N ≥ 3 points in a least‐squares sense.

    Parameters
    ----------
    *pts : sequence of (x, y) pairs
        The clicked points.

    Returns
    -------
    center : (cx, cy)
    radius : r
    """
    # number of points
    N = len(pts)
    if N < 3:
        raise ValueError("Need at least three points to define a circle")

    # build the linear system for x^2 + y^2 + D x + E y + F = 0
    A = np.zeros((N, 3), dtype=float)
    b = np.zeros((N,),    dtype=float)
    for i, (x, y) in enumerate(pts):
        A[i, 0] = x
        A[i, 1] = y
        A[i, 2] = 1
        b[i]    = -(x*x + y*y)

    # solve normal equations: (A^T A) θ = A^T b
    # — more numerically stable to use lstsq, but this matches ATA inv form exactly:
    # ATA = A.T @ A
    # ATb = A.T @ b
    # θ = np.linalg.solve(ATA, ATb)
    θ, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = θ

    # recover circle parameters
    cx = -D/2
    cy = -E/2
    r  = np.sqrt(cx*cx + cy*cy - F)

    return (cx, cy), r

