# Prefer GPU if CuPy is available; otherwise use a NumPy-compatible shim.
import numpy as np
try:
    import cupy as _cupy  # type: ignore

    _cupy.cuda.runtime.getDeviceCount()
    cp = _cupy
    USING_CUPY = True
except Exception as exc:  # pragma: no cover - hardware dependent
    print(f"CuPy unavailable, falling back to NumPy backend: {exc}")
    USING_CUPY = False

    class _NumpyCompat:
        def __getattr__(self, name):
            return getattr(np, name)

        def asarray(self, a, dtype=None):
            return np.asarray(a, dtype=dtype)

        def asnumpy(self, a):
            return np.asarray(a)

        def get(self, a):
            return a

    cp = _NumpyCompat()  # type: ignore

def svd_foreground_cuda(
    video_FHW: cp.ndarray,
    rank: int = 2,
    center: str | None = "median",
    bkg_frame_limit: int = -1,
    return_bg: bool = False,
) -> cp.ndarray | tuple[cp.ndarray, cp.ndarray]:
    """Low-rank background removal via truncated SVD on a (F, H, W) video cube."""
    F, H, W = video_FHW.shape
    X = video_FHW.reshape(F, -1).T  # (HW, F)

    if center == "median":
        if bkg_frame_limit == -1:
            bias = cp.median(X, axis=1, keepdims=True)
        else:
            bias = cp.median(X[:, :bkg_frame_limit], axis=1, keepdims=True)
        Xc = X - bias
    elif center == "mean":
        if bkg_frame_limit == -1:
            bias = cp.mean(X, axis=1, keepdims=True)
        else:
            bias = cp.mean(X[:, :bkg_frame_limit], axis=1, keepdims=True)
        Xc = X - bias
    else:
        bias = 0.0
        Xc = X

    U, s, Vt = cp.linalg.svd(Xc, full_matrices=False)
    r = min(rank, s.size)
    Uk = U[:, :r]
    sk = s[:r]
    Vtk = Vt[:r, :]

    Lc = (Uk * sk) @ Vtk
    S = Xc - Lc
    fg_FHW = S.T.reshape(F, H, W)

    if return_bg:
        bg = (Lc + bias).T.reshape(F, H, W)
        return fg_FHW, bg
    return fg_FHW


def godec_like(
    video_FHW: cp.ndarray,
    rank: int = 2,
    lam: float = 2.5,
    iters: int = 8,
    center: str | None = "median",
    return_bg: bool = False,
) -> cp.ndarray | tuple[cp.ndarray, cp.ndarray]:
    """Robust PCA style foreground extraction (GoDec-style alternating minimisation)."""
    F, H, W = video_FHW.shape
    X = video_FHW.reshape(F, -1).T

    if center == "median":
        bias = cp.median(X, axis=1, keepdims=True)
        Xc = X - bias
    elif center == "mean":
        bias = cp.mean(X, axis=1, keepdims=True)
        Xc = X - bias
    else:
        bias = 0.0
        Xc = X

    L = cp.zeros_like(Xc)
    S = cp.zeros_like(Xc)

    for _ in range(iters):
        U, s, Vt = cp.linalg.svd(Xc - S, full_matrices=False)
        r = min(rank, s.size)
        L = (U[:, :r] * s[:r]) @ Vt[:r, :]

        R = Xc - L
        med = cp.median(cp.abs(R))
        tau = lam * 1.4826 * med + 1e-8
        S = cp.sign(R) * cp.maximum(cp.abs(R) - tau, 0.0)

    fg = (Xc - L).T.reshape(F, H, W)
    if return_bg:
        bg = (L + bias).T.reshape(F, H, W)
        return fg, bg
    return fg