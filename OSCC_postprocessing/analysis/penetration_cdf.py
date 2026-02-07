import numpy as xp
from OSCC_postprocessing.analysis.multihole_utils import resolve_backend

use_gpu, triangle_backend, xp = resolve_backend(use_gpu="auto", triangle_backend="auto")

def monotone_non_decreasing(x, axis=-1):
    """
    Prefix max / monotone non-decreasing transform along `axis`.
    Works for numpy and cupy backends.
    """
    try:
        # NumPy OK; some CuPy builds may raise NotImplementedError
        return xp.maximum.accumulate(x, axis=axis)
    except (NotImplementedError, AttributeError):
        # CuPy fallback: use maximum_filter1d to compute prefix max on GPU
        import cupy as cp
        from cupyx.scipy.ndimage import maximum_filter1d

        x = cp.asarray(x)
        n = x.shape[axis]
        # origin=-(n-1) aligns the window's right edge to current index => covers [0..i]
        return maximum_filter1d(x, size=n, axis=axis, origin=-(n - 1), mode="nearest")



def penetration_cdf_front(I, mask=None, q=0.97, min_x=0):
    """
    I: (T,X)
    mask: (T,X) {0,1} 可选，只在喷雾区域内累计
    q: cumulative quantile, e.g. 0.95~0.995
    """
    I = xp.asarray(I, xp.float32)
    T, X = I.shape

    # 每帧背景抹掉：用分位数当背景（比中位数更适合你这种“大片暗背景+亮喷雾”）
    bg = xp.quantile(I, 0.10, axis=1, keepdims=True)
    S = I - bg
    S[S < 0] = 0

    if mask is not None:
        S = S * mask.astype(xp.float32)

    S[:, :min_x] = 0

    cdf = xp.cumsum(S, axis=1)
    tot = cdf[:, -1] + 1e-6
    target = q * tot

    xhat = xp.argmax(cdf >= target[:, None], axis=1).astype(xp.int32)
    # 若某帧 tot≈0（几乎没亮度），argmax 会返回 0；你可后处理
    return xhat