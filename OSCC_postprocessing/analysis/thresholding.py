from __future__ import annotations

import numpy as np

from OSCC_postprocessing.utils.backend import get_array_module, get_cupy


def triangle_binarize_gpu(px_range_cp, ignore_zeros: bool = False):
    """Histogram-based triangle threshold implemented fully on the GPU."""
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("CuPy is required for triangle_binarize_gpu")
    x = px_range_cp
    if ignore_zeros:
        posmask = x > 0
        nz = x[posmask]
    else:
        nz = x.ravel()

    if nz.size == 0:
        return cp.zeros_like(x, dtype=cp.bool_)

    vmin = nz.min()
    vmax = nz.max()
    scale = 255.0 / (float(vmax - vmin) + 1e-12)
    u8 = cp.floor((nz - vmin) * scale).astype(cp.uint8, copy=False)

    hist, _ = cp.histogram(u8, bins=256, range=(0, 255))
    nzbins = cp.nonzero(hist)[0]
    i0, i1 = int(nzbins[0]), int(nzbins[-1])
    imax = int(hist.argmax())

    iend = i0 if (imax - i0) > (i1 - imax) else i1
    lo, hi = (iend, imax) if iend < imax else (imax, iend)
    xs = cp.arange(lo, hi + 1)
    ys = hist[xs].astype(cp.float32)
    x0, y0 = float(imax), float(hist[imax])
    x1, y1 = float(iend), float(hist[iend])
    denom = cp.hypot(y1 - y0, x1 - x0) + 1e-12
    d = cp.abs((y0 - y1) * xs + (x1 - x0) * ys + (x0 * y1 - x1 * y0)) / denom
    t_idx = int(xs[int(d.argmax())])

    if ignore_zeros:
        mask_full = cp.zeros_like(x, dtype=cp.bool_)
        mask_full[posmask] = u8 > t_idx
        return mask_full
    return (u8 > t_idx).reshape(x.shape)

__all__ = ["triangle_binarize_gpu", "get_array_module"]
