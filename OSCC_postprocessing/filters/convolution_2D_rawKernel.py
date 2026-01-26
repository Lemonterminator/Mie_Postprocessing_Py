import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as cnd

def _mode_map(mode: str) -> str:
    mode = (mode or "").lower()
    # 你之前用的 "edge" 更接近 ndimage 的 nearest
    if mode in ["edge", "nearest"]:
        return "nearest"
    if mode in ["reflect", "mirror"]:
        return "reflect"
    if mode in ["wrap"]:
        return "wrap"
    if mode in ["constant", "const", "zeros", "zero"]:
        return "constant"
    # 默认给个安全值
    return "nearest"

def _separable_factors_from_svd(kernel_gpu: cp.ndarray, tol: float = 1e-10):
    """
    若 kernel rank=1，则返回 (kH_vec, kW_vec)，满足 kernel ≈ outer(kH_vec, kW_vec)
    否则返回 None
    """
    U, S, Vt = cp.linalg.svd(kernel_gpu, full_matrices=False)
    rank = int(cp.sum(S > tol).get())  # 取回到 CPU 做分支
    if rank != 1:
        return None

    s0 = S[0]
    # 令 kernel = (U[:,0]*sqrt(s0)) outer (Vt[0,:]*sqrt(s0))
    a = U[:, 0] * cp.sqrt(s0)     # (kH,)
    b = Vt[0, :] * cp.sqrt(s0)    # (kW,)
    return a.astype(cp.float32), b.astype(cp.float32)

def convolution_2D_cupy(video, kernel, mode="edge", tol=1e-10, cval=0.0):
    """
    video: (F,H,W) array-like
    kernel: (kH,kW) array-like, odd sizes recommended
    mode: "edge"/"reflect"/"constant"/"wrap"
    return: cupy.ndarray (F,H,W) float32
    """
    video_gpu = cp.asarray(video, dtype=cp.float32)
    kernel_gpu = cp.asarray(kernel, dtype=cp.float32)

    if video_gpu.ndim != 3:
        raise ValueError(f"video must be (F,H,W), got shape={video_gpu.shape}")
    if kernel_gpu.ndim != 2:
        raise ValueError(f"kernel must be (kH,kW), got shape={kernel_gpu.shape}")

    F, H, W = video_gpu.shape
    kH, kW = kernel_gpu.shape
    if (kH % 2) != 1 or (kW % 2) != 1:
        raise ValueError(f"kernel size should be odd, got {kernel_gpu.shape}")

    nd_mode = _mode_map(mode)

    # 判 separable
    fac = _separable_factors_from_svd(kernel_gpu, tol=tol)

    out = cp.empty_like(video_gpu, dtype=cp.float32)

    if fac is not None:
        ky, kx = fac  # ky: (kH,), kx: (kW,)
        # 2D separable conv: 先沿 x (axis=1 in 2D img?) 注意 img shape (H,W): axis=1 是 W
        # 再沿 y (axis=0) 是 H
        for f in range(F):
            tmp = cnd.convolve1d(video_gpu[f], kx, axis=1, mode=nd_mode, cval=cval)
            out[f] = cnd.convolve1d(tmp,       ky, axis=0, mode=nd_mode, cval=cval)
    else:
        # 非 separable：直接 2D 卷积
        for f in range(F):
            out[f] = cnd.convolve(video_gpu[f], kernel_gpu, mode=nd_mode, cval=cval)

    return out




def make_kernel(
    name: str,
    wsize: int,
    sigma: float = None,
    direction: str = None,
    normalize: bool = True,
):
    """
    name:
        "gaussian"
        "sobel"
        "prewitt"
        "laplacian"
        "log"   (Laplacian of Gaussian)
    wsize:
        odd integer
    sigma:
        spatial scale (required for gaussian / log / sobel)
    direction:
        "x" or "y" (for sobel / prewitt)
    """

    if wsize % 2 != 1:
        raise ValueError("wsize must be odd")

    k = wsize // 2
    ax = np.arange(-k, k + 1, dtype=np.float32)

    # --------------------------------------------------
    # Gaussian (separable but returned as 2D)
    # --------------------------------------------------
    if name.lower() == "gaussian":
        if sigma is None:
            raise ValueError("sigma required for gaussian")

        g = np.exp(-(ax**2) / (2 * sigma**2))
        g /= g.sum()
        K = np.outer(g, g)

    # --------------------------------------------------
    # Sobel (Gaussian smoothing × first derivative)
    # --------------------------------------------------
    elif name.lower() == "sobel":
        if sigma is None:
            raise ValueError("sigma required for sobel")
        if direction not in ("x", "y"):
            raise ValueError("direction must be 'x' or 'y'")

        g = np.exp(-(ax**2) / (2 * sigma**2))
        g /= g.sum()

        dg = -ax * np.exp(-(ax**2) / (2 * sigma**2))
        dg /= np.sum(np.abs(dg))  # scale-stable derivative

        if direction == "x":
            K = np.outer(g, dg)
        else:
            K = np.outer(dg, g)

    # --------------------------------------------------
    # Prewitt (box smoothing × derivative)
    # --------------------------------------------------
    elif name.lower() == "prewitt":
        if direction not in ("x", "y"):
            raise ValueError("direction must be 'x' or 'y'")

        smooth = np.ones_like(ax) / wsize
        deriv = ax / np.sum(np.abs(ax))

        if direction == "x":
            K = np.outer(smooth, deriv)
        else:
            K = np.outer(deriv, smooth)

    # --------------------------------------------------
    # Laplacian (discrete, isotropic)
    # --------------------------------------------------
    elif name.lower() == "laplacian":
        if wsize != 3:
            raise ValueError("laplacian usually defined for wsize=3")

        K = np.array(
            [[0,  1, 0],
             [1, -4, 1],
             [0,  1, 0]],
            dtype=np.float32
        )

    # --------------------------------------------------
    # LoG: Laplacian of Gaussian (not separable)
    # --------------------------------------------------
    elif name.lower() == "log":
        if sigma is None:
            raise ValueError("sigma required for log")

        xx, yy = np.meshgrid(ax, ax, indexing="ij")
        r2 = xx**2 + yy**2
        s2 = sigma**2

        K = (r2 - 2*s2) / (s2**2) * np.exp(-r2 / (2*s2))
        K -= K.mean()  # zero DC

    else:
        raise ValueError(f"Unknown kernel name: {name}")

    if normalize and name.lower() in ("gaussian",):
        K /= K.sum()

    return K.astype(np.float32)
