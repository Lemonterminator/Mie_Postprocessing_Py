import numpy as np
# Optional GPU acceleration via CuPy.
# Fall back to NumPy/SciPy on machines without CUDA (e.g. laptops).
try:  # pragma: no cover - runtime hardware dependent
    import cupy as cp  # type: ignore
    # from cupyx.scipy.ndimage import median_filter  # type: ignore
    CUPY_AVAILABLE = True
except Exception:  # ImportError, CUDA failure, etc.
    cp = np  # type: ignore
    # from scipy.ndimage import median_filter  # type: ignore
    cp.asnumpy = lambda x: x  # type: ignore[attr-defined]
    CUPY_AVAILABLE = False

def angle_signal_density(video, x0, y0, N_bins: int = 360):
    """Compute signal density versus angle for an image or video.

    Parameters
    ----------
    video : ndarray
        A single image ``(H, W)`` or a stack of frames ``(F, H, W)``.
    x0, y0 : float
        Coordinates of the center point used to compute the angle.
    N_bins : int, optional
        Number of angular bins spanning ``0``–``360`` degrees.

    Returns
    -------
    bins : ndarray of shape ``(N_bins,)``
        Bin centers in degrees.
    signal : ndarray
        Summed pixel values for each bin. Shape is ``(F, N_bins)`` when a video
        is provided or ``(N_bins,)`` for a single image.
    density : ndarray
        Average pixel value in each bin. Same shape as ``signal``.
    """

    arr = np.asarray(video)
    is_image = (arr.ndim == 2)
    if is_image:
        arr = arr[None]

    F, H, W = arr.shape
    y_idx, x_idx = np.indices((H, W))
    dx = x_idx - x0
    dy = y_idx - y0
    theta = np.arctan2(dy, dx)
    theta_deg = np.degrees(theta) % 360


    # Bin edges + per-pixel bin index
    edges = np.linspace(0, 360, N_bins + 1)
    bin_width = edges[1] - edges[0]
    bin_centers = edges[:-1] + 0.5 * bin_width
    inds = np.digitize(theta_deg, edges) - 1  # 0..N_bins-1
    inds = inds.ravel()

    # Pixel counts per bin (same for all frames)
    counts = np.bincount(inds, minlength=N_bins)

    # Grouped sum by bin for each frame (vectorized over pixels; loop only over frames)
    signal = np.empty((F, N_bins), dtype=np.result_type(arr.dtype, np.float64))
    flat = arr.reshape(F, -1)
    for f in range(F):
        signal[f] = np.bincount(inds, weights=flat[f], minlength=N_bins)

    # Avoid divide-by-zero
    density = signal / np.maximum(counts, 1)[None, :]

    if is_image:
        return bin_centers, signal[0], density[0]
    return bin_centers, signal, density


def plot_angle_signal_density(bins, signal, *, log=False, plume_angles=None):
    """Visualize angular signal density for a video or single frame.

    Parameters
    ----------
    bins : ndarray
        Bin centers in degrees produced by :func:`angle_signal_density`.
    signal : ndarray
        Array of shape ``(frames, bins)`` with per-frame summed signal.
    log : bool, optional
        If ``True`` use ``log1p`` before plotting.
    plume_angles : sequence of float, optional
        Angles in degrees indicating plume directions. Vertical lines are drawn
        at these angles.
    """
    import matplotlib.pyplot as plt

    arr = np.log1p(signal) if log else signal
    arr2d = arr if arr.ndim == 2 else arr[None]
    frames, n_bins = arr2d.shape

    bins_ext = np.concatenate((bins - 360, bins))
    arr_ext = np.concatenate((arr2d, arr2d), axis=1)

    X, Y = np.meshgrid(bins_ext, np.arange(frames))

    fig, (ax_heat, ax_contour) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    im = ax_heat.imshow(
        arr_ext,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[bins_ext[0], bins_ext[-1], 0, frames],
    )
    fig.colorbar(im, ax=ax_heat, label="Log Signal" if log else "Signal")
    ax_heat.set_ylabel("Frame index")
    ax_heat.set_title("Angular Signal Density Heatmap")
    ax_heat.set_xlim(-180, 180)

    cont = ax_contour.contourf(X, Y, arr_ext, levels=15, cmap="viridis")
    fig.colorbar(cont, ax=ax_contour, label="Log Signal" if log else "Signal")
    ax_contour.set_xlabel("Angle (degrees)")
    ax_contour.set_ylabel("Frame index")
    ax_contour.set_title("Angular Signal Density Contour")
    ax_heat.set_xlim(-180, 180)

    if plume_angles is not None:
        for ang in plume_angles:
            for shift in (-360, 0):
                ax_contour.axvline(ang + shift, color="cyan", linestyle="--")
                ax_heat.axvline(ang + shift, color="cyan", linestyle="--")

    plt.tight_layout()
    plt.show()



def angle_signal_density_cupy(video, x0, y0, N_bins: int = 360):
    arr = cp.asarray(video)
    is_image = (arr.ndim == 2)
    if is_image:
        arr = arr[None]

    F, H, W = arr.shape
    y_idx, x_idx = cp.indices((H, W))
    dx = x_idx - x0
    dy = y_idx - y0
    theta_deg = (cp.degrees(cp.arctan2(dy, dx)) % 360).astype(cp.float32)

    edges = cp.linspace(0, 360, N_bins + 1, dtype=cp.float32)
    bin_width = edges[1] - edges[0]
    bin_centers = cp.asnumpy(edges[:-1] + 0.5 * bin_width)

    inds = cp.digitize(theta_deg, edges) - 1
    inds = inds.ravel()

    counts = cp.bincount(inds, minlength=N_bins)
    flat = arr.reshape(F, -1)

    signal = cp.empty((F, N_bins), dtype=cp.float32)
    for f in range(F):  # GPU kernel inside; loop over frames is fine
        signal[f] = cp.bincount(inds, weights=flat[f], minlength=N_bins)

    density = signal / cp.maximum(counts, 1)[None, :]

    if is_image:
        return bin_centers, cp.asnumpy(signal[0]), cp.asnumpy(density[0])
    return bin_centers, cp.asnumpy(signal), cp.asnumpy(density)

def angle_signal_density_auto(video, x0, y0, N_bins: int = 360):
    if CUPY_AVAILABLE:
        return angle_signal_density_cupy(video, x0, y0, N_bins = 360)
    else:
        return angle_signal_density(video, x0, y0, N_bins = 360)
