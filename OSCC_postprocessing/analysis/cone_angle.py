import numpy as np
import matplotlib.pyplot as plt
# Optional GPU acceleration via CuPy.
# Fall back to NumPy/SciPy on machines without CUDA (e.g. laptops).
try:  # pragma: no cover - runtime hardware dependent
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA path could not be detected.*",
            category=UserWarning,
        )
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


def snap_angles_to_local_maxima(
    angles,
    ang_int_sum,
    *,
    window_deg: float = 2.0,
    bins: int | None = None,
    preserve_reference_side: bool = True,
):
    """Snap each angle to the nearest strong local maximum within a small window.

    Parameters
    ----------
    angles : array-like of float
        Input angles in degrees. Values can be negative; wrap-around is handled.
    ang_int_sum : array-like
        1D angular intensity profile (e.g. ``xp.sum(total_angular_signal_density, axis=0)``).
    window_deg : float, optional
        Half-window size in degrees used for local peak search around each angle.
    bins : int, optional
        Number of angular bins covering 360 degrees. If omitted, inferred from
        ``len(ang_int_sum)``.
    preserve_reference_side : bool, optional
        If ``True``, negative inputs keep a negative representation when
        appropriate, while non-negative inputs are returned in ``[0, 360)``.

    Returns
    -------
    ndarray
        Refined angles in degrees, one per input angle.
    """
    if window_deg < 0:
        raise ValueError("window_deg must be non-negative")

    signal = np.asarray(cp.asnumpy(ang_int_sum), dtype=float).ravel()
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("ang_int_sum must be a non-empty 1D array")

    n_bins = int(signal.size if bins is None else bins)
    if n_bins <= 0:
        raise ValueError("bins must be positive")
    if signal.size != n_bins:
        raise ValueError("len(ang_int_sum) must match bins")

    angles_arr = np.asarray(angles, dtype=float).ravel()
    if angles_arr.size == 0:
        return angles_arr.copy()

    # Circular local maxima mask.
    is_local_max = (signal >= np.roll(signal, 1)) & (signal >= np.roll(signal, -1))

    deg_per_bin = 360.0 / n_bins
    window_bins = int(np.ceil(window_deg / deg_per_bin))

    refined = np.empty_like(angles_arr)

    for i, angle in enumerate(angles_arr):
        angle_mod = angle % 360.0
        center_idx = int(np.round(angle_mod / deg_per_bin)) % n_bins

        offsets = np.arange(-window_bins, window_bins + 1, dtype=int)
        idx_window = (center_idx + offsets) % n_bins

        local_idx = idx_window[is_local_max[idx_window]]
        candidates = local_idx if local_idx.size else idx_window

        candidate_vals = signal[candidates]
        best_val = candidate_vals.max()
        ties = candidates[candidate_vals == best_val]

        # Break ties by nearest circular distance to the original center index.
        d1 = (ties - center_idx) % n_bins
        d2 = (center_idx - ties) % n_bins
        circular_dist = np.minimum(d1, d2)
        best_idx = int(ties[np.argmin(circular_dist)])

        snapped = best_idx * deg_per_bin
        if preserve_reference_side:
            if angle < 0:
                branch_options = np.array([snapped - 360.0, snapped])
                snapped = branch_options[np.argmin(np.abs(branch_options - angle))]
            else:
                snapped = snapped % 360.0

        refined[i] = snapped

    return refined



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
    bin_centers = cp.asarray(edges[:-1] + 0.5 * bin_width)

    inds = cp.digitize(theta_deg, edges) - 1
    inds = inds.ravel()

    counts = cp.bincount(inds, minlength=N_bins)
    flat = arr.reshape(F, -1)

    signal = cp.empty((F, N_bins), dtype=cp.float32)
    for f in range(F):  # GPU kernel inside; loop over frames is fine
        signal[f] = cp.bincount(inds, weights=flat[f], minlength=N_bins)

    density = signal / cp.maximum(counts, 1)[None, :]

    if is_image:
        return bin_centers, cp.asarray(signal[0]), cp.asarray(density[0])
    return bin_centers, cp.asarray(signal), cp.asarray(density)

def angle_signal_density_auto(video, x0, y0, N_bins: int = 360):
    if CUPY_AVAILABLE:
        return angle_signal_density_cupy(video, x0, y0, N_bins=N_bins)
    else:
        return angle_signal_density(video, x0, y0, N_bins=N_bins)
