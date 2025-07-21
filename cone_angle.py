import numpy as np


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
    if arr.ndim == 2:
        arr = arr[None]  # treat image as 1-frame video

    frames, h, w = arr.shape

    y_idx, x_idx = np.indices((h, w))
    dx = x_idx - x0
    dy = y_idx - y0
    theta = np.arctan2(dy, dx)
    theta_deg = np.degrees(theta) % 360

    bins = np.linspace(0, 360, N_bins + 1)
    inds = np.digitize(theta_deg, bins) - 1  # 0..N_bins-1

    flat = arr.reshape(frames, -1)
    signal = np.zeros((frames, N_bins), dtype=arr.dtype)
    counts = np.zeros(N_bins, dtype=int)

    for b in range(N_bins):
        mask = (inds == b).ravel()
        if np.any(mask):
            signal[:, b] = flat[:, mask].sum(axis=1)
            counts[b] = mask.sum()

    density = signal / (counts + 1e-9)

    if video.ndim == 2:
        return bins[:-1], signal[0], density[0]
    return bins[:-1], signal, density


def plot_angle_signal_density(bins, signal, *, log=False):
    """Visualize angular signal density for a video.

    Parameters
    ----------
    bins : ndarray
        Bin centers in degrees produced by :func:`angle_signal_density`.
    signal : ndarray
        Array of shape ``(frames, bins)`` with per-frame summed signal.
    log : bool, optional
        If ``True`` use ``log1p`` before plotting.
    """
    import matplotlib.pyplot as plt

    arr = np.log1p(signal) if log else signal
    frames, n_bins = arr.shape

    X, Y = np.meshgrid(bins, np.arange(frames))

    fig, (ax_heat, ax_contour) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    im = ax_heat.imshow(
        arr,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[bins[0], bins[-1], 0, frames],
    )
    fig.colorbar(im, ax=ax_heat, label="Log Signal" if log else "Signal")
    ax_heat.set_ylabel("Frame index")
    ax_heat.set_title("Angular Signal Density Heatmap")

    cont = ax_contour.contourf(X, Y, arr, levels=15, cmap="viridis")
    fig.colorbar(cont, ax=ax_contour, label="Log Signal" if log else "Signal")
    ax_contour.set_xlabel("Angle (degrees)")
    ax_contour.set_ylabel("Frame index")
    ax_contour.set_title("Angular Signal Density Contour")

    plt.tight_layout()
    plt.show()