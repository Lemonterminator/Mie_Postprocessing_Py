import numpy as np

def angle_signal_density(img, x0, y0, N_bins=360):
    h, w = img.shape
    y_idx, x_idx = np.indices((h, w))
    dx = x_idx - x0
    dy = y_idx - y0
    theta = np.arctan2(dy, dx)
    theta_deg = np.degrees(theta) % 360

    bins = np.linspace(0, 360, N_bins+1)
    inds = np.digitize(theta_deg, bins) - 1  # 0…N-1

    signal = np.zeros(N_bins)
    counts = np.zeros(N_bins)
    for b in range(N_bins):
        mask = (inds == b)
        signal[b] = img[mask].sum()
        counts[b] = mask.sum()

    density = signal / (counts + 1e-9)
    return bins[:-1], signal, density
