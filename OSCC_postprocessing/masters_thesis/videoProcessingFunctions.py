"""Video processing helpers used by the packaged Masters-thesis pipeline."""

from __future__ import annotations

import cv2
import numpy as np


def createBackgroundMask(first_frame: np.ndarray, threshold: int = 10) -> np.ndarray:
    return (first_frame > threshold).astype(np.uint8) * 255


def findFirstFrame(video: np.ndarray, threshold: int = 10) -> int:
    nframes = video.shape[0]
    for i in range(1, nframes):
        if video[i].mean() > threshold:
            return i
    return 0


def applyCLAHE(video: np.ndarray, clipLimit: float = 2.0, tileGridSize: tuple[int, int] = (8, 8)) -> np.ndarray:
    nframes = video.shape[0]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    out = np.empty_like(video)
    for i in range(nframes):
        out[i] = clahe.apply(video[i])
    return out


def tags_segmentation(
    spray_img: np.ndarray,
    background_img: np.ndarray,
    cell_size: int = 5,
    n_bins: int = 9,
    norm_order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    def get_gradient_statistics_vectors(img: np.ndarray) -> np.ndarray:
        img_gamma = np.sqrt(img.astype(np.float32))

        gx = cv2.copyMakeBorder(img_gamma, 0, 0, 1, 1, cv2.BORDER_REPLICATE)
        gx = gx[:, 2:] - gx[:, :-2]

        gy = cv2.copyMakeBorder(img_gamma, 1, 1, 0, 0, cv2.BORDER_REPLICATE)
        gy = gy[2:, :] - gy[:-2, :]

        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        h, w = img.shape
        statistics_volume = np.zeros((h, w, n_bins), dtype=np.float32)
        bin_width = 360.0 / n_bins

        for i in range(n_bins):
            lower = i * bin_width
            upper = (i + 1) * bin_width
            bin_mask = (angle >= lower) & (angle < upper)
            bin_magnitude = np.where(bin_mask, magnitude, 0)
            statistics_volume[:, :, i] = cv2.boxFilter(
                bin_magnitude,
                -1,
                (cell_size, cell_size),
                normalize=False,
            )

        sum_v = np.sum(statistics_volume, axis=2, keepdims=True)
        return np.divide(
            statistics_volume,
            sum_v,
            out=np.zeros_like(statistics_volume),
            where=sum_v != 0,
        )

    vn_spray = get_gradient_statistics_vectors(spray_img)
    vn_bg = get_gradient_statistics_vectors(background_img)

    diff = np.abs(vn_spray - vn_bg)
    if norm_order == 1:
        diff_map = np.sum(diff, axis=2)
    else:
        diff_map = np.power(np.sum(np.power(diff, norm_order), axis=2), 1.0 / norm_order)

    diff_map_8u = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary_mask = cv2.threshold(diff_map_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask, diff_map
