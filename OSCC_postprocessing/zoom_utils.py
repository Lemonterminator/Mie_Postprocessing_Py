"""Helpers for simple integer-based zoom operations."""

from PIL import Image
import numpy as np
import cv2


def enlarge_image(img: Image.Image, factor: int) -> Image.Image:
    """Return ``img`` scaled by ``factor`` using nearest-neighbor tiling.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image to scale.
    factor : int
        Integer zoom factor. Values less than or equal to 1 return the original
        image.
    """
    if factor <= 1:
        return img

    arr = np.array(img)
    if arr.size == 0:
        # Nothing to scale
        return img

    # Use OpenCV's resize for better performance with nearest interpolation
    arr = cv2.resize(
        arr,
        (arr.shape[1] * factor, arr.shape[0] * factor),
        interpolation=cv2.INTER_NEAREST,
    )
    return Image.fromarray(arr, mode=img.mode)