from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_fill_holes

from OSCC_postprocessing.utils.backend import get_cupy


def binary_fill_holes_cpu(mask, mode="3D"):
    """Fill binary mask holes in a 3D volume or frame-wise 2D mode."""
    mask_bool = np.asarray(mask, dtype=bool)
    if mode.upper() == "3D" or mask_bool.ndim < 3:
        return binary_fill_holes(mask_bool)

    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[1, :, :] = True
    return binary_fill_holes(mask_bool, structure=struct)


def fallback_fill_holes(mask_cp, mode="3D"):
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("CuPy is not available; cannot convert GPU mask to CPU.")

    filled = binary_fill_holes_cpu(cp.asnumpy(mask_cp), mode=mode)
    return cp.asarray(filled)


def binary_fill_holes_gpu(mask_cp, mode="3D"):
    """GPU hole filling with CPU fallback if cupyx lacks support."""
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("CuPy is not available; cannot fill holes on GPU.")

    import cupyx.scipy.ndimage as cndi  # type: ignore

    try:
        if mode.upper() == "3D" or mask_cp.ndim < 3:
            return cndi.binary_fill_holes(mask_cp)
        struct = cp.zeros((3, 3, 3), dtype=bool)
        struct[1, :, :] = True
        return cndi.binary_fill_holes(mask_cp, structure=struct)
    except Exception:
        return fallback_fill_holes(mask_cp, mode=mode)


def penetration_gpu(bw_cp):
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("CuPy is not available; cannot compute GPU penetration.")

    arr = bw_cp.astype(bool)
    any_true = arr.any(axis=1)
    rev_idx = arr.shape[1] - 1 - arr[:, ::-1].argmax(axis=1)
    rev_idx = rev_idx.astype(int)
    return cp.where(any_true, rev_idx, 0)


def regionprops_gpu(label_img, intensity_img=None):
    """Minimal regionprops-like utility for CuPy labels."""
    cp = get_cupy()
    if cp is None:
        raise RuntimeError("CuPy is not available; cannot run GPU regionprops.")

    import cupyx.scipy.ndimage as cndi  # type: ignore

    label_img = cp.asarray(label_img)
    if label_img.size == 0:
        return []

    label_ids = cp.unique(label_img)
    label_ids = label_ids[label_ids > 0]
    if label_ids.size == 0:
        return []

    ones = cp.ones_like(label_img, dtype=cp.float32)
    areas = cndi.sum(ones, label_img, label_ids)

    valid_mask = areas > 0
    label_ids = label_ids[valid_mask]
    areas = areas[valid_mask]
    if label_ids.size == 0:
        return []

    centroids = cndi.center_of_mass(ones, label_img, label_ids)

    coords = cp.argwhere(label_img > 0)
    if coords.size == 0:
        return []
    labels_flat = label_img[label_img > 0]

    coords_cpu = coords.get()
    labels_cpu = labels_flat.get()

    bboxes = defaultdict(lambda: [None, None])
    for idx, lab in zip(coords_cpu, labels_cpu):
        lab = int(lab)
        if bboxes[lab][0] is None:
            bboxes[lab][0] = list(idx)
            bboxes[lab][1] = list(idx)
        else:
            bmin, bmax = bboxes[lab]
            for i, v in enumerate(idx):
                if v < bmin[i]:
                    bmin[i] = v
                if v > bmax[i]:
                    bmax[i] = v

    label_ids_cpu = label_ids.get()
    areas_cpu = areas.get()
    centroids_cpu = [tuple(float(c) for c in cp.asnumpy(cen)) for cen in centroids]

    props = []
    for i, lab in enumerate(label_ids_cpu):
        bbox_min, bbox_max = bboxes[int(lab)]
        bbox = tuple(bbox_min + [x + 1 for x in bbox_max])
        props.append(
            {
                "label": int(lab),
                "area": float(areas_cpu[i]),
                "centroid": centroids_cpu[i],
                "bbox": bbox,
            }
        )
    return props


__all__ = [
    "binary_fill_holes_cpu",
    "binary_fill_holes_gpu",
    "fallback_fill_holes",
    "penetration_gpu",
    "regionprops_gpu",
]
