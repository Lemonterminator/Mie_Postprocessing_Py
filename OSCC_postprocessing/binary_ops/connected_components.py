"""Connected-component and region-property utilities for binary masks.

This module serves two roles:

1. Core binary-mask post-processing:
   - keep only the largest connected component
   - convert column occupancy into penetration indices
2. Lightweight region-measurement helpers:
   - a 3D ``regionprops``-like wrapper for NumPy and CuPy inputs

The CPU implementations are the default and should be treated as the reference
behavior. GPU variants are acceleration-oriented wrappers that preserve the
same high-level API when CuPy / cuCIM are available.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import generate_binary_structure
from skimage import measure

from ._backend import (
    CUCIM_AVAILABLE,
    CUPY_AVAILABLE,
    cndi,
    cp,
    cucim_measure,
    return_like_input,
    to_cupy,
    to_numpy_host,
)

__all__ = [
    "keep_largest_component",
    "keep_largest_component_nd",
    "keep_largest_component_cuda",
    "keep_largest_component_nd_cuda",
    "regionprops_3d",
    "reconstruct_blob",
    "penetration_bw_to_index",
]


def _normalize_connectivity(ndim: int, connectivity: int | None) -> int:
    """Validate and normalize connectivity for a given dimensionality."""
    normalized = ndim if connectivity is None else int(connectivity)
    if not (1 <= normalized <= ndim):
        raise ValueError(f"connectivity must be in [1, {ndim}], got {normalized}")
    return normalized


def _largest_component_from_labeled(labels, num_features: int, dtype):
    """Extract the largest non-background component from a labeled array."""
    if num_features == 0:
        return np.zeros_like(labels, dtype=dtype)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return (labels == counts.argmax()).astype(dtype)


def keep_largest_component(bw, connectivity=2):
    """Keep the largest connected component in a 2D binary mask.

    Parameters
    ----------
    bw:
        2D binary-like array. Any non-zero value is treated as foreground.
    connectivity:
        ``1`` for 4-connectivity, ``2`` for 8-connectivity.
    """
    bw_array = np.asarray(bw)
    binary_mask = bw_array.astype(bool, copy=False)
    structure = (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        if connectivity == 1
        else np.ones((3, 3), dtype=bool)
    )
    labeled, num_features = ndimage.label(binary_mask, structure=structure)
    return _largest_component_from_labeled(labeled, int(num_features), bw_array.dtype)


def keep_largest_component_nd(bw, connectivity=None):
    """Keep the largest connected component in an nD binary mask.

    ``connectivity`` follows SciPy / skimage conventions:
    for 3D data ``1/2/3`` maps to 6/18/26-neighborhoods.
    """
    bw_array = np.asarray(bw)
    binary_mask = bw_array.astype(bool, copy=False)
    ndim = binary_mask.ndim
    connectivity = _normalize_connectivity(ndim, connectivity)

    structure = ndimage.generate_binary_structure(ndim, connectivity)
    labeled, num_features = ndimage.label(binary_mask, structure=structure)
    return _largest_component_from_labeled(labeled, int(num_features), bw_array.dtype)


def _generate_neighbor_offsets(ndim, connectivity):
    """Generate local offsets for iterative GPU label propagation.

    This helper is kept for backward compatibility and experiments. The main
    GPU path prefers cuCIM / cupyx labeling when available.
    """
    offsets = []
    for offset in product((-1, 0, 1), repeat=ndim):
        if all(value == 0 for value in offset):
            continue
        if sum(abs(int(value)) for value in offset) <= int(connectivity):
            offsets.append(tuple(int(value) for value in offset))
    return offsets


def _slices_for_offset(offset, shape):
    """Create matching source/destination slices for a given integer offset."""
    src = []
    dst = []
    for delta, extent in zip(offset, shape):
        if delta == 0:
            src.append(slice(0, extent))
            dst.append(slice(0, extent))
        elif delta > 0:
            src.append(slice(0, extent - delta))
            dst.append(slice(delta, extent))
        else:
            delta = -delta
            src.append(slice(delta, extent))
            dst.append(slice(0, extent - delta))
    return tuple(src), tuple(dst)


def _gpu_label_propagation(binary_mask, connectivity):
    """Fallback GPU connected-components via iterative label propagation.

    This implementation is intentionally conservative and is not the preferred
    production path. It exists as a pure-CuPy fallback when faster labeling
    backends are unavailable.
    """
    shape = binary_mask.shape
    labels = cp.zeros(shape, dtype=cp.int32)
    if binary_mask.any():
        labels[binary_mask] = cp.arange(1, int(binary_mask.sum()) + 1, dtype=cp.int32)
    if labels.max() == 0:
        lin_ids = cp.arange(labels.size, dtype=cp.int32).reshape(shape) + 1
        labels = cp.where(binary_mask, lin_ids, 0)

    offsets = _generate_neighbor_offsets(binary_mask.ndim, connectivity)
    changed = True
    iterations = 0
    while changed and iterations < 10_000:
        changed = False
        iterations += 1
        # Propagate the minimum label across all connected neighbors until the
        # label field converges.
        for offset in offsets:
            src_slice, dst_slice = _slices_for_offset(offset, shape)
            neighbor = labels[src_slice]
            current = labels[dst_slice]
            both_fg = (neighbor > 0) & (current > 0)
            if not both_fg.any():
                continue
            propagated = cp.where(both_fg, cp.minimum(current, neighbor), current)
            if (propagated != current).any():
                changed = True
                labels[dst_slice] = propagated
    return labels


def keep_largest_component_cuda(bw, connectivity=2):
    """GPU-aware 2D largest-component selection with CPU fallback.

    If ``bw`` is a NumPy array, the return value is NumPy. If ``bw`` is a CuPy
    array, the return value stays on GPU.
    """
    if not CUPY_AVAILABLE:
        return keep_largest_component(bw, connectivity=connectivity)

    bw_gpu = to_cupy(bw)
    binary_mask = bw_gpu != 0

    if CUCIM_AVAILABLE:
        labeled, num_features = cucim_measure.label(
            binary_mask,
            connectivity=connectivity,
            return_num=True,
        )
        if int(num_features) == 0:
            return return_like_input(cp.zeros_like(binary_mask, dtype=bool), bw)
    else:
        largest_np = keep_largest_component(cp.asnumpy(binary_mask), connectivity=connectivity)
        return return_like_input(cp.asarray(largest_np), bw)

    labels_fg = labeled[binary_mask]
    uniq, counts = cp.unique(labels_fg, return_counts=True)
    largest_label = uniq[cp.argmax(counts)]
    return return_like_input(labeled == largest_label, bw)


def keep_largest_component_nd_cuda(bw, connectivity=None):
    """GPU-aware nD largest-component selection with CPU fallback."""
    if not CUPY_AVAILABLE or not isinstance(bw, cp.ndarray):
        return keep_largest_component_nd(bw, connectivity=connectivity)

    binary_mask = bw != 0
    ndim = binary_mask.ndim
    connectivity = _normalize_connectivity(ndim, connectivity)

    structure = cndi.generate_binary_structure(ndim, connectivity)
    labels, num_features = cndi.label(binary_mask, structure=structure)
    if int(num_features) == 0:
        return return_like_input(cp.zeros_like(binary_mask, dtype=bool), bw)

    labels_fg = labels[binary_mask].ravel()
    counts = cp.bincount(labels_fg)
    if counts.size <= 1:
        return return_like_input(cp.zeros_like(binary_mask, dtype=bool), bw)
    counts[0] = 0
    return return_like_input(labels == int(cp.argmax(counts)), bw)


def _regionprops_3d_to_df(props):
    """Convert region-property dictionaries into a flat DataFrame.

    The column naming stays explicit instead of storing tuples so the output is
    easier to serialize to CSV / parquet and simpler for downstream notebooks.
    """
    if not props:
        return pd.DataFrame()

    rows = []
    for prop in props:
        row = {"label": prop["label"]}
        if "volume" in prop:
            row["volume"] = prop["volume"]
        if "centroid" in prop:
            centroid = prop["centroid"]
            row["centroid_0"], row["centroid_1"], row["centroid_2"] = centroid
        if "bbox" in prop:
            bbox = prop["bbox"]
            row["bbox_min_0"], row["bbox_min_1"], row["bbox_min_2"] = bbox[:3]
            row["bbox_max_0"], row["bbox_max_1"], row["bbox_max_2"] = bbox[3:]
        rows.append(row)
    return pd.DataFrame(rows)


def regionprops_3d(
    bw,
    connectivity=None,
    volume=True,
    centroid=False,
    bbox=False,
    as_dataframe=True,
    return_labels=False,
):
    """MATLAB-like ``regionprops`` for a 3D binary volume.

    Parameters
    ----------
    bw:
        3D binary-like array of shape ``(F, H, W)``.
    connectivity:
        Connectivity in ``[1, 3]``. ``None`` defaults to full 3D connectivity.
    volume, centroid, bbox:
        Toggle which per-component measurements are computed.
    as_dataframe:
        When ``True``, return a flat pandas DataFrame. Otherwise return a list
        of dictionaries.
    return_labels:
        When ``True``, also return the full label volume.
    """
    if bw.ndim != 3:
        raise ValueError(f"regionprops_3d expects a 3D array, got ndim={getattr(bw, 'ndim', '?')}")

    try:
        connectivity = _normalize_connectivity(3, connectivity)
    except ValueError:
        connectivity = 3

    if CUPY_AVAILABLE and isinstance(bw, cp.ndarray):
        # GPU path: keep all heavy operations on device, but move sparse bbox
        # bookkeeping to host because it is simpler and not performance-critical.
        binary = bw != 0
        structure = cndi.generate_binary_structure(3, connectivity)
        labels, num_features = cndi.label(binary, structure=structure)
        if int(num_features) == 0:
            empty = _regionprops_3d_to_df([]) if as_dataframe else []
            return (empty, labels) if return_labels else empty

        label_ids = cp.unique(labels)
        label_ids = label_ids[label_ids > 0]
        ones = cp.ones_like(labels, dtype=cp.float32)
        areas = cndi.sum(ones, labels, label_ids) if volume else None
        if volume:
            valid = areas > 0
            label_ids = label_ids[valid]
            areas = areas[valid]

        centroids = cndi.center_of_mass(ones, labels, label_ids) if centroid else None
        bboxes = None
        if bbox:
            coords_cpu = cp.argwhere(labels > 0).get()
            labels_cpu = labels[labels > 0].get()
            bboxes = defaultdict(lambda: [None, None])
            for idx, lab in zip(coords_cpu, labels_cpu):
                lab = int(lab)
                if bboxes[lab][0] is None:
                    bboxes[lab][0] = list(idx)
                    bboxes[lab][1] = list(idx)
                else:
                    lower, upper = bboxes[lab]
                    for axis, value in enumerate(idx):
                        if value < lower[axis]:
                            lower[axis] = value
                        if value > upper[axis]:
                            upper[axis] = value

        props = []
        for idx, label_id in enumerate(label_ids.get()):
            prop = {"label": int(label_id)}
            if volume and areas is not None:
                value = areas[idx]
                prop["volume"] = float(value.get() if hasattr(value, "get") else value)
            if centroid and centroids is not None:
                prop["centroid"] = tuple(float(v) for v in cp.asnumpy(centroids[idx]))
            if bbox and bboxes is not None:
                lower, upper = bboxes[int(label_id)]
                prop["bbox"] = tuple(lower + [v + 1 for v in upper])
            props.append(prop)

        result = _regionprops_3d_to_df(props) if as_dataframe else props
        return (result, labels) if return_labels else result

    # CPU path: rely on SciPy for labeling and skimage for per-region geometry.
    binary = np.asarray(bw != 0, dtype=bool)
    structure = generate_binary_structure(3, connectivity)
    labels_np, _ = ndimage.label(binary, structure=structure)
    props = []
    for region in measure.regionprops(labels_np):
        prop = {"label": int(region.label)}
        if volume:
            prop["volume"] = int(region.area)
        if centroid:
            prop["centroid"] = tuple(float(value) for value in region.centroid)
        if bbox:
            prop["bbox"] = tuple(region.bbox)
        props.append(prop)
    result = _regionprops_3d_to_df(props) if as_dataframe else props
    return (result, labels_np) if return_labels else result


def reconstruct_blob(labels, label_id):
    """Reconstruct one labeled component as a binary mask."""
    return labels == label_id


def penetration_bw_to_index(bw):
    """Return the last occupied x-index for each row of a boolean mask.

    This is commonly used on column-summed plume masks:
    ``penetration_bw_to_index(mask.sum(axis=1) > threshold)``.

    Rows without any foreground return ``-1``.
    """
    arr = np.asarray(bw, dtype=bool)
    any_true = arr.any(axis=1)
    last_true = arr.shape[1] - 1 - arr[:, ::-1].argmax(axis=1)
    last_true[~any_true] = -1
    return last_true


# Backward-compatible aliases imported by the legacy ``functions_bw`` shim.
__regionprops_3d_to_df = _regionprops_3d_to_df
_to_numpy_host = to_numpy_host
_to_cupy = to_cupy
_return_like_input = return_like_input
