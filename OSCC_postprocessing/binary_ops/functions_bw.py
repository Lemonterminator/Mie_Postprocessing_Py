"""Backward-compatible re-export layer for binary post-processing utilities.

This module used to contain the full implementation. The code now lives in
smaller focused modules, but the public names remain here so older scripts
and notebooks continue to import successfully.
"""

from ._backend import CUCIM_AVAILABLE, CUPY_AVAILABLE, cndi, cp
from .boundaries import (
    _boundary_points_one_frame,
    bw_boundaries_all_plumes,
    bw_boundaries_single_plume,
    bw_boundaries_xband_filter__single_plume,
    bw_boundaries_xband_filter_all_plumes,
)
from .boundary_io import load_boundary_file
from .connected_components import (
    __regionprops_3d_to_df,
    _generate_neighbor_offsets,
    _gpu_label_propagation,
    _return_like_input,
    _slices_for_offset,
    _to_cupy,
    _to_numpy_host,
    keep_largest_component,
    keep_largest_component_cuda,
    keep_largest_component_nd,
    keep_largest_component_nd_cuda,
    penetration_bw_to_index,
    reconstruct_blob,
    regionprops_3d,
)
from .feature_extraction import spary_features_from_bw_video
from .thresholding import (
    _fill_frame,
    _triangle_threshold_from_hist,
    apply_hole_filling,
    apply_hole_filling_video,
    apply_morph_open,
    apply_morph_open_video,
    binarize_video_global_threshold,
    calculate_bw_area,
    fill_video_holes_gpu,
    fill_video_holes_parallel,
    mask_video,
    triangle_binarize_from_float,
    triangle_binarize_u8,
)

# Historical aliases kept for old notebooks and third-party scripts.
triangle_binarize = triangle_binarize_from_float
bw_boundaries_all_points = bw_boundaries_all_plumes
bw_boundaries_xband_filter = bw_boundaries_xband_filter_all_plumes

__all__ = [
    "CUCIM_AVAILABLE",
    "CUPY_AVAILABLE",
    "cndi",
    "cp",
    "_boundary_points_one_frame",
    "_fill_frame",
    "_generate_neighbor_offsets",
    "_gpu_label_propagation",
    "_return_like_input",
    "_slices_for_offset",
    "_to_cupy",
    "_to_numpy_host",
    "_triangle_threshold_from_hist",
    "__regionprops_3d_to_df",
    "apply_hole_filling",
    "apply_hole_filling_video",
    "apply_morph_open",
    "apply_morph_open_video",
    "binarize_video_global_threshold",
    "bw_boundaries_all_points",
    "bw_boundaries_all_plumes",
    "bw_boundaries_single_plume",
    "bw_boundaries_xband_filter",
    "bw_boundaries_xband_filter__single_plume",
    "bw_boundaries_xband_filter_all_plumes",
    "calculate_bw_area",
    "fill_video_holes_gpu",
    "fill_video_holes_parallel",
    "keep_largest_component",
    "keep_largest_component_cuda",
    "keep_largest_component_nd",
    "keep_largest_component_nd_cuda",
    "load_boundary_file",
    "mask_video",
    "penetration_bw_to_index",
    "reconstruct_blob",
    "regionprops_3d",
    "spary_features_from_bw_video",
    "triangle_binarize",
    "triangle_binarize_from_float",
    "triangle_binarize_u8",
]
