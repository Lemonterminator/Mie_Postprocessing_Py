"""Binary mask operations used throughout OSCC post-processing."""

from .boundary_io import load_boundary_file
from .connected_components import (
    keep_largest_component,
    keep_largest_component_cuda,
    keep_largest_component_nd,
    keep_largest_component_nd_cuda,
    penetration_bw_to_index,
    reconstruct_blob,
    regionprops_3d,
)
from .feature_extraction import (
    extract_single_plume_features,
    processing_from_binarized_video,
    spary_features_from_bw_video,
)
from .morphological import (
    close_video_parallel,
    dilate_video_parallel,
    erode_video_parallel,
    open_video_parallel,
)
from .functions_bw import *  # noqa: F401,F403
from .masking import (
    generate_angular_mask_from_tf,
    generate_plume_mask,
    generate_ring_mask,
    periodic_true_segment_angles,
    periodic_true_segment_lengths,
)
from .thresholding import (
    binarize_video_global_threshold,
    fill_video_holes_parallel,
    mask_video,
    triangle_binarize_from_float,
    triangle_binarize_u8,
)

__all__ = [
    "binarize_video_global_threshold",
    "close_video_parallel",
    "dilate_video_parallel",
    "erode_video_parallel",
    "extract_single_plume_features",
    "fill_video_holes_parallel",
    "generate_angular_mask_from_tf",
    "generate_plume_mask",
    "generate_ring_mask",
    "keep_largest_component",
    "keep_largest_component_cuda",
    "keep_largest_component_nd",
    "keep_largest_component_nd_cuda",
    "load_boundary_file",
    "mask_video",
    "penetration_bw_to_index",
    "periodic_true_segment_angles",
    "periodic_true_segment_lengths",
    "reconstruct_blob",
    "regionprops_3d",
    "open_video_parallel",
    "processing_from_binarized_video",
    "spary_features_from_bw_video",
    "triangle_binarize_from_float",
    "triangle_binarize_u8",
]
