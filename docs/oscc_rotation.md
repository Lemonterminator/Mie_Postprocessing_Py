# OSCC Rotation Stack

## Purpose

This document summarizes the current structure of `OSCC_postprocessing.rotation`
after the legacy `rotate_crop.py` and `functions_rotation.py` cleanup. It is
the release-facing reference for which rotation entry points are still
supported and how the layers relate to each other.

## Module Layout

### `OSCC_postprocessing.rotation.segment_ops`

This is the high-level orchestration layer used by pipelines and analysis code.
It owns the geometry and helper surface for plume-oriented segmentation:

- `generate_CropRect`
- `generate_plume_mask`
- `rotate_video_auto`
- `rotate_all_segments_auto`

`generate_plume_mask` is not implemented locally in this module anymore. It is
re-exported from `OSCC_postprocessing.binary_ops.masking` so the library keeps
one canonical definition for angular plume masking. The module otherwise does
not implement interpolation itself and delegates all remapping to the canonical
affine engines described below.

### `OSCC_postprocessing.rotation.rotate_with_alignment_cpu`

This is the canonical NumPy implementation. It provides:

- affine construction helpers such as `build_rotation_affine`
- inverse-map builders such as `build_nozzle_rotation_maps` and `build_rotation_roi_maps_numpy`
- frame and stack remappers for nearest, bilinear, bicubic, and lanczos3 sampling
- high-level CPU workflows:
  - `rotate_video_about_center_numpy`
  - `rotate_video_nozzle_at_0_half_numpy`

Use this module when you need explicit access to inverse-coordinate maps or
deterministic CPU behaviour for testing and debugging.

### `OSCC_postprocessing.rotation.rotate_with_alignment`

This is the CuPy backend counterpart. It mirrors the CPU implementation closely
so higher-level code can keep the same geometry and interpolation assumptions.
Its main high-level entry points are:

- `rotate_video_about_center_cupy`
- `rotate_video_nozzle_at_0_half_cupy`

## Sign Conventions

There are two sign conventions worth documenting because they were inherited
from the historical API surface.

### Full-frame rotation

`segment_ops.rotate_video_auto(video, angle)` forwards `angle` directly into the
canonical affine rotation engine. Positive values follow the affine convention
implemented by `build_rotation_affine`.

### Segment extraction

`segment_ops.rotate_all_segments_auto(video, angles, crop, centre, mask=None)`
preserves the historical segment orientation used by the deleted
`rotate_crop.py`. Internally that means each requested segment angle is negated
before calling the canonical remapper.

This asymmetry is intentional. It keeps existing plume ordering and orientation
stable for downstream analysis code.

### Nozzle alignment

`rotate_video_nozzle_at_0_half_numpy` and `rotate_video_nozzle_at_0_half_cupy`
interpret `offset_deg` as the measured nozzle offset to remove. Internally they
rotate by `-offset_deg`, preserving the previous GUI calibration semantics.

## Recommended Usage

### For plume-segment analysis

Use the package exports:

```python
from OSCC_postprocessing.rotation import (
    generate_CropRect,
    generate_plume_mask,
    rotate_all_segments_auto,
)
```

### For nozzle-centering workflows

Prefer the backend-specific entry point:

```python
from OSCC_postprocessing.rotation.rotate_with_alignment_cpu import (
    rotate_video_nozzle_at_0_half_numpy,
)
```

Switch to the CuPy module only when the input stack already lives on GPU.

### For debugging affine geometry

Use the lower-level modules directly and inspect the returned `mapx` and `mapy`
arrays from:

- `rotate_video_about_center_numpy`
- `rotate_video_about_center_cupy`
- `rotate_video_nozzle_at_0_half_numpy`
- `rotate_video_nozzle_at_0_half_cupy`

These maps can also be visualized with `plot_inverse_maps`.

## Breaking Changes

The following modules were removed and are no longer supported import paths:

- `OSCC_postprocessing.rotation.rotate_crop`
- `OSCC_postprocessing.rotation.functions_rotation`

Callers must migrate to:

- `OSCC_postprocessing.rotation.segment_ops`
- `OSCC_postprocessing.rotation.rotate_with_alignment_cpu`
- `OSCC_postprocessing.rotation.rotate_with_alignment`

No compatibility shim is kept under the deleted filenames.

## Validation Notes

The migration was validated against pre-deletion baselines:

- crop-rectangle generation matches exactly
- plume-mask generation matches exactly
- nozzle-centering bool remap matches exactly
- float-valued rotation paths remain numerically close to the removed OpenCV
  wrapper implementation

A very small bool edge difference remains for one full-frame rotation smoke
test. The discrepancy is limited to border sampling at a couple of pixels and
comes from the difference between the old OpenCV warp path and the unified
inverse-map remap path.
