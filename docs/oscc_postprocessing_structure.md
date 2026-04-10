# OSCC_postprocessing Structure

## Overview

`OSCC_postprocessing` is a mixed scientific-computing package for spray-image postprocessing. It currently bundles four kinds of responsibility in one namespace:

- domain analysis: penetration, cone angle, hysteresis, nozzle geometry, multi-hole processing
- image processing: denoising, thresholding, morphology, optical flow, rotation/cropping
- data access and persistence: cine/dewe readers, async saving
- infrastructure: backend selection, scaling, package/runtime helpers

That breadth is useful for notebooks, but it also means coupling has grown over time. Several modules still mix algorithm code, backend dispatch, plotting-oriented diagnostics, and notebook-era assumptions.

## Current Package Map

### `analysis/`

Domain-facing logic. This is the part most users likely think of as the main library API.

- `multihole_processing.py`: shared preprocessing and penetration extraction for multi-hole spray videos
- `single_plume.py`, `penetration_cdf.py`, `cone_angle.py`, `hysteresis.py`: derived metrics and signal extraction
- `thresholding.py`: canonical triangle-threshold entry point and array-backend reexports
- `multihole_utils.py`: compatibility shim for older notebooks; prefer the canonical modules above
- `backend.py`: shared backend selection helpers
- `nozzle.py`, `circ_calculator.py`, `regression.py`, `video_utils.py`: supporting geometry and analysis utilities

Risk:
- some files still combine orchestration and low-level math in one module
- some APIs return backend-dependent array types while others silently convert to NumPy

### `filters/`

Image/video denoising and local transforms.

- `stdfilt.py`: local standard deviation filter with SciPy/CuPy routing
- `bilateral_filter_rawKernel.py`: 2D/3D bilateral filtering
- `convolution_2d.py`: frame-wise 2D convolution helpers
- `video_filters.py`, `svd_background_removal.py`: filtering pipelines and background suppression

Risk:
- several modules still own their own backend dispatch and fallback rules
- naming is partly implementation-shaped (`*_rawKernel.py`) rather than API-shaped

### `binary_ops/`

Binary mask creation and shape extraction.

- thresholding, connected components, morphology, masking, boundaries, and binary metrics
- `thresholding.py` now owns the unified triangle-threshold implementation used by both CPU and GPU call sites

Risk:
- functionality is split across several overlapping files
- CPU/GPU variants are not always presented through one stable public wrapper

### `rotation/`

Rotation, crop, and alignment logic for plume-centric coordinate systems.

Risk:
- multiple files appear to overlap in responsibility between alignment strategy and transformation kernels

### `motion/`

Optical-flow wrappers for classical and learned methods.

Risk:
- hardware/runtime setup and algorithm API live close together, which makes testing and fallback handling harder

### `cine/`, `dewe/`, `io/`

Data ingress and asynchronous output utilities.

Risk:
- these are infrastructure modules but live beside image-analysis algorithms in the same top-level package

### `utils/`

Cross-cutting support utilities.

- `backend.py` is now the correct place for unified NumPy/CuPy routing
- other files cover scaling, packaging, SAM runtime helpers, and general utilities

## Main Structural Issues

1. Backend selection is historically duplicated.

There has already been progress moving this into `utils/backend.py`, but some modules still perform local CuPy detection or bind `xp` at import time.

2. Public API is not clearly separated from implementation detail.

Files like `*_rawKernel.py` expose both user-facing functions and backend-specific internals. That makes the package harder to navigate and harder to test systematically.

3. Analysis modules mix pipeline orchestration with primitive operations.

For example, one file may both manage per-plume workflow and contain low-level threshold/edge logic. That raises cognitive load and makes reuse harder.

4. Array-type policy is inconsistent.

Some functions return NumPy unconditionally, some preserve CuPy, and some let `use_gpu` decide. This is workable in notebooks but brittle in libraries.

5. Historical notebook assumptions leak into modules.

Examples include silent printing, implicit shape conventions, and compatibility helpers kept only because old notebooks imported them directly.

For the triangle-threshold workflow specifically, the current canonical import is
`OSCC_postprocessing.analysis.thresholding.triangle_binarize`. The legacy
`triangle_binarize_gpu` name is retained only as a compatibility alias while
the examples and notebooks are migrated.

## Recommended Reorganization

### Phase 1: Stabilize interfaces

- Keep `utils/backend.py` as the only backend detection layer.
- Add clear docstrings for top-level/public functions first.
- Define one array-return policy for each subpackage:
  - either preserve input backend
  - or always return NumPy
  - but do not mix policies without explicit naming

### Phase 2: Separate API from implementation

Reshape by responsibility rather than implementation mechanism.

Suggested direction:

- `filters/bilateral.py`
  public bilateral API
- `filters/_bilateral_cuda.py`
  RawKernel implementation details
- `filters/std.py`
  public std filter API
- `filters/_std_cuda.py`
  CUDA kernel details

Likewise for binary operations and rotation.

### Phase 3: Split orchestration from primitives in `analysis/`

Suggested decomposition:

- `analysis/preprocess.py`
  background subtraction, normalization, mask generation
- `analysis/penetration.py`
  TD-map generation and penetration extraction
- `analysis/cone_angle.py`
  cone-angle specific methods only
- `analysis/hysteresis.py`
  1D signal-state logic only
- `analysis/nozzle.py`
  geometric/nozzle coordinate helpers only

The current `multihole_processing.py` can then become a thin pipeline wrapper around smaller primitives.

### Phase 4: Move I/O-adjacent code out of the scientific core

If the package continues to grow, consider a clearer split such as:

- `OSCC_postprocessing.core`
  reusable algorithms
- `OSCC_postprocessing.io`
  readers/writers/savers
- `OSCC_postprocessing.apps` or `pipelines`
  notebook- and workflow-oriented orchestration

This will make the scientific core easier to test and publish independently.

## Concrete Next Steps

1. Finish documentation pass on public functions in:
   - `analysis/`
   - `filters/`
   - `binary_ops/`

2. Remove remaining local backend detectors and convert them to `utils.backend`.

3. Introduce internal modules prefixed with `_` for CUDA kernels and backend-specific helpers.

4. Add a small API guide that names the preferred public entry points, so notebooks stop importing implementation files directly.

5. Add lightweight smoke tests per subpackage:
   - import test
   - NumPy-path functional test
   - CuPy-path smoke test where available

## Suggested Priority Order

- first: `binary_ops`, `filters`, `rotation` backend cleanup
- second: split `analysis/multihole_processing.py` into preprocessing and penetration layers
- third: consolidate duplicated thresholding and morphology entry points
- fourth: reduce notebook-era compatibility aliases once callers are migrated
