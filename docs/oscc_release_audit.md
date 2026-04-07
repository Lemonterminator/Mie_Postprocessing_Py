# OSCC_postprocessing Release Audit

## Goal

This document tracks release-oriented cleanup for `OSCC_postprocessing` so the library can be audited subpackage by subpackage without losing scope.

Audit rules for each subpackage:

1. inventory public and internal functions
2. check real repo call sites
3. delete only functions with safe replacement or no remaining use
4. unify legacy backend/runtime paths where possible
5. add or improve docstrings on public and math-heavy helpers
6. run import and smoke validation after edits

## Package Inventory

- `analysis`: 15 files, 61 defs/classes
- `binary_ops`: 12 files, 69 defs/classes
- `cine`: 3 files, 12 defs/classes
- `dewe`: 5 files, 18 defs/classes
- `filters`: 6 files, 48 defs/classes
- `io`: 4 files, 5 defs/classes
- `masters_thesis`: 10 files, 41 defs/classes
- `metrics`: 2 files, 4 defs/classes
- `morphology`: 2 files, 0 defs/classes
- `motion`: 3 files, 21 defs/classes
- `playback`: 2 files, 6 defs/classes
- `rotation`: 5 files, 44 defs/classes
- `utils`: 7 files, 21 defs/classes

## Audit Status

### Done

- `rotation`
  - removed OpenCV CUDA/OpenCL rotation paths
  - deleted legacy modules `functions_rotation.py` and `rotate_crop.py`
  - replaced them with `rotation/segment_ops.py` plus the canonical `rotate_with_alignment(_cpu).py` engines
  - migrated in-repo Python callers to the new entry points
  - this is an explicit breaking change for users importing the deleted module paths directly
  - verified imports and direct smoke calls

- `playback`
  - no safe public-function deletions in this pass
  - refactored duplicated boundary overlay logic into shared helpers
  - added module-level and function-level docstrings for release readability
  - preserved existing public API names

### In Progress

- overall package-wide audit tracker and sequencing

### Not Started

- `analysis`
- `binary_ops`
- `cine`
- `dewe`
- `filters`
- `io`
- `masters_thesis`
- `metrics`
- `morphology`
- `motion`
- `utils`

## Decisions Logged

### Rotation

- `rotate_with_alignment.py` and `rotate_with_alignment_cpu.py` remain the canonical CuPy/NumPy remap implementations.
- `segment_ops.py` is the only remaining high-level crop/segment helper surface.
- callers that previously imported `rotation.rotate_crop` or `rotation.functions_rotation` must migrate to `rotation.segment_ops` or `rotate_with_alignment_cpu.py`.
- release-facing rotation usage and architecture notes now live in `docs/oscc_rotation.md`.

### Playback

- `save_video_with_boundaries_cv2` is currently retained even though it has no direct in-repo caller, because it is a plausible public utility for release users and has a clear, stable responsibility.

## Next Recommended Batch

1. `filters`
   - continue removing implementation-shaped legacy wrappers
   - normalize backend selection onto `utils.backend`
   - complete docstrings on convolution / background-removal modules

2. `binary_ops`
   - identify overlapping threshold / morphology entry points
   - clarify public API vs internal helpers

3. `analysis`
   - split or at least document orchestration-heavy modules further
   - remove remaining compatibility imports like `analysis.backend` where safe
