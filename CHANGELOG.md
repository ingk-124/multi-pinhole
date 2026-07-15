# Changelog

## 0.7.2

- Added `World.project()` for camera-summed or per-Eye forward projection.
- Added `World.backproject()` for the corresponding discrete adjoint.
- Added explicit cache and input-shape validation without triggering implicit
  projection construction.

Backprojection applies `P.T`; it is not an inverse reconstruction.

## 0.7.1

- Reduced wall-visibility time and peak memory with bounded point/triangle
  processing and vectorized triangle-ray pair intersections.
- Skipped Eye-rejected and aperture-occluded points in subsequent mesh tests,
  while preserving the existing center-Eye visibility model.
- Added regression coverage for preflight visibility reuse during projection
  construction and accelerated the same path for partial sub-voxel samples.
- Added reproducible preflight and partial-visibility benchmarks, including a
  full d=10 MST reference run.

Visibility work sizes remain internal implementation details. The optimized
path preserves reference visibility masks and projection matrices.

## 0.7.0

- Added fixed, capped adaptive, and uncapped ideal source-resolution policies.
- Added projection preflight estimates and optical chunk ordering.
- Integrated detector cell-area overlap and local finite-Eye etendue evaluation.
- Preserved trilinear source interpolation with per-sample voxel-volume weights.
- Moved reproducible performance and accuracy experiments to `benchmarks/`.
- Updated the projection roadmap and archived the deferred QA/PSF compression design.

This is a breaking pre-1.0 release: projection construction now requires an
explicit source resolution and uses `res_mode` to select the resolution policy.
