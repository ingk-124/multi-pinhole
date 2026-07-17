# Changelog

## 0.8.1

- Added stateless physical/normalized coordinate queries on `Voxel`, inverse
  keyword-coordinate conversion with NumPy broadcasting, and a shared
  immutable coordinate-type registry while preserving the configured
  `normalized_coordinates()` profile API.
- Added `Voxel.center_interpolator()` for scalar or vector fields sampled at
  gravity centers. Projection-specific volume-weighted source quadrature now
  uses a private builder; the former public-looking method remains as a
  compatibility wrapper.

## 0.8.0

- Split the optics implementation into responsibility-focused modules:
  `eye`, `aperture`, `screen`, `camera`, and `rays`. The public package and
  `multi_pinhole.core` compatibility facade continue to expose the same class
  objects and legacy pickle/dill globals.
- Moved geometry-only visibility calculations into the private
  `multi_pinhole._visibility` module. `World` retains public methods, scene
  state, cache ownership, and projection invalidation.
- Moved optical-bin source quadrature and sparse projection assembly into the
  private `multi_pinhole._projection_matrix` module. `World` retains camera
  selection, cache validation and assignment, serialization, and the tightly
  coupled adaptive contiguous-voxel orchestration.
- Preserved detector/source integration formulas, visibility ordering,
  projection/backprojection behavior, public signatures and defaults, and
  projection cache schema version 3.
- Added decomposition, legacy serialization, fixed-value visibility,
  serial/parallel projection, optical/contiguous assembly, partial-volume,
  and architecture regressions.

## 0.7.3

- Standardized public API docstrings toward NumPy style and documented
  defaults, shapes, units, exceptions, and callable contracts.
- Documented detector and source quadrature, visibility-boundary
  approximations, and the limits of adaptive source-resolution heuristics.
- Replaced the README sample with a runnable projection/preflight/project/
  backproject workflow and synchronized the English and Japanese guidance.
- Added documentation regression checks for public docstrings, Markdown
  links, stale source-line citations, and an executed README workflow.
- Fixed spherical polar-angle calculation so the reference radius scales only
  the dimensionless radius and no longer contaminates ``theta``.
- Preserved projection, etendue, visibility, and serialization behavior,
  public names/signatures/import paths, and projection-cache schema version 3.

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
