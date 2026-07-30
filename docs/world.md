# World projection model

> **Level 2 — scientific model.** Use the [overview](overview.md) for routine
> calculations. This page is for choosing source resolution, interpreting
> visibility and projection values, or auditing numerical approximations.
> Sections explicitly marked **Implementation detail** may be skipped.

The `multi_pinhole.world` module brings together voxels, cameras, and
optional occluders (STL "walls") into a simulated scene, and computes the
two things a `World` exists for: **which voxels each camera eye can see**,
and **the sparse matrix that maps voxel intensities to detector pixel
intensities**. This document explains those two computations — visibility
and projection assembly — step by step, grounded in the actual algorithm in
`multi_pinhole/world.py`, and ends with a worked example of the full
pipeline.

## Reading map

| Need | Read |
|---|---|
| Know what is cached or saved | Scene lifecycle |
| Interpret visibility 0/1/2 | Visibility model |
| Choose `res`, `partial_res`, or adaptive mode | Projection accuracy |
| Understand `P`, `P.T`, and array shapes | Projection API |
| Audit chunking and sparse assembly | Implementation details |

## Scene lifecycle

### Constructing a World

`World.__init__` accepts optional voxel, camera, wall, and `inside_func`
arguments. Absent inputs fall back to
defaults (a blank `Voxel()`, no cameras, no walls) and are immediately wired
back to the world (`voxel.set_world(self)`, `camera.set_world(self)`) so
they can request shared state such as visibility results. Cameras are
normalized into a stable-key mapping (`self._cameras`). A list receives keys
from `range(len(cameras))`; a dictionary retains explicit keys such as
`{"left": camera_left, "right": camera_right}`. Removing a camera does not
renumber the remaining keys, and `add_camera(key, camera)` requires an
explicit key. `world.cameras` exposes this registry as a read-only mapping;
updates go through `add_camera`, `change_camera`, and `remove_camera`. An
explicit key-reset helper is left as future work. The world
allocates parallel per-camera dictionaries using those same keys to cache visibility
flags (`_visible_vertices`, `_visible_voxels`) and projection matrices
(`_projection` per eye, `_P_matrix` aggregated across eyes) — all
initialized to `None` until the corresponding computation runs. Providing
`inside_func` seeds the inside-vertex mask right away by calling
`set_inside_vertices`; otherwise it remains lazily initialized to "all
vertices inside" (see `inside_vertices` below).

Walls are normalized to a list of `stl.mesh.Mesh` objects. When changed
(the `walls` setter) they trigger cache invalidation, refresh pre-computed
mesh bounds via `update_min`/`update_max`, and store combined axis-aligned
limits for later plotting (`wall_ranges`).

### Introspection and persistence

`camera_info` and `voxel_info` summarize the registered sensors and grid.
Scene construction and complete calculation checkpoints have separate
formats:

- `World.from_config(path_or_mapping)` and `world.to_config(path)` use the
  cache-free JSON scene schema. See [the config reference](config.md).
- `world.save(path)`, `World.inspect_archive(path)`, and `World.load(path)`
  use a versioned archive containing a plain JSON manifest and a dill
  payload. `save_world` and `load_world` remain compatibility aliases. See
  [the serialization reference](serialization.md).

The archive manifest separates the library version, World serialization
schema, and projection cache schema. Schema-3 projection caches are reused.
Loading an incompatible projection cache schema keeps reusable visibility
results but invalidates `_projection` and `_P_matrix` so they are recomputed
safely. Legacy direct-dill Worlds can be loaded and immediately saved as
archives. Pickle/dill can execute arbitrary code: only load trusted files.

Property setters for
cameras, voxels, and walls reuse cached visibility/projection data when
possible but otherwise call `_invalidate_visibility_cache()`, which resets
`_visible_vertices`, `_visible_voxels`, `_projection`, and `_P_matrix` back
to `None` placeholders so the next query recomputes from
scratch.

## Visibility model

For ordinary use, call `find_visible_voxels(camera_idx)`. The returned
`(N_eye, N_voxel)` array uses `0` for hidden, `1` for partly visible, and `2`
for fully visible. The classification is based on the eight voxel corners.
A partly visible voxel is sampled again during projection.

This corner classification is bookkeeping, not an analytic visible-volume
calculation. Boundaries are ultimately approximated at subvoxel sample
centers, so a resolution sweep is required when wall, aperture, or `inside`
boundaries materially affect the result.

> **Implementation detail:** the rest of this section explains how point,
> vertex, and voxel masks are produced. Skip to
> [Projection accuracy and API](#projection-accuracy-and-api) if you only need
> to choose numerical settings.

`World` owns scene state and the per-camera visibility caches, while the
private `multi_pinhole._visibility` module contains the geometry-to-mask
calculations. `World.find_visible_points` remains the public entry point: it
resolves the camera key and prepares camera-coordinate points and walls.
Vertex and voxel helpers return new masks to `World`, which remains solely
responsible for assigning `_visible_vertices` and `_visible_voxels` and for
invalidating projection caches when scene geometry changes.

Visibility is computed at three granularities, each building on the last:
**points → vertices → voxels**.

### `find_visible_points`: the core per-eye visibility test

`find_visible_points(points, camera_idx, eye_idx=None)` is the primitive
that everything else calls. For the
selected camera and each of its eyes, it:

1. Converts `points` (world coordinates) into that camera's frame via
   `camera.world2camera`, and copies each wall mesh into the same frame
   (`stl_utils.copy_model(wall, -camera_position, rotation.T)`), so
   everything downstream is compared in one consistent frame.
2. Marks a point as (tentatively) visible only if it is in front of the eye
   along the optical axis: `camera_points[:, 2] >= eye.position[-1]`.
3. For every `Aperture` on the camera, runs
   `stl_utils.check_visible(mesh_obj=aperture.stl_model, start=eye.position,
   grid_points=camera_points, behind_start_included=True)` and requires the
   point to clear **all** apertures (`np.all(..., axis=0)`) — apertures are
   opaque surfaces that block a ray unless it passes through the modeled
   opening. This mirrors exactly how `Camera.calc_image_vec` treats
   apertures (see `docs/core.md`), which is what keeps voxel-level
   visibility consistent with per-ray rendering.
4. For every wall mesh, runs the same `check_visible` test (this time
   without `behind_start_included`, since walls are ordinary opaque
   geometry rather than an aperture plane) and ANDs the result in.

The result is a `(N_eye, N_points)` boolean matrix. `check_visible` itself
is a two-stage geometric test (cone prefilter, then exact
Möller–Trumbore ray-triangle intersection) implemented in
`multi_pinhole.utils.stl_utils` — see `docs/utilities.md` for exactly how it
determines whether a segment from the eye to a point crosses a mesh.

### From points to vertices to voxels

Testing every voxel's interior directly would be expensive, so the world
tests the voxel grid's **vertices** once and reuses the result for every
voxel that shares a vertex:

* `_find_visible_vertices` calls `find_visible_points` on the world's grid
  vertices (`self.voxel.grid`), but only for vertices flagged `True` in
  `inside_vertices` — vertices outside the modeled volume are left `False`
  without ever being ray-traced. Results are cached per camera in
  `_visible_vertices` as a `(N_eye, N_grid_vertices)` boolean
  array.
* `find_visible_voxels` aggregates that per-vertex result to each voxel's 8
  corners (`self.voxel.vertices_indices`) and reports one of three states
  per `(eye, voxel)` pair:

  * **`0` — not visible**: none of the voxel's 8 corner vertices are
    visible.
  * **`1` — partially visible**: some but not all corners are visible (the
    voxel straddles an occlusion boundary, e.g. an aperture edge or a wall
    silhouette).
  * **`2` — fully visible**: all 8 corners are visible; the projection
    pipeline below can then skip re-testing this voxel's interior and
    integrate it directly.

`set_inside_vertices(function)` is how a caller defines the "modeled
volume" in the first place: `function` is evaluated on the voxel grid's
`(x, y, z)` coordinates and must return a boolean mask over grid vertices
(e.g. "inside the torus", "inside the vacuum vessel"); vertices outside
that mask are excluded from visibility/projection work entirely, which is
both a correctness tool (don't render emission from outside the physical
device) and a significant performance optimization for grids that are
mostly empty space.

## Projection accuracy and API

`set_projection_matrix(res, ...)` is the entry point that turns a `Voxel`
grid and a set of visible voxels into the sparse matrix that maps voxel
intensities to detector signal, for every camera and every
eye. For each `(camera, eye)` pair it
calls `_calc_voxel_image_for_eye`, then aggregates all eyes on a camera into
that camera's pixel-space `P_matrix`.

Source volume integration uses composite midpoint quadrature at subvoxel
centers. Emission is trilinearly interpolated from voxel-center values and
each sample is weighted by owner-voxel volume divided by its sample count.
It preserves a constant field and reproduces affine fields in interior cells;
the outer half-cell is clamped to the nearest center. Partial visibility and
`inside` boundaries are Boolean tests at sample centers, not analytic cut-cell
integrals.

Adaptive resolution is a geometry heuristic based on local perspective
scale. It is not a bound on image error. `point_source_threshold` is not an
error tolerance, and `partial_res` does not guarantee boundary accuracy.
Final scientific results should be checked with a resolution sweep.

Before starting an expensive build, use the same source-resolution settings
with `preflight_projection`:

```python
work = world.preflight_projection(
    res=5,
    res_mode="auto",
    partial_res=3,
)
print(work.summary())
print(work.total_samples_upper_bound)
```

The report separates fully and partially visible voxels for every eye, lists
the selected full-voxel resolution buckets, and reports the ideal-resolution
percentiles and clipped-axis count for adaptive runs. Its total is an exact
count for the full-voxel source samples plus a conservative pre-mask upper
bound for partial voxels. It is not a runtime or sparse-`nnz` prediction.
Preflight computes and caches voxel visibility, but does not construct or
modify `projection` or `P_matrix`; a subsequent build can reuse that
visibility result.

After construction, `world.project(emission, camera_idx, eye_idx=None)`
applies the cached camera-summed matrix, or one Eye matrix when `eye_idx` is
given. `world.backproject(image, camera_idx, eye_idx=None)` applies its
transpose. Both accept either a vector or a column-wise batch: shapes
`(N_voxel, N_rhs)` and `(N_pixel, N_rhs)` preserve `N_rhs` in the result.
Backprojection is the discrete adjoint `P.T @ image`, not an inverse
reconstruction. Neither method starts an implicit projection build; they
raise `RuntimeError` when the requested matrix is not cached.

> **Ordinary scientific use can stop here.** The remainder explains how this
> contract is implemented and optimized.

## Implementation details

`World` owns projection settings and cache lifecycle. The private
`multi_pinhole._projection_matrix` module performs optical-bin quadrature and
sparse assembly from explicit geometry inputs, returning CSR matrices without
mutating a `World` cache. The remaining contiguous-voxel path stays in
`World` because adaptive scheduling, visibility callbacks, progress, and
parallel task lifecycle are still coordinated there.

### `_calc_voxel_image_for_eye`: fully-visible vs. partially-visible voxels

This is the core, and most expensive, computation in the
module. After computing voxel
visibility (see above), it splits voxels into two groups and handles them
differently, because a fully-visible voxel doesn't need any further
ray-tracing:

* **Fully visible voxels (`vis_flag == 2`)**: sample `res` sub-voxel points
  per voxel (`Voxel.get_sub_voxel_centers`), project *all* of them through
  the eye with `Camera.calc_image_vec(..., check_visibility=False)`
  (visibility is already known, so the expensive aperture/wall occlusion
  test is skipped), and combine the sub-voxel image with an interpolation
  matrix `S` (see below) to produce one column per voxel.
* **Partially visible voxels (`vis_flag == 1`)**: sample the same sub-voxel
  points, but first re-run `find_visible_points` on those specific
  sub-voxel centers (since some of the parent voxel's interior may be
  occluded even though not all 8 corners agree), mask out invisible
  samples, and only project the surviving ones.

Both paths route through `_sub_voxel_interpolator_matrix`, which builds the
matrix `S` mapping values at voxel centers directly to weighted sub-voxel
samples.  An interior sample uses trilinear weights from at most eight
neighboring voxel centers; samples in the outer half of a boundary voxel are
clamped to the nearest center.  Each row is scaled by
`voxel.volume / samples_per_voxel`. That scaling is what turns a sum over a
voxel's sub-voxel rows into an approximation of the *integral* over the
voxel's volume — increasing `res` refines the quadrature without changing the
total integrated signal. Direct center interpolation also reproduces affine
emission profiles in the grid interior without the wider smoothing stencil of
the former center-to-vertex-to-sub-voxel interpolation.

This is composite midpoint quadrature at subvoxel centers. Constant fields are
preserved, and affine fields are reproduced in interior cells; outer half-cells
clamp to the nearest voxel center. Partial visibility and the inside boundary
are Boolean tests at sample centers, not analytic cut-volume integration.
Boundary accuracy therefore depends on `partial_res`; use a resolution sweep
and check convergence when boundary accuracy matters.

Concretely, for a batch of voxels the persistent per-eye pixel projection is

```
P_eye = T_pixel_from_subpixel @ calc_image_vec(eye, sub_voxel_centers) @ S
```

where `calc_image_vec` (from `docs/core.md`) is the `(N_subpixel,
N_sub_voxel_samples)` ray-tracing/rasterization matrix, `T` is the exact
`(N_pixel, N_subpixel)` detector-binning matrix, and `S` is the
`(N_sub_voxel_samples, N_voxel_batch)` interpolation/integration matrix.
Thus `P_eye` has shape `(N_pixel, N_voxel_batch)`. Applying `T` before
persistent sparse assembly retains subpixel quadrature accuracy without
retaining subpixel projection rows.

### Chunking and parallelism

Materializing `calc_image_vec` for every voxel's sub-voxel samples at once
can blow up memory (each ray can touch many subpixels). To bound this, the
function:

1. **Estimates sparsity** by running `calc_image_vec` on a small random
   sample of voxels (20, or fewer if there aren't that many) and measuring
   the average number of non-zero entries (`nnz`) per voxel.
2. **Picks a batch size** from a sample of the point, image, interpolation,
   and result matrices. The estimated transient bytes stay within
   `max_working_memory` (one billion bytes by default) across the bounded
   in-flight task set. The legacy `max_nnz` guard remains as a second cap.
3. **Processes chunks either serially or via a `ThreadPoolExecutor`**
   (`n_jobs > 1`), keeping at most twice `n_jobs` tasks in flight. Each task
   creates its sample points and interpolation matrix inside the worker and returns a COO
   `(data, row, col)` triplet rather than a full sparse matrix object, and
   `_process_parallel_chunks` drains completed futures immediately,
   consolidating buffered results when their array bytes reach a limit derived
   from `max_working_memory`, rather than after a fixed number of chunks. The
   buffered triplets are folded into a running sparse sum to bound peak memory
   rather than holding every chunk's result simultaneously.

This entire dance (steps 1-3, run separately for the full-visibility and
partial-visibility voxel groups) exists purely as a memory/throughput
trade-off — the mathematical result is the same sparse matrix regardless of
`n_jobs` or `max_nnz`; only the computation is chunked, not the answer.

`res` is mandatory. `res_mode="fixed"` uses it directly, while
`res_mode="auto"` interprets it as an axis-wise ceiling for fully-visible
voxels. The
voxel circumsphere is projected with the local worst-direction perspective
magnification, including the off-axis `1/cos(theta)` factor, and normalized by
the local finite-Eye PSF/detector scale. `point_source_threshold` defaults to
`1/8`. It selects res 1 when the complete voxel is negligible and determines a
near-cubic ideal axis-wise resolution otherwise. Voxels are bucketed by the
clipped `(r_x, r_y, r_z)`. This is a geometry heuristic, not a bound on image
error. Uncapped work requires the explicit combination
`res=None, res_mode="ideal"` and an explicit fixed `partial_res`.
Partially-visible voxels remain fixed because visibility is discontinuous;
with `fixed` or `auto`, omitted `partial_res` reuses `res`. A small fixed
`partial_res` does not provide an error bound for an arbitrarily positioned
visibility/inside boundary; validate it separately or specify a conservative
value for the geometry being integrated.

The scale is evaluated from local perspective geometry at the voxel center,
not as a rigorous upper bound over the whole finite voxel. It can underestimate
the projected size of a large voxel close to an Eye. `ideal` means uncapped
heuristic work, not ideal numerical accuracy, and `point_source_threshold` is
a sampling policy rather than an image-error tolerance. A capped axis may not
reach the recommendation; invalid geometry falls back to the configured
resolution.

Each projected subvoxel image is immediately passed through the screen's
`transform_matrix` (subpixel → pixel binning, from `docs/core.md`). Per-eye
pixel-space results are stored in `self._projection[camera_idx][eye_idx]`.
`set_projection_matrix` sums all eyes into `self._P_matrix[camera_idx]`.
Subpixel rows are transient integration data, not a persistent projection.

### `trace_line`: projecting a handful of points without building the full matrix

For quick checks (e.g. "where does this specific point land on the
screen?") without running the full projection pipeline,
`trace_line(points, camera_idx, eye_idx, coord_type)` projects `points`
through one eye and returns either camera-plane `XY` coordinates or screen
`UV` pixel coordinates. Unlike
`calc_image_vec`, it does not run aperture/wall visibility checks or
rasterize onto subpixels — it is a thin wrapper around
`Eye.calc_rays`, useful for debugging geometry rather than for rendering.

## Related task guides

The executable workflow is kept in one place:
[Overview and first projection](overview.md). Plotting camera orientation,
voxel fields, and detector images is kept in
[Visualization](visualization.md).
