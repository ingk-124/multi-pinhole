# World Module Guide

The `multi_pinhole.world` module brings together voxels, cameras, and
optional occluders (STL "walls") into a simulated scene, and computes the
two things a `World` exists for: **which voxels each camera eye can see**,
and **the sparse matrix that maps voxel intensities to detector pixel
intensities**. This document explains those two computations — visibility
and projection assembly — step by step, grounded in the actual algorithm in
`multi_pinhole/world.py`, and ends with a worked example of the full
pipeline.

## Helper Utilities

Internal helpers smooth over array book-keeping:

* `multi_pinhole.utils.type_check_and_list` (imported from
  `multi_pinhole.utils`, not defined in this module) coerces a single object
  or list into a list while enforcing a required element type, with an
  optional default for `None`.【F:multi_pinhole/utils/__init__.py†L24-L54】 It
  backs `walls` and related homogeneous-list inputs. Camera registration has
  its own stable-key normalization described below.
* `type_list`, defined locally in this module, offers similar
  single-object-or-list normalization but is not currently called anywhere
  in `world.py`; it is effectively dead code left over from before
  `type_check_and_list` was factored out into
  `multi_pinhole.utils`.【F:multi_pinhole/world.py†L36-L74】
* `_blocks_lengths` and `_slice_blocks` operate on collections of point and
  sparse-matrix blocks, exposing lightweight slicing for later projection
  work.【F:multi_pinhole/world.py†L77-L148】

## Constructing a World

`World.__init__` accepts optional voxel, camera, wall, and `inside_func`
arguments.【F:multi_pinhole/world.py†L162-L228】 Absent inputs fall back to
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

## Scene Introspection and Persistence

`camera_info` and `voxel_info` summarize the registered sensors and grid.
`save_world`/`load_world` serialize complete scenes with `dill`
(`pickle`-compatible but able to serialize the closures used internally,
e.g. by `coordinate_transform`), making it easy to checkpoint long-running
simulations. Serialized projection caches carry an explicit schema version;
loading a legacy or incompatible version keeps reusable visibility results but
invalidates `_projection` and `_P_matrix` so they are recomputed safely.
【F:multi_pinhole/world.py†L283-L311】 Property setters for
cameras, voxels, and walls reuse cached visibility/projection data when
possible but otherwise call `_invalidate_visibility_cache()`, which resets
`_visible_vertices`, `_visible_voxels`, `_projection`, and `_P_matrix` back
to `None` placeholders so the next query recomputes from
scratch.【F:multi_pinhole/world.py†L376-L387】

## Visibility Evaluation

Visibility is computed at three granularities, each building on the last:
**points → vertices → voxels**.

### `find_visible_points`: the core per-eye visibility test

`find_visible_points(points, camera_idx, eye_idx=None)` is the primitive
that everything else calls.【F:multi_pinhole/world.py†L639-L716】 For the
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
  array.【F:multi_pinhole/world.py†L718-L770】
* `find_visible_voxels` aggregates that per-vertex result to each voxel's 8
  corners (`self.voxel.vertices_indices`) and reports one of three states
  per `(eye, voxel)` pair:【F:multi_pinhole/world.py†L772-L804】

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

## Projection Assembly

`set_projection_matrix(res, ...)` is the entry point that turns a `Voxel`
grid and a set of visible voxels into the sparse matrix that maps voxel
intensities to detector signal, for every camera and every
eye.【F:multi_pinhole/world.py†L1182-L1239】 For each `(camera, eye)` pair it
calls `_calc_voxel_image_for_eye`, then aggregates all eyes on a camera into
that camera's pixel-space `P_matrix`.

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
transpose. Backprojection is the discrete adjoint `P.T @ image`, not an
inverse reconstruction. Neither method starts an implicit projection build;
they raise `RuntimeError` when the requested matrix is not cached.

### `_calc_voxel_image_for_eye`: fully-visible vs. partially-visible voxels

This is the core, and most expensive, computation in the
module.【F:multi_pinhole/world.py†L806-L1145】 After computing voxel
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
`UV` pixel coordinates.【F:multi_pinhole/world.py†L1147-L1180】 Unlike
`calc_image_vec`, it does not run aperture/wall visibility checks or
rasterize onto subpixels — it is a thin wrapper around
`Eye.calc_rays`, useful for debugging geometry rather than for rendering.

## Visualization

`draw_camera_orientation` overlays voxel bounds, camera poses (delegating
to each `Camera.draw_camera_orientation`), and any registered walls in one
3D Matplotlib plot. Axis limits default to the union of voxel and wall
ranges (inflated by 10%) but can be overridden via keyword
arguments.【F:multi_pinhole/world.py†L1241-L1307】

## Worked example: visibility → projection → image

Putting the pipeline together end to end (see
`examples/small_voxel_projection.py` for the runnable version, and
`docs/overview.md` for the full setup code):

1. `world = World(voxel=voxel, cameras=[camera])` registers a 3×3×3 voxel
   grid and one camera with one pinhole eye.
2. `world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))`
   marks every grid vertex as "inside" — with no `inside_func`, the same
   default would apply automatically, but this makes it explicit.
3. `world.set_projection_matrix(res=1, parallel=1)` triggers, per eye:
   * `_find_visible_vertices` ray-traces the grid's 4×4×4 = 64 vertices
     against the camera's aperture (no walls here), caching a
     `(1, 64)` boolean array.
   * `find_visible_voxels` aggregates those 64 vertex results across each
     of the 27 voxels' 8 corners, producing a `(1, 27)` array of
     0/1/2 visibility flags.
   * `_calc_voxel_image_for_eye` samples 1 sub-voxel center per voxel
     (`res=1`, so the sub-voxel center is just the voxel's own gravity
     center), projects fully-visible voxels' centers through the eye with
     `calc_image_vec`, and (for any partially-visible voxels) re-checks
     visibility at the sample level before projecting.
   * Pixel binning is applied while each chunk is assembled. The per-eye
     pixel matrix lands in `world.projection[0][0]` (shape
     `(N_pixel, 27)`); `set_projection_matrix` sums the eye matrices into
     `world.P_matrix[0]`, also with shape `(N_pixel, 27)`.
4. Given any 27-element `emission` vector, `world.P_matrix[0] @ emission`
   is the simulated pixel image — no further ray-tracing needed unless the
   geometry (camera, voxel grid, apertures, walls, or inside-vertex mask)
   changes, in which case the corresponding caches are invalidated and the
   next `set_projection_matrix` call recomputes only what changed.
