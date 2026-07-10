# World Module Guide

The `multi_pinhole.world` module brings together voxels, cameras, and optional
occluders to form a simulated scene. Besides maintaining references between
components, it provides utilities for discovering visibility relationships,
assembling sparse projection matrices, persisting scenarios, and visualising the
setup.

## Helper Utilities
Two internal helpers smooth over array book-keeping:

* `type_list` coerces a single object or list into a list while enforcing a
  required element type. It backs several setters so that callers can supply
  either single instances or collections.
* `_blocks_lengths` and `_slice_blocks` operate on collections of point and
  sparse-matrix blocks, exposing lightweight slicing for later projection work.

## Constructing a World
`World.__init__` accepts optional voxel, camera, wall, and `inside_func`
arguments. Absent inputs fall back to defaults and are immediately wired back to
the world so they can request shared state. Cameras are normalised into an index
mapping (`{int: Camera}`) and populate parallel dictionaries that cache
per-camera visibility flags, per-eye projection matrices, and camera-level
projection operators. Providing `inside_func` seeds the inside-vertex mask right
away; otherwise it remains lazily initialised to “all vertices inside”.

Walls are normalised to a list of `stl.mesh.Mesh` objects. When changed they
trigger cache invalidation, refresh pre-computed mesh bounds via `update_min`
and `update_max`, and store combined axis-aligned limits for later plotting.

## Scene Introspection and Persistence
`camera_info` and `voxel_info` summarise the registered sensors and grid.
`save_world`/`load_world` serialise complete scenes with `dill`, making it easy
to checkpoint long-running simulations. Property setters for cameras, voxels,
and walls reuse cached visibility/projection data when possible but otherwise
invalidate stale matrices so dependent calculations stay coherent.

## Visibility Evaluation
`set_inside_vertices` lets callers define the active volume by passing a
callable evaluated over the voxel grid. `_find_visible_points` converts points
into a camera’s frame, tests them against each eye, aperture, and wall, and
returns a boolean mask. `_find_visible_vertices` iterates that routine across
all cameras (or a specific one) and caches the per-eye, per-vertex results.
`find_visible_voxels` then aggregates vertex visibility into voxel-level flags
indicating invisible, partially visible, or fully visible voxels for every eye.

## Projection Assembly
`set_projection_matrix` coordinates the expensive projection pipeline. For each
camera eye, `_calc_voxel_image_for_eye` prepares sub-voxel interpolators,
evaluates visibility, and multiplies sparse operators to build a voxel-to-
subpixel matrix. Workloads are chunked adaptively based on estimated sparsity
and can run in parallel through a `ThreadPoolExecutor` (or serially if only one
job is requested). Per-eye results accumulate in `_projection`, while combined
screen-space matrices live in `_P_matrix` after applying the screen transform.

## Visualisation
`draw_camera_orientation` offers a convenient 3D plot that overlays voxel
bounds, camera poses, and any registered walls. Axis limits default to the union
of voxel and wall ranges (inflated slightly) but can be overridden via keyword
arguments.
