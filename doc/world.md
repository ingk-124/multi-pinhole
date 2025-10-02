# World Module Guide

The `multi_pinhole.world` module coordinates cameras, voxels, and occluding geometry into a complete simulation scene. It exposes helpers for managing world assets, persisting scenarios, and rasterizing voxel emission into camera images.

## Construction
`World` accepts optional voxel, camera, wall, and inside-function inputs. Missing values are replaced with defaults, after which each component is linked back to the world so they can request shared state when projecting rays.【F:multi_pinhole/world.py†L38-L118】 Cameras are normalized into a dictionary keyed by index, letting the world track per-camera visibility masks and projection matrices alongside the optical hardware.【F:multi_pinhole/world.py†L76-L118】

Walls, when provided, are validated as STL meshes and cached together with their bounding boxes, allowing later visibility checks to skip empty scenes quickly.【F:multi_pinhole/world.py†L119-L151】【F:multi_pinhole/world.py†L252-L276】 The constructor also primes bookkeeping for inside-vertex masks, per-eye projections, and sparse image operators so subsequent calls can reuse or refresh cached data as needed.【F:multi_pinhole/world.py†L99-L151】 Supplying an `inside_func` immediately seeds the inside-vertex mask, which is otherwise inferred lazily.【F:multi_pinhole/world.py†L153-L171】

## Scene Introspection
Utility methods such as `camera_info` and `voxel_info` summarize the active optics and grid, while the `save_world`/`load_world` pair persists complete scenarios using `dill` so experiments can be checkpointed and restored.【F:multi_pinhole/world.py†L173-L220】 Rich property setters ensure updates propagate appropriately: replacing the voxel clears cached projections, changing cameras preserves any reusable matrices, and swapping wall meshes refreshes visibility tables.【F:multi_pinhole/world.py†L222-L321】【F:multi_pinhole/world.py†L330-L384】

## Visibility Management
World maintains boolean masks that describe which voxel vertices are inside the region of interest and which are visible from each eye. `set_inside_vertices` evaluates a user-supplied predicate over the voxel grid, while `_find_visible_points` transforms candidate points into camera coordinates, checks them against aperture and wall meshes, and returns per-eye visibility flags.【F:multi_pinhole/world.py†L386-L480】【F:multi_pinhole/world.py†L432-L474】 The higher-level `_find_visible_vertices` orchestrates this process for all cameras, caching the results so expensive ray-casting runs are reused unless forced to refresh.【F:multi_pinhole/world.py†L482-L546】

## Projection Assembly
Once visibility is known, `_find_visible_voxels` and `_calculate_projection_matrix` (invoked through `set_projection_matrix`) build sparse matrices that map voxel emission to detector subpixels.【F:multi_pinhole/world.py†L548-L748】【F:multi_pinhole/world.py†L748-L889】 The implementation batches sub-voxel centers, queries each camera for projected footprints, and multiplies the resulting sparse operators to yield the final projection. Work chunks are sized adaptively based on sampling density so that large scenes remain tractable even with many voxels.【F:multi_pinhole/world.py†L700-L889】 Parallel execution via `joblib` allows heavy projection calculations to utilize multiple CPU cores when requested.【F:multi_pinhole/world.py†L818-L889】

## Accessing Results
`set_projection_matrix` populates `World._projection` with per-camera, per-eye sparse matrices and caches a combined `P_matrix` when all pieces are available.【F:multi_pinhole/world.py†L891-L949】 Callers can then retrieve visible voxels, projection blocks, and the assembled operator to analyze or render simulated images without recomputing the expensive geometry pipeline.【F:multi_pinhole/world.py†L222-L251】【F:multi_pinhole/world.py†L891-L949】 Ancillary helpers expose inside-vertex masks and wall ranges so downstream code can cull off-screen geometry or track simulation metadata.【F:multi_pinhole/world.py†L201-L251】
