# Utilities Overview

The `multi_pinhole.utils` package (imported as `from multi_pinhole.utils
import ...` or `from multi_pinhole.utils import stl_utils`, etc. — it was
moved under `multi_pinhole` to avoid colliding with unrelated top-level
`utils` packages) collects helper functions that support the optical
simulation without cluttering the core camera logic. The most
computationally interesting piece is the STL geometry toolkit, which
implements the actual ray/mesh occlusion test used throughout
`multi_pinhole.core` and `multi_pinhole.world`; this document explains that
algorithm in detail, plus the simpler collection and logging helpers.

## Collection Helpers
`multi_pinhole.utils.type_check_and_list` normalizes optional constructor
arguments into lists while validating element types. It also accepts a
default fallback when `None` is provided, allowing world and camera
constructors to treat single objects and lists
uniformly.

## Console Wrappers
`multi_pinhole.utils.my_stdio` wraps common iteration primitives with
optional progress displays. `my_print` gates log messages on a `show` flag,
`my_range` and `my_tqdm` defer to `tqdm` versions when verbosity is enabled,
and `my_zip` pairs iterables with progress bars via `tzip`. These adapters
let long-running geometry routines expose progress feedback without
hard-coding dependencies on `tqdm` in calling code.

## STL Geometry Toolkit

`multi_pinhole.utils.stl_utils` builds and analyzes the triangle meshes used
by apertures and world walls.

### Constructing an aperture mesh

`shape_check(shape, size)` normalizes a shape keyword (`circle`, `ellipse`,
`rectangle`, `square`) and its size specification into a canonical
`(shape, (height, width))` pair, expanding a bare scalar to both
axes.
`generate_aperture_stl(shape, size, resolution, max_size)` then builds an
actual flat mesh in the `z = 0` plane:

1. Sample `resolution` points along the shape's boundary (a parametric
   ellipse `(a·cos t, b·sin t)`, or the four edges of a rectangle).
2. Fill in a coarse `10×10` background grid of points out to `max_size`
   (default `1.5×` the aperture size) *excluding* points already inside the
   shape, so the surrounding "opaque" material is represented too — this is
   what makes the mesh useful as an occluder, not just an outline.
3. Feed all points (boundary + background) to `make_2D_surface`, which
   Delaunay-triangulates them (`scipy.spatial.Delaunay`) and keeps only the
   triangles whose centroid lies **outside** the aperture opening (the
   `condition` predicate) — i.e. the mesh models the *opaque* material
   around the hole, not the hole itself. Rays that reach the hole simply
   never intersect any triangle in this mesh.

`rotate_model` and `copy_model` duplicate meshes and apply translations or
Euler-angle rotations without mutating the
originals.

### Visibility / occlusion testing

Determining whether a mesh blocks the line from an eye to a candidate point
is the single most performance-sensitive computation in the package (it
runs for every voxel vertex, and again for every sub-voxel sample of every
partially-visible voxel — see `docs/world.md`). It is implemented as a
two-stage test to avoid running an exact ray-triangle intersection against
every triangle for every point:

**Stage 1 — bounding-cone prefilter (`delta_cone_apply` /
`delta_cone_prepare`).** For each triangle `(a, b, c)` and a shared apex
(the eye position), the three "cone" planes through the apex and each edge
(`oa×ob`, `ob×oc`, `oc×oa`) are computed and oriented so that the
half-spaces they define agree with the triangle's own third vertex. A
candidate point lies inside the cone spanned by that triangle exactly when
it is on the same side of all three planes as the triangle interior
(`p · n ≥ 0` for all three oriented normals).
This is a cheap, batched (`einsum`/matrix-multiply) test that can rule out
the vast majority of (triangle, point) pairs — a point can only possibly be
occluded by a triangle if it lies inside that triangle's cone from the
eye's point of view. Results are stored as a sparse `(M_triangles,
N_points)` boolean matrix, batched over points to bound memory.

**Stage 2 — exact ray-triangle intersection (`check_intersection`).** Only
for the (triangle, point) pairs that survived stage 1, `check_intersection`
runs the standard **Möller–Trumbore algorithm**: with triangle edges
`e₁ = b−a`, `e₂ = c−a`, segment direction `d = point−start`, and offset
`r = start−a`, it solves `start + t·d = a + u·e₁ + v·e₂` for the barycentric
coordinates `(u, v)` and the parametric distance `t` via
`u = (r·(d×e₂))/det`, `v = (r·(e₁×d))/det`, `t = (r·(e₁×e₂))/det`, where
`det = −d·(e₁×e₂)`. The segment crosses the triangle iff
`u ≥ 0`, `v ≥ 0`, `u+v ≤ 1`, and `0 < t ≤ 1` (numerical tolerance `eps` is
applied to all comparisons).
The `behind_start_included` flag relaxes the `t > 0` lower bound — either
to `-inf` (a `True` boolean, used so aperture occlusion also blocks light
paths that would pass through the aperture plane *behind* the eye, matching
how `Camera.calc_image_vec` treats apertures) or to a signed distance
computed from a numeric value (used to allow a bounded region behind the
apex, e.g. so a screen surface right behind a lens doesn't self-occlude).

**`check_visible(mesh_obj, start, grid_points, ...)`** combines both
stages: it runs the cone prefilter for the whole mesh at once, then for
each triangle that has any candidate points inside its cone, runs
`check_intersection` against just those candidates, marking any point that
intersects **any** triangle as occluded (`visible=False`). A point that
never falls inside any triangle's cone is visible without ever running the
exact intersection test. This
is what `Camera.calc_image_vec` (per-aperture occlusion, `docs/core.md`)
and `World.find_visible_points` (per-aperture and per-wall occlusion,
`docs/world.md`) both call.

An older, simpler implementation (`check_visible_old`) is retained in the
module but not used by the current pipeline; it performs the same
Möller–Trumbore test without the cone prefilter, and is presumably kept for
reference/testing rather than production use — the module does not document
why it was superseded beyond the evident performance motivation.

### Visualization and parametric surfaces

`show_stl`/`plotly_show_stl` render meshes in Matplotlib/Plotly for
debugging, and `torus`/`sphere`/`meshed_surface` are simple parametric
surface generators usable as wall geometry.

These routines are consumed by `World` and `Camera` to test aperture and wall occlusion efficiently during projection.
