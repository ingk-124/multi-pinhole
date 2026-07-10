# Project Overview

## Purpose

`multi_pinhole` simulates X-ray pinhole-camera imaging of a plasma (built for
the MST reversed-field-pinch experiment, but not specific to it): given a 3D
emission profile defined on a voxel grid and one or more cameras (each with
several pinhole or concave-lens "eyes"), it computes the sparse linear
operator that maps voxel intensities to detector-pixel intensities. That
operator is the thing you actually want out of the simulation — once you
have it, "rendering an image" of any emission profile is a single sparse
matrix-vector product, and, conversely, image inversion (tomography) is a
linear inverse problem against the same matrix.

The package is organized around four coordinate systems — world, camera,
pinhole/eye, and screen/image — that are formalized in
`multi_pinhole.core`.【F:multi_pinhole/core.py†L54-L106】 Every geometric
calculation in the package is a composition of transforms between these
frames; `docs/core.md` walks through that chain in detail.

## Key Components

- **Core optics** (`multi_pinhole.core`) — `Eye` (a single pinhole/lens
  channel), `Aperture` (an occluding shape, analytic or STL), `Screen` (the
  pixelated detector plane and its rasterizer), and `Camera` (which groups
  eyes/apertures/screen and places them in world space). See
  `docs/core.md`.【F:multi_pinhole/core.py†L126-L1727】
- **Voxel modeling** (`multi_pinhole.voxel`) — a Cartesian voxel grid
  (`Voxel`) plus synthetic-profile helpers for toroidal plasma emission. See
  the "Voxel grid geometry" section below.【F:multi_pinhole/voxel.py†L307-L1367】
- **Coordinate transforms** (`multi_pinhole.coordinates`) — pure functions
  that reinterpret Cartesian voxel-grid points in cylindrical, toroidal, or
  spherical terms, purely for *evaluating a profile*; the grid itself is
  always Cartesian.【F:multi_pinhole/coordinates.py†L1-L124】
- **World orchestration** (`multi_pinhole.world`) — `World` binds a `Voxel`,
  one or more `Camera` instances, and optional STL `walls` into a scene; it
  computes per-eye visibility and assembles the voxel-to-screen projection
  matrix. See `docs/world.md`.【F:multi_pinhole/world.py†L151-L1595】

## Typical Workflow

1. **Describe the scene.** Build a `Voxel` grid — either directly from axis
   arrays, or with `Voxel.uniform_voxel(ranges, shape)` for an evenly-spaced
   Cartesian box. Optionally evaluate toroidal or poloidal profiles through
   `multi_pinhole.profiles`, and load STL `walls` that should occlude rays.
2. **Configure optics.** Create one or more `Eye` objects (pinhole position,
   focal length, aperture size/shape), pair them with `Aperture` geometry,
   and attach them to a `Screen` (physical size, pixel grid, subpixel
   refinement).
3. **Assemble a `Camera`** from the eyes/apertures/screen, and place it in
   world space with `camera_position` and a rotation.
4. **Build a `World`** from the voxel grid and camera(s), and mark which
   voxel vertices are physically "inside" the volume of interest with
   `World.set_inside_vertices(...)` (vertices outside are skipped by the
   visibility/projection steps below — this is how, e.g., a torus-shaped
   plasma volume within a rectangular voxel box is expressed).
5. **Compute visibility and the projection matrix.**
   `World.set_projection_matrix()` determines, for every camera eye, which
   voxels are visible (unobstructed by apertures or walls) and builds the
   sparse `(N_pixel, N_voxel)` matrix `world.P_matrix[camera_idx]`. See the
   worked example below and `docs/world.md` for the full pipeline.
6. **Render or invert.** Given a voxel-intensity vector `emission`
   (`shape (N_voxel,)`), `world.P_matrix[camera_idx] @ emission` is the
   simulated pixel image; `world.projection[camera_idx][eye_idx] @ emission`
   is the finer subpixel image before pixel-binning.

### Worked example: from an empty `World` to a rendered image

This follows `examples/small_voxel_projection.py`, trimmed to the essential
calls:

```python
from multi_pinhole import Aperture, Camera, Eye, Screen, Voxel, World
import numpy as np

# 1. A 3x3x3 voxel grid spanning [-3, 3] mm on each axis.
voxel = Voxel.uniform_voxel(ranges=[[-3, 3], [-3, 3], [-3, 3]], shape=[3, 3, 3])

# 2-3. One pinhole eye, one circular aperture, a small screen, assembled into a Camera.
camera = Camera(
    eyes=[Eye(eye_type="pinhole", eye_shape="circle", eye_size=1.0,
              focal_length=12.0, position=[0.0, 0.0])],
    apertures=Aperture(shape="circle", size=6.0, position=[0.0, 0.0, 25.0],
                        resolution=24, max_size=24.0),
    screen=Screen(screen_shape="rectangle", screen_size=[12.0, 12.0],
                  pixel_shape=(8, 8), subpixel_resolution=2),
    camera_position=[0.0, 0.0, -60.0],
)

# 4. Bind the voxel grid and camera into a World, and mark all vertices "inside".
world = World(voxel=voxel, cameras=[camera], verbose=0)
world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))

# 5. Compute visibility + the sparse voxel-to-screen projection matrix.
world.set_projection_matrix(res=1, verbose=0, parallel=1)

# 6. Render: pick an emission value per voxel, then one sparse matvec per image.
emission = np.exp(-((voxel.gravity_center[:, 0] / 2.2) ** 2
                    + (voxel.gravity_center[:, 1] / 1.8) ** 2
                    + (voxel.gravity_center[:, 2] / 2.6) ** 2))
pixel_image = world.P_matrix[0] @ emission          # shape (N_pixel,)
subpixel_image = world.projection[0][0] @ emission  # shape (N_subpixel,)
```

Internally, step 5 (`set_projection_matrix`) is the expensive part: for each
camera eye it (a) classifies every voxel as invisible/partially/fully
visible by ray-tracing its 8 corner vertices against every aperture and
wall, (b) for visible voxels, samples sub-voxel points, projects them
through the eye with `Camera.calc_image_vec` (the pinhole-projection +
rasterization pipeline from `docs/core.md`), and (c) integrates those
sub-voxel samples back into one weight per voxel. `docs/world.md` documents
each of these sub-steps.

## Voxel grid geometry

A `Voxel` is a rectilinear (not necessarily uniformly-spaced) 3D grid,
defined by three 1D axis arrays `x_axis`, `y_axis`, `z_axis` of grid-line
positions.【F:multi_pinhole/voxel.py†L370-L450】 From those axes,
`Voxel.update()` derives everything else in a vectorized way (no Python
loops over voxels):【F:multi_pinhole/voxel.py†L924-L992】

* **Grid points** are the `(N_x+1) × (N_y+1) × (N_z+1)` Cartesian product of
  the three axes, flattened in `z`-fastest, then `y`, then `x` order (i.e.
  linear index `n = k + N_z'·(j + N_y'·i)` for grid shape
  `(N_x', N_y', N_z')`).
* **Voxels** are the `N_x × N_y × N_z` cells between consecutive grid lines.
  Voxel `(i, j, k)`'s 8 corner vertices are obtained by adding a fixed
  offset pattern (`{0,1} × {0,1} × {0,1}` combinations, expressed as linear
  index offsets `{0, 1, N_z', N_z'+1, N_z'·N_y', ...}`) to that voxel's base
  linear grid index — this is a pure index-arithmetic trick that avoids
  building an explicit `(N_voxel, 8, 3)` array of vertex coordinates unless
  a caller actually asks for `Voxel.vertices`.
* **Volume** of each voxel is the product of its three edge lengths
  (`dx · dy · dz`), and its **gravity center** is the midpoint of its 8
  corners — both computed per-axis and broadcast, not per-voxel.
* **Sub-voxel sampling.** For interpolation/integration (used heavily by
  `World`'s projection pipeline — see `docs/world.md`), a voxel can be
  subdivided into an `res = (x_res, y_res, z_res)` grid of sub-voxel sample
  points. `interpolate_matrix_from_vertices(res)` builds a matrix of
  trilinear interpolation weights: each sub-voxel sample point at fractional
  position `(a, b, c)` within the parent voxel (`a, b, c ∈ [0, 1]`) is
  assigned to a weighted combination of the voxel's 8 corner vertex values,
  with weights `(1−a)(1−b)(1−c)`, `(1−a)(1−b)c`, ..., `abc` — the standard
  trilinear interpolation basis.【F:multi_pinhole/voxel.py†L39-L77】

### Coordinate transforms for profile evaluation

The grid itself is always Cartesian; `Voxel.normalized_coordinates()`
optionally *reinterprets* Cartesian points (by default, the voxel gravity
centers) in a different coordinate system, purely so that profile functions
can be written in terms that are natural for the device's symmetry.
【F:multi_pinhole/voxel.py†L749-L772】
`multi_pinhole.coordinates` implements five such transforms (all taking
Cartesian `(x, y, z)` and returning normalized coordinates):

* **Cartesian** — just rescales each axis by half its configured extent.
* **Cylindrical** `(r, theta, z)` — `r = sqrt(x²+y²)/a`,
  `theta = atan2(y, x)`, `z` rescaled by `h/2`.
* **Torus** `(r, theta, phi)` — for a torus of major radius `R_0` and minor
  radius `a`: `R = sqrt(x²+y²)`, `r = sqrt((R−R_0)² + z²)/a`,
  `theta = atan2(z, R−R_0)` (poloidal angle, `0` on the outboard midplane),
  `phi = atan2(−y, x)` (toroidal angle, increasing clockwise viewed from
  `+z`). `torus_inverse` is the same construction with both angles flipped
  in sign/reference (`theta` referenced to the inboard midplane, `phi`
  counter-clockwise) — both conventions are right-handed
  `(r, theta, phi)`.【F:multi_pinhole/coordinates.py†L32-L82】
* **Spherical** `(r, theta, phi)` — the usual physics convention,
  `r = |x|/a`, `theta = arccos(z/r)`, `phi = atan2(y, x)`.

`multi_pinhole.profiles` provides composable helpers for evaluating synthetic
toroidal and poloidal profiles on top of these coordinates, including shifted
polar coordinates, kinked/flattened radial coordinates, and thin wrappers that
evaluate profile functions directly on torus-coordinate `Voxel` instances.
These helpers are meant for test profiles and reusable profile models; plotting,
fitting, and experiment-specific diagnostics should live outside the core
profile API.

## Notable Capabilities

- Cameras support multiple simultaneous eyes (multi-pinhole imaging), each
  with independent position, focal length, aperture shape/size, and
  wavelength range.
- Apertures accept analytic shapes (circle/ellipse/rectangle) or arbitrary
  STL meshes, and are used as hard occluders: `check_visible` performs a
  two-stage (cone prefilter + Möller–Trumbore ray-triangle intersection)
  visibility test between an eye and each candidate point — see
  `docs/utilities.md`.
- Screen rasterization (`Screen.ray2image_grid`) uses sparse CSR/CSC
  matrices scaled by etendue weights, so millions of rays can be
  accumulated into a subpixel image without ever materializing a dense
  array — see `docs/core.md`.
- `World.set_projection_matrix` parallelizes the expensive sub-voxel
  sampling/projection work across a `ThreadPoolExecutor`, chunked
  adaptively by estimated sparsity to bound memory use — see
  `docs/world.md`.

## Extensibility

The project layers utility functions (STL processing in
`multi_pinhole.utils.stl_utils`, progress-aware logging in
`multi_pinhole.utils.my_stdio`) beneath the main classes, so new optical
elements or custom workflows can reuse the existing coordinate transforms,
visibility checks, and visualization routines without re-implementing the
underlying geometry. See `docs/utilities.md` for those building blocks.
