# Overview and first projection

> **Audience:** first-time users through researchers applying an established
> camera geometry. This page deliberately omits ray–triangle and sparse-matrix
> implementation details. Read [Level 2](README.md#level-2--understand-the-scientific-model)
> only when you need to validate the numerical model.

## What the library computes

`multi_pinhole` maps a three-dimensional voxel emission field to a pinhole
camera detector image. Geometry is compiled once into a sparse operator

$$
\mathbf{g}=\mathbf{P}\mathbf{f},
$$

where `f` is emission per voxel and `g` is detector signal per pixel. Once
`P` exists, changing the emission requires only `World.project`, not another
ray trace.

Five objects are enough for normal use:

| Object | Meaning |
|---|---|
| `Voxel` | Cartesian cells carrying the emission |
| `Eye` | One pinhole or finite-eye channel |
| `Aperture` | The opening and surrounding opaque geometry |
| `Screen` | The pixelated detector plane |
| `Camera` / `World` | One optical assembly / the complete cached scene |

## Recommended workflow

### 1. Define the scene in JSON

```python
from multi_pinhole import World

world = World.from_config("scene.json")
```

The JSON contains voxel ranges, camera pose, eyes, screen, apertures, and an
optional wall. Follow [Building a World JSON configuration](world-config-guide.md);
use the [schema reference](config.md) only for exhaustive key and type rules.

### 2. Inspect geometry before expensive work

Check camera direction, wall ports, consistent length units, and the emitting
`inside` region. `inside` selects source volume; it is not an opaque wall.

```python
world.find_visible_voxels("main", verbose=1)
work = world.preflight_projection(res=3, partial_res=3)
print(work.summary())
```

Visibility states are `0=hidden`, `1=partly visible`, and `2=fully visible`.
Preflight estimates work without building the projection matrix.

### 3. Build the projection

```python
world.set_projection_matrix(
    res=3,
    partial_res=3,
    parallel=4,
    verbose=1,
)
```

Larger resolutions refine source integration and cost more. They are not an
error guarantee; final analyses should check convergence, especially for
partly visible voxels crossing wall or aperture boundaries.

### 4. Project emission

```python
import numpy as np

x, y, z = world.voxel.gravity_center.T
emission = np.exp(-((x / 100) ** 2 + (y / 100) ** 2 + (z / 150) ** 2))

image = world.project(emission, camera_idx="main")
world.cameras["main"].screen.show_image(image)
```

Emission may have shape `(N_voxel,)`, or `(N_voxel, N_time)` for a batch.
Use zero, not NaN, outside the modeled source: non-finite values propagate
through matrix multiplication. See [Visualization](visualization.md) for 3D
and slice plots.

### 5. Save expensive results when needed

```python
world.save("checkpoint.mpw")
```

JSON stores scene inputs. A `.mpw` archive stores visibility and projection
caches. See [Serialization](serialization.md) for compatibility and security.

## Coordinates: the minimum needed

- World and voxel grids are Cartesian `(x, y, z)`.
- Camera, eye, and screen frames are normally handled by the library.
- Profile evaluation may reinterpret the same Cartesian points as cylindrical,
  toroidal, or poloidal Cartesian coordinates; the grid itself stays Cartesian.
- Angle signs and origins matter. Confirm the selected convention in
  `Voxel.to_coordinates()` and [the coordinate section](core.md#coordinate-frames-users-need).

## Common mistakes

- Mixing millimetres and metres; the library does not attach units.
- Confusing `inside` (source selection) with `wall` (occlusion).
- Filling inactive emission with NaN instead of zero.
- Assuming screen arrays have generic image-row orientation; prefer
  `Screen.show_image`.
- Trusting one source resolution without a convergence check.
- Expecting JSON to contain calculated caches.

> **Ordinary usage can stop here.** Choose a deeper page only for the task at
> hand.

## Where to go next

- Build camera JSON → [World configuration guide](world-config-guide.md)
- Evaluate plasma profiles or interpolate an R–Z section →
  [Coordinates, profiles, and interpolation](coordinates-profiles.md)
- Plot results → [Visualization](visualization.md)
- Understand pinhole equations and detector integration → [Core optics](core.md)
- Audit visibility and source integration → [World projection](world.md)
- Look up exact JSON keys → [Config reference](config.md)
