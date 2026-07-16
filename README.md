# multi-pinhole

This repository contains code and resources for Multi-pinhole imaging simulation and reconstruction.

## Contents
- `multi_pinhole/`: Python package to simulate multi-pinhole imaging.
- `multi_pinhole/utils/`: Package-internal utility functions for data processing and visualization.
- `docs/`: Documentation files and user guides.
- `docs/ja/projection-roadmap.md`: Projection-matrix validation and improvement backlog.
- `docs/ja/projection-compression-future.md`: Deferred QA/PSF compression design notes.
- `examples/`: Example and analysis scripts demonstrating how to use the package.
- `benchmarks/`: Reproducible performance and numerical-accuracy experiments.

## Architecture and compatibility

The public classes remain available from `multi_pinhole`. Optics
implementations live in `eye`, `aperture`, `screen`, `camera`, and `rays`;
`multi_pinhole.core` remains a compatibility facade exposing the same class
objects for legacy imports and serialized globals. `World` remains in
`multi_pinhole.world` and owns public orchestration, scene state,
serialization, and caches. Private `_visibility` and `_projection_matrix`
modules contain calculations that can return masks or sparse matrices without
owning those caches. Projection cache schema 3 remains compatible with 0.7.3.

## Installation
To install the package, clone the repository and run:

```bash
git clone https://github.com/ingk-124/multi-pinhole.git
cd multi-pinhole
pip install -e .
```

## Minimal projection workflow

Lengths must use one consistent unit throughout a model; this example uses
millimetres (mm). Profile helpers live in the submodule and are imported with
`from multi_pinhole import profiles` rather than as top-level functions.

```python
import numpy as np
from multi_pinhole import Eye, Screen, Camera, Voxel, World

# Build a simple camera.
eye = Eye(position=(0., 0.), focal_length=10., eye_size=0.5)
screen = Screen("square", 20., pixel_shape=(4, 4), subpixel_resolution=2)
camera = Camera(eyes=[eye], apertures=[], screen=screen,
                camera_position=(0., 0., -20.))
voxel = Voxel.uniform_voxel(((-1., 1.),) * 3, shape=(2, 2, 2))
world = World(voxel=voxel, cameras={"main": camera}, verbose=0)

# Boolean tests are evaluated at voxel vertices and sample centers.
world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))

# res is the composite-midpoint source resolution on each voxel axis.
work = world.preflight_projection(res=1, res_mode="fixed", verbose=0)
print(work.summary())
world.set_projection_matrix(res=1, res_mode="fixed", parallel=1, verbose=0)

# Entries are emission values at voxel centers. Volume integration is already
# included in the projection matrix construction.
emission = np.ones(voxel.N)
image = world.project(emission, camera_idx="main")
adjoint = world.backproject(image, camera_idx="main")
assert image.shape == (screen.N_pixel,)
assert adjoint.shape == (voxel.N,)
```

`project` applies a previously constructed matrix; it never builds one
implicitly. `backproject` applies the discrete adjoint `P.T` and is not an
inverse reconstruction. See [the world guide](docs/world.md) for resolution
policies and [the core guide](docs/core.md) for detector integration.

## Classes
- `World`: Represents the 3D world to be imaged.
- `Camera`: Simulates a multi-pinhole camera system.
- `Voxel`: Represents a 3D voxel grid for the imaging volume.

`Camera` holds the following independent components (they are not Python
subclasses of `Camera`):
- `Eye`: Models of the pinholes and their arrangement in the camera.
- `Screen`: Represents the imaging sensor where the projections are captured.
- `Aperture`: Models of the apertures to control an image size.

## Voxel Coordinates
`Voxel` grids are Cartesian. The `coordinate_type` option controls only how
Cartesian points are converted by `voxel.normalized_coordinates()` for profile
evaluation:

```python
voxel = Voxel.uniform_voxel(
    ranges=[[-2010, 2010], [-2010, 2010], [-510, 510]],
    shape=[201, 201, 51],
    coordinate_type="torus",
    coordinate_parameters=dict(R_0=1500, a=500),
)
r, theta, phi = voxel.normalized_coordinates().T
```

For `coordinate_type="torus"`, `theta=0` is the outboard midplane and `phi`
increases clockwise when viewed from `+z`, making `(r, theta, phi)` right-handed.
Use `coordinate_type="torus_inverse"` when both angular directions should be
reversed: `theta=0` at the inboard midplane and counter-clockwise `phi`.
