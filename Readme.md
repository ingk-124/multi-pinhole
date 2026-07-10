# multi-pinhole

This repository contains code and resources for Multi-pinhole imaging simulation and reconstruction.

## Contents
- `multi_pinhole/`: Python package to simulate multi-pinhole imaging.
- `multi_pinhole/utils/`: Package-internal utility functions for data processing and visualization.
- `doc/`: Documentation files and user guides.
- `example.py`: Example script demonstrating how to use the package.

## Installation
To install the package, clone the repository and run:

```bash
git clone https://github.com/ingk-124/multi-pinhole.git
cd multi-pinhole
pip install -e .
```

## Minimal usage
The public API can be imported directly from `multi_pinhole`:

```python
import numpy as np
from multi_pinhole import Eye, Screen, Aperture, Camera, Voxel, World

# Build a simple camera.
eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=0.5)
screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(4, 4), subpixel_resolution=20)
aperture = Aperture(shape="circle", size=1.0, position=(0.0, 0.0, 5.0))
camera = Camera(eyes=[eye], apertures=aperture, screen=screen, camera_position=(0.0, 0.0, -10.0))

# Project points expressed in camera coordinates through the eye.
points = np.array([[0.0, 0.0, 20.0], [1.0, 0.0, 20.0]])
rays = eye.calc_rays(points)
print(rays.XY)

# Create the world container used by reconstruction workflows.
voxel = Voxel(
    x_axis=np.linspace(-1.0, 1.0, 3),
    y_axis=np.linspace(-1.0, 1.0, 3),
    z_axis=np.linspace(-1.0, 1.0, 3),
)
world = World(voxel=voxel, cameras=[camera], verbose=0)
print(world)
```

## Classes
- `World`: Represents the 3D world to be imaged.
- `Camera`: Simulates a multi-pinhole camera system.
- `Voxel`: Represents a 3D voxel grid for the imaging volume.

`Camera` class contains the following sub-classes:
- `Eye`: Models of the pinholes and their arrangement in the camera.
- `Screen`: Represents the imaging sensor where the projections are captured.
- `Aperture`: Models of the apertures to control an image size.

## Refactoring notes
- `utils` has been moved under `multi_pinhole.utils` to avoid collisions with unrelated top-level packages named `utils`.
- `multi_pinhole.core` remains the primary implementation module for camera-related classes, while the low-risk `Rays` data container has been split into `multi_pinhole.rays` and re-exported through the existing public API.
- Further low-risk splits can extract `Eye`/optics, `Screen`, `Aperture`, and `Camera` into separate modules while keeping compatibility imports in `multi_pinhole.__init__` and `multi_pinhole.core`.
