# Visualization

> **Level 1 — task guide.** This page is the canonical location for plotting
> voxel emission and detector data. Internal Plotly, Matplotlib, and mesh
> implementation details remain in [Utilities](utilities.md).

## Inspecting camera and wall geometry

Inspect geometry before calculating visibility or a projection matrix. The
highest-level Matplotlib view is:

```python
import matplotlib.pyplot as plt

ax = world.draw_camera_orientation(
    show_fig=False,
    elev=60,
    azim=-30,
    facecolors="lightgray",
    alpha=0.15,
)
plt.show()
```

Despite its name, `World.draw_camera_orientation` shows the complete world
context needed for alignment:

- voxel bounding ranges;
- every camera position and its local X/Y/Z axes;
- every registered wall mesh;
- axis limits expanded to include voxels, walls, and cameras.

It does not draw each camera's eye, aperture, and screen at their physical
size. Inspect those in the camera-local frame:

```python
camera = world.cameras["main"]
ax = camera.draw_optical_system(
    show_focal_length=True,
    show_aperture=True,
    show_screen=True,
)
plt.show()
```

`Camera.draw_optical_system` is the appropriate view for checking eye
positions, focal lengths, aperture support meshes, and screen placement.
`Camera.draw_camera_orientation()` draws only the selected camera's world
position and axes.

For interactive wall and camera-axis inspection, compose the Plotly helpers:

```python
import plotly.graph_objects as go
from multi_pinhole.utils import stl_utils

fig = go.Figure()
for wall in world.walls:
    stl_utils.plotly_show_stl(
        wall,
        fig=fig,
        color="lightgray",
        opacity=0.2,
        show_edges=False,
        show_fig=False,
    )
for camera in world.cameras.values():
    camera.draw_camera_orientation_plotly(
        fig,
        axis_length=100,
        show_fig=False,
    )
fig.show()
```

Use `stl_utils.show_stl(model)` or `stl_utils.plotly_show_stl(model)` when a
wall or custom aperture STL needs to be inspected by itself.

These functions visualize configured geometry only. They do not prove that a
ray passes through an aperture or wall port. After visual inspection, run
`world.find_visible_voxels(camera_idx)` and inspect the resulting 0/1/2
visibility states before starting an expensive projection.

## Voxel emission in 3D

`plot_voxel_volume` accepts the `Voxel` and emission together, validates their
shape, and uses `voxel.gravity_center` in the correct flattening order.

```python
import numpy as np
from multi_pinhole import Voxel
from multi_pinhole.utils.plot import plot_voxel_volume

voxel = Voxel.uniform_voxel(
    ranges=[[-1, 1], [-1, 1], [-1, 1]],
    shape=[30, 30, 30],
)
x, y, z = voxel.gravity_center.T
emission = np.exp(-4 * (x**2 + y**2 + z**2))

fig = plot_voxel_volume(
    voxel,
    emission,
    length_unit="m",
    value_label="Emissivity [W m⁻³]",
    opacity=0.15,
    surface_count=20,
    colorscale="Viridis",
)
fig.show()
```

`emission` may have shape `(N_voxel,)` or `voxel.shape`. Non-finite values are
omitted. Use `mask=world.visible_voxels["camera"].any(axis=0)` to show only
voxels visible to at least one eye, but replace invisible values with zero
before passing emission to `World.project`; NaN propagates through sparse
matrix multiplication.

## Axis-aligned slices

```python
import matplotlib.pyplot as plt
from multi_pinhole.utils.plot import plot_voxel_slice

plot_voxel_slice(
    voxel,
    emission,
    axis="z",
    coordinate=0,
    length_unit="m",
    colorbar_label="Emissivity [W m⁻³]",
)
plt.show()
```

`coordinate` selects the nearest voxel-center plane. Use `index=` for an
exact plane index. The helper uses voxel boundary axes with `pcolormesh`, so
the displayed cells have their physical extent rather than being treated as
point samples.

This helper does not interpolate. For a smooth fixed-toroidal-angle R–Z
section using `Voxel.center_interpolator`, see
[Coordinates, profiles, and interpolation](coordinates-profiles.md#fixed-phi-rz-cross-section).

## Detector images

```python
image = world.project(emission, camera_idx="main")
world.cameras["main"].screen.show_image(image)
```

Screen image arrays use the screen's `(u, v)` convention. Prefer
`Screen.show_image` when possible because it applies the library's pixel
ordering and physical extent. When drawing manually with `pcolormesh`, use
the screen coordinate arrays and explicitly choose the desired vertical
axis direction rather than assuming image-row orientation.

See [`examples/example.py`](../examples/example.py) for an end-to-end
projection and both visualization helpers. The
[`examples/relax`](../examples/relax/README.md) example also shows how to
inspect visibility before building a projection matrix.
