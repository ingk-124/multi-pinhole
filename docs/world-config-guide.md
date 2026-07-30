# Building a World JSON configuration

> **Level 1 — task guide.** Read this in order when creating a scene. It
> explains decisions and common failure modes without listing every schema
> key. Use the [schema reference](config.md) as a lookup table.

This tutorial explains how to write a portable, cache-free scene file for
`World.from_config()`. For every accepted key and validation rule, use the
[schema reference](config.md).

## 1. Start with the voxel grid

All lengths in one file must use the same unit. The library does not attach
units to numbers; the fragments below use millimetres. Start with this
`voxel` field:

```json
"voxel": {
  "type": "uniform",
  "ranges": [[250, 750], [-250, 250], [-250, 250]],
  "shape": [40, 40, 40],
  "coordinate": {
    "type": "torus",
    "parameters": {
      "major_radius": 508,
      "minor_radius": 250
    }
  },
  "sub_voxel_resolution": [1, 1, 1]
}
```

`ranges` contains the `[minimum, maximum]` boundaries for Cartesian `x`, `y`,
and `z`. `shape` is the number of voxels `(N_x, N_y, N_z)`, not the number of
grid vertices. Memory and projection time grow with their product. For a
nonuniform grid, replace `type`, `ranges`, and `shape` with explicit boundary
arrays:

```json
"type": "axes",
"axes": {
  "x": [250, 300, 380, 500, 750],
  "y": [-250, -100, 0, 100, 250],
  "z": [-250, -50, 50, 250]
}
```

Do not mix the two representations. `to_config()` automatically uses the
compact form when all three axes are uniform.

An optional `inside` rule excludes voxels outside the physical plasma:

```json
"inside": {
  "type": "torus",
  "parameters": {
    "major_radius": 508,
    "minor_radius": 250
  }
}
```

This evaluates
`(sqrt(x^2 + y^2) - major_radius)^2 + z^2 <= minor_radius^2` at voxel
vertices. It defines the emitting domain; it is not an opaque wall.

## 2. Add a camera pose

The least error-prone orientation form points the camera at a known world
point:

```json
"cameras": [
  {
    "key": "main",
    "name": "equatorial camera",
    "position": [671.75, 671.75, 0],
    "orientation": {
      "look_point": [359.21, 359.21, 0],
      "right_point": [672.46, 671.04, 0]
    },
    "eyes": [{
      "type": "pinhole",
      "position": [0, 0],
      "focal_length": 20,
      "shape": "circle",
      "size": [0.25, 0.25],
      "wavelength_range": [0.01, 0.1]
    }],
    "screen": {
      "shape": "rectangle",
      "size": [8, 8],
      "pixel_shape": [32, 32],
      "subpixel_resolution": 3
    },
    "apertures": []
  }
]
```

`position`, `look_point`, and `right_point` are absolute world coordinates.
`right_point - position` defines the screen-right direction. It must not be
parallel to the viewing direction. `down_point` may replace `right_point`
when the detector's downward direction is easier to specify. Advanced users
may instead provide a 3-by-3 `rotation_matrix`; do not mix the forms.

Eye and aperture positions are expressed in the camera-local frame.
Check a new pose visually before computing an expensive projection:

```python
import plotly.graph_objects as go
from multi_pinhole import World

world = World.from_config("scene.json")
fig = go.Figure()
world.cameras["main"].draw_camera_orientation_plotly(fig, show_fig=False)
fig.show()
```

## 3. Add an aperture

An analytic aperture describes an opening and generates its surrounding
opaque mesh:

```json
"apertures": [{
  "type": "analytic",
  "shape": "circle",
  "size": [10, 10],
  "position": [0, 0, 80],
  "direction": [0, 0, 1],
  "resolution": 40,
  "max_size": [200, 200]
}]
```

`resolution` samples the opening boundary. Increase it for curved openings;
it affects geometry fidelity and visibility cost. `max_size` is the extent of
the opaque plate around the hole and must cover every relevant ray. It is not
the aperture diameter.

For a measured or CAD aperture, use an STL:

```json
"apertures": [{
  "type": "stl",
  "path": "geometry/aperture.stl",
  "position": [0, 0, 80]
}]
```

Relative paths are resolved from the JSON file. Keep assets beside the
configuration when the scene must be portable.

## 4. Add an optional wall

```json
"walls": [{
  "type": "stl",
  "path": "geometry/vessel.stl"
}]
```

The wall and aperture meshes are hard occluders. A closed vessel without an
opening makes an external camera see nothing. Confirm that the STL contains
the intended port and that the camera points through it. Omitting `wall`
means that only the camera aperture limits visibility.

## 5. Load, inspect, and compute

```python
from multi_pinhole import World

world = World.from_config("scene.json")
world.find_visible_voxels("main", verbose=1)
world.set_projection_matrix(res=3, parallel=4, verbose=1)
```

JSON stores reproducible scene inputs, not computed visibility or projection
caches. Use a versioned `.mpw` archive when those expensive results must be
saved. See [serialization](serialization.md).

The complete working RELAX configuration and inspection script are in
[`examples/relax`](../examples/relax/README.md). Visualization of voxel
profiles and detector images is covered in the
[visualization guide](visualization.md).
