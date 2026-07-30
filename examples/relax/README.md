# RELAX multi-pinhole example

`RELAX_multi_pinhole_imaging.py` reconstructs the useful parts of the former
development demo in `multi_pinhole.world`. It builds two two-eye cameras around
a toroidal plasma, uses the packaged RELAX vessel mesh for wall occlusion, and
projects a simple radial emission profile onto both screens.

Run it from the repository root:

```bash
python examples/relax/RELAX_multi_pinhole_imaging.py
```

The default voxel and subpixel resolutions are intentionally reduced so the
example can be exercised locally. Increase the projection resolution when a
higher-fidelity result is needed:

```bash
python examples/relax/RELAX_multi_pinhole_imaging.py --resolution 3 --parallel 4
```

Use `--no-show` in a headless environment. The vessel geometry is loaded from
`multi_pinhole/data/relax_rotated.stl`; it is not duplicated in this directory.

## JSON World configuration

[`relax_world.json`](relax_world.json) is a complete version-1 World
configuration for a reduced RELAX scene. It uses `R_0 = 508 mm` and
`a = 250 mm`, and places one equatorial camera at `R = 950 mm`,
`phi = +45 deg`, aligned with the port in the packaged RELAX vessel STL.
The camera looks radially inward; image-right follows
increasing toroidal angle and image-down follows decreasing world `Z`.
The JSON demonstrates point-based camera orientation with `look_point` and
`right_point`. Its analytic circular aperture records `resolution = 40` and
`max_size = [200, 200]` so the generated support mesh matches the former
Python-built RELAX example. The voxel uses the compact `"type": "uniform"`
form: `ranges` gives the three Cartesian bounds and `shape` gives the number
of cells, so the boundary coordinates do not need to be listed individually.

Load the JSON without constructing any optics in Python:

```bash
python examples/relax/RELAX_world_from_json.py
```

Add `--project` to build a low-resolution projection and display the image:

```bash
python examples/relax/RELAX_world_from_json.py --project
```

Plot the camera-visible voxels on the voxel-center equatorial plane:

```bash
python examples/relax/RELAX_world_from_json.py \
  --visibility --save relax_visible_voxels.png --no-show
```

The visibility plot uses the public state convention: `0` is not visible,
`1` is partially visible, and `2` is fully visible.

The JSON loads the packaged RELAX vessel STL as a wall. Its built-in `torus`
inside-mask restricts source integration to
`(sqrt(x^2 + y^2) - 508)^2 + z^2 <= 250^2`; the example also sets emission
outside normalized toroidal radius `r = 1` to zero.
