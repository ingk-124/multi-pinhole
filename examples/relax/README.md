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
