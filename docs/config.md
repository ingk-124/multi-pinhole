# World config schema

> **Level 3 — reference.** This page is exhaustive and is not intended as a
> first tutorial. To create a file, follow
> [Building a World JSON configuration](world-config-guide.md), then return
> here only to check exact keys, types, and validation errors.

`World.from_config(path_or_mapping)` reads schema
`multi-pinhole/world-config`, version 1. `world.to_config(path)` writes the
same canonical JSON and also returns the mapping. Config is a declarative
scene description: it never stores visibility, per-Eye projection,
`P_matrix`, or any cache schema.

## Root contract

Every root field is required:

| Field | Contract |
| --- | --- |
| `schema` | Exact string `multi-pinhole/world-config` |
| `schema_version` | Integer `1` |
| `units` | Exact object `{"length": "mm", "angle": "rad"}` |
| `voxel` | Voxel object described below |
| `cameras` | Ordered array; every `key` is a unique string |
| `walls` | Array of `{"type": "stl", "path": "..."}` |
| `inside` | `null` or a safe built-in inside-mask specification |
| `verbose` | Integer |

Unknown and missing fields are errors at every schema level. Numeric values
must be finite. JSON never names a Python import, expression, or callable,
and the loader does not use `eval`.

## Voxel

`voxel.type` selects exactly one grid representation:

- `"uniform"` requires `ranges` with three `[minimum, maximum]` pairs and
  `shape = [N_x, N_y, N_z]`. The loader generates `N_axis + 1` equally spaced
  boundary points along each axis.
- `"axes"` requires `axes` containing explicit, strictly increasing `x`, `y`,
  and `z` boundary arrays. Grid shape and ranges are derived from them.

The two representations are exclusive: `axes` is invalid on a uniform voxel,
and `ranges`/`shape` are invalid on an axes voxel. `to_config` writes the
compact uniform form when all three axes are equally spaced and otherwise
writes explicit axes. `sub_voxel_resolution` contains three positive integers.

`coordinate` has `type` and numeric `parameters`. Supported coordinate names
and parameters are the same as `Voxel`. A Voxel does not persist a rotated
coordinate frame; use the per-call `rotation` argument of coordinate
conversion helpers for an isolated conversion, or align physical scene
geometry in world coordinates.

## Cameras and optics

Each camera contains:

- string `key` and `name`;
- world `position` in mm and exactly one orientation representation:
  - a 3 by 3 world-to-camera `rotation_matrix`; or
  - `orientation` with a world `look_point` and exactly one world
    `right_point` or `down_point`;
- one or more Eyes with `type`, two-dimensional `position`, `focal_length`,
  analytic `shape`, two-dimensional `size`, and `wavelength_range`;
- one Screen with analytic `shape`, two-dimensional `size`, two-dimensional
  positive-integer `pixel_shape`, and positive `subpixel_resolution`;
- an array of analytic Apertures. Version 1 supports `circle`, `ellipse`, and
  `rectangle`, with `position` and `direction`. Optional positive integer
  `resolution` controls boundary discretization (default `20`), while
  `max_size` is either `null` or two positive support-mesh half-extents;
- path-backed STL Apertures with `type`, `path`, and camera-coordinate
  `position`. The STL vertices must already have the desired orientation in
  camera coordinates.

Point-based orientation is converted through
`Camera.set_orientation_from_points()`. Points are absolute world
coordinates, not direction vectors. Supplying both orientation forms,
supplying neither, or giving both lateral points is an error.

An STL Aperture assembled from an in-memory mesh without source-path
provenance cannot be written to JSON; `to_config` raises `WorldConfigError`.

## Paths and inside masks

Wall and STL Aperture paths may be absolute. Relative paths are resolved from
the JSON file directory; for a mapping input they are resolved from the
current directory. When a loaded config is written elsewhere, `to_config`
writes paths relative to the new config directory. A World assembled directly
from mesh objects has no trustworthy source path, so `to_config` fails rather
than claiming a round trip.

Version 1 inside types are:

- `{"type": "all", "parameters": {}}`;
- `{"type": "box", "parameters": {"ranges": [[xmin, xmax], ...]}}`;
- `{"type": "sphere", "parameters": {"center": [x, y, z], "radius": r}}`.
- `{"type": "torus", "parameters": {"major_radius": R0, "minor_radius": a}}`,
  representing `(sqrt(x^2 + y^2) - R0)^2 + z^2 <= a^2`.

Arbitrary Python callables and standalone `inside_vertices` arrays are
deliberately unsupported. Construct them in application code or use a World
archive if exact Python state must be checkpointed.

## Errors and evolution

Invalid input raises `multi_pinhole.config.WorldConfigError`, a `ValueError`
subclass. Unsupported schema versions fail explicitly. Future schema changes
are independent of both the package version and World archive schema.
