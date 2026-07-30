# World config schema

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

`voxel.axes` contains increasing `x`, `y`, and `z` arrays with at least two
entries. `shape` must equal the three axis lengths minus one. `ranges` must
equal their endpoints. `sub_voxel_resolution` contains three positive
integers.

`coordinate` has `type`, numeric `parameters`, and a 3 by 3 world rotation
matrix. The matrix must be orthonormal with determinant +1. Supported
coordinate names and parameters are the same as `Voxel`.

Axes, ranges, and shape are intentionally redundant: disagreement is treated
as corrupt configuration rather than silently choosing one representation.

## Cameras and optics

Each camera contains:

- string `key` and `name`;
- world `position` in mm and a 3 by 3 world-to-camera `rotation_matrix`;
- one or more Eyes with `type`, two-dimensional `position`, `focal_length`,
  analytic `shape`, two-dimensional `size`, and `wavelength_range`;
- one Screen with analytic `shape`, two-dimensional `size`, two-dimensional
  positive-integer `pixel_shape`, and positive `subpixel_resolution`;
- an array of analytic Apertures. Version 1 supports `circle`, `ellipse`, and
  `rectangle`, with `position` and `direction`.

An Aperture created from an arbitrary STL object is not representable as an
analytic aperture and `to_config` raises `WorldConfigError`.

## Paths and inside masks

Wall paths may be absolute. Relative paths are resolved from the JSON file
directory; for a mapping input they are resolved from the current directory.
When a loaded config is written elsewhere, `to_config` writes a relative path
from the new config directory. A World assembled directly from mesh objects
has no trustworthy source path, so `to_config` fails rather than claiming a
round trip.

Version 1 inside types are:

- `{"type": "all", "parameters": {}}`;
- `{"type": "box", "parameters": {"ranges": [[xmin, xmax], ...]}}`;
- `{"type": "sphere", "parameters": {"center": [x, y, z], "radius": r}}`.

Arbitrary Python callables and standalone `inside_vertices` arrays are
deliberately unsupported. Construct them in application code or use a World
archive if exact Python state must be checkpointed.

## Errors and evolution

Invalid input raises `multi_pinhole.config.WorldConfigError`, a `ValueError`
subclass. Unsupported schema versions fail explicitly. Future schema changes
are independent of both the package version and World archive schema.
