"""Strict, dependency-free JSON configuration for :class:`World` scenes.

The configuration format describes scene construction only.  It deliberately
does not contain visibility or projection caches; those belong to the versioned
World archive implemented in :mod:`multi_pinhole.serialization`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from stl import mesh

from .aperture import Aperture
from .camera import Camera
from .eye import Eye
from .screen import Screen
from .voxel import Voxel

if TYPE_CHECKING:
    from .world import World

WORLD_CONFIG_SCHEMA = "multi-pinhole/world-config"
WORLD_CONFIG_SCHEMA_VERSION = 1
WORLD_CONFIG_UNITS = {"length": "mm", "angle": "rad"}


class WorldConfigError(ValueError):
    """Raised when a World configuration violates the public schema."""


def _inside_all(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.ones(np.broadcast(x, y, z).shape, dtype=bool)


def _inside_box(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    ranges: list[list[float]],
) -> np.ndarray:
    bounds = np.asarray(ranges, dtype=float)
    return (
        (x >= bounds[0, 0])
        & (x <= bounds[0, 1])
        & (y >= bounds[1, 0])
        & (y <= bounds[1, 1])
        & (z >= bounds[2, 0])
        & (z <= bounds[2, 1])
    )


def _inside_sphere(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    center: list[float],
    radius: float,
) -> np.ndarray:
    center_array = np.asarray(center, dtype=float)
    return (x - center_array[0]) ** 2 + (y - center_array[1]) ** 2 + (
        z - center_array[2]
    ) ** 2 <= radius**2


_INSIDE_REGISTRY = {
    "all": (_inside_all, set()),
    "box": (_inside_box, {"ranges"}),
    "sphere": (_inside_sphere, {"center", "radius"}),
}
_INSIDE_NAMES = {function: name for name, (function, _) in _INSIDE_REGISTRY.items()}


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorldConfigError(f"{location} must be an object")
    return dict(value)


def _fields(
    value: Any,
    location: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> dict[str, Any]:
    result = _object(value, location)
    missing = required - result.keys()
    unknown = result.keys() - required - optional
    if missing:
        raise WorldConfigError(
            f"{location} is missing required field(s): {', '.join(sorted(missing))}"
        )
    if unknown:
        raise WorldConfigError(
            f"{location} has unknown field(s): {', '.join(sorted(unknown))}"
        )
    return result


def _number(value: Any, location: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorldConfigError(f"{location} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise WorldConfigError(f"{location} must be finite")
    if positive and result <= 0:
        raise WorldConfigError(f"{location} must be positive")
    return result


def _integer(value: Any, location: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorldConfigError(f"{location} must be an integer")
    if positive and value <= 0:
        raise WorldConfigError(f"{location} must be positive")
    return value


def _array(
    value: Any,
    location: str,
    *,
    length: int | None = None,
    positive: bool = False,
) -> list[float]:
    if not isinstance(value, list):
        raise WorldConfigError(f"{location} must be an array")
    if length is not None and len(value) != length:
        raise WorldConfigError(f"{location} must contain exactly {length} values")
    return [
        _number(item, f"{location}[{index}]", positive=positive)
        for index, item in enumerate(value)
    ]


def _integer_array(
    value: Any,
    location: str,
    *,
    length: int,
    positive: bool = False,
) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise WorldConfigError(f"{location} must contain exactly {length} integers")
    return [
        _integer(item, f"{location}[{index}]", positive=positive)
        for index, item in enumerate(value)
    ]


def _matrix3(value: Any, location: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise WorldConfigError(f"{location} must be a 3 by 3 array")
    matrix = [
        _array(row, f"{location}[{index}]", length=3) for index, row in enumerate(value)
    ]
    candidate = np.asarray(matrix)
    if not np.allclose(candidate @ candidate.T, np.eye(3), rtol=0.0, atol=1e-10):
        raise WorldConfigError(f"{location} must be an orthonormal rotation matrix")
    if not np.isclose(np.linalg.det(candidate), 1.0, rtol=0.0, atol=1e-10):
        raise WorldConfigError(f"{location} must have determinant +1")
    return matrix


def _json_value(value: Any, location: str) -> Any:
    """Convert a JSON-compatible numeric value while rejecting code-like data."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _number(
            value.item() if isinstance(value, np.generic) else value, location
        )
    if isinstance(value, (list, tuple, np.ndarray)):
        return [
            _json_value(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise WorldConfigError(f"{location} keys must be strings")
        return {
            key: _json_value(item, f"{location}.{key}") for key, item in value.items()
        }
    raise WorldConfigError(
        f"{location} contains unsupported value {type(value).__name__}"
    )


def _load_voxel(value: Any) -> Voxel:
    data = _fields(
        value,
        "voxel",
        required={"axes", "ranges", "shape", "coordinate"},
        optional={"sub_voxel_resolution"},
    )
    axes_data = _fields(data["axes"], "voxel.axes", required={"x", "y", "z"})
    axes = []
    for name in ("x", "y", "z"):
        axis = _array(axes_data[name], f"voxel.axes.{name}")
        if len(axis) < 2:
            raise WorldConfigError(
                f"voxel.axes.{name} must contain at least two values"
            )
        if np.any(np.diff(axis) <= 0):
            raise WorldConfigError(f"voxel.axes.{name} must be strictly increasing")
        axes.append(axis)

    shape = _integer_array(data["shape"], "voxel.shape", length=3, positive=True)
    expected_shape = [len(axis) - 1 for axis in axes]
    if shape != expected_shape:
        raise WorldConfigError(
            f"voxel.shape must equal axis lengths minus one: expected {expected_shape}, "
            f"got {shape}"
        )
    if not isinstance(data["ranges"], list) or len(data["ranges"]) != 3:
        raise WorldConfigError(
            "voxel.ranges must contain three [minimum, maximum] arrays"
        )
    ranges = [
        _array(bounds, f"voxel.ranges[{index}]", length=2)
        for index, bounds in enumerate(data["ranges"])
    ]
    expected_ranges = [[axis[0], axis[-1]] for axis in axes]
    if not np.allclose(ranges, expected_ranges, rtol=0.0, atol=0.0):
        raise WorldConfigError(
            f"voxel.ranges must match axis endpoints: expected {expected_ranges}"
        )

    coordinate = _fields(
        data["coordinate"],
        "voxel.coordinate",
        required={"type", "parameters", "rotation_matrix"},
    )
    if not isinstance(coordinate["type"], str):
        raise WorldConfigError("voxel.coordinate.type must be a string")
    parameters = _object(coordinate["parameters"], "voxel.coordinate.parameters")
    parameters = {
        key: _number(item, f"voxel.coordinate.parameters.{key}")
        for key, item in parameters.items()
        if isinstance(key, str)
    }
    if len(parameters) != len(coordinate["parameters"]):
        raise WorldConfigError("voxel.coordinate.parameters keys must be strings")
    rotation = _matrix3(
        coordinate["rotation_matrix"], "voxel.coordinate.rotation_matrix"
    )
    resolution = data.get("sub_voxel_resolution", [1, 1, 1])
    resolution = _integer_array(
        resolution, "voxel.sub_voxel_resolution", length=3, positive=True
    )
    try:
        return Voxel(
            x_axis=np.asarray(axes[0]),
            y_axis=np.asarray(axes[1]),
            z_axis=np.asarray(axes[2]),
            coordinate_type=coordinate["type"],
            rotation=np.asarray(rotation),
            coordinate_parameters=parameters,
            sub_voxel_resolution=tuple(resolution),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise WorldConfigError(f"invalid voxel configuration: {error}") from error


def _load_eye(value: Any, location: str) -> Eye:
    data = _fields(
        value,
        location,
        required={
            "type",
            "position",
            "focal_length",
            "shape",
            "size",
            "wavelength_range",
        },
    )
    if data["type"] not in {"pinhole", "concave_lens"}:
        raise WorldConfigError(f"{location}.type is unknown: {data['type']!r}")
    if data["shape"] not in {"circle", "ellipse", "rectangle"}:
        raise WorldConfigError(f"{location}.shape is unknown: {data['shape']!r}")
    position = _array(data["position"], f"{location}.position", length=2)
    focal_length = _number(data["focal_length"], f"{location}.focal_length")
    size = _array(data["size"], f"{location}.size", length=2, positive=True)
    wavelength = _array(
        data["wavelength_range"],
        f"{location}.wavelength_range",
        length=2,
        positive=True,
    )
    eye_size: float | list[float] = size[0] if data["shape"] == "circle" else size
    try:
        return Eye(
            position=position,
            focal_length=focal_length,
            eye_type=data["type"],
            eye_size=eye_size,
            eye_shape=data["shape"],
            wavelength_range=tuple(wavelength),
        )
    except (TypeError, ValueError) as error:
        raise WorldConfigError(f"invalid {location}: {error}") from error


def _load_screen(value: Any, location: str) -> Screen:
    data = _fields(
        value,
        location,
        required={"shape", "size", "pixel_shape", "subpixel_resolution"},
    )
    if data["shape"] not in {"circle", "ellipse", "square", "rectangle"}:
        raise WorldConfigError(f"{location}.shape is unknown: {data['shape']!r}")
    size = _array(data["size"], f"{location}.size", length=2, positive=True)
    screen_size: float | list[float] = (
        size[0] if data["shape"] in {"circle", "square"} else size
    )
    pixel_shape = _integer_array(
        data["pixel_shape"], f"{location}.pixel_shape", length=2, positive=True
    )
    subpixel_resolution = _integer(
        data["subpixel_resolution"],
        f"{location}.subpixel_resolution",
        positive=True,
    )
    try:
        return Screen(
            screen_shape=data["shape"],
            screen_size=screen_size,
            pixel_shape=tuple(pixel_shape),
            subpixel_resolution=subpixel_resolution,
        )
    except (TypeError, ValueError) as error:
        raise WorldConfigError(f"invalid {location}: {error}") from error


def _load_aperture(value: Any, location: str) -> Aperture:
    data = _fields(
        value,
        location,
        required={"type", "shape", "size", "position", "direction"},
    )
    if data["type"] != "analytic":
        raise WorldConfigError(
            f"{location}.type is unknown or unsupported: {data['type']!r}"
        )
    if data["shape"] not in {"circle", "ellipse", "rectangle"}:
        raise WorldConfigError(f"{location}.shape is unknown: {data['shape']!r}")
    size = _array(data["size"], f"{location}.size", length=2, positive=True)
    aperture_size: float | list[float] = size[0] if data["shape"] == "circle" else size
    try:
        return Aperture(
            shape=data["shape"],
            size=aperture_size,
            position=_array(data["position"], f"{location}.position", length=3),
            direction=_array(data["direction"], f"{location}.direction", length=3),
        )
    except (TypeError, ValueError) as error:
        raise WorldConfigError(f"invalid {location}: {error}") from error


def _load_camera(value: Any, index: int) -> tuple[str, Camera]:
    location = f"cameras[{index}]"
    data = _fields(
        value,
        location,
        required={
            "key",
            "name",
            "position",
            "rotation_matrix",
            "eyes",
            "screen",
            "apertures",
        },
    )
    if not isinstance(data["key"], str):
        raise WorldConfigError(f"{location}.key must be a string")
    if not isinstance(data["name"], str):
        raise WorldConfigError(f"{location}.name must be a string")
    if not isinstance(data["eyes"], list) or not data["eyes"]:
        raise WorldConfigError(f"{location}.eyes must be a non-empty array")
    if not isinstance(data["apertures"], list):
        raise WorldConfigError(f"{location}.apertures must be an array")
    eyes = [
        _load_eye(item, f"{location}.eyes[{eye_index}]")
        for eye_index, item in enumerate(data["eyes"])
    ]
    apertures = [
        _load_aperture(item, f"{location}.apertures[{aperture_index}]")
        for aperture_index, item in enumerate(data["apertures"])
    ]
    try:
        camera = Camera(
            eyes=eyes,
            apertures=apertures,
            screen=_load_screen(data["screen"], f"{location}.screen"),
            camera_position=_array(data["position"], f"{location}.position", length=3),
            rotation_matrix=np.asarray(
                _matrix3(data["rotation_matrix"], f"{location}.rotation_matrix")
            ),
            camera_name=data["name"],
        )
    except (TypeError, ValueError) as error:
        raise WorldConfigError(f"invalid {location}: {error}") from error
    return data["key"], camera


def _load_inside(value: Any) -> tuple[Any, dict[str, Any]] | None:
    if value is None:
        return None
    data = _fields(value, "inside", required={"type", "parameters"})
    if data["type"] not in _INSIDE_REGISTRY:
        raise WorldConfigError(f"inside.type is unknown: {data['type']!r}")
    function, required = _INSIDE_REGISTRY[data["type"]]
    parameters = _fields(data["parameters"], "inside.parameters", required=required)
    if data["type"] == "box":
        ranges = parameters["ranges"]
        if not isinstance(ranges, list) or len(ranges) != 3:
            raise WorldConfigError("inside.parameters.ranges must contain three ranges")
        parameters["ranges"] = [
            _array(bounds, f"inside.parameters.ranges[{index}]", length=2)
            for index, bounds in enumerate(ranges)
        ]
    elif data["type"] == "sphere":
        parameters["center"] = _array(
            parameters["center"], "inside.parameters.center", length=3
        )
        parameters["radius"] = _number(
            parameters["radius"], "inside.parameters.radius", positive=True
        )
    return function, parameters


def load_world_config(
    source: str | os.PathLike[str] | Mapping[str, Any],
) -> "World":
    """Construct a World from a path or already-parsed configuration mapping."""
    from .world import World

    if isinstance(source, Mapping):
        raw = dict(source)
        base_directory = Path.cwd()
    elif isinstance(source, (str, os.PathLike)):
        path = Path(source)
        try:
            with path.open(encoding="utf-8") as file:
                raw = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise WorldConfigError(
                f"cannot read World config {path}: {error}"
            ) from error
        base_directory = path.resolve().parent
    else:
        raise TypeError("source must be a path or mapping")

    data = _fields(
        raw,
        "config",
        required={
            "schema",
            "schema_version",
            "units",
            "voxel",
            "cameras",
            "walls",
            "inside",
            "verbose",
        },
    )
    if data["schema"] != WORLD_CONFIG_SCHEMA:
        raise WorldConfigError(f"unknown config schema: {data['schema']!r}")
    if data["schema_version"] != WORLD_CONFIG_SCHEMA_VERSION:
        raise WorldConfigError(
            f"unsupported config schema_version: {data['schema_version']!r}"
        )
    units = _fields(data["units"], "units", required={"length", "angle"})
    if units != WORLD_CONFIG_UNITS:
        raise WorldConfigError(
            f"units must be {WORLD_CONFIG_UNITS!r}; automatic conversion is not supported"
        )
    if not isinstance(data["cameras"], list):
        raise WorldConfigError("cameras must be an array")
    cameras: dict[str, Camera] = {}
    for index, item in enumerate(data["cameras"]):
        key, camera = _load_camera(item, index)
        if key in cameras:
            raise WorldConfigError(f"duplicate camera key: {key!r}")
        cameras[key] = camera

    if not isinstance(data["walls"], list):
        raise WorldConfigError("walls must be an array")
    walls = []
    wall_paths = []
    for index, item in enumerate(data["walls"]):
        wall = _fields(item, f"walls[{index}]", required={"type", "path"})
        if wall["type"] != "stl":
            raise WorldConfigError(f"walls[{index}].type is unknown: {wall['type']!r}")
        if not isinstance(wall["path"], str) or not wall["path"]:
            raise WorldConfigError(f"walls[{index}].path must be a non-empty string")
        resolved = Path(wall["path"])
        if not resolved.is_absolute():
            resolved = base_directory / resolved
        resolved = resolved.resolve()
        try:
            walls.append(mesh.Mesh.from_file(resolved))
        except (OSError, ValueError) as error:
            raise WorldConfigError(
                f"cannot load walls[{index}].path {wall['path']!r}: {error}"
            ) from error
        wall_paths.append(resolved)

    verbose = _integer(data["verbose"], "verbose")
    world = World(
        voxel=_load_voxel(data["voxel"]),
        cameras=cameras,
        walls=walls,
        verbose=verbose,
    )
    inside = _load_inside(data["inside"])
    if inside is not None:
        world.set_inside_vertices(inside[0], **inside[1])
    world._config_wall_paths = tuple(wall_paths)
    return world


def _world_config_mapping(world: "World", target_path: Path | None) -> dict[str, Any]:
    if not all(isinstance(key, str) for key in world.cameras):
        raise WorldConfigError("to_config requires every camera key to be a string")

    voxel = world.voxel
    cameras = []
    for key, camera in world.cameras.items():
        apertures = []
        for index, aperture in enumerate(camera.apertures):
            if aperture.shape not in {"circle", "ellipse", "rectangle"}:
                raise WorldConfigError(
                    f"camera {key!r} aperture {index} is not an analytic aperture"
                )
            apertures.append(
                {
                    "type": "analytic",
                    "shape": aperture.shape,
                    "size": np.asarray(aperture.size, dtype=float).tolist(),
                    "position": np.asarray(aperture.position, dtype=float).tolist(),
                    "direction": np.asarray(aperture.direction, dtype=float).tolist(),
                }
            )
        cameras.append(
            {
                "key": key,
                "name": str(camera._camera_name),
                "position": np.asarray(camera.camera_position, dtype=float).tolist(),
                "rotation_matrix": np.asarray(
                    camera.rotation_matrix, dtype=float
                ).tolist(),
                "eyes": [
                    {
                        "type": eye.eye_type,
                        "position": np.asarray(eye.position[:2], dtype=float).tolist(),
                        "focal_length": float(eye.focal_length),
                        "shape": eye.eye_shape,
                        "size": np.asarray(eye.eye_size, dtype=float).tolist(),
                        "wavelength_range": [
                            float(item) for item in eye.wavelength_range
                        ],
                    }
                    for eye in camera.eyes
                ],
                "screen": {
                    "shape": camera.screen.screen_shape,
                    "size": np.asarray(camera.screen.screen_size, dtype=float).tolist(),
                    "pixel_shape": np.asarray(
                        camera.screen.pixel_shape, dtype=int
                    ).tolist(),
                    "subpixel_resolution": int(camera.screen.subpixel_resolution),
                },
                "apertures": apertures,
            }
        )

    wall_paths = getattr(world, "_config_wall_paths", ())
    if len(wall_paths) != len(world.walls):
        raise WorldConfigError(
            "World contains wall mesh objects without source-path provenance; "
            "they cannot be represented faithfully in a scene config"
        )
    walls = []
    for source in wall_paths:
        source_path = Path(source).resolve()
        output_path: Path | str = source_path
        if target_path is not None:
            output_path = Path(
                os.path.relpath(source_path, start=target_path.resolve().parent)
            )
        walls.append({"type": "stl", "path": str(output_path)})

    if world._inside_function is None:
        if world._inside_vertices is not None:
            raise WorldConfigError(
                "World has an explicit inside_vertices array without a registered "
                "construction function; it cannot be represented in config"
            )
        inside = None
    else:
        name = _INSIDE_NAMES.get(world._inside_function)
        if name is None:
            raise WorldConfigError(
                "World inside_func is an arbitrary callable; only built-in config "
                f"types {sorted(_INSIDE_REGISTRY)} are supported"
            )
        inside = {
            "type": name,
            "parameters": _json_value(world._inside_kwargs, "inside.parameters"),
        }

    return {
        "schema": WORLD_CONFIG_SCHEMA,
        "schema_version": WORLD_CONFIG_SCHEMA_VERSION,
        "units": dict(WORLD_CONFIG_UNITS),
        "voxel": {
            "axes": {
                "x": np.asarray(voxel.x_axis, dtype=float).tolist(),
                "y": np.asarray(voxel.y_axis, dtype=float).tolist(),
                "z": np.asarray(voxel.z_axis, dtype=float).tolist(),
            },
            "ranges": [[float(lower), float(upper)] for lower, upper in voxel.ranges],
            "shape": [int(item) for item in voxel.shape],
            "coordinate": {
                "type": voxel.coordinate_type,
                "parameters": _json_value(
                    voxel.coordinate_parameters, "voxel.coordinate.parameters"
                ),
                "rotation_matrix": np.asarray(
                    voxel._rotation_matrix, dtype=float
                ).tolist(),
            },
            "sub_voxel_resolution": [int(item) for item in voxel.res],
        },
        "cameras": cameras,
        "walls": walls,
        "inside": inside,
        "verbose": int(world.verbose),
    }


def dump_world_config(
    world: "World", destination: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Return a canonical mapping and optionally write it as UTF-8 JSON."""
    path = Path(destination) if destination is not None else None
    mapping = _world_config_mapping(world, path)
    if path is not None:
        try:
            with path.open("w", encoding="utf-8", newline="\n") as file:
                json.dump(mapping, file, indent=2, ensure_ascii=False, allow_nan=False)
                file.write("\n")
        except OSError as error:
            raise WorldConfigError(
                f"cannot write World config {path}: {error}"
            ) from error
    return mapping
