"""Contract tests for the versioned, cache-free World JSON schema."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from stl import mesh

from multi_pinhole import Aperture, Camera, Eye, Screen, Voxel, World
from multi_pinhole.config import (
    WORLD_CONFIG_SCHEMA,
    WORLD_CONFIG_SCHEMA_VERSION,
    WorldConfigError,
)
from multi_pinhole.utils import stl_utils


ROOT = Path(__file__).resolve().parents[1]


def _camera(position=(0.0, 0.0, -20.0)):
    return Camera(
        eyes=[
            Eye(
                position=(0.0, 0.0),
                focal_length=10.0,
                eye_size=0.5,
                wavelength_range=(0.02, 0.2),
            )
        ],
        apertures=[
            Aperture(
                shape="circle",
                size=1.0,
                position=(0.0, 0.0, 5.0),
            )
        ],
        screen=Screen("square", 20.0, pixel_shape=(4, 5), subpixel_resolution=2),
        camera_position=position,
        camera_name="test camera",
    )


def _world():
    voxel = Voxel(
        x_axis=np.array([-2.0, -0.5, 2.0]),
        y_axis=np.array([-1.0, 1.0]),
        z_axis=np.array([3.0, 4.0, 6.0]),
        coordinate_type="torus",
        coordinate_parameters={"major_radius": 1500.0, "minor_radius": 500.0},
        sub_voxel_resolution=(2, 3, 4),
    )
    return World(voxel=voxel, cameras={"main": _camera()}, verbose=0)


def _minimal_mapping():
    return _world().to_config()


def test_minimal_world_json_roundtrip(tmp_path):
    original = _world()
    path = tmp_path / "world.json"

    mapping = original.to_config(path)
    restored = World.from_config(path)

    assert mapping["schema"] == WORLD_CONFIG_SCHEMA
    assert mapping["schema_version"] == WORLD_CONFIG_SCHEMA_VERSION
    assert mapping["units"] == {"length": "mm", "angle": "rad"}
    assert restored.to_config() == mapping
    assert restored.verbose == 0


def test_multiple_camera_keys_and_voxel_coordinate_parameters_roundtrip():
    world = _world()
    world.add_camera("side", _camera(position=(1.0, 2.0, -30.0)))

    restored = World.from_config(world.to_config())

    assert list(restored.cameras) == ["main", "side"]
    assert restored.cameras["side"].camera_position.tolist() == [1.0, 2.0, -30.0]
    assert restored.voxel.res == (2, 3, 4)
    assert restored.voxel.coordinate_type == "torus"
    assert restored.voxel.coordinate_parameters == {
        "major_radius": 1500.0,
        "minor_radius": 500.0,
    }


def test_relative_wall_stl_path_uses_config_directory(tmp_path):
    shutil.copy(ROOT / "examples/mst/MST_wall-mesh.stl", tmp_path / "wall.stl")
    mapping = _minimal_mapping()
    mapping["walls"] = [{"type": "stl", "path": "wall.stl"}]
    source = tmp_path / "scene.json"
    source.write_text(json.dumps(mapping), encoding="utf-8")

    world = World.from_config(source)
    destination = tmp_path / "nested"
    destination.mkdir()
    output = destination / "roundtrip.json"
    roundtrip = world.to_config(output)

    assert len(world.walls) == 1
    assert roundtrip["walls"] == [{"type": "stl", "path": "../wall.stl"}]
    assert len(World.from_config(output).walls) == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "other/world", "unknown config schema"),
        ("schema_version", 999, "unsupported config schema_version"),
    ],
)
def test_unknown_schema_and_version_fail(field, value, message):
    mapping = _minimal_mapping()
    mapping[field] = value
    with pytest.raises(WorldConfigError, match=message):
        World.from_config(mapping)


def test_unknown_field_and_type_fail():
    mapping = _minimal_mapping()
    mapping["surprise"] = True
    with pytest.raises(WorldConfigError, match="unknown field"):
        World.from_config(mapping)

    mapping = _minimal_mapping()
    mapping["cameras"][0]["apertures"][0]["type"] = "python-object"
    with pytest.raises(WorldConfigError, match="unknown or unsupported"):
        World.from_config(mapping)


def test_malformed_voxel_shape_and_missing_field_fail():
    mapping = _minimal_mapping()
    mapping["voxel"]["shape"] = [2, 1, 999]
    with pytest.raises(WorldConfigError, match="axis lengths minus one"):
        World.from_config(mapping)

    mapping = _minimal_mapping()
    del mapping["cameras"][0]["screen"]
    with pytest.raises(WorldConfigError, match="missing required field"):
        World.from_config(mapping)


def test_safe_inside_registry_roundtrip_and_unknown_type_failure():
    mapping = _minimal_mapping()
    mapping["inside"] = {
        "type": "sphere",
        "parameters": {"center": [0.0, 0.0, 4.0], "radius": 2.0},
    }
    world = World.from_config(mapping)

    assert world.to_config()["inside"] == mapping["inside"]
    assert world.inside_vertices.dtype == np.dtype(bool)

    mapping["inside"]["type"] = "import"
    with pytest.raises(WorldConfigError, match="inside.type is unknown"):
        World.from_config(mapping)


def test_arbitrary_inside_callable_and_explicit_mask_fail_to_config():
    world = _world()
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    with pytest.raises(WorldConfigError, match="arbitrary callable"):
        world.to_config()

    world = _world()
    world.inside_vertices = np.ones(world.voxel.N_grid, dtype=bool)
    with pytest.raises(WorldConfigError, match="explicit inside_vertices"):
        world.to_config()


def test_stl_aperture_and_unprovenanced_wall_fail_to_config():
    model = mesh.Mesh.from_file(ROOT / "examples/mst/MST_wall-mesh.stl")
    camera = Camera(
        eyes=[Eye(position=(0.0, 0.0), focal_length=10.0)],
        apertures=[Aperture(stl_model=model)],
        screen=Screen("square", 20.0, pixel_shape=(2, 2)),
        camera_position=(0.0, 0.0, -20.0),
    )
    with pytest.raises(WorldConfigError, match="not an analytic aperture"):
        World(cameras={"main": camera}, verbose=0).to_config()

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wall = stl_utils.make_stl(vertices, np.array([[0, 1, 2]]))
    with pytest.raises(WorldConfigError, match="without source-path provenance"):
        World(cameras={}, walls=wall, verbose=0).to_config()


def test_config_contains_no_visibility_or_projection_cache():
    world = _world()
    world._visible_vertices["main"] = np.ones((1, world.voxel.N_grid), dtype=bool)
    world._visible_voxels["main"] = np.ones((1, world.voxel.N), dtype=np.int8)

    text = json.dumps(world.to_config())

    for forbidden in (
        "visible_vertices",
        "visible_voxels",
        "projection",
        "P_matrix",
        "projection_cache_schema_version",
    ):
        assert forbidden not in text


def test_mapping_input_is_not_mutated():
    mapping = _minimal_mapping()
    original = copy.deepcopy(mapping)

    World.from_config(mapping)

    assert mapping == original
