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


def test_uniform_voxel_uses_compact_canonical_form():
    world = World(
        voxel=Voxel.uniform_voxel(
            ranges=[[-2.0, 2.0], [-1.0, 1.0], [3.0, 6.0]],
            shape=[4, 2, 3],
        ),
        cameras={},
        verbose=0,
    )

    config = world.to_config()

    assert config["voxel"]["type"] == "uniform"
    assert config["voxel"]["ranges"] == [[-2.0, 2.0], [-1.0, 1.0], [3.0, 6.0]]
    assert config["voxel"]["shape"] == [4, 2, 3]
    assert "axes" not in config["voxel"]
    restored = World.from_config(config)
    np.testing.assert_allclose(restored.voxel.x_axis, world.voxel.x_axis)
    np.testing.assert_allclose(restored.voxel.y_axis, world.voxel.y_axis)
    np.testing.assert_allclose(restored.voxel.z_axis, world.voxel.z_axis)
    assert restored.to_config() == config


def test_nonuniform_voxel_uses_explicit_axes_canonical_form():
    config = _minimal_mapping()

    assert config["voxel"]["type"] == "axes"
    assert set(config["voxel"]["axes"]) == {"x", "y", "z"}
    assert "ranges" not in config["voxel"]
    assert "shape" not in config["voxel"]


def test_analytic_aperture_model_settings_roundtrip():
    mapping = _minimal_mapping()
    aperture = mapping["cameras"][0]["apertures"][0]
    aperture["resolution"] = 40
    aperture["max_size"] = [100.0, 120.0]

    world = World.from_config(mapping)
    restored = world.to_config()["cameras"][0]["apertures"][0]

    assert restored["resolution"] == 40
    assert restored["max_size"] == [100.0, 120.0]
    assert world.cameras["main"].apertures[0].model_resolution == 40
    np.testing.assert_array_equal(
        world.cameras["main"].apertures[0].model_max_size,
        [100.0, 120.0],
    )

    aperture["resolution"] = 0
    with pytest.raises(WorldConfigError, match="resolution must be positive"):
        World.from_config(mapping)


def test_analytic_aperture_omitted_model_settings_use_defaults():
    mapping = _minimal_mapping()
    aperture = mapping["cameras"][0]["apertures"][0]
    del aperture["resolution"]
    del aperture["max_size"]

    world = World.from_config(mapping)
    restored = world.to_config()["cameras"][0]["apertures"][0]

    assert restored["resolution"] == 20
    assert restored["max_size"] is None


@pytest.mark.parametrize("lateral_name", ["right_point", "down_point"])
def test_camera_point_orientation_roundtrip(lateral_name):
    mapping = _minimal_mapping()
    camera = mapping["cameras"][0]
    del camera["rotation_matrix"]
    lateral_point = [1.0, 0.0, -20.0]
    if lateral_name == "down_point":
        lateral_point = [0.0, 1.0, -20.0]
    camera["orientation"] = {
        "look_point": [0.0, 0.0, 0.0],
        lateral_name: lateral_point,
    }

    world = World.from_config(mapping)

    np.testing.assert_allclose(world.cameras["main"].rotation_matrix, np.eye(3))
    assert world.to_config()["cameras"][0]["orientation"] == camera["orientation"]


def test_camera_orientation_requires_one_representation_and_lateral_point():
    mapping = _minimal_mapping()
    camera = mapping["cameras"][0]
    camera["orientation"] = {
        "look_point": [0.0, 0.0, 0.0],
        "right_point": [1.0, 0.0, -20.0],
    }
    with pytest.raises(WorldConfigError, match="exactly one"):
        World.from_config(mapping)

    del camera["rotation_matrix"]
    camera["orientation"]["down_point"] = [0.0, 1.0, -20.0]
    with pytest.raises(WorldConfigError, match="exactly one"):
        World.from_config(mapping)

    del camera["orientation"]["right_point"]
    del camera["orientation"]["down_point"]
    with pytest.raises(WorldConfigError, match="exactly one"):
        World.from_config(mapping)

    del camera["orientation"]
    with pytest.raises(WorldConfigError, match="exactly one"):
        World.from_config(mapping)


def test_camera_point_orientation_rejects_degenerate_directions():
    mapping = _minimal_mapping()
    camera = mapping["cameras"][0]
    del camera["rotation_matrix"]
    camera["orientation"] = {
        "look_point": [0.0, 0.0, 0.0],
        "right_point": [0.0, 0.0, 1.0],
    }

    with pytest.raises(WorldConfigError, match="invalid cameras"):
        World.from_config(mapping)


def test_relative_stl_aperture_path_roundtrip(tmp_path):
    shutil.copy(ROOT / "examples/mst/MST_wall-mesh.stl", tmp_path / "aperture.stl")
    mapping = _minimal_mapping()
    mapping["cameras"][0]["apertures"] = [
        {
            "type": "stl",
            "path": "aperture.stl",
            "position": [1.0, 2.0, 3.0],
        }
    ]
    source = tmp_path / "scene.json"
    source.write_text(json.dumps(mapping), encoding="utf-8")

    world = World.from_config(source)
    destination = tmp_path / "nested"
    destination.mkdir()
    output = destination / "roundtrip.json"
    restored = world.to_config(output)

    assert world.cameras["main"].apertures[0].shape == "stl"
    assert restored["cameras"][0]["apertures"] == [
        {
            "type": "stl",
            "path": "../aperture.stl",
            "position": [1.0, 2.0, 3.0],
        }
    ]
    assert World.from_config(output).cameras["main"].apertures[0].shape == "stl"


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


def test_malformed_voxel_variants_and_missing_field_fail():
    mapping = _minimal_mapping()
    mapping["voxel"]["axes"]["x"][1] = mapping["voxel"]["axes"]["x"][0]
    with pytest.raises(WorldConfigError, match="strictly increasing"):
        World.from_config(mapping)

    mapping = _minimal_mapping()
    mapping["voxel"]["type"] = "uniform"
    with pytest.raises(WorldConfigError, match="missing required field"):
        World.from_config(mapping)

    mapping = World(
        voxel=Voxel.uniform_voxel([[-1, 1]] * 3, [2, 2, 2]),
        cameras={},
        verbose=0,
    ).to_config()
    mapping["voxel"]["ranges"][0] = [1.0, -1.0]
    with pytest.raises(WorldConfigError, match="minimum must be less"):
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


def test_torus_inside_registry_roundtrip_and_validation():
    mapping = _minimal_mapping()
    mapping["inside"] = {
        "type": "torus",
        "parameters": {"major_radius": 2.0, "minor_radius": 0.5},
    }

    world = World.from_config(mapping)

    assert world.to_config()["inside"] == mapping["inside"]
    expected = (
        np.hypot(world.voxel.grid[:, 0], world.voxel.grid[:, 1]) - 2.0
    ) ** 2 + world.voxel.grid[:, 2] ** 2 <= 0.5**2
    np.testing.assert_array_equal(world.inside_vertices, expected)

    mapping["inside"]["parameters"]["minor_radius"] = 0.0
    with pytest.raises(WorldConfigError, match="minor_radius must be positive"):
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


def test_unprovenanced_stl_aperture_and_wall_fail_to_config():
    model = mesh.Mesh.from_file(ROOT / "examples/mst/MST_wall-mesh.stl")
    camera = Camera(
        eyes=[Eye(position=(0.0, 0.0), focal_length=10.0)],
        apertures=[Aperture(stl_model=model)],
        screen=Screen("square", 20.0, pixel_shape=(2, 2)),
        camera_position=(0.0, 0.0, -20.0),
    )
    with pytest.raises(WorldConfigError, match="no source-path provenance"):
        World(
            voxel=Voxel.uniform_voxel([[-1, 1]] * 3, [1, 1, 1]),
            cameras={"main": camera},
            verbose=0,
        ).to_config()

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wall = stl_utils.make_stl(vertices, np.array([[0, 1, 2]]))
    with pytest.raises(WorldConfigError, match="without source-path provenance"):
        World(
            voxel=Voxel.uniform_voxel([[-1, 1]] * 3, [1, 1, 1]),
            cameras={},
            walls=wall,
            verbose=0,
        ).to_config()


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


def test_relax_json_example_geometry_and_camera_orientation():
    path = ROOT / "examples/relax/relax_world.json"

    world = World.from_config(path)
    camera = world.cameras["phi=+45deg"]

    assert world.voxel.coordinate_parameters == {
        "major_radius": 508.0,
        "minor_radius": 250.0,
    }
    np.testing.assert_allclose(
        camera.camera_position,
        950.0
        * np.array(
            [
                np.cos(np.deg2rad(45.0)),
                np.sin(np.deg2rad(45.0)),
                0.0,
            ]
        ),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        camera.camera_z,
        np.array([-np.cos(np.deg2rad(45.0)), -np.sin(np.deg2rad(45.0)), 0.0]),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        camera.camera_x,
        np.array([-np.sin(np.deg2rad(45.0)), np.cos(np.deg2rad(45.0)), 0.0]),
        atol=1e-10,
    )
    np.testing.assert_allclose(camera.camera_y, [0.0, 0.0, -1.0], atol=1e-12)
    assert np.any(np.isclose(world.voxel.gravity_center[:, 2], 0.0))
    assert len(world.walls) == 1
    assert world.to_config()["inside"] == {
        "type": "torus",
        "parameters": {"major_radius": 508.0, "minor_radius": 250.0},
    }
