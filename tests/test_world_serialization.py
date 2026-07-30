"""Versioned World archive and legacy migration regressions."""

import json
import zipfile

import dill
import numpy as np
import pytest
from scipy import sparse

import multi_pinhole.serialization as serialization
from multi_pinhole import Camera, Eye, Screen, Voxel, World
from multi_pinhole.__about__ import __version__
from multi_pinhole.serialization import (
    MANIFEST_NAME,
    WORLD_ARCHIVE_FORMAT,
    WORLD_PAYLOAD_NAME,
    WORLD_SCHEMA_VERSION,
    WorldSerializationError,
)
from multi_pinhole.world import PROJECTION_CACHE_SCHEMA_VERSION


def _world_with_caches():
    camera = Camera(
        eyes=[Eye(position=(0.0, 0.0), focal_length=10.0)],
        apertures=[],
        screen=Screen("square", 10.0, pixel_shape=(2, 2)),
        camera_position=(0.0, 0.0, -20.0),
    )
    voxel = Voxel.uniform_voxel(
        ranges=((-1.0, 1.0), (-1.0, 1.0), (1.0, 2.0)),
        shape=(2, 1, 1),
    )
    world = World(voxel=voxel, cameras={"main": camera}, verbose=0)
    world._visible_vertices["main"] = np.array([[True] * voxel.N_grid])
    world._visible_voxels["main"] = np.array([[1, 2]], dtype=np.int8)
    eye_projection = sparse.csr_matrix(
        ([2.0, 3.0], ([0, 1], [0, 1])),
        shape=(camera.screen.N_pixel, voxel.N),
    )
    combined = sparse.csr_matrix(
        ([5.0, 7.0], ([0, 1], [0, 1])),
        shape=(camera.screen.N_pixel, voxel.N),
    )
    world._projection["main"][0] = eye_projection
    world._projection_settings["main"][0] = ("fixture",)
    world._P_matrix["main"] = combined
    return world


def _rewrite_manifest(source, destination, **updates):
    with zipfile.ZipFile(source) as archive:
        manifest = json.loads(archive.read(MANIFEST_NAME))
        payload = archive.read(WORLD_PAYLOAD_NAME)
    manifest.update(updates)
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest))
        archive.writestr(WORLD_PAYLOAD_NAME, payload)


def test_archive_metadata_and_member_contract(tmp_path):
    world = _world_with_caches()
    path = tmp_path / "world.mpw"

    world.save(path)
    metadata = World.inspect_archive(path)

    assert metadata == {
        "format": WORLD_ARCHIVE_FORMAT,
        "world_schema_version": WORLD_SCHEMA_VERSION,
        "library_version": __version__,
        "projection_cache_schema_version": PROJECTION_CACHE_SCHEMA_VERSION,
        "python_version": metadata["python_version"],
        "numpy_version": np.__version__,
        "scipy_version": metadata["scipy_version"],
    }
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {MANIFEST_NAME, WORLD_PAYLOAD_NAME}


def test_inspect_archive_does_not_unpickle(tmp_path, monkeypatch):
    path = tmp_path / "world.mpw"
    _world_with_caches().save(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("inspect_archive unpickled the payload")

    monkeypatch.setattr(serialization.dill, "loads", forbidden)
    monkeypatch.setattr(serialization.dill, "load", forbidden)

    assert World.inspect_archive(path)["format"] == WORLD_ARCHIVE_FORMAT


def test_current_world_roundtrip_preserves_visibility_and_projection(tmp_path):
    world = _world_with_caches()
    path = tmp_path / "world.mpw"
    emission = np.array([1.25, -0.5])
    expected = world.project(emission, "main")

    world.save(path)
    restored = World.load(path)

    np.testing.assert_array_equal(
        restored._visible_vertices["main"], world._visible_vertices["main"]
    )
    np.testing.assert_array_equal(
        restored.visible_voxels["main"], world.visible_voxels["main"]
    )
    np.testing.assert_allclose(
        restored.projection["main"][0].toarray(),
        world.projection["main"][0].toarray(),
    )
    np.testing.assert_allclose(restored.project(emission, "main"), expected)
    assert restored._projection_settings["main"][0] == ("fixture",)


def test_incompatible_manifest_cache_schema_invalidates_only_projection(tmp_path):
    source = tmp_path / "world.mpw"
    incompatible = tmp_path / "incompatible.mpw"
    world = _world_with_caches()
    world.save(source)
    _rewrite_manifest(
        source,
        incompatible,
        projection_cache_schema_version=PROJECTION_CACHE_SCHEMA_VERSION + 1,
    )

    restored = World.load(incompatible)

    np.testing.assert_array_equal(
        restored._visible_vertices["main"], world._visible_vertices["main"]
    )
    np.testing.assert_array_equal(
        restored.visible_voxels["main"], world.visible_voxels["main"]
    )
    assert restored.projection["main"] == [None]
    assert restored.P_matrix["main"] is None
    assert restored.projection_cache_schema_version == PROJECTION_CACHE_SCHEMA_VERSION


def test_unsupported_world_schema_fails_before_unpickling(tmp_path, monkeypatch):
    source = tmp_path / "world.mpw"
    incompatible = tmp_path / "future.mpw"
    _world_with_caches().save(source)
    _rewrite_manifest(
        source,
        incompatible,
        world_schema_version=WORLD_SCHEMA_VERSION + 1,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported archive was unpickled")

    monkeypatch.setattr(serialization.dill, "loads", forbidden)
    with pytest.raises(
        WorldSerializationError, match="unsupported world_schema_version"
    ):
        World.load(incompatible)


def test_legacy_direct_dill_load_and_archive_migration(tmp_path):
    world = _world_with_caches()
    legacy = tmp_path / "legacy.pkl"
    migrated = tmp_path / "migrated.mpw"
    emission = np.array([0.25, 2.0])
    expected = world.project(emission, "main")
    with legacy.open("wb") as file:
        dill.dump(world, file)

    assert World.inspect_archive(legacy) == {
        "format": "legacy-direct-dill",
        "world_schema_version": None,
        "library_version": None,
        "projection_cache_schema_version": None,
        "python_version": None,
        "numpy_version": None,
        "scipy_version": None,
    }
    loaded_legacy = World.load(legacy)
    np.testing.assert_allclose(loaded_legacy.project(emission, "main"), expected)

    loaded_legacy.save(migrated)
    restored = World.load(migrated)

    np.testing.assert_allclose(restored.project(emission, "main"), expected)
    np.testing.assert_array_equal(
        restored.visible_voxels["main"], world.visible_voxels["main"]
    )


@pytest.mark.parametrize("legacy_version", [None, 0, 10_000])
def test_legacy_incompatible_cache_keeps_visibility(tmp_path, legacy_version):
    world = _world_with_caches()
    if legacy_version is None:
        del world._projection_cache_schema_version
    else:
        world._projection_cache_schema_version = legacy_version
    legacy = tmp_path / f"legacy-{legacy_version}.pkl"
    with legacy.open("wb") as file:
        dill.dump(world, file)

    restored = World.load(legacy)

    np.testing.assert_array_equal(
        restored.visible_voxels["main"], world.visible_voxels["main"]
    )
    assert restored.projection["main"] == [None]
    assert restored.P_matrix["main"] is None


def test_save_does_not_mutate_world_or_incompatible_cache_state(tmp_path):
    world = _world_with_caches()
    projection = world.P_matrix["main"]
    projection_data = projection.data.copy()
    world._projection_cache_schema_version = 999

    world.save(tmp_path / "world.mpw")

    assert world.P_matrix["main"] is projection
    np.testing.assert_array_equal(world.P_matrix["main"].data, projection_data)
    assert world.projection_cache_schema_version == 999


def test_save_world_and_load_world_are_archive_compatibility_aliases(tmp_path):
    path = tmp_path / "compatibility-name.pkl"
    world = _world_with_caches()

    world.save_world(path)
    restored = World.load_world(path)

    assert zipfile.is_zipfile(path)
    np.testing.assert_allclose(
        restored.P_matrix["main"].toarray(), world.P_matrix["main"].toarray()
    )


def test_library_world_and_projection_schema_versions_are_independent(tmp_path):
    path = tmp_path / "world.mpw"
    _world_with_caches().save(path)

    metadata = World.inspect_archive(path)

    assert isinstance(metadata["library_version"], str)
    assert metadata["world_schema_version"] == 1
    assert metadata["projection_cache_schema_version"] == 3
