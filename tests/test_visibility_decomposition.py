import inspect
import pickle

import dill
import numpy as np

import multi_pinhole
from multi_pinhole import World
from multi_pinhole import _visibility
from multi_pinhole import world as world_module


class _Eye:
    def __init__(self, z):
        self.position = np.array([0.0, 0.0, z])


def test_world_public_identity_and_signature_remain_in_world_module():
    assert multi_pinhole.World is world_module.World
    assert World.__module__ == "multi_pinhole.world"
    parameters = inspect.signature(World.find_visible_points).parameters
    assert tuple((name, parameter.default) for name, parameter in parameters.items()) == (
        ("self", inspect.Parameter.empty),
        ("points", inspect.Parameter.empty),
        ("camera_idx", inspect.Parameter.empty),
        ("eye_idx", None),
        ("verbose", 1),
    )


def test_legacy_world_pickle_and_dill_globals_resolve():
    payload = b"cmulti_pinhole.world\nWorld\n."
    assert pickle.loads(payload) is World
    assert dill.loads(payload) is World


def test_visibility_module_shares_world_stl_utils_for_monkeypatch_compatibility():
    assert _visibility.stl_utils is world_module.stl_utils


def test_point_visibility_front_mask_has_fixed_shape_and_dtype():
    points = np.array([
        [0.0, 0.0, -2.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    actual = _visibility.calculate_point_visibility(
        camera_points=points,
        eyes=[_Eye(-1.0), _Eye(1.0)],
        eye_indices=[0, 1],
        apertures=[],
        walls_in_camera=[],
        verbose=0,
    )
    expected = np.array([
        [False, True, True],
        [False, False, True],
    ], dtype=bool)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (2, 3)
    assert actual.dtype == np.dtype(bool)


def test_vertex_expansion_and_voxel_classification_have_fixed_values():
    inside = np.array([True, True, True, False, True, True, True, True])
    inside_visibility = np.array([
        [False, True, True, True, True, True, True],
        [True, True, True, True, True, True, True],
        [False, False, False, False, False, False, False],
    ], dtype=bool)
    visible_vertices = _visibility.calculate_visible_vertex_mask(
        inside, inside_visibility,
    )
    expected_vertices = np.array([
        [False, True, True, False, True, True, True, True],
        [True, True, True, False, True, True, True, True],
        [False, False, False, False, False, False, False, False],
    ], dtype=bool)
    np.testing.assert_array_equal(visible_vertices, expected_vertices)

    states = _visibility.classify_visible_voxels(
        visible_vertices,
        np.array([[0, 1, 2, 4, 5, 6, 7], [1, 2, 4, 5, 6, 7, 1]]),
    )
    np.testing.assert_array_equal(
        states,
        np.array([[1, 2], [2, 2], [0, 0]]),
    )
    assert states.dtype == np.dtype(int)
