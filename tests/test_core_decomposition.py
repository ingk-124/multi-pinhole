"""Compatibility regressions for the core module decomposition."""
import inspect
import pickle

import dill

import multi_pinhole
import multi_pinhole.core as core
from multi_pinhole.aperture import Aperture
from multi_pinhole.camera import Camera
from multi_pinhole.eye import Eye
from multi_pinhole.rays import Rays
from multi_pinhole.screen import Screen
from multi_pinhole.world import PROJECTION_CACHE_SCHEMA_VERSION


PUBLIC_CLASSES = (Eye, Aperture, Screen, Camera, Rays)


def _small_camera():
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=0.5)
    screen = Screen("square", 20.0, pixel_shape=(4, 4), subpixel_resolution=2)
    return Camera(
        eyes=[eye], apertures=[], screen=screen,
        camera_position=(0.0, 0.0, 0.0),
    )


def test_top_level_core_and_new_modules_share_class_objects():
    for class_ in PUBLIC_CLASSES:
        assert getattr(multi_pinhole, class_.__name__) is class_
        assert getattr(core, class_.__name__) is class_


def test_decomposed_class_module_paths():
    assert Eye.__module__ == "multi_pinhole.eye"
    assert Aperture.__module__ == "multi_pinhole.aperture"
    assert Screen.__module__ == "multi_pinhole.screen"
    assert Camera.__module__ == "multi_pinhole.camera"
    assert Rays.__module__ == "multi_pinhole.rays"


def test_legacy_core_pickle_and_dill_globals_resolve():
    for class_ in PUBLIC_CLASSES:
        payload = f"cmulti_pinhole.core\n{class_.__name__}\n.".encode("ascii")
        assert pickle.loads(payload) is class_
        assert dill.loads(payload) is class_


def test_new_module_instances_roundtrip_with_pickle_and_dill():
    camera = _small_camera()
    objects = (camera.eyes[0], camera.screen, camera)
    for serializer in (pickle, dill):
        for obj in objects:
            restored = serializer.loads(serializer.dumps(obj))
            assert type(restored) is type(obj)
            assert type(restored).__module__ == type(obj).__module__


def test_public_constructor_parameter_contract_and_defaults():
    expected = {
        Eye: (
            ("position", inspect.Parameter.empty),
            ("focal_length", inspect.Parameter.empty),
            ("eye_type", 1),
            ("eye_size", 0.5),
            ("eye_shape", "circle"),
            ("wavelength_range", (0.01, 0.1)),
        ),
        Aperture: (
            ("shape", None), ("size", None), ("position", None),
            ("direction", None), ("stl_model", None),
            ("stl_args", inspect.Parameter.empty),
        ),
        Screen: (
            ("screen_shape", "square"), ("screen_size", 10),
            ("pixel_shape", (100, 100)), ("subpixel_resolution", 1),
        ),
        Camera: (
            ("eyes", inspect.Parameter.empty),
            ("apertures", inspect.Parameter.empty),
            ("screen", inspect.Parameter.empty),
            ("camera_position", inspect.Parameter.empty),
            ("rotation_matrix", None), ("camera_name", None),
        ),
    }
    for class_, contract in expected.items():
        parameters = inspect.signature(class_).parameters
        assert tuple((name, parameter.default)
                     for name, parameter in parameters.items()) == contract


def test_projection_cache_schema_remains_three():
    assert PROJECTION_CACHE_SCHEMA_VERSION == 3
