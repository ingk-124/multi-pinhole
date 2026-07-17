import dill
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from multi_pinhole import Voxel
from multi_pinhole.coordinates import COORDINATE_TYPES


def _voxel(rotation=None):
    return Voxel.uniform_voxel(
        ranges=((-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0)),
        shape=(2, 2, 2),
        rotation=rotation,
    )


def test_available_coordinate_types_comes_from_coordinates_registry():
    voxel = _voxel()

    assert voxel.available_coordinate_types is COORDINATE_TYPES
    assert voxel.available_coordinate_types == (
        "cartesian", "torus", "torus_inverse", "cylindrical", "spherical",
    )


def test_to_cylindrical_selects_centers_and_vertices_without_normalizing():
    voxel = _voxel()

    centers = voxel.to_coordinates("cylindrical")
    vertices = voxel.to_coordinates("cylindrical", points="vertices")

    np.testing.assert_allclose(centers[:, 0], np.hypot(
        voxel.gravity_center[:, 0], voxel.gravity_center[:, 1],
    ))
    np.testing.assert_allclose(centers[:, 2], voxel.gravity_center[:, 2])
    assert centers.shape == (voxel.N_voxel, 3)
    assert vertices.shape == (voxel.N_grid, 3)
    np.testing.assert_array_equal(
        voxel.to_coordinates("cylindrical", points="gravity_center"), centers,
    )
    np.testing.assert_array_equal(
        voxel.to_coordinates("cylindrical", points="grid"), vertices,
    )


def test_cylindrical_normalization_preserves_existing_scale_definition():
    voxel = _voxel()
    points = np.array([[3.0, 4.0, 5.0]])

    actual = voxel.to_coordinates(
        "cylindrical", points=points, normalized=True,
        radius=10.0, height=20.0,
    )

    np.testing.assert_allclose(actual, [[0.5, np.arctan2(4.0, 3.0), 0.5]])


@pytest.mark.parametrize(
    ("coordinate_type", "parameters", "missing"),
    [
        ("cartesian", {"width": 2.0, "depth": 3.0}, "height"),
        ("cylindrical", {"radius": 2.0}, "height"),
        ("spherical", {}, "radius"),
        ("torus", {"major_radius": 3.0}, "minor_radius"),
        ("torus_inverse", {"major_radius": 3.0}, "minor_radius"),
    ],
)
def test_normalized_forward_conversion_requires_every_scale(
        coordinate_type, parameters, missing):
    voxel = _voxel()

    with pytest.raises(ValueError, match=missing):
        voxel.to_coordinates(
            coordinate_type, points=[[1.0, 0.0, 0.0]],
            normalized=True, **parameters,
        )


def test_from_coordinates_broadcasts_cylindrical_keyword_components():
    voxel = _voxel()
    R = np.array([1.0, 2.0, 3.0])
    Z = np.array([[-1.0], [1.0]])
    phi = np.pi / 2

    xyz = voxel.from_coordinates("cylindrical", R=R, Z=Z, phi=phi)

    assert xyz.shape == (2, 3, 3)
    np.testing.assert_allclose(xyz[..., 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(xyz[..., 1], np.broadcast_to(R, (2, 3)))
    np.testing.assert_allclose(xyz[..., 2], np.broadcast_to(Z, (2, 3)))


def test_from_coordinates_missing_component_error_lists_complete_signature():
    voxel = _voxel()

    with pytest.raises(
        ValueError,
        match="torus_inverse coordinates require components rho, theta, phi; missing rho",
    ):
        voxel.from_coordinates(
            "torus_inverse", theta=0.0, phi=0.0, major_radius=1.5,
        )


def test_from_coordinates_unsupported_error_lists_available_types():
    voxel = _voxel()

    with pytest.raises(ValueError, match="choose one of .*torus_inverse.*cylindrical"):
        voxel.from_coordinates("toroidal_inverse", rho=1.0, theta=0.0, phi=0.0)


def test_normalized_inverse_cylindrical_requires_scales():
    voxel = _voxel()

    with pytest.raises(ValueError, match="radius"):
        voxel.from_coordinates(
            "cylindrical", R=1.0, Z=0.0, phi=0.0,
            normalized=True, height=2.0,
        )
    with pytest.raises(ValueError, match="height"):
        voxel.from_coordinates(
            "cylindrical", R=1.0, Z=0.0, phi=0.0,
            normalized=True, radius=2.0,
        )


def test_coordinate_api_adds_no_serialized_voxel_state_and_dill_roundtrips():
    voxel = Voxel.uniform_voxel(
        ranges=((-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0)),
        shape=(2, 2, 2),
        coordinate_type="torus_inverse",
        coordinate_parameters={"R_0": 1.5, "a": 0.5},
    )
    state_keys = set(voxel.__dict__)

    loaded = dill.loads(dill.dumps(voxel))

    assert set(loaded.__dict__) == state_keys
    np.testing.assert_allclose(
        loaded.normalized_coordinates(), voxel.normalized_coordinates(),
    )
    np.testing.assert_allclose(
        loaded.to_coordinates("cylindrical"),
        voxel.to_coordinates("cylindrical"),
    )


@pytest.mark.parametrize("coordinate_type", ["torus", "torus_inverse"])
@pytest.mark.parametrize("normalized", [False, True])
def test_torus_roundtrip_for_both_angle_conventions(coordinate_type, normalized):
    voxel = _voxel()
    points = np.array([
        [4.2, 1.1, 0.7],
        [-3.8, 0.6, -0.4],
        [0.5, -4.1, 1.3],
    ])
    parameters = {"major_radius": 3.0}
    if normalized:
        parameters["minor_radius"] = 1.7

    coordinates = voxel.to_coordinates(
        coordinate_type, points=points, normalized=normalized, **parameters,
    )
    reconstructed = voxel.from_coordinates(
        coordinate_type,
        rho=coordinates[:, 0], theta=coordinates[:, 1], phi=coordinates[:, 2],
        normalized=normalized, **parameters,
    )

    np.testing.assert_allclose(reconstructed, points, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("normalized", [False, True])
def test_spherical_roundtrip(normalized):
    voxel = _voxel()
    points = np.array([[1.2, -0.8, 2.1], [-2.0, 1.5, -0.7]])
    parameters = {"radius": 4.0} if normalized else {}

    coordinates = voxel.to_coordinates(
        "spherical", points=points, normalized=normalized, **parameters,
    )
    reconstructed = voxel.from_coordinates(
        "spherical", r=coordinates[:, 0], theta=coordinates[:, 1],
        phi=coordinates[:, 2],
        normalized=normalized, **parameters,
    )

    np.testing.assert_allclose(reconstructed, points, rtol=1e-14, atol=1e-14)


def test_coordinate_roundtrip_respects_voxel_rotation():
    rotation = Rotation.from_euler("xyz", [20.0, -15.0, 35.0], degrees=True)
    voxel = _voxel(rotation=rotation)
    points = np.array([[2.0, 1.0, 0.5], [-1.5, 2.2, -0.3]])

    coordinates = voxel.to_coordinates(
        "cylindrical", points=points, normalized=True,
        radius=3.0, height=4.0,
    )
    reconstructed = voxel.from_coordinates(
        "cylindrical", R=coordinates[:, 0], phi=coordinates[:, 1],
        Z=coordinates[:, 2],
        normalized=True, radius=3.0, height=4.0,
    )

    np.testing.assert_allclose(reconstructed, points, rtol=1e-14, atol=1e-14)


def test_arbitrary_point_leading_dimensions_are_preserved():
    voxel = _voxel()
    points = np.arange(24.0).reshape(2, 4, 3) + 1.0

    coordinates = voxel.to_coordinates("cylindrical", points=points)

    assert coordinates.shape == (2, 4, 3)
