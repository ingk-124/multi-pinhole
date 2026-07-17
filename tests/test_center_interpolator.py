import numpy as np
import pytest

from multi_pinhole import Voxel


def _voxel():
    return Voxel.uniform_voxel(
        ranges=((0.0, 4.0), (-2.0, 2.0), (10.0, 14.0)),
        shape=(4, 4, 4),
    )


def _affine(points):
    points = np.asarray(points)
    return 2.0 + 0.5 * points[..., 0] - points[..., 1] + 3.0 * points[..., 2]


def test_center_interpolator_reproduces_affine_field_at_cartesian_points():
    voxel = _voxel()
    values = _affine(voxel.gravity_center)
    interpolator = voxel.center_interpolator(values)
    points = np.array([[1.25, -0.25, 11.75], [2.8, 0.6, 12.2]])

    np.testing.assert_allclose(
        interpolator(points=points), _affine(points), rtol=1e-14, atol=1e-14,
    )


def test_center_interpolator_accepts_grid_shaped_and_vector_values():
    voxel = _voxel()
    scalar = _affine(voxel.gravity_center).reshape(voxel.voxel_shape)
    values = np.stack([scalar, -scalar], axis=-1)
    interpolator = voxel.center_interpolator(values)
    point = np.array([2.0, 0.0, 12.0])

    actual = interpolator(points=point)

    assert actual.shape == (2,)
    np.testing.assert_allclose(actual, [_affine(point), -_affine(point)])


def test_center_interpolator_converts_broadcast_cylindrical_queries():
    voxel = _voxel()
    values = _affine(voxel.gravity_center)
    interpolator = voxel.center_interpolator(values)
    R = np.array([1.5, 2.0])
    phi = np.array([[0.0], [np.pi / 6]])
    Z = 12.0

    actual = interpolator(
        coordinate_type="cylindrical", R=R, phi=phi, Z=Z,
    )
    xyz = voxel.from_coordinates("cylindrical", R=R, phi=phi, Z=Z)

    assert actual.shape == (2, 2)
    np.testing.assert_allclose(actual, _affine(xyz), rtol=1e-14, atol=1e-14)


def test_center_interpolator_accepts_normalized_coordinate_queries():
    voxel = _voxel()
    values = _affine(voxel.gravity_center)
    interpolator = voxel.center_interpolator(values)

    actual = interpolator(
        coordinate_type="cylindrical",
        R=0.5, phi=0.0, Z=1.0,
        normalized=True, radius=4.0, height=24.0,
    )

    np.testing.assert_allclose(actual, _affine([2.0, 0.0, 12.0]))


def test_center_interpolator_forwards_boundary_options():
    voxel = _voxel()
    interpolator = voxel.center_interpolator(
        _affine(voxel.gravity_center), bounds_error=False, fill_value=-99.0,
    )

    assert interpolator(points=[[100.0, 0.0, 12.0]])[0] == -99.0


def test_center_interpolator_rejects_ambiguous_queries():
    voxel = _voxel()
    interpolator = voxel.center_interpolator(_affine(voxel.gravity_center))

    with pytest.raises(ValueError, match="points are required"):
        interpolator()
    with pytest.raises(ValueError, match="mutually exclusive"):
        interpolator(
            points=[[1.0, 0.0, 12.0]],
            coordinate_type="cylindrical", R=1.0, phi=0.0, Z=12.0,
        )
    with pytest.raises(ValueError, match="explicit coordinate_type"):
        interpolator(points=[[1.0, 0.0, 12.0]], R=1.0)


def test_legacy_source_interpolator_is_compatibility_wrapper():
    voxel = _voxel()
    indices = np.array([0, 1])

    legacy = voxel.sub_voxel_interpolator_from_centers(indices, res=2)
    internal = voxel._build_source_quadrature_matrix(indices, res=2)

    np.testing.assert_array_equal(legacy.indptr, internal.indptr)
    np.testing.assert_array_equal(legacy.indices, internal.indices)
    np.testing.assert_allclose(legacy.data, internal.data)
