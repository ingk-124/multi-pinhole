"""Numerical and compatibility boundaries required for the 0.8.0 release."""

import numpy as np
from scipy import sparse

from multi_pinhole import Camera, Eye, Screen, Voxel, World


def _single_pixel_world(z_range):
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=1.0)
    screen = Screen(
        screen_shape="square", screen_size=100.0,
        pixel_shape=(1, 1), subpixel_resolution=1,
    )
    camera = Camera(
        eyes=[eye], apertures=[], screen=screen,
        camera_position=(0.0, 0.0, 0.0),
    )
    voxel = Voxel.uniform_voxel(
        ranges=((-0.5, 0.5), (-0.5, 0.5), z_range),
        shape=(1, 1, 1),
    )
    world = World(voxel=voxel, cameras={"main": camera}, verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    return world


def test_generated_single_pixel_projection_is_csr_and_adjoint():
    world = _single_pixel_world((20.0, 21.0))
    world.set_projection_matrix(res=1, parallel=1, verbose=0)
    matrix = world.projection["main"][0]

    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (1, 1)
    assert matrix.dtype == np.dtype(float)
    assert matrix[0, 0] > 0.0
    emission = np.array([1.25])
    image = np.array([-0.75])
    np.testing.assert_allclose(
        np.dot(world.project(emission, "main"), image),
        np.dot(emission, world.backproject(image, "main")),
        rtol=0.0, atol=0.0,
    )


def test_fully_invisible_voxel_produces_zero_csr_projection():
    world = _single_pixel_world((-2.0, -1.0))
    world.set_projection_matrix(res=1, parallel=1, verbose=0)
    matrix = world.projection["main"][0]

    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (1, 1)
    assert matrix.dtype == np.dtype(float)
    assert matrix.nnz == 0
    np.testing.assert_array_equal(world.visible_voxels["main"], [[0]])


def test_anisotropic_source_quadrature_weights_sum_to_voxel_volume():
    voxel = Voxel.uniform_voxel(
        ranges=((0.0, 2.0), (0.0, 3.0), (0.0, 5.0)),
        shape=(1, 1, 1),
    )
    points = voxel.get_sub_voxel_centers(n=np.array([0]), res=(2, 3, 4))
    weights = voxel._build_source_quadrature_matrix(
        np.array([0]), res=(2, 3, 4), points=points,
    )

    assert sparse.isspmatrix_csr(weights)
    assert weights.shape == (24, 1)
    np.testing.assert_allclose(
        np.asarray(weights.sum(axis=0)).ravel(),
        voxel.volume,
        rtol=0.0, atol=1e-14,
    )
