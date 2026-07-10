import contextlib
import io

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from multi_pinhole import Aperture, Camera, Eye, Screen, Voxel, World, emission_profile
from multi_pinhole.utils import stl_utils


def make_double_pinhole_aperture(size=0.8, offset=1.2, max_size=6.0, resolution=16):
    x_arr = np.linspace(-max_size, max_size, 4)
    y_arr = np.linspace(-max_size, max_size, 4)
    outer_points = np.array(np.meshgrid(x_arr, y_arr, indexing="ij")).reshape(2, -1).T
    t = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    edge_points1 = np.array([size * np.cos(t) - offset, size * np.sin(t)]).T
    edge_points2 = np.array([size * np.cos(t) + offset, size * np.sin(t)]).T
    points = np.vstack([outer_points, edge_points1, edge_points2])

    def condition(x, y):
        c1 = ((x + offset) / size) ** 2 + (y / size) ** 2 <= 1
        c2 = ((x - offset) / size) ** 2 + (y / size) ** 2 <= 1
        return c1 | c2

    return stl_utils.make_2D_surface(points, condition)


def make_mst_like_world():
    aperture_offset = 1.2
    aperture_model = make_double_pinhole_aperture(offset=aperture_offset)
    camera = Camera(
        eyes=[
            Eye(position=(-aperture_offset, 0.0), focal_length=5.0, eye_size=0.5),
            Eye(position=(aperture_offset, 0.0), focal_length=5.0, eye_size=0.5),
        ],
        screen=Screen(screen_shape="rectangle", screen_size=[8.0, 8.0], pixel_shape=(4, 4), subpixel_resolution=5),
        apertures=[Aperture(stl_model=aperture_model, position=[0.0, 0.0, 3.0])],
        camera_position=[0.0, 0.0, -15.0],
    )
    voxel = Voxel.uniform_voxel(
        ranges=[[-2.0, 2.0], [-2.0, 2.0], [-1.0, 1.0]],
        shape=[2, 2, 2],
        coordinate_type="torus_inverse",
        coordinate_parameters=dict(a=1.0, R_0=1.5),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    return world


def test_standard_torus_normalized_coordinates_are_right_handed_by_convention():
    voxel = Voxel.uniform_voxel(
        ranges=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        shape=[1, 1, 1],
        coordinate_type="torus",
        coordinate_parameters=dict(a=1.0, R_0=1.5),
    )
    points = np.array(
        [
            [2.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 1.0],
            [0.0, 1.5, 0.0],
        ]
    )

    normalized = voxel.normalized_coordinates(points)

    np.testing.assert_allclose(normalized[:, 0], np.array([1.0, 1.0, 1.0, 0.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(normalized[:, 1], np.array([0.0, np.pi, np.pi / 2, 0.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(normalized[:, 2], np.array([0.0, 0.0, 0.0, -np.pi / 2]), rtol=1e-12, atol=1e-12)


def test_inverse_torus_normalized_coordinates_reverse_theta_and_phi():
    voxel = Voxel.uniform_voxel(
        ranges=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        shape=[1, 1, 1],
        coordinate_type="torus_inverse",
        coordinate_parameters=dict(a=1.0, R_0=1.5),
    )
    points = np.array(
        [
            [2.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 1.0],
            [0.0, 1.5, 0.0],
        ]
    )

    normalized = voxel.normalized_coordinates(points)

    np.testing.assert_allclose(normalized[:, 0], np.array([1.0, 1.0, 1.0, 0.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(normalized[:, 1], np.array([np.pi, 0.0, np.pi / 2, 0.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(normalized[:, 2], np.array([0.0, 0.0, 0.0, np.pi / 2]), rtol=1e-12, atol=1e-12)


def test_mst_like_double_pinhole_projection_workflow_runs():
    with contextlib.redirect_stdout(io.StringIO()):
        world = make_mst_like_world()
        world.set_projection_matrix(res=1, verbose=0, parallel=1)

    camera = world.cameras[0]
    voxel = world.voxel
    projection = world.P_matrix[0]

    assert len(camera.eyes) == 2
    assert projection.shape == (camera.screen.N_pixel, voxel.N_voxel)
    assert projection.nnz > 0
    assert np.all(projection.tocsc().getnnz(axis=0) > 0)

    r, theta, phi = voxel.normalized_coordinates().T
    f = emission_profile(r, theta, phi)
    image = projection @ f

    assert image.shape == (camera.screen.N_pixel,)
    assert np.isfinite(image).all()
    assert np.any(image > 0)


def test_mst_like_trace_line_and_show_image_paths_are_finite():
    with contextlib.redirect_stdout(io.StringIO()):
        world = make_mst_like_world()
        world.set_projection_matrix(res=1, verbose=0, parallel=1)

    camera = world.cameras[0]
    toroidal_axis = np.array(
        [
            [1.5, 0.0, 0.0],
            [1.25, 0.25, 0.0],
            [1.0, 0.5, 0.0],
        ]
    )
    uv = world.trace_line(toroidal_axis, camera_idx=0, eye_idx=0, coord_type="UV")
    assert uv.shape == (3, 2)
    assert np.isfinite(uv).all()

    r, theta, phi = world.voxel.normalized_coordinates().T
    image = world.P_matrix[0] @ emission_profile(r, theta, phi)
    fig, ax = plt.subplots()
    try:
        returned_ax = camera.screen.show_image(image, ax=ax, colorbar=False, show=False)
        assert returned_ax is ax
    finally:
        plt.close(fig)
