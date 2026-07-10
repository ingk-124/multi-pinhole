import numpy as np

from multi_pinhole import Aperture, Camera, Eye, Screen, Voxel, World


def make_camera():
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=0.5)
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(4, 4), subpixel_resolution=20)
    aperture = Aperture(shape="circle", size=1.0, position=(0.0, 0.0, 5.0))
    return Camera(eyes=[eye], apertures=aperture, screen=screen, camera_position=(0.0, 0.0, -10.0))


def test_eye_screen_voxel_world_initialization():
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=0.5)
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(4, 4), subpixel_resolution=20)
    voxel = Voxel(
        x_axis=np.linspace(-1.0, 1.0, 3),
        y_axis=np.linspace(-1.0, 1.0, 3),
        z_axis=np.linspace(-1.0, 1.0, 3),
    )
    world = World(voxel=voxel, cameras=None, verbose=0)

    assert eye.eye_type == "pinhole"
    assert tuple(screen.pixel_shape) == (4, 4)
    assert voxel.N_voxel == 8
    assert world.voxel is voxel


def test_camera_initialization_preserves_public_api():
    camera = make_camera()

    assert isinstance(camera, Camera)
    assert len(camera.eyes) == 1
    assert isinstance(camera.screen, Screen)


def test_eye_calc_rays_projects_front_points_and_masks_back_points():
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=0.5)
    points = np.array([
        [0.0, 0.0, 20.0],
        [1.0, 0.0, 20.0],
        [0.0, 0.0, -1.0],
    ])

    rays = eye.calc_rays(points)

    np.testing.assert_allclose(rays.XY[:2], np.array([[0.0, 0.0], [-0.5, 0.0]]))
    np.testing.assert_allclose(rays.zoom_rate[:2], np.array([1.5, 1.5]))
    assert rays.front_and_visible.tolist() == [True, True, False]
    assert np.isnan(rays.XY[2]).all()


def test_readme_minimal_sample_runs():
    camera = make_camera()
    points = np.array([[0.0, 0.0, 20.0], [1.0, 0.0, 20.0]])
    rays = camera.eyes[0].calc_rays(points)
    voxel = Voxel(
        x_axis=np.linspace(-1.0, 1.0, 3),
        y_axis=np.linspace(-1.0, 1.0, 3),
        z_axis=np.linspace(-1.0, 1.0, 3),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)

    assert rays.XY.shape == (2, 2)
    assert world.cameras[0] is camera
