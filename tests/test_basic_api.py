import contextlib
import io
import importlib.util
from pathlib import Path

import numpy as np

from multi_pinhole import Aperture, Camera, Eye, Rays, Screen, Voxel, World


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

    # Pinhole coordinates place the eye at z=focal_length, so the second point
    # projects with denominator (20 - 10), not the original camera-coordinate z.
    np.testing.assert_allclose(rays.XY[:2], np.array([[0.0, 0.0], [-1.0, 0.0]]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rays.zoom_rate[:2], np.array([2.0, 2.0]), rtol=1e-12, atol=1e-12)
    assert rays.front_and_visible.tolist() == [True, True, False]
    assert np.isnan(rays.XY[2]).all()


def test_public_imports_are_available():
    import multi_pinhole
    from multi_pinhole import Eye, Screen, Aperture, Camera, Voxel, World

    assert multi_pinhole.Eye is Eye
    assert multi_pinhole.Screen is Screen
    assert multi_pinhole.Aperture is Aperture
    assert multi_pinhole.Camera is Camera
    assert multi_pinhole.Voxel is Voxel
    assert multi_pinhole.World is World


def test_screen_print_settings_does_not_require_color_image_attribute():
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(4, 4), subpixel_resolution=20)

    with contextlib.redirect_stdout(io.StringIO()):
        screen.print_settings()


def test_ray2image_grid_keeps_csc_columns_monotonic_when_invalid_ray_is_in_middle():
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=0.5)
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(4, 4), subpixel_resolution=20)
    rays = Rays(
        Z=np.array([10.0, -1.0, 10.0]),
        XY=np.array([[0.0, 0.0], [np.nan, np.nan], [1.0, 0.0]]),
        zoom_rate=np.array([1.5, np.nan, 1.5]),
        front_and_visible=np.array([True, False, True]),
    )

    mat = screen.ray2image_grid(eye, rays).tocsc()

    assert mat.shape == (screen.N_subpixel, 3)
    assert mat[:, 0].nnz > 0
    assert mat[:, 1].nnz == 0
    assert mat[:, 2].nnz > 0
    assert np.all(np.diff(mat.indptr) >= 0)


def test_ray2image_grid_matches_point_source_pinhole_solid_angle():
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=1.0)
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(600, 600), subpixel_resolution=1)
    points = np.array([
        [0.0, 0.0, 30.0],
        [5.0, 0.0, 30.0],
    ])
    rays = eye.calc_rays(points)

    mat = screen.ray2image_grid(eye, rays).tocsc()
    total_etendue = np.asarray(mat.sum(axis=0)).ravel()

    z = points[:, 2] - eye.position[2]
    rho = np.linalg.norm(points[:, :2] - eye.position[:2], axis=1)
    cos_theta = z / np.sqrt(z ** 2 + rho ** 2)
    pinhole_area = np.pi * (eye.eye_size[0] / 2) ** 2
    expected = pinhole_area * cos_theta ** 3 / (4 * np.pi * z ** 2)

    np.testing.assert_allclose(total_etendue, expected, rtol=3e-2, atol=0.0)


def test_etendue_x_scan_example_matches_analytic_curve(tmp_path):
    example_path = Path(__file__).resolve().parents[1] / "examples" / "verify_etendue_x_scan.py"
    spec = importlib.util.spec_from_file_location("verify_etendue_x_scan", example_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.run(output_dir=tmp_path, n_points=21, pixel_shape=(120, 120), subpixel_resolution=4,
                        axial_distance=100.0, x_extent=30.0)

    np.testing.assert_allclose(result["numerical"], result["analytic"], rtol=4e-2, atol=0.0)
    np.testing.assert_allclose(result["numerical_screen"], result["analytic_screen"], rtol=1e-6, atol=1e-18)
    assert np.max(np.abs(result["relative_error"])) < 0.04
    assert abs(result["spot_relative_error"]) < 0.04
    assert abs(result["screen_relative_error"]) < 0.04
    assert result["P_sources"].shape == (result["numerical_screen"].size, result["source_strength"].size)
    assert result["output_path"].is_file()
    assert result["spot_output_path"].is_file()
    assert result["screen_output_path"].is_file()
    assert result["profile_output_path"].is_file()


def test_multiple_apertures_block_if_any_aperture_intersects(monkeypatch):
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=0.5)
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(4, 4), subpixel_resolution=20)
    apertures = [
        Aperture(shape="circle", size=1.0, position=(0.0, 0.0, 5.0)),
        Aperture(shape="circle", size=1.0, position=(1.0, 0.0, 5.0)),
    ]
    camera = Camera(eyes=[eye], apertures=apertures, screen=screen, camera_position=(0.0, 0.0, 0.0))
    points = np.array([
        [0.0, 0.0, 20.0],
        [1.0, 0.0, 20.0],
        [2.0, 0.0, 20.0],
    ])

    visibility_by_aperture = [
        np.array([True, False, True]),
        np.array([True, True, False]),
    ]

    def fake_check_visible(*args, **kwargs):
        return visibility_by_aperture.pop(0)

    monkeypatch.setattr("multi_pinhole.core.stl_utils.check_visible", fake_check_visible)

    mat = camera.calc_image_vec(0, points, check_visibility=True).tocsc()

    assert mat[:, 0].nnz > 0
    assert mat[:, 1].nnz == 0
    assert mat[:, 2].nnz == 0


def test_world_multiple_apertures_match_camera_blocking_rule(monkeypatch):
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=0.5)
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(4, 4), subpixel_resolution=20)
    apertures = [
        Aperture(shape="circle", size=1.0, position=(0.0, 0.0, 5.0)),
        Aperture(shape="circle", size=1.0, position=(1.0, 0.0, 5.0)),
    ]
    camera = Camera(eyes=[eye], apertures=apertures, screen=screen, camera_position=(0.0, 0.0, 0.0))
    voxel = Voxel(
        x_axis=np.linspace(-1.0, 1.0, 2),
        y_axis=np.linspace(-1.0, 1.0, 2),
        z_axis=np.linspace(-1.0, 1.0, 2),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    points = np.array([
        [0.0, 0.0, 20.0],
        [1.0, 0.0, 20.0],
        [2.0, 0.0, 20.0],
    ])
    visibility_by_aperture = [
        np.array([True, False, True]),
        np.array([True, True, False]),
    ]

    def fake_check_visible(*args, **kwargs):
        return visibility_by_aperture.pop(0)

    monkeypatch.setattr("multi_pinhole.world.stl_utils.check_visible", fake_check_visible)

    visible = world.find_visible_points(points, camera_idx=0, eye_idx=0, verbose=0)

    np.testing.assert_array_equal(visible, np.array([[True, False, False]]))


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


def test_small_projection_matrix_case_completes_and_preserves_sparse_columns():
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=0.5)
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(4, 4), subpixel_resolution=20)
    aperture = Aperture(shape="circle", size=50.0, position=(0.0, 0.0, 5.0))
    camera = Camera(eyes=[eye], apertures=aperture, screen=screen, camera_position=(0.0, 0.0, -20.0))
    voxel = Voxel(
        x_axis=np.linspace(-1.0, 1.0, 2),
        y_axis=np.linspace(-1.0, 1.0, 2),
        z_axis=np.linspace(-1.0, 1.0, 2),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)

    world.set_projection_matrix(res=1, verbose=0, parallel=1)

    assert world.projection[0][0].shape == (screen.N_subpixel, voxel.N_voxel)
    assert world.P_matrix[0].shape == (screen.N_pixel, voxel.N_voxel)
    assert world.projection[0][0].tocsc()[:, 0].nnz > 0


def test_projection_matrix_is_stable_when_subvoxel_resolution_changes():
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=1.0)
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(80, 80), subpixel_resolution=1)
    aperture = Aperture(shape="circle", size=50.0, position=(0.0, 0.0, 5.0))
    camera = Camera(eyes=[eye], apertures=aperture, screen=screen, camera_position=(0.0, 0.0, -20.0))
    voxel = Voxel(
        x_axis=np.array([-0.1, 0.1]),
        y_axis=np.array([-0.1, 0.1]),
        z_axis=np.array([-0.1, 0.1]),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))

    totals = []
    for res in [1, 2, 3]:
        world.set_projection_matrix(res=res, verbose=0, parallel=1, force=True)
        totals.append(float(world.P_matrix[0][:, 0].sum()))

    np.testing.assert_allclose(totals, np.full(3, totals[0]), rtol=5e-2, atol=0.0)


def test_partial_voxel_inside_mask_scales_integrated_light():
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=1.0)
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(80, 80), subpixel_resolution=1)
    aperture = Aperture(shape="circle", size=50.0, position=(0.0, 0.0, 5.0))
    camera = Camera(eyes=[eye], apertures=aperture, screen=screen, camera_position=(0.0, 0.0, 0.0))

    def make_world():
        voxel = Voxel(
            x_axis=np.array([-0.1, 0.1]),
            y_axis=np.array([-0.1, 0.1]),
            z_axis=np.array([30.0, 30.2]),
        )
        return World(voxel=voxel, cameras=[camera], verbose=0)

    full_world = make_world()
    full_world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))
    full_world.set_projection_matrix(res=2, partial_res=2, verbose=0, parallel=1)
    full_total = float(full_world.P_matrix[0][:, 0].sum())

    half_world = make_world()
    half_world.set_inside_vertices(lambda x, y, z: x >= 0.0)
    half_world.set_projection_matrix(res=2, partial_res=2, verbose=0, parallel=1)
    half_total = float(half_world.P_matrix[0][:, 0].sum())

    assert full_world.visible_voxels[0][0, 0] == 2
    assert half_world.visible_voxels[0][0, 0] == 1
    np.testing.assert_allclose(half_total / full_total, 0.5, rtol=2e-2, atol=0.0)


def test_partial_res_refines_only_partial_voxel_quadrature():
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=1.0)
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(80, 80), subpixel_resolution=1)
    aperture = Aperture(shape="circle", size=50.0, position=(0.0, 0.0, 5.0))
    camera = Camera(eyes=[eye], apertures=aperture, screen=screen, camera_position=(0.0, 0.0, 0.0))
    voxel = Voxel(
        x_axis=np.array([-0.1, 0.1]),
        y_axis=np.array([-0.1, 0.1]),
        z_axis=np.array([30.0, 30.2]),
    )
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: x >= 0.0)

    totals = []
    for partial_res in [2, 4]:
        world.set_projection_matrix(res=1, partial_res=partial_res, verbose=0, parallel=1, force=True)
        totals.append(float(world.P_matrix[0][:, 0].sum()))

    assert world.projection[0][0].shape == (screen.N_subpixel, voxel.N_voxel)
    assert world.P_matrix[0].shape == (screen.N_pixel, voxel.N_voxel)
    np.testing.assert_allclose(totals[1], totals[0], rtol=3e-2, atol=0.0)


def test_small_voxel_projection_example_draws_outputs(tmp_path):
    example_path = Path(__file__).resolve().parents[1] / "examples" / "small_voxel_projection.py"
    spec = importlib.util.spec_from_file_location("small_voxel_projection", example_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.run(output_dir=tmp_path)
    world = result["world"]
    camera = world.cameras[0]
    voxel = world.voxel
    projection = world.P_matrix[0].tocsc()
    subpixel_projection = world.projection[0][0].tocsc()

    assert projection.shape == (camera.screen.N_pixel, voxel.N_voxel)
    assert subpixel_projection.shape == (camera.screen.N_subpixel, voxel.N_voxel)
    assert projection.nnz > 0
    assert subpixel_projection.nnz > 0
    assert result["pixel_image"].shape == (camera.screen.N_pixel,)
    assert result["subpixel_image"].shape == (camera.screen.N_subpixel,)
    assert np.isfinite(result["pixel_image"]).all()
    assert np.isfinite(result["subpixel_image"]).all()
    assert np.any(result["pixel_image"] > 0)
    assert result["geometry_path"].is_file()
    assert result["projection_path"].is_file()
