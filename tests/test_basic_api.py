import contextlib
import io
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

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


def test_uniform_voxel_from_centers_preserves_requested_gravity_center_ranges():
    cx = np.linspace(-2.0, 2.0, 5)
    cy = np.linspace(10.0, 14.0, 3)
    cz = np.linspace(-1.5, 1.5, 7)

    voxel = Voxel.uniform_voxel_from_centers(
        ranges=[[-2.0, 2.0], [10.0, 14.0], [-1.5, 1.5]],
        shape=[5, 3, 7],
    )

    np.testing.assert_allclose(voxel.cx_axis, cx)
    np.testing.assert_allclose(voxel.cy_axis, cy)
    np.testing.assert_allclose(voxel.cz_axis, cz)
    assert voxel.shape == (5, 3, 7)
    np.testing.assert_allclose(voxel.ranges, [(-2.5, 2.5), (9.0, 15.0), (-1.75, 1.75)])


def test_uniform_voxel_from_centers_rejects_single_center_axis():
    with pytest.raises(ValueError, match="at least two"):
        Voxel.uniform_voxel_from_centers(
            ranges=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            shape=[1, 2, 2],
        )


def test_camera_initialization_preserves_public_api():
    camera = make_camera()

    assert isinstance(camera, Camera)
    assert len(camera.eyes) == 1
    assert isinstance(camera.screen, Screen)


def test_single_pinhole_factory_builds_camera_in_reference_pose():
    aperture = Aperture(shape="circle", size=1.0, position=(0.0, 0.0, 5.0))

    camera = Camera.single_pinhole(
        focal_length=25.0,
        eye_size=1.0,
        screen_size=61 * 0.13,
        pixel_shape=(61, 61),
        subpixel_resolution=5,
        apertures=aperture,
    )

    assert len(camera.eyes) == 1
    np.testing.assert_allclose(camera.eyes[0].position, np.array([0.0, 0.0, 25.0]))
    np.testing.assert_array_equal(camera.screen.pixel_shape, np.array([61, 61]))
    np.testing.assert_allclose(camera.camera_position, np.zeros(3))
    np.testing.assert_allclose(camera.rotation_matrix, np.eye(3))


def test_camera_absolute_pose_and_relative_translation_methods():
    camera = make_camera()
    rotation = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()

    returned = camera.set_camera_position([1.0, 2.0, 3.0]).set_rotation_matrix(rotation)
    assert returned is camera

    camera.translate_world([4.0, 5.0, 6.0])
    np.testing.assert_allclose(camera.camera_position, np.array([5.0, 7.0, 9.0]))

    camera.translate_camera([2.0, 0.0, 0.0])
    np.testing.assert_allclose(
        camera.camera_position,
        np.array([5.0, 7.0, 9.0]) + rotation.T @ np.array([2.0, 0.0, 0.0]),
        atol=1e-12,
    )


def test_set_orientation_from_look_and_right_builds_world_to_camera_matrix():
    camera = make_camera().set_orientation(look=[0.0, 1.0, 0.0], right=[1.0, 0.0, 0.0])

    np.testing.assert_allclose(camera.camera_x, np.array([1.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(camera.camera_y, np.array([0.0, 0.0, -1.0]), atol=1e-12)
    np.testing.assert_allclose(camera.camera_z, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_set_orientation_from_look_and_down_builds_same_frame():
    camera = make_camera().set_orientation(look=[0.0, 1.0, 0.0], down=[0.0, 0.0, -1.0])

    np.testing.assert_allclose(camera.rotation_matrix, np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]), atol=1e-12)


def test_set_orientation_rejects_parallel_axes():
    with pytest.raises(ValueError, match="right must be a non-zero"):
        make_camera().set_orientation(look=[0.0, 0.0, 1.0], right=[0.0, 0.0, 2.0])


def test_set_orientation_from_world_points_uses_camera_position_as_origin():
    camera = make_camera().set_camera_position([10.0, 20.0, 30.0])

    returned = camera.set_orientation_from_points(
        look_point=[10.0, 25.0, 30.0],
        right_point=[14.0, 22.0, 30.0],
    )

    assert returned is camera
    np.testing.assert_allclose(camera.camera_x, np.array([1.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(camera.camera_y, np.array([0.0, 0.0, -1.0]), atol=1e-12)
    np.testing.assert_allclose(camera.camera_z, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_set_orientation_from_points_requires_position_to_be_set_first():
    camera = make_camera().set_camera_position([10.0, 20.0, 30.0])

    with pytest.raises(ValueError, match="look must be a non-zero"):
        camera.set_orientation_from_points(
            look_point=camera.camera_position,
            right_point=[11.0, 20.0, 30.0],
        )


def test_set_rotation_matrix_rejects_non_rotation_matrix():
    with pytest.raises(ValueError, match="orthonormal"):
        make_camera().set_rotation_matrix(np.diag([1.0, 1.0, 2.0]))


def test_world_registration_freezes_camera_and_child_geometry():
    camera = make_camera()
    world = World(cameras=[camera], verbose=0)

    assert camera.world is world
    assert camera.frozen
    assert camera.eyes[0].frozen
    assert camera.screen.frozen
    assert camera.apertures[0].frozen
    assert isinstance(camera.eyes, tuple)
    assert isinstance(camera.apertures, tuple)

    with pytest.raises(RuntimeError, match="Camera geometry is frozen"):
        camera.translate_world([1.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="Screen geometry is frozen"):
        camera.screen.subpixel_resolution = 2
    with pytest.raises(RuntimeError, match="Aperture geometry is frozen"):
        camera.apertures[0].set_model()
    with pytest.raises(ValueError, match="read-only"):
        camera.camera_position[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        camera.eyes[0].position[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        camera.screen.pixel_position[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        camera.screen.transform_matrix.data[0] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        camera.apertures[0].stl_model.data[0] = camera.apertures[0].stl_model.data[0]


def test_world_camera_add_change_and_remove_manage_registration():
    world = World(cameras=None, verbose=0)
    first = make_camera()
    second = make_camera().set_camera_position([1.0, 2.0, 3.0])

    world.add_camera("main", first)
    assert first.world is world
    assert first.frozen
    assert world.cameras["main"] is first

    world.change_camera("main", second)
    assert first.world is None
    assert first.frozen
    assert second.world is world
    assert second.frozen
    assert world.cameras["main"] is second
    assert world.visible_voxels["main"] is None
    assert world.projection["main"] == [None]
    assert world.P_matrix["main"] is None

    world.remove_camera("main")
    assert second.world is None
    assert second.frozen
    assert world.cameras == {}


def test_frozen_camera_can_be_shared_by_multiple_worlds():
    camera = make_camera()
    first_world = World(cameras=[camera], verbose=0)
    second_world = World(cameras=[camera], verbose=0)

    assert camera.frozen
    assert first_world in camera.worlds
    assert second_world in camera.worlds

    first_world.remove_camera(0)
    assert camera.world is second_world
    assert camera.worlds == (second_world,)


def test_world_accepts_keyed_camera_dict_and_preserves_keys_after_removal():
    left = make_camera()
    right = make_camera().set_camera_position([1.0, 0.0, 0.0])
    world = World(cameras={"left": left, "right": right}, verbose=0)

    assert list(world.cameras) == ["left", "right"]
    assert world.cameras["left"] is left
    assert world.cameras["right"] is right

    world.remove_camera("left")

    assert list(world.cameras) == ["right"]
    assert world.cameras["right"] is right
    assert list(world.projection) == ["right"]
    assert list(world.P_matrix) == ["right"]


def test_world_list_camera_keys_are_not_renumbered_after_removal():
    cameras = [make_camera(), make_camera(), make_camera()]
    world = World(cameras=cameras, verbose=0)

    world.remove_camera(1)

    assert list(world.cameras) == [0, 2]
    assert world.cameras[0] is cameras[0]
    assert world.cameras[2] is cameras[2]


def test_add_camera_requires_unique_explicit_key():
    world = World(cameras={"left": make_camera()}, verbose=0)

    with pytest.raises(TypeError):
        world.cameras["right"] = make_camera()

    with pytest.raises(KeyError, match="already registered"):
        world.add_camera("left", make_camera())

    world.add_camera("right", make_camera())
    assert list(world.cameras) == ["left", "right"]


def test_projection_pipeline_supports_string_camera_key():
    camera = make_camera()
    voxel = Voxel(
        x_axis=np.array([-0.1, 0.1]),
        y_axis=np.array([-0.1, 0.1]),
        z_axis=np.array([20.0, 20.2]),
    )
    world = World(voxel=voxel, cameras={"right": camera}, verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))

    world.set_projection_matrix(res=1, verbose=0, parallel=1)

    assert world.projection["right"][0].shape == (camera.screen.N_subpixel, voxel.N_voxel)
    assert world.P_matrix["right"].shape == (camera.screen.N_pixel, voxel.N_voxel)


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


def test_profile_helpers_are_available_from_profiles_module():
    import multi_pinhole
    from multi_pinhole import profiles

    assert not hasattr(multi_pinhole, "emission_profile")
    assert profiles.flattening_profile is multi_pinhole.profiles.flattening_profile
    assert profiles.axisymmetric_profile is multi_pinhole.profiles.axisymmetric_profile


def test_voxel_vertex_coordinate_properties_initialize_vertices_lazily():
    voxel = Voxel(
        x_axis=np.array([0.0, 1.0]),
        y_axis=np.array([2.0, 3.0]),
        z_axis=np.array([4.0, 5.0]),
    )

    assert voxel.vx.shape == (1, 8)
    assert voxel.vy.shape == (1, 8)
    assert voxel.vz.shape == (1, 8)
    np.testing.assert_allclose(voxel.vx, voxel.vertices[..., 0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(voxel.vy, voxel.vertices[..., 1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(voxel.vz, voxel.vertices[..., 2], rtol=0.0, atol=0.0)


def test_screen_print_settings_does_not_require_color_image_attribute():
    screen = Screen(screen_shape="square", screen_size=20.0, pixel_shape=(4, 4), subpixel_resolution=20)

    with contextlib.redirect_stdout(io.StringIO()):
        screen.print_settings()


@pytest.mark.parametrize("eye_shape", ["circle", "rectangle"])
def test_ray2image_grid_keeps_csc_columns_monotonic_when_invalid_ray_is_in_middle(eye_shape):
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=0.5,
              eye_shape=eye_shape)
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


def test_sub_voxel_centers_match_sub_voxel_objects():
    voxel = Voxel(
        x_axis=np.array([-1.0, 0.0, 2.0]),
        y_axis=np.array([-2.0, 1.0, 3.0]),
        z_axis=np.array([10.0, 11.0, 13.0]),
    )
    indices = np.array([0, 3, 5])
    expected = np.concatenate([sv.gravity_center for sv in voxel.get_sub_voxel(n=indices, res=(2, 3, 4))], axis=0)
    actual = voxel.get_sub_voxel_centers(n=indices, res=(2, 3, 4))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_get_sub_voxel_centers_does_not_mutate_voxel_resolution():
    voxel = Voxel.uniform_voxel(ranges=((-1.0, 1.0),) * 3, shape=(2, 2, 2),
                                sub_voxel_resolution=3)

    centers = voxel.get_sub_voxel_centers(n=np.array([0, 1]), res=2)

    assert voxel.res == (3, 3, 3)
    assert centers.shape == (2 * 2 ** 3, 3)


def test_center_sub_voxel_interpolator_preserves_constants_and_is_sparse():
    voxel = Voxel.uniform_voxel(ranges=((-1.0, 1.0),) * 3, shape=(3, 3, 3))
    voxel_indices = np.array([0, 13, 26])
    matrix = voxel.sub_voxel_interpolator_from_centers(n=voxel_indices, res=2)
    expected_scale = np.repeat(voxel.volume[voxel_indices] / 2 ** 3, 2 ** 3)

    np.testing.assert_allclose(matrix @ np.ones(voxel.N), expected_scale,
                               rtol=0.0, atol=1e-15)
    assert matrix.shape == (voxel_indices.size * 2 ** 3, voxel.N)
    assert matrix.getnnz(axis=1).max() <= 8
    assert np.all(matrix.data >= 0.0)


def test_center_sub_voxel_interpolator_reproduces_affine_profile_interior():
    voxel = Voxel.uniform_voxel(ranges=((0.0, 3.0),) * 3, shape=(3, 3, 3))
    center_voxel = np.ravel_multi_index((1, 1, 1), voxel.shape)
    points = voxel.get_sub_voxel_centers(n=np.array([center_voxel]), res=3)
    matrix = voxel.sub_voxel_interpolator_from_centers(
        n=np.array([center_voxel]), res=3, points=points,
    )
    center_profile = 2.0 + 0.5 * voxel.gravity_center[:, 0] \
        - 0.25 * voxel.gravity_center[:, 1] + 0.75 * voxel.gravity_center[:, 2]
    expected = 2.0 + 0.5 * points[:, 0] - 0.25 * points[:, 1] + 0.75 * points[:, 2]
    expected *= voxel.volume[center_voxel] / 3 ** 3

    np.testing.assert_allclose(matrix @ center_profile, expected, rtol=1e-14, atol=1e-14)


def test_parallel_projection_matches_serial_projection():
    eye = Eye(position=(0.0, 0.0), focal_length=10.0, eye_size=2.0)
    screen = Screen(screen_shape="square", screen_size=20.0,
                    pixel_shape=(8, 8), subpixel_resolution=2)
    aperture = Aperture(shape="circle", size=50.0, position=(0.0, 0.0, 5.0))
    camera = Camera(eyes=[eye], apertures=aperture, screen=screen,
                    camera_position=(0.0, 0.0, -20.0))
    voxel = Voxel.uniform_voxel(ranges=((-1.0, 1.0),) * 3, shape=(3, 3, 3))
    world = World(voxel=voxel, cameras=[camera], verbose=0)
    world.set_inside_vertices(lambda x, y, z: np.ones_like(x, dtype=bool))

    world.set_projection_matrix(res=2, verbose=0, parallel=1, force=True)
    expected = world.projection[0][0].copy()
    world.set_projection_matrix(res=2, verbose=0, parallel=4, force=True,
                                max_working_memory=10_000)

    np.testing.assert_allclose(world.projection[0][0].toarray(), expected.toarray(),
                               rtol=1e-12, atol=1e-14)


def test_projection_rejects_nonpositive_working_memory():
    voxel = Voxel.uniform_voxel(ranges=((-1.0, 1.0),) * 3, shape=(2, 2, 2))
    world = World(voxel=voxel, cameras=[make_camera()], verbose=0)

    with pytest.raises(ValueError, match="max_working_memory"):
        world.set_projection_matrix(res=1, verbose=0, parallel=1,
                                    max_working_memory=0)


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
