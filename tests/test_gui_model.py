import numpy as np
import pytest

from multi_pinhole import Aperture, Camera
from multi_pinhole.gui.model import (
    CameraPoseDraft,
    sample_view_boundary,
    single_pinhole_from_form,
    view_boundary_segments,
)


def make_camera():
    return Camera.single_pinhole(
        focal_length=25.0,
        eye_size=1.0,
        screen_size=[8.0, 10.0],
        pixel_shape=(8, 10),
        subpixel_resolution=2,
        screen_shape="rectangle",
        apertures=Aperture(shape="circle", size=1.5, position=[0, 0, 13]),
        camera_name="test",
    ).set_camera_position([10.0, 20.0, 30.0])


def test_camera_pose_draft_builds_new_camera_with_shared_optics():
    source = make_camera()
    draft = CameraPoseDraft.from_camera(source)
    draft.position[:] = [11.0, 22.0, 33.0]
    draft.yaw = 5.0

    updated = draft.build_camera()

    assert updated is not source
    assert updated.screen is source.screen
    assert updated.eyes[0] is source.eyes[0]
    assert updated.apertures[0] is source.apertures[0]
    np.testing.assert_allclose(updated.camera_position, [11.0, 22.0, 33.0])
    assert not np.allclose(updated.rotation_matrix, source.rotation_matrix)


def test_camera_pose_draft_can_replace_frozen_world_camera():
    from multi_pinhole import World

    source = make_camera()
    world = World(cameras={"main": source}, verbose=0)
    draft = CameraPoseDraft.from_camera(source)
    draft.position[0] += 2.0

    replacement = draft.build_camera()
    world.change_camera("main", replacement)

    assert world.cameras["main"] is replacement
    assert replacement.frozen
    np.testing.assert_allclose(replacement.camera_position, [12.0, 20.0, 30.0])
    assert world.P_matrix["main"] is None


def test_draft_orientation_from_points_aims_exactly():
    draft = CameraPoseDraft.from_camera(make_camera())  # position [10, 20, 30]
    draft.yaw = 30.0  # leftover local angles must be replaced by aiming

    draft.set_orientation_from_points(
        [10.0, 20.0, 130.0], right_point=[500.0, 21.0, 30.0]
    )

    assert draft.yaw == draft.pitch == draft.roll == 0.0
    camera = draft.build_camera()
    np.testing.assert_allclose(camera.camera_z, [0.0, 0.0, 1.0], atol=1e-12)
    # right point is projected perpendicular to look, so its Y offset is dropped
    np.testing.assert_allclose(
        camera.camera_x, [490.0, 1.0, 0.0] / np.linalg.norm([490.0, 1.0, 0.0]), atol=1e-12
    )
    # the look point lands exactly on the camera +Z axis
    local = camera.world2camera(np.array([[10.0, 20.0, 130.0]]))
    np.testing.assert_allclose(local, [[0.0, 0.0, 100.0]], atol=1e-9)


def test_draft_translate_camera_moves_along_local_axes():
    draft = CameraPoseDraft.from_camera(make_camera())  # position [10, 20, 30]
    # aim +X so the optical axis is world X, image-right is world -Z
    draft.set_orientation_from_points(
        [510.0, 20.0, 30.0], right_point=[10.0, 20.0, -470.0]
    )

    draft.translate_camera([0.0, 0.0, 100.0])  # advance along the optical axis
    np.testing.assert_allclose(draft.position, [110.0, 20.0, 30.0], atol=1e-12)

    draft.translate_camera([50.0, 0.0, 0.0])  # step image-right (world -Z)
    np.testing.assert_allclose(draft.position, [110.0, 20.0, -20.0], atol=1e-12)

    with pytest.raises(ValueError):
        draft.translate_camera([1.0, 2.0])


def test_draft_translate_camera_uses_local_angle_offsets():
    draft = CameraPoseDraft.from_camera(make_camera())  # identity base rotation
    draft.yaw = 90.0  # rotate camera axes 90 degrees about local Z

    draft.translate_camera([100.0, 0.0, 0.0])

    # local X maps to world -Y after the +90 degree yaw (zyx euler, row-axes)
    np.testing.assert_allclose(draft.position, [10.0, -80.0, 30.0], atol=1e-9)


def test_draft_orientation_from_points_supports_down_and_rejects_degenerate():
    draft = CameraPoseDraft.from_camera(make_camera())
    draft.set_orientation_from_points([10.0, 20.0, 130.0], down_point=[10.0, 520.0, 30.0])
    camera = draft.build_camera()
    np.testing.assert_allclose(camera.camera_y, [0.0, 1.0, 0.0], atol=1e-12)

    with pytest.raises(ValueError):
        draft.set_orientation_from_points([10.0, 20.0, 130.0])
    with pytest.raises(ValueError):
        # lateral point collinear with the look direction
        draft.set_orientation_from_points(
            [10.0, 20.0, 130.0], right_point=[10.0, 20.0, 230.0]
        )


def test_single_pinhole_from_form_builds_reference_camera():
    camera = single_pinhole_from_form(
        name=" side ",
        focal_length="25",
        eye_size="1.0",
        screen_size=7.5,
        pixels="61",
        subpixel_resolution=5,
        aperture_size=1.8,
        aperture_z="13",
    )

    np.testing.assert_allclose(camera.camera_position, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(camera.rotation_matrix, np.eye(3))
    np.testing.assert_array_equal(camera.screen.pixel_shape, (61, 61))
    assert len(camera.eyes) == 1
    np.testing.assert_allclose(camera.apertures[0].position, [0.0, 0.0, 13.0])


def test_single_pinhole_from_form_supports_screen_shapes():
    rectangle = single_pinhole_from_form(
        name="rect",
        focal_length=25.0,
        eye_size=1.0,
        screen_shape="rectangle",
        screen_size="8",
        screen_width="10",
        pixels=8,
        pixels_width=10,
        subpixel_resolution=2,
        aperture_size=1.8,
        aperture_z=13.0,
    )
    assert rectangle.screen.screen_shape == "rectangle"
    np.testing.assert_allclose(rectangle.screen.screen_size, [8.0, 10.0])
    np.testing.assert_array_equal(rectangle.screen.pixel_shape, (8, 10))

    circle = single_pinhole_from_form(
        name="circ",
        focal_length=25.0,
        eye_size=1.0,
        screen_shape="circle",
        screen_size=7.5,
        screen_width=999.0,  # ignored: circle takes a scalar diameter
        pixels=61,
        subpixel_resolution=5,
        aperture_size=1.8,
        aperture_z=13.0,
    )
    assert circle.screen.screen_shape == "circle"
    np.testing.assert_allclose(circle.screen.screen_size, [7.5, 7.5])
    np.testing.assert_array_equal(circle.screen.pixel_shape, (61, 61))


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "  "},
        {"focal_length": "-5"},
        {"focal_length": "not-a-number"},
        {"pixels": "61.5"},
        {"pixels": 0},
        {"subpixel_resolution": -1},
        {"aperture_size": 0.0},
        {"aperture_z": "inf"},
        {"screen_shape": "triangle"},
        {"screen_shape": "rectangle", "screen_width": -1.0},
        {"screen_shape": "ellipse", "pixels_width": 0},
    ],
)
def test_single_pinhole_from_form_rejects_bad_values(overrides):
    values = dict(
        name="cam",
        focal_length=25.0,
        eye_size=1.0,
        screen_size=7.5,
        pixels=61,
        subpixel_resolution=5,
        aperture_size=1.8,
        aperture_z=13.0,
    )
    values.update(overrides)
    with pytest.raises(ValueError):
        single_pinhole_from_form(**values)


def test_view_boundary_is_world_transformed_and_unoccluded():
    camera = make_camera()
    boundary = sample_view_boundary(camera, distance=100.0, samples=12)
    segments = view_boundary_segments(camera, distance=100.0, samples=12)

    assert boundary.shape == (12, 3)
    assert segments.shape == (12, 2, 3)
    np.testing.assert_allclose(segments[:, 1], boundary)
    np.testing.assert_allclose(segments[:, 0], np.tile([10.0, 20.0, 55.0], (12, 1)))
    np.testing.assert_allclose(np.linalg.norm(boundary - segments[:, 0], axis=1), 100.0)


def test_view_boundary_rejects_invalid_inputs():
    camera = make_camera()
    for samples in (0, 3):
        try:
            sample_view_boundary(camera, samples=samples)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid samples should fail")
