from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("trame.app")
pytest.importorskip("vtkmodules")

from trame.app import get_server

from multi_pinhole.gui.app import create_app


def make_app(name, **kwargs):
    return create_app(get_server(name, client_type="vue3"), **kwargs)


def test_gui_app_loads_wall_and_applies_camera_pose():
    wall_path = Path(__file__).resolve().parents[1] / "examples" / "mst" / "MST_wall-mesh.stl"
    app = make_app("wall-pose", wall_path=wall_path)

    assert len(app.world.walls) == 1
    assert len(app.wall_actors) == 1
    assert len(app.cone_actors["camera"]) == 2

    app.state.position_x = 25.0
    app.state.position_y = -10.0
    app.state.position_z = 5.0
    app.state.yaw = 2.0
    app.preview_pose()
    assert "stale" in app.state.status
    assert all(actor.GetVisibility() == 0 for actor in app.cone_actors["camera"])

    app.apply_pose()
    camera = app.world.cameras["camera"]
    np.testing.assert_allclose(camera.camera_position, [25.0, -10.0, 5.0])
    assert camera.frozen
    assert "boundary current" in app.state.status
    assert "Applied #1" in app.state.status
    assert all(actor.GetVisibility() == 1 for actor in app.cone_actors["camera"])
    assert app.state.applied_revision == 1
    assert app.state.yaw == app.state.pitch == app.state.roll == 0.0
    np.testing.assert_allclose(app.draft.base_rotation, camera.rotation_matrix)

    app.state.interaction_mode = "pan"
    app.set_interaction_mode("pan")
    assert app.state.interactor_settings[0]["action"] == "Pan"
    assert app.state.interactor_settings[1]["action"] == "Pan"
    assert app.state.interactor_settings[2]["action"] == "Zoom"


def test_gui_app_adds_switches_and_removes_cameras():
    app = make_app("camera-mgmt")

    app.state.new_camera_name = "side"
    app.add_camera()
    assert set(app.world.cameras) == {"camera", "side"}
    assert app.state.camera_keys == ["camera", "side"]
    assert app.state.active_camera == "side"
    assert set(app.cone_actors) == {"camera", "side"}
    np.testing.assert_allclose(app.world.cameras["side"].camera_position, [0.0, 0.0, 0.0])

    app.state.position_x = 40.0
    app.preview_pose()
    app.apply_pose()
    np.testing.assert_allclose(app.world.cameras["side"].camera_position, [40.0, 0.0, 0.0])
    np.testing.assert_allclose(
        app.world.cameras["camera"].camera_position, [0.0, 0.0, -100.0]
    )

    app.set_active_camera("camera")
    assert app.active_key == "camera"
    assert app.state.position_z == -100.0

    app.state.new_camera_name = "side"
    app.add_camera()
    assert "already registered" in app.state.error
    assert set(app.world.cameras) == {"camera", "side"}

    app.state.new_camera_name = "wide"
    app.state.new_screen_shape = "rectangle"
    app.state.new_screen_size = 8.0
    app.state.new_screen_width = 12.0
    app.state.new_pixels = 40
    app.state.new_pixels_w = 60
    app.add_camera()
    assert app.state.error == ""
    wide = app.world.cameras["wide"]
    assert wide.screen.screen_shape == "rectangle"
    np.testing.assert_allclose(wide.screen.screen_size, [8.0, 12.0])
    np.testing.assert_array_equal(wide.screen.pixel_shape, (40, 60))
    app.remove_active_camera()

    app.set_active_camera("side")
    app.remove_active_camera()
    assert set(app.world.cameras) == {"camera"}
    assert app.state.active_camera == "camera"
    assert set(app.cone_actors) == {"camera"}

    app.remove_active_camera()
    assert set(app.world.cameras) == {"camera"}
    assert "at least one camera" in app.state.error


def test_gui_app_aims_camera_at_world_points():
    app = make_app("aim-points")

    # camera at [0, 0, -100]; aim +X with image-right pointing to -Z
    app.state.look_x, app.state.look_y, app.state.look_z = 100.0, 0.0, -100.0
    app.state.lateral_kind = "right"
    app.state.lateral_x, app.state.lateral_y, app.state.lateral_z = 0.0, 0.0, -1100.0
    app.orient_from_points()
    assert app.state.error == ""
    assert app.state.yaw == app.state.pitch == app.state.roll == 0.0
    assert "Aimed at look point" in app.state.status

    app.apply_pose()
    camera = app.world.cameras["camera"]
    np.testing.assert_allclose(camera.camera_z, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(camera.camera_x, [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(camera.camera_position, [0.0, 0.0, -100.0])

    # degenerate lateral point (collinear with look) reports an error
    app.state.lateral_x, app.state.lateral_y, app.state.lateral_z = 200.0, 0.0, -100.0
    app.orient_from_points()
    assert app.state.error != ""


def test_gui_app_translates_in_camera_frame():
    app = make_app("camera-step")

    # default camera at [0, 0, -100] with identity rotation: local Z is world Z
    app.state.step_z = 30.0
    app.translate_camera_frame()
    assert app.state.error == ""
    assert app.state.position_z == -70.0
    app.translate_camera_frame()
    assert app.state.position_z == -40.0
    assert "stale" in app.state.status

    app.state.step_z = "not-a-number"
    app.translate_camera_frame()
    assert app.state.error != ""
    assert app.state.position_z == -40.0


def test_gui_app_saves_and_loads_world_project(tmp_path):
    app = make_app("project-save")
    app.state.new_camera_name = "side"
    app.add_camera()
    app.state.position_x = 40.0
    app.preview_pose()
    app.apply_pose()

    project = tmp_path / "world.pkl"
    app.state.project_path = str(project)
    app.save_project()
    assert app.state.error == ""
    assert project.is_file()

    other = make_app("project-load")
    other.state.project_path = str(project)
    other.load_project()
    assert other.state.error == ""
    assert set(other.world.cameras) == {"camera", "side"}
    assert set(other.cone_actors) == {"camera", "side"}
    assert other.state.camera_keys == ["camera", "side"]
    np.testing.assert_allclose(
        other.world.cameras["side"].camera_position, [40.0, 0.0, 0.0]
    )

    other.state.project_path = str(tmp_path / "missing.pkl")
    other.load_project()
    assert "existing World project" in other.state.error
