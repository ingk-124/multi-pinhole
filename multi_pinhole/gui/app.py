"""trame/VTK MVP for building and aligning a multi-pinhole World.

Launch with ``multi-pinhole-gui`` after installing ``multi-pinhole[gui]``.
The MVP imports wall STL files from a local path, manages multiple
single-pinhole cameras under stable World keys, aligns the active camera with
an unoccluded view boundary, and saves or reloads the World project.
"""

from pathlib import Path

import numpy as np
from stl import mesh

from ..core import Aperture, Camera
from ..world import World
from .model import CameraPoseDraft, single_pinhole_from_form


def _require_gui():
    try:
        from trame.app import TrameApp, get_server
        from trame.ui.vuetify3 import SinglePageWithDrawerLayout
        from trame.widgets import vtk as vtk_widgets
        from trame.widgets import vuetify3 as v3
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa: F401
        from vtkmodules.vtkRenderingCore import vtkRenderWindow, vtkRenderer
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "The GUI dependencies are not installed. Run `pip install -e '.[gui]'`."
        ) from exc
    return (
        TrameApp, get_server, SinglePageWithDrawerLayout, vtk_widgets, v3,
        vtkRenderWindow, vtkRenderer,
    )


def default_camera():
    """Return a GUI-editable single-pinhole camera in a useful reference pose."""
    return Camera.single_pinhole(
        focal_length=25.0,
        eye_size=1.0,
        screen_size=7.5,
        pixel_shape=(61, 61),
        subpixel_resolution=5,
        apertures=Aperture(shape="circle", size=1.8, position=[0, 0, 13]),
        camera_name="camera",
    ).set_camera_position([0.0, 0.0, -100.0])


def create_app(server=None, *, wall_path=None):
    """Create the trame application and return its app instance."""
    (
        TrameApp,
        get_server,
        SinglePageWithDrawerLayout,
        vtk_widgets,
        v3,
        vtkRenderWindow,
        vtkRenderer,
    ) = _require_gui()
    from .vtk_scene import camera_actors, view_cone_actor, wall_actor

    class WorldBuilderApp(TrameApp):
        def __init__(self, app_server=None):
            super().__init__(app_server or get_server(client_type="vue3"))
            self.renderer = vtkRenderer()
            self.renderer.SetBackground(0.08, 0.09, 0.12)
            self.render_window = vtkRenderWindow()
            self.render_window.AddRenderer(self.renderer)
            self.world = World(cameras={"camera": default_camera()}, verbose=0)
            self.active_key = "camera"
            self.draft = CameraPoseDraft.from_camera(self.world.cameras["camera"])
            self.wall_actors = []
            self.camera_actors = {}
            self.cone_actors = {}
            self.applied_revision = 0
            self.view = None
            self._keys_by_label = {}
            self._set_initial_state()
            self._rebuild_scene()
            self.renderer.ResetCamera()
            self._sync_camera_state()
            self._bind_state()
            self._build_ui()
            if wall_path:
                self.state.wall_path = str(Path(wall_path).resolve())
                self.load_wall()

        def _set_initial_state(self):
            camera = self.world.cameras["camera"]
            self.state.position_x, self.state.position_y, self.state.position_z = map(float, camera.camera_position)
            self.state.yaw = self.state.pitch = self.state.roll = 0.0
            self.state.wall_path = ""
            self.state.wall_units = "mm"
            self.state.boundary_samples = 32
            self.state.boundary_distance = 1000.0
            self.state.status = "Geometry current · boundary current"
            self.state.pivot = "World origin"
            self.state.applied_revision = 0
            self.state.interaction_mode = "orbit"
            self.state.interactor_settings = self._interaction_settings("orbit")
            self.state.error = ""
            self.state.camera_keys = ["camera"]
            self.state.active_camera = "camera"
            self.state.new_camera_name = ""
            self.state.new_focal_length = 25.0
            self.state.new_eye_size = 1.0
            self.state.new_screen_shape = "square"
            self.state.new_screen_size = 7.5
            self.state.new_screen_width = 7.5
            self.state.new_pixels = 61
            self.state.new_pixels_w = 61
            self.state.new_subpixel = 5
            self.state.new_aperture_size = 1.8
            self.state.new_aperture_z = 13.0
            self.state.project_path = "world.pkl"
            self.state.look_x = 0.0
            self.state.look_y = 0.0
            self.state.look_z = 0.0
            self.state.lateral_kind = "right"
            self.state.lateral_x = 1000.0
            self.state.lateral_y = 0.0
            self.state.lateral_z = 0.0
            self.state.step_x = 0.0
            self.state.step_y = 0.0
            self.state.step_z = 0.0
            self.state.open_panels = ["cameras", "position", "orientation"]

        @staticmethod
        def _interaction_settings(mode):
            primary = {"orbit": "Rotate", "pan": "Pan", "zoom": "Zoom"}[mode]
            return [
                {"button": 1, "action": primary},
                {"button": 2, "action": "Pan"},
                {"button": 3, "action": "Zoom", "scrollEnabled": True},
            ]

        def _draft_from_state(self):
            self.draft.position[:] = [self.state.position_x, self.state.position_y, self.state.position_z]
            self.draft.yaw = float(self.state.yaw)
            self.draft.pitch = float(self.state.pitch)
            self.draft.roll = float(self.state.roll)
            return self.draft

        def _sync_camera_state(self):
            self._keys_by_label = {str(key): key for key in self.world.cameras}
            self.state.camera_keys = list(self._keys_by_label)
            self.state.active_camera = str(self.active_key)

        def _load_pose_state(self, camera):
            self.state.position_x, self.state.position_y, self.state.position_z = map(
                float, camera.camera_position
            )
            self.state.yaw = self.state.pitch = self.state.roll = 0.0

        def _add_camera_actors(self, key, camera):
            self.camera_actors[key] = camera_actors(camera)
            for actor in self.camera_actors[key]:
                self.renderer.AddActor(actor)
            self.cone_actors[key] = list(
                view_cone_actor(
                    camera,
                    distance=float(self.state.boundary_distance),
                    samples=int(self.state.boundary_samples),
                )
            )
            for actor in self.cone_actors[key]:
                self.renderer.AddActor(actor)

        def _remove_camera_actors(self, key):
            for actor in self.camera_actors.pop(key):
                self.renderer.RemoveActor(actor)
            for actor in self.cone_actors.pop(key):
                self.renderer.RemoveActor(actor)

        def _rebuild_scene(self):
            for key in list(self.camera_actors):
                self._remove_camera_actors(key)
            for key, camera in self.world.cameras.items():
                self._add_camera_actors(key, camera)

        def _rebuild_wall_actors(self):
            for actor in self.wall_actors:
                self.renderer.RemoveActor(actor)
            self.wall_actors = [wall_actor(wall) for wall in self.world.walls]
            for actor in self.wall_actors:
                self.renderer.AddActor(actor)

        def _replace_camera_actor(self, camera):
            key = self.active_key
            for actor in self.camera_actors.get(key, ()):
                self.renderer.RemoveActor(actor)
            self.camera_actors[key] = camera_actors(camera)
            for actor in self.camera_actors[key]:
                self.renderer.AddActor(actor)

        def _replace_cone(self, camera):
            key = self.active_key
            for actor in self.cone_actors.get(key, ()):
                self.renderer.RemoveActor(actor)
            self.cone_actors[key] = list(
                view_cone_actor(
                    camera,
                    distance=float(self.state.boundary_distance),
                    samples=int(self.state.boundary_samples),
                )
            )
            for actor in self.cone_actors[key]:
                self.renderer.AddActor(actor)

        def _update_view(self):
            self.renderer.Modified()
            self.render_window.Modified()
            try:
                if self.view is not None:
                    self.view.update()
                else:
                    self.ctrl.view_update()
            except (AttributeError, TypeError):
                pass

        def _set_cone_visible(self, visible):
            for actor in self.cone_actors.get(self.active_key, ()):
                actor.SetVisibility(bool(visible))

        def _set_pivot(self, point, label):
            self.state.pivot = label
            self.renderer.GetActiveCamera().SetFocalPoint(*map(float, point))
            self.renderer.ResetCameraClippingRange()
            if self.view is not None and self.server.protocol:
                self.server.js_call(
                    "world_view", "setCamera",
                    {"centerOfRotation": tuple(map(float, point))},
                )
            self._update_view()

        def set_interaction_mode(self, interaction_mode=None, **_):
            mode = interaction_mode or self.state.interaction_mode
            self.state.interactor_settings = self._interaction_settings(mode)

        def pivot_world(self, **_):
            self._set_pivot((0.0, 0.0, 0.0), "World origin")

        def pivot_camera(self, **_):
            self._set_pivot(self._draft_from_state().position, "Camera position")

        def _activate_camera(self, key):
            self.active_key = key
            camera = self.world.cameras[key]
            self.draft = CameraPoseDraft.from_camera(camera)
            self._load_pose_state(camera)
            self._sync_camera_state()
            self._set_cone_visible(True)

        def set_active_camera(self, active_camera=None, **_):
            label = active_camera or self.state.active_camera
            key = self._keys_by_label.get(label)
            if key is None or key == self.active_key:
                return
            self._activate_camera(key)
            self.state.status = f"Editing camera {label!r} · geometry current"
            self.state.error = ""
            self._update_view()

        def add_camera(self):
            try:
                label = str(self.state.new_camera_name).strip()
                if label in self._keys_by_label:
                    raise ValueError(f"camera key {label!r} is already registered")
                camera = single_pinhole_from_form(
                    name=label,
                    focal_length=self.state.new_focal_length,
                    eye_size=self.state.new_eye_size,
                    screen_shape=self.state.new_screen_shape,
                    screen_size=self.state.new_screen_size,
                    screen_width=self.state.new_screen_width,
                    pixels=self.state.new_pixels,
                    pixels_width=self.state.new_pixels_w,
                    subpixel_resolution=self.state.new_subpixel,
                    aperture_size=self.state.new_aperture_size,
                    aperture_z=self.state.new_aperture_z,
                )
                self.world.add_camera(label, camera)
                self._add_camera_actors(label, camera)
                self.renderer.ResetCameraClippingRange()
                self._activate_camera(label)
                self.state.new_camera_name = ""
                self.state.status = f"Added camera {label!r} · now editing it"
                self.state.error = ""
                self._update_view()
            except (TypeError, ValueError, KeyError) as exc:
                self.state.error = str(exc)

        def remove_active_camera(self):
            if len(self.world.cameras) <= 1:
                self.state.error = "The World needs at least one camera; add another before removing this one."
                return
            key = self.active_key
            self.world.remove_camera(key)
            self._remove_camera_actors(key)
            self._activate_camera(next(iter(self.world.cameras)))
            self.state.status = (
                f"Removed camera {str(key)!r} · editing {self.state.active_camera!r}"
            )
            self.state.error = ""
            self._update_view()

        def translate_camera_frame(self):
            try:
                draft = self._draft_from_state()
                draft.translate_camera(
                    [
                        float(self.state.step_x),
                        float(self.state.step_y),
                        float(self.state.step_z),
                    ]
                )
                self.state.position_x, self.state.position_y, self.state.position_z = map(
                    float, draft.position
                )
                self.preview_pose()
            except (TypeError, ValueError) as exc:
                self.state.error = str(exc)

        def orient_from_points(self):
            try:
                draft = self._draft_from_state()
                look_point = [
                    float(self.state.look_x),
                    float(self.state.look_y),
                    float(self.state.look_z),
                ]
                lateral_point = [
                    float(self.state.lateral_x),
                    float(self.state.lateral_y),
                    float(self.state.lateral_z),
                ]
                if self.state.lateral_kind == "right":
                    draft.set_orientation_from_points(look_point, right_point=lateral_point)
                else:
                    draft.set_orientation_from_points(look_point, down_point=lateral_point)
                self.state.yaw = self.state.pitch = self.state.roll = 0.0
                self.preview_pose()
                suffix = (
                    "cone and boundary stale until Apply"
                    if "stale" in self.state.status else "geometry current"
                )
                self.state.status = f"Aimed at look point (exact) · {suffix}"
            except (TypeError, ValueError) as exc:
                self.state.error = str(exc)

        def preview_pose(self, **_):
            try:
                camera = self._draft_from_state().build_camera()
                self._replace_camera_actor(camera)
                current = self.world.cameras[self.active_key]
                is_current = (
                    np.allclose(camera.camera_position, current.camera_position)
                    and np.allclose(camera.rotation_matrix, current.rotation_matrix)
                )
                self._set_cone_visible(is_current)
                self.state.status = (
                    "Geometry current · boundary current"
                    if is_current else "Camera draft · cone and boundary stale"
                )
                self.state.error = ""
                self._update_view()
            except (TypeError, ValueError) as exc:
                self.state.error = str(exc)

        def apply_pose(self):
            try:
                replacement = self._draft_from_state().build_camera()
                self.world.change_camera(self.active_key, replacement)
                self._replace_camera_actor(replacement)
                self._replace_cone(replacement)
                self._set_cone_visible(True)
                self.applied_revision += 1
                self.draft = CameraPoseDraft.from_camera(replacement)
                self._load_pose_state(replacement)
                self.state.applied_revision = self.applied_revision
                self.state.status = (
                    f"Applied #{self.applied_revision} to {self.state.active_camera!r} · "
                    f"boundary current "
                    f"({int(self.state.boundary_samples)} rays, no wall intersections)"
                )
                self.state.error = ""
                self._update_view()
            except (TypeError, ValueError) as exc:
                self.state.error = str(exc)

        def revert_pose(self):
            camera = self.world.cameras[self.active_key]
            self.draft = CameraPoseDraft.from_camera(camera)
            self._load_pose_state(camera)
            self._replace_camera_actor(camera)
            self._set_cone_visible(True)
            self.state.status = "Geometry current · boundary current"
            self.state.error = ""
            self._update_view()

        def load_wall(self):
            try:
                path = Path(self.state.wall_path).expanduser().resolve()
                if path.suffix.lower() != ".stl" or not path.is_file():
                    raise ValueError("Select an existing .stl file")
                wall = mesh.Mesh.from_file(path)
                scale = {"mm": 1.0, "cm": 10.0, "m": 1000.0}[self.state.wall_units]
                if scale != 1.0:
                    wall.vectors[:] *= scale
                    wall.update_normals()
                self.world.walls = wall
                self._rebuild_wall_actors()
                self.renderer.ResetCamera()
                self.state.status = f"Loaded wall: {path.name} ({self.state.wall_units})"
                self.state.error = ""
                self._update_view()
            except (OSError, KeyError, ValueError) as exc:
                self.state.error = str(exc)

        def save_project(self):
            try:
                if not str(self.state.project_path).strip():
                    raise ValueError("Enter a project file path")
                path = Path(self.state.project_path).expanduser().resolve()
                self.world.save_world(path)
                self.state.status = f"Saved World project: {path.name}"
                self.state.error = ""
            except (OSError, TypeError, ValueError) as exc:
                self.state.error = str(exc)

        def load_project(self):
            try:
                path = Path(self.state.project_path).expanduser().resolve()
                if not path.is_file():
                    raise ValueError("Select an existing World project file")
                world = World.load_world(path)
                if not isinstance(world, World):
                    raise ValueError("The file does not contain a World project")
                if not world.cameras:
                    raise ValueError("The loaded World has no cameras")
                self.world = world
                self._rebuild_wall_actors()
                self._rebuild_scene()
                self._activate_camera(next(iter(world.cameras)))
                self.renderer.ResetCamera()
                self.state.status = f"Loaded World project: {path.name}"
                self.state.error = ""
                self._update_view()
            except Exception as exc:  # dill raises many distinct unpickling errors
                self.state.error = str(exc)

        def _bind_state(self):
            self.state.change(
                "position_x", "position_y", "position_z", "yaw", "pitch", "roll"
            )(self.preview_pose)
            self.state.change("interaction_mode")(self.set_interaction_mode)
            self.state.change("active_camera")(self.set_active_camera)

        def _pose_control(self, label, state_name, minimum, maximum, step, color=None):
            with v3.VRow(dense=True, classes="ma-0 align-center"):
                with v3.VCol(cols=2, classes="pa-1"):
                    v3.VLabel(label, color=color)
                with v3.VCol(cols=6, classes="pa-1"):
                    v3.VSlider(
                        v_model=(state_name,), min=minimum, max=maximum, step=step,
                        hide_details=True, density="compact", color=color,
                    )
                with v3.VCol(cols=4, classes="pa-1"):
                    v3.VTextField(
                        v_model=(state_name,), type="number", step=step,
                        hide_details=True, density="compact", color=color,
                    )

        def _number_row(self, *fields):
            with v3.VRow(dense=True, classes="ma-0"):
                for label, name in fields:
                    with v3.VCol(classes="pa-1"):
                        v3.VTextField(
                            label=label, v_model=(name,), type="number",
                            density="compact", hide_details=True,
                        )

        def _build_ui(self):
            with SinglePageWithDrawerLayout(self.server) as self.ui:
                self.ui.title.set_text("Multi-pinhole World Builder")
                with self.ui.toolbar:
                    v3.VSpacer()
                    with v3.VBtnToggle(
                        v_model=("interaction_mode",), mandatory=True, density="compact",
                    ):
                        v3.VBtn(
                            icon="mdi-orbit", value="orbit",
                            v_tooltip_bottom="'Orbit'",
                        )
                        v3.VBtn(
                            icon="mdi-pan", value="pan",
                            v_tooltip_bottom="'Pan'",
                        )
                        v3.VBtn(
                            icon="mdi-magnify-plus-outline", value="zoom",
                            v_tooltip_bottom="'Zoom'",
                        )
                    v3.VBtn("Apply", color="primary", click=self.apply_pose)
                    v3.VBtn(
                        icon="mdi-crop-free", click=self.ctrl.reset_camera,
                        v_tooltip_bottom="'Reset view'",
                    )
                with self.ui.drawer as drawer:
                    drawer.width = 390
                    with v3.VExpansionPanels(
                        v_model=("open_panels",), multiple=True, variant="accordion",
                    ):
                        with v3.VExpansionPanel(value="wall", title="Wall STL"):
                            with v3.VExpansionPanelText():
                                v3.VTextField(
                                    label="Local STL path", v_model=("wall_path",),
                                    density="compact", hide_details=True, classes="ma-2",
                                )
                                v3.VSelect(
                                    label="Source units", items=("['mm', 'cm', 'm']",),
                                    v_model=("wall_units",), density="compact", hide_details=True, classes="ma-2",
                                )
                                v3.VBtn("Import STL", block=True, classes="ma-2", click=self.load_wall)
                        with v3.VExpansionPanel(value="cameras", title="Cameras"):
                            with v3.VExpansionPanelText():
                                v3.VSelect(
                                    label="Active camera", items=("camera_keys",),
                                    v_model=("active_camera",), density="compact", hide_details=True, classes="ma-2",
                                )
                                with v3.VRow(dense=True, classes="ma-0"):
                                    with v3.VCol(cols=6, classes="pa-1"):
                                        v3.VTextField(
                                            label="New camera name", v_model=("new_camera_name",),
                                            density="compact", hide_details=True,
                                        )
                                    with v3.VCol(cols=3, classes="pa-1"):
                                        v3.VBtn("Add", block=True, color="primary", click=self.add_camera)
                                    with v3.VCol(cols=3, classes="pa-1"):
                                        v3.VBtn("Remove", block=True, click=self.remove_active_camera)
                                self._number_row(
                                    ("Focal length [mm]", "new_focal_length"),
                                    ("Eye size [mm]", "new_eye_size"),
                                )
                                v3.VSelect(
                                    label="Screen shape",
                                    items=("['square', 'rectangle', 'circle', 'ellipse']",),
                                    v_model=("new_screen_shape",),
                                    density="compact", hide_details=True, classes="ma-2",
                                )
                                self._number_row(
                                    ("Screen height / diameter [mm]", "new_screen_size"),
                                    ("Pixels H", "new_pixels"),
                                )
                                with v3.VRow(
                                    dense=True, classes="ma-0",
                                    v_show="['rectangle', 'ellipse'].includes(new_screen_shape)",
                                ):
                                    with v3.VCol(classes="pa-1"):
                                        v3.VTextField(
                                            label="Screen width [mm]", v_model=("new_screen_width",),
                                            type="number", density="compact", hide_details=True,
                                        )
                                    with v3.VCol(classes="pa-1"):
                                        v3.VTextField(
                                            label="Pixels W", v_model=("new_pixels_w",),
                                            type="number", density="compact", hide_details=True,
                                        )
                                self._number_row(
                                    ("Subpixel resolution", "new_subpixel"),
                                    ("Aperture diameter [mm]", "new_aperture_size"),
                                )
                                self._number_row(("Aperture Z [mm]", "new_aperture_z"))
                                v3.VAlert(
                                    text=(
                                        "New cameras start at the world origin in the reference pose; "
                                        "move them with the Position controls. Square and circle screens "
                                        "use the height field as width or diameter."
                                    ),
                                    type="info", variant="tonal", density="compact", classes="ma-2",
                                )
                        with v3.VExpansionPanel(value="position", title="Position"):
                            with v3.VExpansionPanelText():
                                v3.VListSubheader("World coordinates [mm]")
                                self._pose_control("X", "position_x", -5000, 5000, 0.1, "red")
                                self._pose_control("Y", "position_y", -5000, 5000, 0.1, "green")
                                self._pose_control("Z", "position_z", -5000, 5000, 0.1, "blue")
                                v3.VListSubheader("Camera-frame step [mm]")
                                self._number_row(
                                    ("ΔX · image right", "step_x"),
                                    ("ΔY · image down", "step_y"),
                                    ("ΔZ · optical axis", "step_z"),
                                )
                                v3.VBtn(
                                    "Translate in camera frame", block=True, classes="ma-2",
                                    click=self.translate_camera_frame,
                                )
                                v3.VAlert(
                                    text=(
                                        "Moves along the current draft orientation (Camera.translate_camera). "
                                        "Each click applies the step again, so repeated clicks walk the camera."
                                    ),
                                    type="info", variant="tonal", density="compact", classes="ma-2",
                                )
                        with v3.VExpansionPanel(value="orientation", title="Orientation"):
                            with v3.VExpansionPanelText():
                                v3.VListSubheader("Local angles [degree]")
                                self._pose_control("Yaw", "yaw", -180, 180, 0.1)
                                self._pose_control("Pitch", "pitch", -180, 180, 0.1)
                                self._pose_control("Roll", "roll", -180, 180, 0.1)
                                v3.VListSubheader("Aim at world points")
                                self._number_row(
                                    ("Look at X", "look_x"),
                                    ("Y", "look_y"),
                                    ("Z", "look_z"),
                                )
                                v3.VSelect(
                                    label="Lateral constraint", items=("['right', 'down']",),
                                    v_model=("lateral_kind",), density="compact", hide_details=True, classes="ma-2",
                                )
                                self._number_row(
                                    ("Point X", "lateral_x"),
                                    ("Y", "lateral_y"),
                                    ("Z", "lateral_z"),
                                )
                                v3.VBtn("Aim camera", block=True, classes="ma-2", click=self.orient_from_points)
                                v3.VAlert(
                                    text=(
                                        "The optical axis passes exactly through the look point from the "
                                        "current position. The lateral point sets image-right or image-down "
                                        "and is projected perpendicular to the look direction, so it only "
                                        "needs to be approximate. Yaw/pitch/roll reset to zero for fine-tuning."
                                    ),
                                    type="info", variant="tonal", density="compact", classes="ma-2",
                                )
                        with v3.VExpansionPanel(value="viewopts", title="View & boundary"):
                            with v3.VExpansionPanelText():
                                v3.VListSubheader("Orbit pivot")
                                with v3.VRow(classes="ma-1"):
                                    v3.VBtn("World origin", click=self.pivot_world)
                                    v3.VBtn("Camera", click=self.pivot_camera)
                                v3.VAlert(
                                    title="Orbit pivot", text=("pivot",), type="info", variant="tonal",
                                    density="compact", classes="ma-2",
                                )
                                v3.VAlert(
                                    text="Choose Orbit, Pan, or Zoom in the top-right toolbar. Mouse wheel always zooms.",
                                    type="info", variant="tonal", density="compact", classes="ma-2",
                                )
                                v3.VSelect(
                                    label="Boundary outline points", items=("[16, 32, 64]",),
                                    v_model=("boundary_samples",), density="compact", hide_details=True, classes="ma-2",
                                )
                                v3.VAlert(
                                    text="Samples the detector perimeter for view-cone smoothness only; it does not affect visibility.",
                                    type="info", variant="tonal", density="compact", classes="ma-2",
                                )
                                v3.VTextField(
                                    label="Boundary distance [mm]", type="number",
                                    v_model=("boundary_distance",), density="compact", hide_details=True, classes="ma-2",
                                )
                        with v3.VExpansionPanel(value="project", title="World project"):
                            with v3.VExpansionPanelText():
                                v3.VTextField(
                                    label="Project file (.pkl)", v_model=("project_path",),
                                    density="compact", hide_details=True, classes="ma-2",
                                )
                                with v3.VRow(classes="ma-1"):
                                    v3.VBtn("Save World", click=self.save_project)
                                    v3.VBtn("Load World", click=self.load_project)
                    with v3.VRow(classes="ma-1"):
                        v3.VBtn("Apply & Boundary", color="primary", click=self.apply_pose)
                        v3.VBtn("Revert", click=self.revert_pose)
                    v3.VAlert(text=("status",), type="info", density="compact", classes="ma-2")
                    v3.VAlert(
                        title="Applied Camera revision", text=("applied_revision",),
                        type="success", density="compact", classes="ma-2",
                    )
                    v3.VAlert(
                        text=("error",), type="error", density="compact", classes="ma-2",
                        v_show="error",
                    )
                with self.ui.content:
                    with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                        view = vtk_widgets.VtkLocalView(
                            self.render_window,
                            ref="world_view",
                            interactor_settings=("interactor_settings",),
                        )
                        self.view = view
                        self.ctrl.view_update = view.update
                        self.ctrl.reset_camera = view.reset_camera
                        self.ctrl.on_server_ready.add(view.update)
                        self.ctrl.on_server_ready.add(self.pivot_world)

    return WorldBuilderApp(server)


def main():
    """Run the optional local GUI."""
    _, get_server, *_ = _require_gui()
    server = get_server(client_type="vue3")
    server.cli.add_argument("--wall", help="STL wall to import at startup")
    args, _ = server.cli.parse_known_args()
    create_app(server, wall_path=args.wall)
    server.start()


if __name__ == "__main__":  # pragma: no cover
    main()
