# World Builder GUI (MVP)

The optional World Builder GUI provides a VTK-backed view for importing a wall
STL, building a World with multiple single-pinhole Cameras under stable keys,
aligning each camera, and saving or reloading the World project. It is an
early MVP on the GUI development branch; voxel fields and coordinate slices
will be added incrementally.

## Installation

Install the package with its optional GUI dependencies:

```bash
pip install -e '.[gui]'
```

The normal package dependencies do not include VTK or trame, so library-only
users are unaffected.

## Launch

Start with an empty wall scene:

```bash
python -m multi_pinhole.gui.app
```

Or load the MST example wall at startup:

```bash
python -m multi_pinhole.gui.app --wall examples/mst/MST_wall-mesh.stl
```

An installed package also provides the equivalent `multi-pinhole-gui` command.

## Current workflow

The drawer is organized into collapsible panels (Wall STL, Cameras, Position,
Orientation, View & boundary, World project); Cameras, Position, and
Orientation start expanded. The Apply/Revert buttons and the status, revision,
and error read-outs stay visible below the panels.

1. Enter a local STL path, choose its source units, and click **Import STL**.
   STL files carry no unit metadata; imported geometry is converted to the
   package's millimetre convention.
2. Manage cameras under **Cameras**. The World starts with one camera under
   the stable key `"camera"`. Type a name, adjust the optics form (focal
   length, eye size, screen shape and size, pixel counts, subpixel resolution,
   and circular-aperture diameter and Z offset), and click **Add** to register
   a new single-pinhole camera through `World.add_camera()`. The **Screen
   shape** selector supports square, rectangle, circle, and ellipse; rectangle
   and ellipse reveal separate width and Pixels W fields, while square and
   circle use the height field as the width or diameter. New cameras start at
   the world origin in the reference pose. **Remove** deletes the active
   camera; the last remaining camera cannot be removed. Switch the editing
   target with the **Active camera** selector — every camera stays visible in
   the 3D view.
3. Adjust the active Camera's world position with the color-coded X/Y/Z slider
   or numeric field. X is red, Y is green, and Z is blue in both the controls
   and 3D view.

   To move in the camera frame instead, enter a step under **Position ·
   camera-frame step** and click **Translate in camera frame**
   (`Camera.translate_camera()`): ΔX steps image-right, ΔY image-down, and ΔZ
   advances along the optical axis, all using the current draft orientation
   including any yaw/pitch/roll adjustment. Each click applies the step again,
   so repeated clicks walk the camera.
4. Adjust camera-local yaw, pitch, and roll over the full -180 to +180 degree
   range. During editing only the lightweight Camera actor moves; the existing
   view cone is hidden and marked stale.

   For exact alignment, use **Orientation · aim at world points** instead of
   the incremental angles: enter a world-coordinate **look point** and a
   lateral point (image-**right** or image-**down**, selectable), then click
   **Aim camera**. The optical axis passes exactly through the look point from
   the current draft position (`Camera.set_orientation_from_points()`); the
   lateral point is projected perpendicular to the look direction, so it only
   needs to be approximate. Yaw/pitch/roll reset to zero afterwards and remain
   available for local fine-tuning on top of the aimed base orientation. Note
   that moving the camera position afterwards keeps the orientation fixed and
   does not re-aim automatically.
5. Click **Apply & Boundary**. The GUI creates a new Camera, replaces the
   active stable key through `World.change_camera()`, and updates the view
   cone and detector-boundary samples. The applied revision number is shown so
   it is clear that the action completed.
6. Click **Revert** to discard the current draft and return to the last applied
   Camera pose.
7. Save or restore the whole World under **World project**. **Save World**
   serializes the World (cameras, walls, voxel, caches) with
   `World.save_world()`; **Load World** restores a saved project with
   `World.load_world()` and rebuilds the scene. The dill-based format is a
   full Python snapshot: load only files you created yourself, and expect
   compatibility to track the package version.

Use **World origin** or **Camera** under Orbit pivot to keep 3D rotation fixed
around a predictable point instead of relying on the scene's automatic pivot.
The top-right toolbar switches the primary left-drag action between Orbit, Pan,
and Zoom. Middle drag remains available for Pan and the mouse wheel for Zoom.
Rendering stays in the browser through VTK.js so interaction does not require
streaming a new image from Python for every frame.

The **Boundary outline points** setting controls only how densely the detector
perimeter is sampled for the lightweight view-cone outline. It does not change
wall/aperture visibility or projection accuracy.

The boundary preview deliberately performs no wall or aperture intersection.
It samples rays from the selected Eye away from the detector boundary, making
alignment responsive and keeping the expensive visibility calculation a
separate future action.

## Implemented architecture

- `multi_pinhole.gui.model.CameraPoseDraft` holds mutable GUI pose state without
  mutating a frozen Camera.
- `single_pinhole_from_form()` validates loose (string) web-form values and
  builds a reference-pose single-pinhole Camera, GUI framework independent.
- `sample_view_boundary()` produces GUI-independent unoccluded boundary points.
- `multi_pinhole.gui.vtk_scene` adapts numpy-stl walls, Camera geometry, and the
  sampled cone to VTK actors.
- `multi_pinhole.gui.app` binds those pieces to trame controls and keeps
  per-stable-key actor maps so every registered Camera stays in the scene.

Next planned capabilities are Voxel construction, World and Camera-coordinate
slices, scalar `f` cell data, and visibility inspection.
