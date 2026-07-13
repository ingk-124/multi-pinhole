"""VTK scene adapter used by the optional world-builder GUI."""

import numpy as np

from .model import sample_view_boundary, view_boundary_segments


def _vtk_imports():
    try:
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkLine, vtkPolyData, vtkTriangle
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
    except ImportError as exc:  # pragma: no cover - depends on optional GUI extras
        raise RuntimeError(
            "VTK is required for the GUI; install with `pip install -e '.[gui]'`"
        ) from exc
    return vtkPoints, vtkCellArray, vtkLine, vtkPolyData, vtkTriangle, vtkActor, vtkPolyDataMapper


def _line_polydata(segments):
    vtkPoints, vtkCellArray, vtkLine, vtkPolyData, _, _, _ = _vtk_imports()
    segments = np.asarray(segments, dtype=float)
    points = vtkPoints()
    lines = vtkCellArray()
    for segment in segments:
        start_id = points.InsertNextPoint(*segment[0])
        stop_id = points.InsertNextPoint(*segment[1])
        line = vtkLine()
        line.GetPointIds().SetId(0, start_id)
        line.GetPointIds().SetId(1, stop_id)
        lines.InsertNextCell(line)
    data = vtkPolyData()
    data.SetPoints(points)
    data.SetLines(lines)
    return data


def _actor(polydata, color, *, opacity=1.0, line_width=1.0):
    *_, vtkActor, vtkPolyDataMapper = _vtk_imports()
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetLineWidth(line_width)
    return actor


def wall_actor(stl_mesh, *, color=(0.72, 0.74, 0.78), opacity=0.28):
    """Convert a numpy-stl mesh to a translucent VTK actor."""
    vtkPoints, vtkCellArray, _, vtkPolyData, vtkTriangle, _, _ = _vtk_imports()
    points = vtkPoints()
    triangles = vtkCellArray()
    for vector in np.asarray(stl_mesh.vectors, dtype=float):
        triangle = vtkTriangle()
        for corner, point in enumerate(vector):
            point_id = points.InsertNextPoint(*point)
            triangle.GetPointIds().SetId(corner, point_id)
        triangles.InsertNextCell(triangle)
    data = vtkPolyData()
    data.SetPoints(points)
    data.SetPolys(triangles)
    return _actor(data, color, opacity=opacity)


def camera_actors(camera, *, axis_length=80.0):
    """Create world-space actors for Camera axes, screen, and optical axis.

    Plain ``vtkActor`` instances are returned instead of a ``vtkAssembly``
    because trame's local-view serializer silently drops assemblies, which
    would leave the camera invisible in the browser.
    """
    origin = np.asarray(camera.camera_position, dtype=float)
    axes = np.vstack([camera.camera_x, camera.camera_y, camera.camera_z])
    actors = []
    axis_colors = (
        (0.90, 0.12, 0.12),  # X: red
        (0.12, 0.72, 0.22),  # Y: green
        (0.12, 0.38, 0.95),  # Z: blue
    )
    for axis, color in zip(axes, axis_colors):
        segment = [[origin, origin + axis_length * axis]]
        actors.append(_actor(_line_polydata(segment), color, line_width=4.0))

    height, width = camera.screen.screen_size
    corners_local = np.array(
        [
            [-width / 2, -height / 2, 0],
            [width / 2, -height / 2, 0],
            [width / 2, height / 2, 0],
            [-width / 2, height / 2, 0],
        ]
    )
    corners = corners_local @ camera.rotation_matrix + origin
    optical_segments = [[corners[i], corners[(i + 1) % 4]] for i in range(4)]
    for eye in camera.eyes:
        eye_world = eye.position @ camera.rotation_matrix + origin
        optical_segments.append([origin, eye_world])
    actors.append(
        _actor(_line_polydata(optical_segments), (0.95, 0.55, 0.18), line_width=3.0)
    )
    return actors


def view_cone_actor(camera, *, distance=1000.0, samples=32):
    """Create lightweight, unoccluded view-cone and boundary actors."""
    endpoints = sample_view_boundary(camera, distance=distance, samples=samples)
    rays = view_boundary_segments(camera, distance=distance, samples=samples)
    boundary_segments = [
        [endpoints[index], endpoints[(index + 1) % len(endpoints)]]
        for index in range(len(endpoints))
    ]
    ray_actor = _actor(_line_polydata(rays[:: max(1, samples // 8)]), (0.25, 0.72, 0.92), opacity=0.35)
    boundary_actor = _actor(_line_polydata(boundary_segments), (0.25, 0.72, 0.92), line_width=2.0)
    return ray_actor, boundary_actor
