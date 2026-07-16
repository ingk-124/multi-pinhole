"""Private visibility calculations used by :mod:`world`.

The helpers do not own or mutate ``World`` or projection-cache state.  They
may, however, trigger lazy geometry preparation on objects supplied by the
caller, such as :meth:`multi_pinhole.aperture.Aperture.set_model`.

The helpers in this module consume prepared geometry and return masks.  Cache
ownership, camera-key resolution, and scene mutation remain responsibilities
of :class:`multi_pinhole.world.World`.
"""

from collections.abc import Sequence

import numpy as np

from .utils import stl_utils
from .utils.my_stdio import my_print


def calculate_point_visibility(
        camera_points: np.ndarray,
        eyes: Sequence,
        eye_indices: Sequence[int],
        apertures: Sequence,
        walls_in_camera: Sequence,
        verbose: int = 1,
) -> np.ndarray:
    """Return per-eye visibility for points expressed in camera coordinates.

    Occluders are applied in the established order: the eye front plane,
    apertures, then walls.  Each geometry check receives only points still
    active after the preceding check.
    """
    visible = np.zeros((len(eye_indices), camera_points.shape[0]), dtype=bool)

    for i, eye_index in enumerate(eye_indices):
        eye = eyes[eye_index]
        visible[i] = camera_points[:, 2] >= eye.position[-1]
        my_print(
            f"checking visible points for eye {eye_index + 1}/{len(eyes)}",
            show=verbose > 0,
        )
        my_print("-" * 15, show=verbose > 0)

        my_print("--- checking for apertures ---", show=verbose > 0)
        for aperture_index, aperture in enumerate(apertures):
            active = np.flatnonzero(visible[i])
            if active.size == 0:
                break
            if aperture.stl_model is None:
                aperture.set_model()
            aperture_visible = stl_utils.check_visible(
                mesh_obj=aperture.stl_model,
                start=eye.position,
                grid_points=camera_points[active],
                verbose=verbose,
                behind_start_included=True,
            )
            visible[i, active[~aperture_visible]] = False
            my_print(
                f"{aperture_index + 1}/{len(apertures)} done",
                show=verbose > 0,
            )
            my_print("-" * 15, show=verbose > 0)

        my_print("--- checking for walls ---", show=verbose > 0)
        for wall_index, wall_in_camera in enumerate(walls_in_camera):
            active = np.flatnonzero(visible[i])
            if active.size == 0:
                break
            wall_visible = stl_utils.check_visible(
                mesh_obj=wall_in_camera,
                start=eye.position,
                grid_points=camera_points[active],
                verbose=verbose,
            )
            visible[i, active[~wall_visible]] = False
            my_print(
                f"{wall_index + 1}/{len(walls_in_camera)} done",
                show=verbose > 0,
            )
            my_print("-" * 15, show=verbose > 0)

    return visible


def calculate_visible_vertex_mask(
        inside_vertices: np.ndarray,
        inside_visibility: np.ndarray,
) -> np.ndarray:
    """Expand visibility for inside vertices to the complete vertex grid."""
    visible_vertices = np.zeros(
        (inside_visibility.shape[0], inside_vertices.size),
        dtype=bool,
    )
    visible_vertices[:, inside_vertices] = inside_visibility
    return visible_vertices


def classify_visible_voxels(
        visible_vertices: np.ndarray,
        vertices_indices: np.ndarray,
) -> np.ndarray:
    """Classify voxels as invisible (0), partial (1), or fully visible (2)."""
    voxel_vertices = visible_vertices[:, vertices_indices]
    conditions_any = np.any(voxel_vertices, axis=-1).astype(int)
    conditions_all = np.all(voxel_vertices, axis=-1).astype(int)
    return conditions_any + conditions_all
