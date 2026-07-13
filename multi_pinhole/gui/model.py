"""GUI-independent camera editing and lightweight view-cone geometry."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..core import Aperture, Camera


@dataclass
class CameraPoseDraft:
    """Editable pose layered on top of an immutable Camera configuration.

    ``yaw``, ``pitch``, and ``roll`` are camera-local incremental rotations in
    degrees.  They are intentionally kept separate from ``base_rotation`` so a
    GUI can expose zero-centered adjustment controls and revert cheaply.
    """

    source: Camera
    position: np.ndarray
    base_rotation: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    @classmethod
    def from_camera(cls, camera: Camera) -> "CameraPoseDraft":
        """Create an editable pose snapshot from ``camera``."""
        if not isinstance(camera, Camera):
            raise TypeError("camera must be a Camera")
        return cls(
            source=camera,
            position=np.array(camera.camera_position, dtype=float, copy=True),
            base_rotation=np.array(camera.rotation_matrix, dtype=float, copy=True),
        )

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).copy()
        self.base_rotation = np.asarray(self.base_rotation, dtype=float).copy()
        if self.position.shape != (3,):
            raise ValueError("position must have shape (3,)")
        if self.base_rotation.shape != (3, 3):
            raise ValueError("base_rotation must have shape (3, 3)")

    @property
    def rotation_matrix(self) -> np.ndarray:
        """World-to-camera rotation including the local GUI adjustments."""
        delta = Rotation.from_euler(
            "zyx", [self.yaw, self.pitch, self.roll], degrees=True
        ).as_matrix()
        return delta @ self.base_rotation

    def set_orientation_from_points(self, look_point, *, right_point=None, down_point=None):
        """Aim the draft at world points, replacing the local angle offsets.

        The current draft position is the ray origin, so set the position
        first.  A temporary unfrozen Camera reuses the validation and
        orthonormal construction of ``Camera.set_orientation_from_points``:
        the look direction is honored exactly while the lateral point is
        projected perpendicular to it.  The result becomes the new
        ``base_rotation`` and yaw/pitch/roll reset to zero for fine-tuning.
        """
        camera = self.build_camera()
        camera.set_orientation_from_points(
            look_point, right_point=right_point, down_point=down_point
        )
        self.base_rotation = np.array(camera.rotation_matrix, dtype=float, copy=True)
        self.yaw = self.pitch = self.roll = 0.0
        return self

    def translate_camera(self, offset) -> "CameraPoseDraft":
        """Translate the draft by an offset expressed in camera coordinates.

        The offset uses the full draft orientation (base rotation plus the
        current yaw/pitch/roll), matching ``Camera.translate_camera``: X is
        image-right, Y is image-down, and Z advances along the optical axis.
        """
        camera = self.build_camera()
        camera.translate_camera(offset)
        self.position[:] = camera.camera_position
        return self

    def build_camera(self) -> Camera:
        """Build a new mutable Camera while reusing immutable optics.

        Eye, Screen, and Aperture instances are safe to share after World
        registration because their geometry is frozen.  The new Camera owns a
        fresh pose and is frozen when passed to :meth:`World.change_camera`.
        """
        return Camera(
            eyes=self.source.eyes,
            apertures=self.source.apertures,
            screen=self.source.screen,
            camera_position=self.position,
            rotation_matrix=self.rotation_matrix,
            camera_name=getattr(self.source, "_camera_name", None),
        )


def _positive_number(value, label: str) -> float:
    """Convert a loose form value to a positive finite float."""
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a positive number")
    return number


def _positive_count(value, label: str) -> int:
    """Convert a loose form value to a positive integer."""
    number = float(value)
    if not number.is_integer() or number < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(number)


def single_pinhole_from_form(
    *,
    name,
    focal_length,
    eye_size,
    screen_shape="square",
    screen_size,
    screen_width=None,
    pixels,
    pixels_width=None,
    subpixel_resolution,
    aperture_size,
    aperture_z,
) -> Camera:
    """Build a reference-pose single-pinhole Camera from loose GUI form values.

    Web form fields deliver numbers as strings, so every value is validated
    and converted here before touching the Camera constructor.  ``screen_size``
    is the height (or diameter); ``screen_width`` is consulted only for the
    two-dimensional shapes (rectangle, ellipse) as :class:`Screen` requires a
    scalar for square and circle.  ``pixels_width`` defaults to ``pixels`` for
    a square pixel grid.  The result is in the single-pinhole reference pose
    (origin position, identity rotation) and ready for interactive placement.
    """
    label = str(name).strip()
    if not label:
        raise ValueError("camera name must not be empty")
    shape = str(screen_shape).strip().lower()
    if shape not in ("square", "rectangle", "circle", "ellipse"):
        raise ValueError("screen shape must be square, rectangle, circle, or ellipse")
    height = _positive_number(screen_size, "screen size")
    if shape in ("rectangle", "ellipse"):
        width = _positive_number(
            screen_size if screen_width is None else screen_width, "screen width"
        )
        size = [height, width]
    else:
        size = height
    rows = _positive_count(pixels, "pixels (height)")
    cols = rows if pixels_width is None else _positive_count(pixels_width, "pixels (width)")
    aperture = Aperture(
        shape="circle",
        size=_positive_number(aperture_size, "aperture size"),
        position=[0.0, 0.0, _positive_number(aperture_z, "aperture Z")],
    )
    return Camera.single_pinhole(
        focal_length=_positive_number(focal_length, "focal length"),
        eye_size=_positive_number(eye_size, "eye size"),
        screen_shape=shape,
        screen_size=size,
        pixel_shape=(rows, cols),
        subpixel_resolution=_positive_count(subpixel_resolution, "subpixel resolution"),
        apertures=aperture,
        camera_name=label,
    )


def _screen_boundary(screen, samples: int) -> np.ndarray:
    """Return ``samples`` camera-local points around the detector boundary."""
    if not isinstance(samples, int) or samples < 4:
        raise ValueError("samples must be an integer of at least 4")
    height, width = np.asarray(screen.screen_size, dtype=float)
    if screen.screen_shape in ("circle", "ellipse"):
        angle = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        return np.column_stack(
            [0.5 * width * np.cos(angle), 0.5 * height * np.sin(angle), np.zeros(samples)]
        )

    distance = np.linspace(0.0, 4.0, samples, endpoint=False)
    points = np.empty((samples, 3), dtype=float)
    for index, perimeter in enumerate(distance):
        side = int(perimeter)
        fraction = perimeter - side
        if side == 0:
            x, y = -width / 2 + width * fraction, -height / 2
        elif side == 1:
            x, y = width / 2, -height / 2 + height * fraction
        elif side == 2:
            x, y = width / 2 - width * fraction, height / 2
        else:
            x, y = -width / 2, height / 2 - height * fraction
        points[index] = (x, y, 0.0)
    return points


def sample_view_boundary(
    camera: Camera,
    *,
    eye_index: int = 0,
    distance: float = 1000.0,
    samples: int = 32,
) -> np.ndarray:
    """Sample a cheap, unoccluded field-of-view boundary in world coordinates.

    The result contains one endpoint per detector-boundary sample. Rays start
    at the selected eye and continue away from the screen.  No aperture or wall
    intersection is evaluated, making this suitable for interactive alignment.
    """
    if not isinstance(camera, Camera):
        raise TypeError("camera must be a Camera")
    if not 0 <= eye_index < len(camera.eyes):
        raise IndexError("eye_index is out of range")
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("distance must be a positive finite value")

    eye = np.asarray(camera.eyes[eye_index].position, dtype=float)
    screen_points = _screen_boundary(camera.screen, samples)
    directions = eye[None, :] - screen_points
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    endpoints_camera = eye[None, :] + distance * directions
    return endpoints_camera @ camera.rotation_matrix + camera.camera_position[None, :]


def view_boundary_segments(camera: Camera, **kwargs) -> np.ndarray:
    """Return eye-to-boundary line segments with shape ``(samples, 2, 3)``."""
    eye_index = kwargs.get("eye_index", 0)
    endpoints = sample_view_boundary(camera, **kwargs)
    eye_camera = np.asarray(camera.eyes[eye_index].position, dtype=float)
    eye_world = eye_camera @ camera.rotation_matrix + camera.camera_position
    starts = np.broadcast_to(eye_world, endpoints.shape)
    return np.stack([starts, endpoints], axis=1)
