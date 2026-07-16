"""Eye geometry and ray generation."""
from numbers import Number
from typing import List, Tuple, Union

import numpy as np
from typing_extensions import Literal

from .rays import Rays

Vector2DLike = Union[np.ndarray, List[Number], Tuple[Number, Number]]

class Eye:
    """A single pinhole or concave-lens optical channel of a camera.

    An ``Eye`` is used to calculate the matrices for transportation and
    projection: it converts points already expressed in the camera
    coordinate system into rays landing on the screen (see
    :meth:`camera2eye` and :meth:`calc_rays`). See ``__init__`` below for the
    full parameter and attribute reference.
    """

    def __init__(self,
                 position: Vector2DLike,
                 focal_length: float,
                 eye_type: Literal["pinhole", "concave_lens", 1, 2] = 1,
                 eye_size: Union[float, Vector2DLike] = 0.5,
                 eye_shape: Literal["circle", "ellipse", "rectangle"] = "circle",
                 wavelength_range: Tuple[float, float] = (0.01, 0.1), ):
        """Eye class

        Eye class is used to calculate matrices for transportation and projection.

        Parameters
        ----------
        focal_length : float
            focal length of the pinhole or lens in millimeters
            if eye_type is "pinhole", focal length is the distance between the screen and the pinhole (positive value)
            if eye_type is "concave_lens", focal length is the lens's focal length (negative value)
        position : Tuple[float, float, float]
            3D vector (X_eye, Y_eye, focal_length) from the origin of the camera coordinate system
            If the eye_type is "pinhole", the position of the eye will be set to (X_eye, Y_eye, focal_length) automatically.

        eye_type : Literal["pinhole", "concave_lens", 1, 2], optional (default is 1)
            type of the eye ('pinhole'(1) or 'concave_lens'(2))
        eye_size : float or Iterable[float, float], optional (default is 0.5)
            size of the eye in millimeters (float or tuple of two floats)

            - if eye_shape is "circle", eye_size is the diameter of the eye
            - if eye_shape is "ellipse", eye_size is the semi-major axis a and semi-minor axis b of the eye
            - if eye_shape is "rectangle", eye_size is the height and width of the eye
        eye_shape : Literal["circle", "rectangle"], optional (default is "circle")
            shape of the eye ("circle", "ellipse", or "rectangle") (default is "circle")
        wavelength_range : Tuple[float, float], optional (default is (0.01, 0.1))
            wavelength range of the light in nanometers

        Raises
        ------
        ValueError
            if eye_type is not "pinhole", "concave_lens", 1, or 2
            if eye_shape is not "circle" or "rectangle"
            if eye_size is not float when eye_shape is "circle"
            if eye_size is not float or tuple of two floats when eye_shape is "rectangle"

        Notes
        -----
        Earlier revisions of this docstring described ``T``/``P`` "transportation"
        and "projection" matrix attributes; those were never implemented as
        attributes on this class (position/frame conversion is done directly in
        :meth:`camera2eye` and :meth:`calc_rays` instead), so they have been
        removed from this docstring to avoid documenting nonexistent behavior.

        Examples
        --------
        >>> pinhole = Eye(eye_type="pinhole", position=(5, 0), focal_length=20, eye_size=0.5, eye_shape="circle")
        >>> pinhole.print_settings()
        eye_type: pinhole
        position: [ 5  0 20]
        focal_length: 20
        eye_size: [0.5 0.5]
        eye_shape: circle
        principal_point: [ 5  0 20]
        >>> lens = Eye(eye_type="concave_lens", position=(5, 0), focal_length=-40,
        ...            eye_size=(10, 12), eye_shape="rectangle")
        >>> lens.print_settings()
        eye_type: concave_lens
        position: [5 0 0]
        focal_length: -40
        eye_size: [10 12]
        eye_shape: rectangle
        principal_point: [ 5  0 40]
        """

        if eye_type == "pinhole" or eye_type == 1:
            self._eye_type = "pinhole"
            if focal_length <= 0:
                raise ValueError("when eye_type is 'pinhole', focal_length must be positive")
            self._focal_length = focal_length
            self._position = np.array([position[0], position[1], self._focal_length])
        elif eye_type == "concave_lens" or eye_type == 2:
            self._eye_type = "concave_lens"
            if focal_length >= 0:
                raise ValueError("when eye_type is 'concave_lens', focal_length must be negative")
            self._focal_length = focal_length
            self._position = np.array([position[0], position[1], 0])
        else:
            raise ValueError("eye_type must be 'pinhole(1)' or 'concave_lens(2)'")

        self._principal_point = np.array([position[0], position[1], abs(self._focal_length)])
        if eye_shape in ["circle", "ellipse", "rectangle"]:
            self._eye_shape = eye_shape
        else:
            raise ValueError("eye_shape must be 'circle', 'ellipse', or 'rectangle'")

        if self._eye_shape == "circle":
            if isinstance(eye_size, Number):
                self._eye_size = np.array([eye_size, eye_size])
            else:
                raise ValueError("if eye_shape is 'circle', eye_size must be float")
        else:
            if type(eye_size) in [list, tuple, np.ndarray] and len(eye_size) == 2:
                self._eye_size = np.array(eye_size)
            elif isinstance(eye_size, Number):
                self._eye_size = np.array([eye_size, eye_size])
            else:
                raise ValueError(f"if eye_shape is '{self._eye_shape}', eye_size must be float or array of two floats")

        self._wavelength_range = wavelength_range
        self._camera = None
        self._frozen = False

    def __eq__(self, other):
        """bool: Return ``True`` when two eyes share identical intrinsic settings."""
        if isinstance(other, Eye):
            for k in self.__dict__.keys():
                if k in ("_camera", "_frozen"):
                    continue
                elif not np.all(self.__dict__[k] == other.__dict__[k]):
                    return False
            return True
        else:
            return False

    def camera2eye(self, points_in_camera: np.ndarray):
        """Convert points in camera coordinate to eye coordinate

        Parameters
        ----------
        points_in_camera : np.ndarray
            points in camera coordinate (n, 3)

        Returns
        -------
        points_in_eye : np.ndarray
            points in eye coordinate (n, 3)
        """

        return points_in_camera - self.position.reshape((1, 3))  # (n, 3)

    def calc_rays(self, points_in_camera: np.ndarray, visible: np.ndarray = None, front_only: bool = True):
        """Rays: Project world points through the eye onto the screen plane.

        Parameters
        ----------
        points_in_camera : np.ndarray
            points in eye coordinate (n, 3)
        visible : np.ndarray, optional
            visible points in camera coordinate (n, )
            If visible is None, all points are considered to be visible
        front_only : bool, optional
            If True, discard samples located behind the optical center.

        Returns
        -------
        Rays
            Ray bundle storing axial distance, projected coordinates, zoom factor, and visibility mask.

        Notes
        -----
        Rays are calculated only for points in front of the eye (z > 0)
        They are calculated as follows:
            1. Convert points in camera coordinate to eye coordinate
            2. Get conditions of points in front of the eye (z > 0) and visible
            3. Calculate the center of the projected image on the screen (x/z * f + t_x, y/z * f + t_y)
            4. Calculate the zoom rate of the projected image on the screen (1 + f/z)
        """
        # camera coordinate -> eye coordinate
        points_in_camera = points_in_camera if points_in_camera.ndim == 2 else points_in_camera.reshape((1, 3))
        points_in_eye = self.camera2eye(points_in_camera)  # (n, 3)
        Z = points_in_eye[:, 2]  # (n, )

        visible = np.ones(points_in_camera.shape[0], dtype=bool) if visible is None else visible
        if front_only:
            front_and_visible = (Z > 0) & visible  # (n, ) True if the point is in front of the eye and visible
        else:
            front_and_visible = visible

        XY = np.tile(np.zeros_like(Z) * np.nan, (2, 1)).T
        zoom_rate = np.zeros_like(Z) * np.nan
        XY[front_and_visible] = (-points_in_eye[front_and_visible, :2] / Z[front_and_visible, None] * self.focal_length
                                 + self.principal_point[None, :2])  # (n_front, 2)
        zoom_rate[front_and_visible] = 1 + self.focal_length / Z[front_and_visible]  # (n_front, )

        return Rays(Z=Z, XY=XY, zoom_rate=zoom_rate, front_and_visible=front_and_visible)

    def set_camera(self, camera_obj):
        """None: Register the parent :class:`Camera` that owns this eye.

        Parameters
        ----------
        camera_obj : Camera
            Camera instance providing coordinate transforms for the eye.
        """
        self._camera = camera_obj

    @property
    def frozen(self):
        """bool: Whether this eye's geometry is immutable."""
        return self._frozen

    def freeze(self):
        """Freeze the eye geometry and make its public arrays read-only."""
        if not self._frozen:
            self._position.setflags(write=False)
            self._principal_point.setflags(write=False)
            self._eye_size.setflags(write=False)
            self._frozen = True
        return self

    @property
    def eye_type(self):
        """str: Kind of optical element (``"pinhole"`` or ``"concave_lens"``)."""
        return self._eye_type

    @property
    def eye_size(self):
        """np.ndarray: Aperture extents along the camera ``x`` and ``y`` axes in millimeters."""
        return self._eye_size

    @property
    def eye_shape(self):
        """str: Geometric outline applied when rasterizing the pupil."""
        return self._eye_shape

    @property
    def focal_length(self):
        """float: Signed focal length measured along the optical axis in millimeters."""
        return self._focal_length

    @property
    def position(self):
        """np.ndarray: Eye origin expressed as ``(x, y, z)`` in camera coordinates."""
        return self._position

    @property
    def principal_point(self):
        """np.ndarray: Principal point location ``(x, y, z)`` defining the imaging center."""
        return self._principal_point

    @property
    def camera(self):
        """Camera or None: Parent camera providing transforms, or ``None`` if detached."""
        return self._camera

    @property
    def wavelength_range(self):
        """Tuple[float, float]: Minimum and maximum supported wavelengths in meters."""
        return self._wavelength_range

    def print_settings(self):
        """None: Print the configured eye properties for debugging."""
        print("eye_type:", self.eye_type)
        print("position:", self.position)
        print("focal_length:", self.focal_length)
        print("eye_size:", self.eye_size)
        print("eye_shape:", self.eye_shape)
        print("principal_point:", self.principal_point)
