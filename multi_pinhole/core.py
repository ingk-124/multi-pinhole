# PYTHON CODE
# this code is based on PEP8 style guide
# docstrings are based NumPy style
from dataclasses import dataclass
from numbers import Number
from typing import Tuple, List, Union

import mpl_toolkits.mplot3d.art3d as art3d
# import libraries
import numpy as np
import plotly.graph_objects as go
from stl import mesh
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from numba import njit
from scipy import sparse
from scipy.spatial.transform import Rotation
from typing_extensions import Literal

from utils import stl_utils
from utils.my_stdio import my_tqdm

# TODO: add docstring, type hints, and tests(<- additional, help me copilot!)
# TODO: refactor variable names

# type aliases
# any length vector like object (accepts numpy.ndarray, list, tuple) (including 0D, 1D, 2D, 3D, etc.)
VectorLike = Union[np.ndarray, List[Number], Tuple[Number], Number]

Vector2DLike = Union[np.ndarray, List[Number], Tuple[Number, Number]]
# 3D vector like object (accepts numpy.ndarray, list, tuple)
Vector3DLike = Union[np.ndarray, List[Number], Tuple[Number, Number, Number]]
# Matrix like object (2D array) (accepts numpy.ndarray, list of list, tuple of tuple, etc.)
MatrixLike = Union[np.ndarray, List[List[Number]], Tuple[List[Number]], Tuple[List[Number]], Tuple[Tuple[Number]]]


# Multiple-Pinhole Camera Simulation Code Document
# -----------------------------------------------
# A camera with multiple eyes (pinhole or concave lens) is called a "multiple-pinhole camera."
#
# 1. Coordinate Systems
# There are four coordinate systems; the world, camera, pinhole, and image coordinate.
#
# 　## 1.1. The World Coordinate System
# The world coordinate system (x,y,z) is the primary coordinate system of this code.
# Every object of this simulation is in the world coordinate system.
# This coordinate system is defined as Cartesian and limited with (x_min,x_max), (y_min,y_max), and (z_min,z_max).
# The coordinates are also expressed in cylindrical or torus coordinates.
# Radii or angles normalize the axes of the systems.
# Which coordinate system should be used is specified by the "coordinate" argument.

# ### 1.1.1. The Cylindrical Coordinate System
# The cylindrical coordinate system (r, theta, z) is set so that the z-axis coincides with the z-axis of the Cartesian coordinate system.
# The origin is the center of gravity of a cylinder of radius a and height 2*h, and the r- and z-axes are normalized by a and h, respectively.
# The theta is 0<=theta<2pi from the x-axis clockwise. (r=sqrt(x\^2+y^2)/a, theta=arctan(y/x))

# ### 1.1.2. The Torus Coordinate System
# The torus coordinate system (r, theta, phi) is set on a torus obtained by rotating around the z-axis a circle of radius a_0 whose center is on the xy-plane and R_0 away from the z-axis.
# The radii R_0 and a_0 are called "major" and "minor" radii. The ratio R_0/a_0 is known as the "aspect ratio."
# "Poloidal" angle theta represents rotation around the tube, whereas "toroidal" angle phi represents rotation around the torus' axis of revolution.
# The circle of radius a_0 is called a "poloidal cross-section."
# The path of the center of the circle is the toroidal axis and is on the equatorial plane (the same as xy-plane).
# In each poloidal cross-section, r and theta are 2D polar coordinates, where the origin is at the toroidal axis.
# The toroidal and poloidal angles are referenced to the x-axis and xy-plane inward of the torus, respectively.
# The r-axis is normalized by radius a_0.
# (phi=arctan(y/x), R=sqrt(x\^2+y^2), theta=arctan((R-R_0)/z), r=sqrt((R-R_0)\^2+z^2)/a_0)

# ## 1.2. The Camera Coordinate System
# The camera has a screen, and the main optical axis is defined to be perpendicular to and point out the screen. The camera coordinate system (X, Y, Z) is right-handed Cartesian, and the Z axis is the main optical axis. The origin of the coordinate system is the center of the screen. The X- and Y-axes face horizontally to the right and vertically downward, respectively.
# The camera position and orientation are specified by three 3D vectors in the world coordinate system (Cartesian); "position," "look," and "right." The "position" is the position in the world coordinates of the origin of the camera coordinates. The "look" and "right" are vectors that point from the origin of the camera coordinate system toward the main optical axis and the X-axis, respectively.
#
# ## 1.3. The Pinhole Coordinate System
# The camera has multiple "eyes," pinholes, or concave lenses. These eyes are placed in different positions and have different focal lengths. The eye's position is determined by 2D vectors (X_h, Y_h) in the camera coordinate system, and the focal length f (must be positive value). The flag "eye_type" is a flag to specify eye type, "pinhole," or "concave lens."
# The pinhole coordinate system (X', Y', Z') is set so that the origin of the coordinate system is translated parallel to the principal point of the eye. (X'=X-X_h, Y'=Y-Y_h, Z'=Z-f)
#
# ### 1.3.1. Pinhole
# The focal length is the distance between the screen and the pinhole.
# The eye position and principal point are both (X_h, Y_h, f).
#
# ### 1.3.2. Concave Lens
# The focal length is the lens's focal length, and the screen is the surface of the lens.
# The eye position and principal point are (X_h, Y_h, 0) and (X_h, Y_h, f).
#
# ## 1.4. The Image Coordinate System
# The image coordinate system (u,v) is a 2D Cartesian coordinate on the screen, originating at the screen's upper left.
# The u- and v-axis represent vertical and horizontal, respectively.
# In the image coordinate, the screen's center point is expressed as (u_c,v_c).
# (Thus, the origin of the image coordinate is (-u_c,-v_c) in the camera coordinate system). (u=Y+u_c, v=X+v_c)

# --------------------------------

# class tree
# Camera
#  - Eye
#    - Pinhole
#    - ConcaveLens
#  - Screen
#  - Aperture

# MARK: rays class
@dataclass(frozen=True)
class Rays:
    """Rays class

    Parameters
    ----------
    Z : np.ndarray
        distance from eye to light source along the main optical axis (n, )
    XY : np.ndarray
        position of the light spot on the screen (n, 2)
    zoom_rate : np.ndarray
        zoom rate of the light spot on the screen (n, )
        It expressed as 1 + f / Z.
    front_and_visible : np.ndarray
        boolean array (n, ) True if the point is in front of the eye and visible
    """

    Z: np.ndarray
    XY: np.ndarray
    zoom_rate: np.ndarray
    front_and_visible: np.ndarray

    @property
    def n(self):
        """int: Total number of sampled rays contained in this instance."""
        return self.Z.size

    @property
    def n_visible(self):
        """int: Count of rays that are both in front of the eye and marked visible."""
        return self.front_and_visible.nonzero()[0].size

    # allow indexing and slicing
    def __getitem__(self, key):
        # for slicing
        return Rays(Z=self.Z[key],
                    XY=self.XY[key],
                    zoom_rate=self.zoom_rate[key],
                    front_and_visible=self.front_and_visible[key])

    def __len__(self):
        return self.n

# MARK: Eye class
# --------------------------------
# Eye class is used to calculate matrices for transportation and projection
# Eye class has two attributes; transportation matrix T and projection matrix P
# transportation matrix T is used to transport 3D points in the camera coordinate system to 3D points in the coordinate system of the pinhole
# projection matrix P is used to project 3D points in the coordinate system of the pinhole to 2D points in the screen coordinate system
class Eye:
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

        Attributes
        ----------
        T : numpy.ndarray
            transportation matrix
        P : numpy.ndarray
            projection matrix

        Raises
        ------
        ValueError
            if eye_type is not "pinhole", "concave_lens", 1, or 2
            if eye_shape is not "circle" or "rectangle"
            if eye_size is not float when eye_shape is "circle"
            if eye_size is not float or tuple of two floats when eye_shape is "rectangle"

        Examples
        --------
        >>> pinhole = Eye(eye_type="pinhole", position=(5, 0), focal_length=20, eye_size=0.5, eye_shape="circle")
        >>> pinhole.print_settings()
        eye_type: pinhole
        position: (5, 0, 0)
        focal_length: 20
        eye_size: 0.5
        eye_shape: circle
        principal_point: (5, 0, 20)
        >>> pinhole.T
        array([[ 1.,  0.,  0., -5.],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  1., -20.],
                [ 0.,  0.,  0.,  1.]])
        >>> pinhole.P
        array([[ -20.,    0.,    0.,    0.],
                [   0.,  -20.,    0.,    0.],
                [   0.,    0.,    1.,    0.]])
        >>> lens = Eye(eye_type="concave_lens", position=(5, 0), focal_length=40,
        >>>             eye_size=(10, 12), eye_shape="rectangle")
        >>> lens.print_settings()
        eye_type: concave_lens
        position: (5, 0, 0)
        focal_length: 40
        eye_size: (10, 12)
        eye_shape: rectangle
        principal_point: (5, 0, 40)
        >>> lens.T
        array([[ 1.,  0.,  0., -5.],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  1.,  -40.],
                [ 0.,  0.,  0.,  1.]])
        >>> lens.P
        array([[ 40.,    0.,    0.,    0.],
                [   0.,  40.,    0.,    0.],
                [   0.,    0.,    1.,    0.]])
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

    def __eq__(self, other):
        """bool: Return ``True`` when two eyes share identical intrinsic settings."""
        if isinstance(other, Eye):
            for k in self.__dict__.keys():
                if k == "_camera":
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


# Aperture class
# --------------------------------
# Aperture class is used to create an aperture object.
# Aperture object is given as geometry information or stl model.
# if geometry information is given, stl model is generated automatically.
# if you set stl model, geometry is not used
class Aperture:
    def __init__(self,
                 shape: Literal["circle", "ellipse", "rectangle"] = None,
                 size: Union[Number, Vector2DLike] = None,
                 position: Vector3DLike = None,
                 direction: Vector3DLike = None,
                 stl_model: mesh.Mesh = None,
                 **stl_args):
        """Create an aperture object

        Parameters
        ----------
        position : Vector3DLike
            Aperture position in the camera coordinate system (origin is the center of the screen)
        direction : Vector3DLike
            Aperture direction in the camera coordinate system (default: [0, 0, 1])
        shape : Literal["circle", "ellipse", "rectangle"]
            Aperture shape
        size : Union[Number, Vector2DLike]
            Aperture size
            If shape is "circle", size must be float (radius)
            If shape is "ellipse", size must be array of two floats (semi-major axis a and semi-minor axis b)
            If shape is "rectangle", size must be array of two floats (height, width)
        """

        self._stl_model = None
        self._position = np.array(position) if position is not None else np.array([0, 0, 0])
        self._direction = np.array(direction) if direction is not None else np.array([0, 0, 1])

        if isinstance(stl_model, mesh.Mesh):
            self._shape = "stl"
            self._size = None
            self._stl_model = stl_utils.copy_model(stl_model)
            self._stl_model.translate(self._position)
        else:
            if (shape is None) and (size is None):
                pass
            else:
                self._shape, self._size = stl_utils.shape_check(shape, size)
                self.set_model(**stl_args)

        # self._stl_model = stl_model

    def __eq__(self, other):
        """bool: Compare aperture geometry, ignoring linked STL mesh objects."""
        if isinstance(other, Aperture):
            for k in self.__dict__.keys():
                if k == "_stl_model":
                    continue
                elif not np.all(self.__dict__[k] == other.__dict__[k]):
                    return False
            return True
        else:
            return False

    def set_model(self, resolution: int = 20, max_size: Union[Number, Vector2DLike] = None):
        """Set stl model

        Parameters
        ----------
        resolution : int, optional (default is 20)
            Resolution of the stl model
        max_size : Union[Number, Vector2DLike], optional (default is None)
            Maximum size of the stl model
            If max_size is None, the size of the stl model is 1.5 times the size of the aperture

        Returns
        -------
        self : Aperture
        """
        if self._shape == "stl":
            return self

        self._stl_model = stl_utils.generate_aperture_stl(shape=self._shape, size=self._size,
                                                          resolution=resolution, max_size=max_size)
        self._stl_model.translate(self._position)
        # self._stl_model.rotate([0, 0, 1], np.pi / 2)
        return self

    #  properties
    @property
    def position(self):
        """np.ndarray: Aperture center in camera coordinates as ``(x, y, z)``."""
        return self._position

    @property
    def direction(self):
        """np.ndarray: Unit vector indicating aperture normal in camera space."""
        return self._direction

    @property
    def shape(self):
        """str: Shape keyword such as ``"circle"``, ``"ellipse"``, ``"rectangle"`` or ``"stl"``."""
        return self._shape

    @property
    def size(self):
        """np.ndarray or None: Characteristic dimensions of the aperture opening in millimeters."""
        return self._size

    @property
    def stl_model(self):
        """mesh.Mesh or None: Triangulated aperture surface when generated from STL."""
        return self._stl_model

    #  methods
    def print_info(self):
        """None: Print the aperture's spatial configuration and dimensions."""
        print("position:", self.position)
        print("shape:", self.shape)
        print("size:", self.size)


# Screen class
# --------------------------------
# Screen class is used to create a screen object
# the image coordinate system is defined on the screen as follows:
#   origin: upper left corner of the screen
#   u-axis: vertical direction and points to the bottom of the screen (same direction as Y-axis of the camera coordinate system)
#   v-axis: horizontal direction and points to the right of the screen (same direction as X-axis of the camera coordinate system)
#   caution: the order of axes is different from the camera coordinate system (Y-axis -> u-axis, X-axis -> v-axis)
# the center of screen is at the origin of the camera coordinate system, so the position is always [0, 0, 0]
# screen shape is circle, ellipse or rectangle
# image area is defined as the rectangle circumscribed by the screen
# image size is the size of the image area and the unit is mm
# pixels are defined on the image area and the number of pixels is given as (U_p, V_p) in the image coordinate system
# the size of pixel is calculated by image size / number of pixels
# To simulate more realistic images, subpixel resolution can be used
# pixels and subpixels are discretized and noted as (i, j) and (i', j') respectively
class Screen:
    """Screen class

    Screen class is used to create a screen object
    """

    def __init__(self,
                 screen_shape: Literal["circle", "ellipse", "square", "rectangle"] = "square",
                 screen_size: Union[Number, Vector2DLike] = 10,
                 pixel_shape: Tuple[int, int] = (100, 100),
                 subpixel_resolution: int = 1):
        """Create a screen object

        Parameters
        ----------
        screen_shape : Literal["circle", "ellipse", "square", "rectangle"]
            Screen shape (default: "square")
        screen_size : Union[Number, Vector2DLike]
            Screen size (default: 10)
            If screen_shape is "circle" or "square", screen_size must be float (diameter or width)
            If screen_shape is "ellipse" or "rectangle", screen_size must be array of two floats (height, width)
        pixel_shape : Vector2DLike
            Pixel shape (U_p, V_p)
        N_pixel : int
            Number of pixels (U_p * V_p)
        subpixel_resolution : int, optional
            Subpixel resolution, by default 1

        Raises
        ------
        ValueError
            If screen_shape is not "circle", "ellipse" or "rectangle"
        ValueError
            If screen_size is not positive number or array of two positive numbers
        ValueError
            If pixel_shape is not positive integer
        ValueError
            If subpixel_resolution is not positive integer


        Notes
        -----
        The image coordinate system is defined on the screen as follows:
            origin: upper left corner of the screen
            u-axis: vertical direction and points to the bottom of the screen (same direction as Y-axis of the camera coordinate system)
            v-axis: horizontal direction and points to the right of the screen (same direction as X-axis of the camera coordinate system)
            caution: the order of axes is different from the camera coordinate system (Y-axis -> u-axis, X-axis -> v-axis)
        The center of screen is at the origin of the camera coordinate system, so the position is always [0, 0, 0]
        Screen shape is circle, ellipse or rectangle
        Image area is defined as the rectangle circumscribed by the screen
        Image size is the size of the image area and the unit is mm
        Pixels are defined on the image area and the number of pixels is given as (U_p, V_p) in the image coordinate system
        The size of pixel is calculated by image size / number of pixels
        To simulate more realistic images, subpixel resolution can be used
        Pixels and subpixels are discretized and noted as (i, j) and (i', j') respectively

        Examples
        --------
        >>> screen = Screen(screen_shape="circle",screen_size=100.0,pixel_shape=[100,50])
        screen_size=screen.print_settings(),pixel_shape=)
        screen_shape: circle
        screen_size: 100.0
        pixel_shape: [100 50]
        subpixel_resolution: 1
        pixel_size: [1. 1.]
        subpixel_size: [1. 1.]
        >>> screen.subpixel_resolution=4
        >>> screen.print_settings()
        screen_shape: circle
        screen_size: 100.0
        pixel_shape: [100 50]
        subpixel_resolution: 4
        pixel_size: [1. 1.]
        subpixel_size: [0.25 0.25]
        """

        # set the screen shape and size
        self._screen_shape, self._screen_size = stl_utils.shape_check(screen_shape, screen_size)

        # set the pixel shape, pixel size and pixel position
        self._pixel_shape = np.array(pixel_shape, dtype=int)
        self._N_pixel = self._pixel_shape[0] * self._pixel_shape[1]
        self._pixel_size = self._screen_size / self._pixel_shape
        self._A_pixel = self._pixel_size[0] * self._pixel_size[1]
        self._pixel_position = self.positions(pixel_shape=self._pixel_shape, pixel_size=self._pixel_size)

        self._pixel_image_size = (self._pixel_shape[0], self._pixel_shape[1])

        # set the image (masked array, mask is True if the pixel is outside the screen)
        self._pixel_image_mask = self.image_mask(self._pixel_position)

        # variables related to subpixel resolution
        self._subpixel_resolution = None
        self._subpixel_shape = None
        self._N_subpixel = None
        self._subpixel_size = None
        self._subpixel_image_size = None
        self._subpixel_image_mask = None
        self._subpixel_position = None
        self._transform_matrix = None

        self.subpixel_resolution = subpixel_resolution

    def __eq__(self, other):
        """bool: Check if two screens share identical discretisation parameters."""
        if isinstance(other, Screen):
            for k in self.__dict__.keys():
                if isinstance(self.__dict__[k], sparse.spmatrix):
                    if not np.all(self.__dict__[k].data == other.__dict__[k].data):
                        return False
                else:
                    if not np.all(self.__dict__[k] == other.__dict__[k]):
                        return False
            return True

        else:
            return False

    #  properties
    @property
    def screen_shape(self):
        """str: Geometric outline of the active display surface."""
        return self._screen_shape

    @property
    def screen_size(self):
        """np.ndarray: Physical height and width of the screen in millimeters."""
        return self._screen_size

    @property
    def pixel_shape(self):
        """np.ndarray: Count of pixels along the ``u`` and ``v`` axes."""
        return self._pixel_shape

    @property
    def N_pixel(self):
        """int: Total number of discrete pixels on the screen."""
        return self._N_pixel

    @property
    def pixel_size(self):
        """np.ndarray: Pixel pitch along ``u`` and ``v`` directions in millimeters."""
        return self._pixel_size

    @property
    def A_pixel(self):
        """float: Pixel area in square millimeters."""
        return self._A_pixel

    @property
    def pixel_position(self):
        """np.ndarray: Coordinates of each pixel center in ``(u, v)`` order."""
        return self._pixel_position

    @property
    def subpixel_resolution(self):
        """int: Number of sub-divisions applied per pixel edge."""
        return self._subpixel_resolution

    @subpixel_resolution.setter
    def subpixel_resolution(self, subpixel_resolution):
        """None: Update the subpixel refinement factor and recompute caches.

        Parameters
        ----------
        subpixel_resolution : int
            Positive integer denoting how many subpixels subdivide each pixel axis.
        """
        # set the subpixel resolution
        if not isinstance(subpixel_resolution, int) or subpixel_resolution < 1:
            raise ValueError("subpixel_resolution must be integer and larger than 1")
        self._subpixel_resolution = subpixel_resolution
        self._set_variables()

    @property
    def subpixel_shape(self):
        """np.ndarray: Subpixel lattice dimensions in ``(u, v)`` order."""
        return self._subpixel_shape

    @property
    def N_subpixel(self):
        """int: Total number of subpixels composing the discretised screen."""
        return self._N_subpixel

    @property
    def subpixel_size(self):
        """np.ndarray: Size of each subpixel in millimeters along ``u`` and ``v``."""
        return self._subpixel_size

    @property
    def A_subpixel(self):
        """float: Subpixel area in square millimeters."""
        return self._A_subpixel

    @property
    def subpixel_position(self):
        """np.ndarray: Center coordinates for all subpixels in ``(u, v)`` order."""
        return self._subpixel_position

    @property
    def pixel_image_mask(self):
        """np.ndarray: Boolean mask indicating pixels outside the active region."""
        return self._pixel_image_mask

    @property
    def subpixel_image_mask(self):
        """np.ndarray: Boolean mask identifying subpixels clipped by the screen shape."""
        return self._subpixel_image_mask

    @property
    def transform_matrix(self):
        """sparse.csr_matrix or None: Mapping from subpixels to their parent pixels."""
        return self._transform_matrix

    #  methods
    def _set_variables(self):
        """None: Populate cached geometry derived from the current subpixel resolution."""
        # set the subpixel shape, subpixel size and subpixel position
        self._subpixel_shape = self._pixel_shape * self._subpixel_resolution
        self._N_subpixel = self._subpixel_shape[0] * self._subpixel_shape[1]
        self._subpixel_size = self._screen_size / self._subpixel_shape
        self._subpixel_position = self.positions(pixel_shape=self._subpixel_shape, pixel_size=self._subpixel_size)
        self._A_subpixel = self._subpixel_size[0] * self._subpixel_size[1]

        self._subpixel_image_mask = self.image_mask(self._subpixel_position)
        indices = np.arange(self._N_subpixel
                            ).reshape(self._pixel_shape[0], self._subpixel_resolution,
                                      self._pixel_shape[1], self._subpixel_resolution).transpose(0, 2, 1, 3).ravel()
        indptr = np.arange(self.N_pixel + 1) * self._subpixel_resolution ** 2
        self._transform_matrix = sparse.csr_matrix((np.ones(self._N_subpixel), indices, indptr),
                                                   shape=(self._N_pixel, self._N_subpixel))

    def positions(self, pixel_shape, pixel_size):
        """Calculate the center (u, v) of each pixel

        Parameters
        ----------
        pixel_shape : Tuple[int, int]
            shape of the pixel (U_p, V_p)
        pixel_size : Tuple[float, float]
            size of the pixel (d_u, d_v)

        Returns
        -------
        uv : np.ndarray
            center of each pixel (u, v) (shape: (U_p * V_p, 2))
        """

        u_axis = np.linspace(pixel_size[0] / 2, self._screen_size[0] - pixel_size[0] / 2, pixel_shape[0])
        v_axis = np.linspace(pixel_size[1] / 2, self._screen_size[1] - pixel_size[1] / 2, pixel_shape[1])

        u, v = np.meshgrid(u_axis, v_axis, indexing="ij")
        return np.stack([u, v], axis=-1).reshape((-1, 2))

    def image_mask(self, position: np.ndarray):
        """Set the mask of the image

        Parameters
        ----------
        position : np.ndarray
            position of each pixel or subpixel

        Returns
        -------
        mask : np.ndarray
            mask of the image

        Notes
        -----
        If screen shape is rectangle, no pixels are masked.
        If screen shape is circle or ellipse, pixels outside the screen are masked.
        """
        # set the mask of the image
        # outside the screen is masked
        # position of each pixel and subpixel can be loaded from self._pixel_position and self._subpixel_position
        if self._screen_shape == "rectangle":
            return False
        else:
            mask = np.linalg.norm((position - self._screen_size / 2) / (self._screen_size / 2), axis=1) > 1
            return mask

    def cosine(self, eye: Eye):
        """Calculate cosine of each pixel

        Parameters
        ----------
        eye : Eye object
            eye projecting the rays to the screen

        Returns
        -------
        cosine : np.ndarray
            cosine of each pixel (shape: (U_p * V_p, ))

        Notes
        -----
        Cosine is calculated as:
            1. Calculate the distance between the center of each pixel and the eye

        """

        uv = self.subpixel_position - (eye.position[:2][::-1] + self._screen_size / 2)  # (U_p * V_p, 2)
        tangent = np.linalg.norm(uv, axis=-1) / eye.focal_length  # (U_p * V_p, )
        return 1 / np.sqrt(1 + tangent ** 2)  # (U_p * V_p, )

    def etendue_per_subpixel(self, eye: Eye):
        """Calculate etendue of each subpixel

        Parameters
        ----------
        eye : Eye object
            eye projecting the rays to the screen

        Returns
        -------
        etendue_per_subpixel : np.ndarray
            etendue of each subpixel (shape: (U_p * V_p, ))

        Notes
        -----
        Etendue `G_subpix` is expressed as:
            G_subpix = A_subpix * cos(theta)**4 / 4*pi
        """
        return self._A_subpixel * (self.cosine(eye) ** 4) / (4 * np.pi)  # (U_p * V_p, )

    def ray2image_grid(self, eye: Eye, rays: Rays, verbose=0):
        """Convert rays to image vectors (grid-localized; consistent with ray2image)

        Parameters
        ----------
        eye : Eye object
            eye projecting the rays to the screen
        rays : Rays object
            rays from the light source to the eye
        verbose : int, optional (default is 0)
            verbose level for parallel calculation
        parallel : int, optional (default is 0)
            number of parallel processes for parallel calculation (0: no parallel calculation)

        Returns
        -------
        image_vectors : sparse.csr_matrix
            Spot position matrix (shape: (N_subpixel, n)) in compressed sparse row format.
        """
        if rays.n == 0:
            # if no rays, return empty matrix (N_subpixel, 0)
            return sparse.csc_matrix((self.N_subpixel, 0), dtype=float)

        uv = self.xy2uv(rays.XY)  # (n, 2) image coords
        spot_size = eye.eye_size[None, :] * rays.zoom_rate[:, None]  # (n, 2)
        half = 0.5 * spot_size

        # subpixel axis (must match positions())
        U_sub, V_sub = int(self._subpixel_shape[0]), int(self._subpixel_shape[1])  # number of sub-pixels
        du, dv = float(self._subpixel_size[0]), float(self._subpixel_size[1])  # size of sub-pixels
        u_axis = np.linspace(du * 0.5, self._screen_size[0] - du * 0.5, U_sub)  # u positions of sub-pixels (U_sub,)
        v_axis = np.linspace(dv * 0.5, self._screen_size[1] - dv * 0.5, V_sub)  # v positions of sub-pixels (V_sub,)
        u_min, u_max = u_axis[0], u_axis[-1]
        v_min, v_max = v_axis[0], v_axis[-1]
        # AABB means "Axis-Aligned Bounding Box"
        # pre-clip rays whose spot AABB doesn't touch the screen rect (same effect as原実装のNaN→False)

        # +----> (v-axis)
        # |
        # v (u-axis)
        #
        # uv[:, 1] - half[:, 1]         uv[:, 1] + half[:, 1]
        # |                             |
        # +-------- _..-===-.._ --------+-- uv[:, 0] - half[:, 0]
        # |     .~˙             ˙~.     |
        # |  .'                     '.  |
        # | /                         \ |
        # |∫                           l|
        # ├·             + <-center    ·┤   u = uv[:, 0], v = uv[:, 1]
        # |l                   (u,v)   ∫|   width = half[:, 0] * 2, height = half[:, 1] * 2
        # | \                         / |
        # |  '.                     .'  | <- Bounding Box
        # |     ˙~.             .~˙     |
        # +-------- ¯''-===-''¯ --------+-- uv[:, 0] + half[:, 0]
        #

        inside_screen = ((uv[:, 0] + half[:, 0] >= u_min) &  # u + half_width >= u_min -> left side of screen
                         (uv[:, 0] - half[:, 0] <= u_max) &  # u - half_width <= u_max -> right side of screen
                         (uv[:, 1] + half[:, 1] >= v_min) &  # v + half_height >= v_min -> top side of screen
                         (uv[:, 1] - half[:, 1] <= v_max))  # v - half_height <= v_max -> bottom side of screen

        if not np.any(inside_screen):
            # no spots on screen
            return sparse.csc_matrix((self.N_subpixel, rays.n), dtype=float)

        use_ellipse = eye.eye_shape in ("circle", "ellipse")

        if not (use_ellipse or eye.eye_shape == "rectangle"):
            raise ValueError("eye_shape must be 'circle', 'ellipse', or 'rectangle'")

        u_center, v_center = uv[:, 0], uv[:, 1]
        a_u, a_v = half[:, 0], half[:, 1]
        u_low = np.clip(u_center - a_u, u_min, u_max)
        u_high = np.clip(u_center + a_u, u_min, u_max)
        v_low = np.clip(v_center - a_v, v_min, v_max)
        v_high = np.clip(v_center + a_v, v_min, v_max)

        valid = np.nonzero(inside_screen)[0]

        i_min = np.clip(np.floor(u_low / du - 0.5), 0, U_sub - 1).astype(np.int32)
        i_max = np.clip(np.ceil(u_high / du - 0.5), 0, U_sub - 1).astype(np.int32)
        j_min = np.clip(np.floor(v_low / dv - 0.5), 0, V_sub - 1).astype(np.int32)
        j_max = np.clip(np.ceil(v_high / dv - 0.5), 0, V_sub - 1).astype(np.int32)

        if use_ellipse:
            @njit(cache=True)
            def indexer(u_ax, v_ax, uc, vc, au, av, i_min_r, j_min_r, _V_sub):
                u_n_sq = ((u_ax - uc) / au) ** 2  # (I_r,)
                v_n_sq = ((v_ax - vc) / av) ** 2  # (J_r,)
                inside = (u_n_sq[:, None] + v_n_sq[None, :]) < 1.0
                if not inside.any():
                    return np.empty((0,), dtype=np.int32)
                ii, jj = np.nonzero(inside)
                return ((i_min_r + ii) * _V_sub + (j_min_r + jj)).astype(np.int32)

        else:
            @njit(cache=True)
            def indexer(u_ax, v_ax, uc, vc, au, av, i_min_r, j_min_r, _V_sub):
                u_n = np.abs((u_ax - uc) / au)  # (I_r,)
                v_n = np.abs((v_ax - vc) / av)  # (J_r,)
                u_cond = u_n < 0.5
                v_cond = v_n < 0.5
                inside = u_cond[:, None] & v_cond[None, :]
                if not inside.any():
                    return np.empty((0,), dtype=np.int32)
                ii, jj = np.nonzero(inside)
                return ((i_min_r + ii) * _V_sub + (j_min_r + jj)).astype(np.int32)

        out_pix = [indexer(u_axis[i_min[r]:i_max[r] + 1],
                           v_axis[j_min[r]:j_max[r] + 1],
                           uv[r, 0], uv[r, 1],
                           half[r, 0], half[r, 1],
                           i_min[r], j_min[r], V_sub) for r in
                   my_tqdm(valid, desc="ray to image", disable=verbose <= 0)]
        pix_sizes = [p.size for p in out_pix]
        pixel_indices = np.concatenate(out_pix)
        indptr = np.zeros(rays.n + 1, dtype=np.int32)
        indptr[valid + 1] = np.cumsum(pix_sizes)
        last_filled = np.nonzero(indptr)[0][-1]
        indptr[last_filled + 1:] = indptr[last_filled]
        etendue_per_ray = (1.0 / (rays.zoom_rate * (rays.Z ** 2))).astype(np.float32)
        etendue_per_subpixel = self.etendue_per_subpixel(eye).astype(np.float32)
        data = etendue_per_subpixel[pixel_indices]
        mat = sparse.csc_matrix((data, pixel_indices, indptr), shape=(self.N_subpixel, rays.n)).tocsr()
        mat.data = mat.data * etendue_per_ray[mat.indices]
        return mat  # (N_subpixel, n) csr matrix

    def xy2uv(self, xy: np.ndarray):
        """Convert uv vectors in the camera coordinate to the image coordinate

        Parameters
        ----------
        xy : np.ndarray
            xy vectors in the camera coordinate (n, 2)

        Returns
        -------
        uv : np.ndarray
            uv vectors in the image coordinate (n, 2)
        """
        if xy.ndim == 1:
            return xy[::-1] + self._screen_size / 2
        elif xy.ndim == 2:
            return xy[:, ::-1] + self._screen_size / 2
        else:
            raise ValueError("xy must be 1D or 2D array")

    def uv2subpixel_index(self, light_points: np.ndarray, intensity: np.ndarray):
        """Tuple[np.ndarray, np.ndarray]: Convert image-plane samples into subpixel indices.

        Parameters
        ----------
        light_points : np.ndarray
            Light points (u, v) in the image coordinate system (shape: (N, 2))
        intensity : np.ndarray
            Corresponding light intensity values with shape ``(N,)`` or ``(N, C)``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Subpixel indices ``(M, 2)`` inside the screen and the filtered intensities.
        """

        # calculate subpixel index
        subpixel_indices = np.floor(
            (light_points + self._screen_size / 2) / self._pixel_size * self._subpixel_resolution)

        # if subpixel index is out of range, remove it and its intensity
        mask = np.all([subpixel_indices[:, 0] >= 0, subpixel_indices[:, 0] < self._subpixel_shape[0],
                       subpixel_indices[:, 1] >= 0, subpixel_indices[:, 1] < self._subpixel_shape[1]], axis=0)
        return subpixel_indices[mask], intensity[mask]

    def subpixel_to_pixel(self, subpixel_image: np.ndarray = None) -> np.ndarray:
        """Convert subpixel image to pixel image

        Parameters
        ----------
        subpixel_image : np.ndarray, optional (default: None)
            Subpixel image
            If image is color image, shape must be (N_subpixel, 3) or (subpixel_shape[0], subpixel_shape[1], 3)
            If image is grayscale image, shape must be (N_subpixel, ) or (subpixel_shape[0], subpixel_shape[1])
            If None, self.subpixel_image is used.

        Returns
        -------
        pixel_image : np.ndarray
            Pixel image
            If image is color image, shape is (pixel_shape[0], pixel_shape[1], 3)
            If image is grayscale image, shape is (pixel_shape[0], pixel_shape[1])
        """
        # check image dimensions
        if subpixel_image is None:
            subpixel_image = self._subpixel_image
        if subpixel_image.shape[0] == self._N_subpixel:
            pass
        elif subpixel_image.shape == self._subpixel_shape:
            subpixel_image = subpixel_image.flatten()
        else:
            raise ValueError(f"image size must be {self._subpixel_shape}")

        # Convert subpixel image to pixel image by averaging subpixels
        # pixel_image = np.zeros((*self._pixel_shape, 3) if color_flag else self._pixel_shape)
        # for i in range(self._subpixel_resolution):
        #     for j in range(self._subpixel_resolution):
        #         pixel_image += subpixel_image[i::self._subpixel_resolution, j::self._subpixel_resolution]
        # pixel_image /= self._subpixel_resolution ** 2
        # return pixel_image
        return self._transform_matrix.dot(subpixel_image)

    def show_image(self, image: np.ndarray = None, ax: plt.Axes = None,
                   block: bool = True, pixel_image: bool = False, pm: bool = False,
                   colorbar: bool = True, masked: bool = False, **kwargs):
        """Show image

        Parameters
        ----------
        image : np.ndarray, optional
            Image to show
        ax : plt.Axes, optional
            Axes to show image, by default None (new figure is created)
        block : bool, optional
            If True, the function does not return until the window is closed, by default True
        pixel_image : bool, optional
            If True, given image is converted to pixel image, by default False
        pm : bool, optional
            If True, the image is shown in the range [-vmax, vmax], by default False
        colorbar : bool, optional
            If True, colorbar is shown, by default True
        masked : bool, optional
            If True, masked pixels are shown in gray, by default False
        **kwargs : dict
            Keyword arguments for plt.pcolormesh

        Returns
        -------
        ax : plt.Axes
            Axes to show image
        """

        image = image.toarray() if sparse.issparse(image) else image
        if pixel_image:
            image = self.subpixel_to_pixel(image) if image.size == self._N_subpixel else image

        if image.size == self._N_pixel:
            UV = self.pixel_position.T.reshape((2, *self.pixel_shape))
            image_shape = self.pixel_shape
            mask = self._pixel_image_mask
        elif image.size == self._N_subpixel:
            UV = self.subpixel_position.T.reshape((2, *self.subpixel_shape))
            image_shape = self.subpixel_shape
            mask = self._subpixel_image_mask
        else:
            raise ValueError(f"image size must be {self._pixel_shape} (pixel) or {self._subpixel_shape} (subpixel)")

        image = np.array(image).reshape(image_shape)

        if masked:
            image = np.ma.masked_array(image, mask=mask.reshape(image_shape))

        ax = plt.subplot() if ax is None else ax
        vmax = np.max(np.abs(image))
        kwargs.setdefault("vmin", -vmax if pm else 0)
        kwargs.setdefault("vmax", vmax)
        _ = ax.pcolormesh(UV[1], UV[0], image, linewidth=0, rasterized=True, **kwargs)
        ax.set_aspect(1)
        ax.set_xlim(0, self._screen_size[1])
        ax.set_ylim(self._screen_size[0], 0)
        if colorbar:
            ax.figure.colorbar(_, ax=ax)
        ax.set_xlabel("v (mm)")
        ax.set_ylabel("u (mm)")

        plt.show(block=block)
        return ax

    def print_settings(self) -> None:
        """None: Display key screen discretisation parameters."""
        print("screen_shape: {}".format(self._screen_shape))
        print("pixel_shape: {}".format(self._pixel_shape))
        print("subpixel_resolution: {}".format(self._subpixel_resolution))
        print("color_image: {}".format(self._color_image))
        print("pixel_size: {}".format(self._pixel_size))
        print("subpixel_size: {}".format(self._subpixel_size))


# Camera class
# --------------------------------
# Camera class is used to create a camera object
# camera position, look, up and right are given in the world coordinate system
# camera right or up direction are only one of them is specified.
# if both are given, ValueError is raised
# if both are not given, they are calculated automatically.
# if the look vector and z-axis of the world coordinate system are not parallel:
#   right is calculated by cross product of look vector and z-axis of the world coordinate system
# if the look vector and z-axis of the world coordinate system are parallel:
#   right is set to be the x-axis
class Camera:
    def __init__(self,
                 eyes: List[Eye],
                 apertures: Union[Aperture, List[Aperture]],
                 screen: Screen,
                 camera_position: Tuple[float, float, float],
                 rotation_matrix: np.ndarray = None,
                 camera_name: str = None):
        """Camera
        
        Parameters
        ----------
        eyes : List[Eye]
            list of eyes
        apertures : Aperture or List[Aperture]
            aperture or list of apertures
        screen : Screen
            screen
        camera_position : Tuple[float, float, float]
            camera position in the world coordinate system (origin of the camera coordinate system)
        rotation_matrix : np.ndarray, optional
            rotation matrix from the world coordinate system to the camera coordinate system, by default None
        camera_name : str, optional
            camera name, by default None

        Raises
        ------
        ValueError
            if every eye_type is not 'pinhole' or 'concave_lens'
            if Eyes are not located on the same plane when eye_type is 'concave_lens'

        """

        self._camera_name = camera_name if camera_name is not None else "camera"
        self._world = None
        self._eyes = eyes if isinstance(eyes, list) else [eyes, ]
        if len({eye.eye_type for eye in self._eyes}) == 1:
            self._eye_type = self._eyes[0].eye_type
        else:
            raise ValueError("eye_type of all eyes should be the same")
        self._apertures = apertures if isinstance(apertures, list) else [apertures, ]
        self._screen = screen
        self._camera_position = np.array(camera_position)
        self._rotation_matrix = np.eye(3) if rotation_matrix is None else rotation_matrix
        if self._screen.subpixel_size.max() < np.min([eye.eye_size for eye in self._eyes]):
            print("ok")
        else:
            # warning
            raise ValueError(f"subpixel size of the screen must be smaller than eye size of all eyes "
                             f"(subpixel_size: {self._screen.subpixel_size}, "
                             f"smallest eye size: {np.min([eye.eye_size for eye in self._eyes])})")

        print("Camera is created.")

    def __repr__(self):
        """str: Render a concise textual summary of the camera configuration."""
        return f"Camera(eye_type={self._eye_type}, camera_position={self._camera_position})"

    def __eq__(self, other):
        """bool: Equality comparison based on optics, sensors, and pose."""
        if not isinstance(other, Camera):
            return False
        else:
            for k in self.__dict__.keys():
                if k == "_world":
                    continue
                elif k == "_eyes":
                    if len(self.__dict__[k]) != len(other.__dict__[k]):
                        return False
                    elif not all([eye1 == eye2 for eye1, eye2 in zip(self.__dict__[k], other.__dict__[k])]):
                        return False
                elif k == "_apertures":
                    if len(self.__dict__[k]) != len(other.__dict__[k]):
                        return False
                    elif not all([aperture1 == aperture2 for aperture1, aperture2 in
                                  zip(self.__dict__[k], other.__dict__[k])]):
                        return False
                elif k == "_screen":
                    if self.__dict__[k] != other.__dict__[k]:
                        return False
                else:
                    if not np.all(self.__dict__[k] == other.__dict__[k]):
                        return False
            d1 = self.__dict__.copy()
            d1.pop("_world")
            d1.pop("_eyes")
            d1.pop("_apertures")
            d1.pop("_screen")

            d2 = other.__dict__.copy()
            d2.pop("_world")
            d2.pop("_eyes")
            d2.pop("_apertures")
            d2.pop("_screen")

            return all([np.all(v1 == v2) for (v1, v2) in zip(d1.values(), d2.values())]) and \
                all([eye1 == eye2 for eye1, eye2 in zip(self._eyes, other._eyes)]) and \
                all([aperture1 == aperture2 for aperture1, aperture2 in zip(self._apertures, other._apertures)]) and \
                self._screen == other._screen

    @property
    def eye_type(self):
        """str: Shared optical mode for all eyes (``"pinhole"`` or ``"concave_lens"``)."""
        return self._eye_type

    @property
    def eyes(self):
        """List[Eye]: Collection of configured eyes mounted on the camera."""
        return self._eyes

    @property
    def apertures(self):
        """List[Aperture]: Aperture geometries paired with the camera."""
        return self._apertures

    @property
    def screen(self):
        """Screen: Display surface receiving projected rays."""
        return self._screen

    @property
    def camera_position(self):
        """numpy.ndarray: camera position in the world coordinate system."""
        return self._camera_position

    @property
    def camera_x(self):
        """numpy.ndarray: camera right direction in the world coordinate system."""
        # X-direction of the camera coordinate system in the world coordinate system
        return self.rotation_matrix.T @ np.array([1, 0, 0]).ravel()

    @property
    def camera_y(self):
        """numpy.ndarray: camera up direction in the world coordinate system."""
        # Y-direction of the camera coordinate system in the world coordinate system
        return self.rotation_matrix.T @ np.array([0, 1, 0]).ravel()

    @property
    def camera_z(self):
        """numpy.ndarray: camera look direction in the world coordinate system."""
        # Z-direction of the camera coordinate system in the world coordinate system
        return self.rotation_matrix.T @ np.array([0, 0, 1]).ravel()

    @property
    def rotation_matrix(self):
        """numpy.ndarray: rotation matrix from the world coordinate system to the camera coordinate system."""
        return self._rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, rotation_matrix):
        """None: Override the world-to-camera rotation matrix.

        Parameters
        ----------
        rotation_matrix : np.ndarray
            ``3×3`` orthonormal rotation transforming world vectors into camera coordinates.
        """
        self._rotation_matrix = rotation_matrix

    @property
    def world(self):
        """World or None: Scene graph currently observed by the camera."""
        return self._world

    def set_world(self, world_obj):
        """None: Register the :class:`World` scene providing geometry and emitters.

        Parameters
        ----------
        world_obj : World
            World instance this camera observes.
        """
        self._world = world_obj

    def move_camera(self, camera_position):
        """Camera: Update the camera origin within the world coordinate system.

        Parameters
        ----------
        camera_position : Tuple[float, float, float]
            camera position in the world coordinate system

        Returns
        -------
        Camera
            The camera instance for fluent-style chaining.
        """
        self._camera_position = np.array(camera_position)
        return self

    def set_rotation_matrix(self, order, angle, degrees=True):
        """Camera: Set the world-to-camera rotation using Euler angles.

        Parameters
        ----------
        order : str
            order of rotation, e.g. 'xyz'
        angle : float
            angle of rotation in degree
        degrees : bool, optional
            if True, angle is in degree, else in radian, by default True

        Returns
        -------
        Camera
            The camera instance with the updated rotation matrix.
        """
        self._rotation_matrix = Rotation.from_euler(order, angle, degrees=degrees).as_matrix()
        return self

    def world2camera(self, points):
        """transform points from the world coordinate system to the camera coordinate system

        Parameters
        ----------
        points : numpy.ndarray
            points in the world coordinate system (shape: (n, 3))

        Returns
        -------
        numpy.ndarray
            points in the camera coordinate system (shape: (n, 3))
        """
        return (self.rotation_matrix @ (points - self.camera_position[None, :]).T).T

    def add_eye(self, eye):
        """Add an eye to the camera.

        Parameters
        ----------
        eye : Eye
            an eye object

        Raises
        ------
        ValueError
            if eye_type of the new eye is different from the other eyes
        """
        if self._eye_type is None:
            self._eye_type = eye.eye_type
        elif self._eye_type != eye.eye_type:
            raise ValueError("eye_type of the new eye is different from the other eyes")
        self._eyes.append(eye)

    def add_aperture(self, aperture):
        """Add an aperture to the camera.

        Parameters
        ----------
        aperture : Aperture
            an aperture object
        """
        self._apertures.append(aperture)

    def calc_image_vec(self, eye_num, points, verbose: int = 0, check_visibility: bool = True):
        """sparse.csr_matrix: Assemble ray hits into a sparse image vector.

        Parameters
        ----------
        eye_num : int
            index of the eyes
        points : numpy.ndarray
            points in the world coordinate system (shape: (n, 3))
        verbose : int, optional
            Verbosity level forwarded to progress reporters.
        check_visibility : bool, optional
            Whether to cull rays occluded by apertures before projection.

        Returns
        -------
        sparse.csr_matrix
            Image mapping matrix ``(N_subpixel, n)`` describing which subpixels are reached by each ray.
            The intensity information must be calculated separately.
        """
        eye = self._eyes[eye_num]
        points_in_camera = self.world2camera(points)
        if check_visibility:
            visible_list = [stl_utils.check_visible(mesh_obj=aperture.stl_model,
                                                    start=eye.position,
                                                    grid_points=points_in_camera,
                                                    behind_start_included=True) for aperture in self._apertures]
            visible = np.any(visible_list, axis=0)
            rays = eye.calc_rays(points_in_camera, visible)
        else:
            rays = eye.calc_rays(points_in_camera)

        # print("ray2image start")
        # res1 = self.screen.ray2image(eye, rays, parallel=parallel)
        # res2 = self.screen.ray2image2(eye, rays, parallel=parallel)
        mat = self.screen.ray2image_grid(eye, rays, verbose=verbose)
        return mat

    def draw_optical_system(self, ax=None, show_focal_length=True, show_aperture=True, show_screen=True,
                            X_lim=None, Y_lim=None, Z_lim=None):
        """matplotlib.axes.Axes: Visualise optical elements in a 3D Matplotlib scene.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target 3D axis; a new figure is created when omitted.
        show_focal_length : bool, optional
            Toggle rendering of eye focal length vectors.
        show_aperture : bool, optional
            Toggle drawing aperture planes and models.
        show_screen : bool, optional
            Toggle rendering of the imaging screen.
        X_lim : Tuple[float, float], optional
            Manual bounds for the x-axis of the plot.
        Y_lim : Tuple[float, float], optional
            Manual bounds for the y-axis of the plot.
        Z_lim : Tuple[float, float], optional
            Manual bounds for the z-axis of the plot.
        
        Returns
        -------
        matplotlib.axes.Axes
            Axis containing the rendered optical setup.
        """
        # draw the optical system
        # note: axes in the figure are not equal to axes in the camera coordinate system
        # x-axis in the figure is Z-axis in the camera coordinate system
        # y-axis in the figure is X-axis in the camera coordinate system
        # z-axis in the figure is -Y-axis in the camera coordinate system
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')

        # set axes limits
        X_lim = (min([*[eye.position[0] for eye in self._eyes],
                      *[aperture.stl_model.x.min() if aperture.stl_model else
                        -aperture.size[0] / 2 + aperture.position[0] for aperture in self._apertures],
                      -self._screen.screen_size[1] / 2]),
                 max([*[eye.position[0] for eye in self._eyes],
                      *[aperture.stl_model.x.max() if aperture.stl_model else
                        aperture.size[0] / 2 + aperture.position[0] for aperture in self._apertures],
                      self._screen.screen_size[1] / 2])) if X_lim is None else X_lim
        Y_lim = (min([*[eye.position[1] for eye in self._eyes],
                      *[aperture.stl_model.y.min() if aperture.stl_model else
                        -aperture.size[1] / 2 + aperture.position[1] for aperture in self._apertures],
                      -self._screen.screen_size[0] / 2]),
                 max([*[eye.position[1] for eye in self._eyes],
                      *[aperture.stl_model.y.max() if aperture.stl_model else
                        aperture.size[1] / 2 + aperture.position[1] for aperture in self._apertures],
                      self._screen.screen_size[0] / 2])) if Y_lim is None else Y_lim
        Z_lim = (0,
                 max(*[aperture.position[2] for aperture in self._apertures],
                     *[eye.position[2] for eye in self._eyes])) if Z_lim is None else Z_lim

        ax.set_xlim(-Z_lim[1] * 0.1, Z_lim[1] * 1.1)
        ax.set_ylim(1.1 * X_lim[0], 1.1 * X_lim[1])
        ax.set_zlim(1.1 * Y_lim[0], 1.1 * Y_lim[1])
        ax.invert_yaxis()
        ax.invert_zaxis()

        # set axes labels
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        ax.set_zlabel("Y")

        ax.set_box_aspect((Z_lim[1] - Z_lim[0], X_lim[1] - X_lim[0], Y_lim[1] - Y_lim[0]))
        ax.set_title("Optical system")

        # draw the origin of the camera coordinate system
        ax.scatter(0, 0, 0, c="k", zdir="z")
        # draw eyes
        for eye in self._eyes:
            # draw eye position
            ax.scatter(eye.position[0], eye.position[1], eye.position[2], c="k", marker="x", s=50, zdir="x")
            # draw eye principal point
            ax.scatter(eye.principal_point[0], eye.principal_point[1], eye.principal_point[2], edgecolors="r",
                       facecolors="none", marker="o", s=50, zdir="x")
            if show_focal_length:
                # draw eye focal length
                ax.quiver(eye.principal_point[2], eye.principal_point[0], eye.principal_point[1],
                          -eye.focal_length, 0, 0, color="r")
            if self._eye_type == "pinhole":
                pass
            elif self._eye_type == "lens":
                # draw eye shape
                if eye.eye_shape == "rectangle":
                    # patch of rectangle
                    patch2d = Rectangle((eye.position[0] - eye.eye_size[0] / 2,
                                         eye.position[1] - eye.eye_size[1] / 2),
                                        eye.eye_size[0], eye.eye_size[1])
                else:
                    # patch of ellipse
                    patch2d = Ellipse((eye.position[0], eye.position[1]), eye.eye_size[0], eye.eye_size[1])
                # transform patch to 3D
                ax.add_collection3d(col=patch2d, zs=eye.position[2], zdir="x")
        # draw screen
        if show_screen:
            # draw screen shape
            # center of the screen is at the origin of the camera coordinate system
            if self._screen.screen_shape == "rectangle":
                # patch of rectangle
                patch2d = Rectangle(-np.array(self._screen.screen_size[::-1]) / 2, *self._screen.screen_size[::-1],
                                    facecolor="orange", edgecolor="k", alpha=0.5, linewidth=2)
            else:
                # patch of ellipse
                patch2d = Ellipse((0, 0), *self._screen.screen_size * 2,
                                  facecolor="orange", edgecolor="k", alpha=0.5, linewidth=2)
            # transform patch to 3D
            ax.add_patch(patch2d)
            art3d.pathpatch_2d_to_3d(patch2d, z=0, zdir="x")
        # draw apertures
        if show_aperture:
            for aperture in self._apertures:
                # draw aperture position
                ax.scatter(aperture.position[0], aperture.position[1], aperture.position[2], c="k", marker="x", s=100,
                           zdir="x")
                # show aperture stl model
                if aperture.stl_model is not None:
                    tmp_model = stl_utils.rotate_model(aperture.stl_model, matrix=[[0, 1, 0],
                                                                                   [0, 0, 1],
                                                                                   [1, 0, 0]])
                    stl_utils.show_stl(tmp_model, ax=ax, alpha=0.5, facecolors="orange", edgecolors="k", lw=0.5)
                else:
                    # draw aperture shape
                    if aperture.shape == "rectangle":
                        # patch of rectangle
                        patch2d = Rectangle((aperture.position[0] - aperture.size[0] / 2,
                                             aperture.position[1] - aperture.size[1] / 2),
                                            *aperture.size,
                                            facecolor="none", edgecolor="k", linewidth=2)
                    else:
                        # patch of ellipse
                        patch2d = Ellipse(aperture.position, *aperture.size * 2,
                                          facecolor="none", edgecolor="k", linewidth=2)
                    # TODO: aperture rotation (allow to set arbitrary normal vector) (future work)
                    # rotation matrix is not specified only one vector, so... how to rotate?
                    # ->
                    # transform patch to 3D
                    ax.add_patch(patch2d)
                    art3d.pathpatch_2d_to_3d(patch2d, z=aperture.position[2], zdir="x")

                    # draw camera coordinate system (length is 0.8*axes limit)
        _ = np.mean([ax.get_ylim()[0], ax.get_zlim()[0]]) * 0.8
        # X-axis in the camera coordinate system (y-axis in the figure)
        ax.quiver(0, 0, 0, 0, _, 0, color="r")
        # Y-axis in the camera coordinate system (-z-axis in the figure)
        ax.quiver(0, 0, 0, 0, 0, _, color="g")
        # Z-axis in the camera coordinate system (x-axis in the figure)
        ax.quiver(0, 0, 0, _, 0, 0, color="b")

        return ax

    # todo: make plotly version of draw_optical_system

    def draw_camera_orientation_plotly(self, fig=None, **kwargs):
        """go.Figure: Render camera axes within Plotly for interactive viewing.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure, optional
            Existing figure to append orientation glyphs to.
        **kwargs
            Additional keyword arguments forwarded to :func:`stl_utils.plotly_show_axes`.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure augmented with camera orientation geometry.
        """
        fig = go.Figure() if fig is None else fig
        stl_utils.plotly_show_axes(R=self.rotation_matrix, fig=fig, origin=self.camera_position, name="camera",
                                   **kwargs)
        return fig

    def draw_camera_orientation(self, ax=None):
        """matplotlib.axes.Axes: Plot camera axes relative to the world frame.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axis to draw on; a new orthographic subplot is created when omitted.

        Returns
        -------
        matplotlib.axes.Axes
            Axis showing both camera and world coordinate frames.
        """
        if ax is None:
            ax = plt.subplot(projection="3d", proj_type="ortho")
        # set axes limits
        lim = np.max(np.abs(self.camera_position))

        ax.set_xlim(-1.1 * lim, 1.1 * lim)
        ax.set_ylim(-1.1 * lim, 1.1 * lim)
        ax.set_zlim(-1.1 * lim, 1.1 * lim)
        ax.set_box_aspect((1, 1, 1))
        # set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # draw camera coordinate system in the world coordinate system
        # x, y, z axes in the world coordinate system is the same as axes in the figure
        # draw camera position
        ax.scatter(*self.camera_position, c="k", marker="x", s=100)
        arrow_length = np.linalg.norm(self.camera_position) * 0.2

        # draw camera X-axis
        ax.quiver(*self.camera_position, *(self.camera_x * arrow_length), color="r")
        # draw camera Y-axis
        ax.quiver(*self.camera_position, *(self.camera_y * arrow_length), color="g")
        # draw camera Z-axis
        ax.quiver(*self.camera_position, *(self.camera_z * arrow_length), color="b")

        # draw world coordinate system
        arrow_length = np.mean([ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]]) * 0.8
        ax.quiver(0, 0, 0, arrow_length, 0, 0, color="k")
        ax.quiver(0, 0, 0, 0, arrow_length, 0, color="k")
        ax.quiver(0, 0, 0, 0, 0, arrow_length, color="k")

        return ax

    def print_settings(self):
        """None: Emit a formatted summary of camera, eye, screen, and aperture settings."""
        print("Camera settings:")
        print(f"Camera position: {self._camera_position}")
        print(f"Eye type: {self._eye_type}")
        print(f"Eye position: {[eye.position for eye in self._eyes]}")
        print(f"Eye focal length: {[eye.focal_length for eye in self._eyes]}")
        print(f"Eye shape: {[eye.eye_shape for eye in self._eyes]}")
        print(f"Eye size: {[eye.eye_size for eye in self._eyes]}")
        print(f"Screen shape: {self._screen.screen_shape}")
        print(f"Screen size: {self._screen.screen_size}")
        print(f"pixel shape: {self._screen.pixel_shape}")
        print(f"pixel size: {self._screen.pixel_size}")
        print(f"Aperture position: {[aperture.position for aperture in self._apertures]}")
        print(f"Aperture shape: {[aperture.shape for aperture in self._apertures]}")
        print(f"Aperture size: {[aperture.size for aperture in self._apertures]}")


if __name__ == '__main__':
    # test Camera class
    camera = Camera(eyes=[Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.1, focal_length=15, position=[5, 0]),
                          Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.1, focal_length=15, position=[-5, 5])],
                    screen=Screen(screen_shape="circle", screen_size=17, pixel_shape=(100, 100)),
                    apertures=[
                        Aperture(shape="circle", size=20, position=[0, 0, 80]).set_model(resolution=40, max_size=50),
                        Aperture(shape="circle", size=60, position=[0, 10, 120])],
                    camera_position=[0, 900, 0])
    camera.set_rotation_matrix("xyz", (90, 0, 0), degrees=True)

    camera.print_settings()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    obj = camera.apertures[0].stl_model
    obj2 = stl_utils.rotate_model(obj, order="xyz", angles=(60, 0, 0), origin=(0, 0, 0))
    stl_utils.show_stl(obj2, ax, facecolors="orange", edgecolors="k", alpha=0.5, linewidth=1)
    # plt.show()
    ax = camera.draw_optical_system(ax=ax)
    camera.draw_camera_orientation()
    plt.show()

    # aper = Aperture(shape="circle", size=30, position=[0, 0, 80]).set_model(resolution=40, max_size=50)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # test3.show_stl(ax, aper.stl_model, azim=0, elev=90)
    # ax.scatter(aper.stl_model.x, aper.stl_model.y, aper.stl_model.z, c="r", marker="o", s=10)
    # plt.show()
    f = Filter(material="Al", thickness=1, photon_energy_range=[10, 1e4])
