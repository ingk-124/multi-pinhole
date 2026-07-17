"""Detector geometry, overlap integration, and spot rasterization."""
from numbers import Number
from typing import List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from scipy import sparse
from typing_extensions import Literal

from .eye import Eye
from .rays import Rays
from .utils import stl_utils
from .utils.my_stdio import my_tqdm

Vector2DLike = Union[np.ndarray, List[Number], Tuple[Number, Number]]

@njit(cache=True, nogil=True, inline="always")
def _unit_circle_primitive(x):
    """Integral of ``sqrt(1 - x**2)`` from zero to x on the unit circle."""
    x = min(1.0, max(-1.0, x))
    return 0.5 * (x * np.sqrt(max(0.0, 1.0 - x * x)) + np.arcsin(x))


@njit(cache=True, nogil=True)
def _unit_circle_rectangle_overlap(x0, x1, y0, y1):
    """Return the exact area shared by a unit circle and an axis-aligned rectangle."""
    x0 = max(-1.0, x0)
    x1 = min(1.0, x1)
    y0 = max(-1.0, y0)
    y1 = min(1.0, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    nearest_x = 0.0
    if x1 < 0.0:
        nearest_x = x1
    elif x0 > 0.0:
        nearest_x = x0
    nearest_y = 0.0
    if y1 < 0.0:
        nearest_y = y1
    elif y0 > 0.0:
        nearest_y = y0
    if nearest_x * nearest_x + nearest_y * nearest_y >= 1.0:
        return 0.0

    if (x0 * x0 + y0 * y0 <= 1.0 and
            x0 * x0 + y1 * y1 <= 1.0 and
            x1 * x1 + y0 * y0 <= 1.0 and
            x1 * x1 + y1 * y1 <= 1.0):
        return (x1 - x0) * (y1 - y0)

    # The vertical overlap changes expression only where a horizontal
    # rectangle edge crosses the circle. Split there and integrate the circle
    # arc analytically on each interval.
    breaks = np.empty(7, dtype=np.float64)
    n_breaks = 2
    breaks[0] = x0
    breaks[1] = x1
    # The circle height has its maximum at zero. Splitting there avoids using
    # the tangency point itself to classify a symmetric interval.
    if x0 < 0.0 < x1:
        breaks[n_breaks] = 0.0
        n_breaks += 1
    for y in (y0, y1):
        if -1.0 < y < 1.0:
            crossing = np.sqrt(max(0.0, 1.0 - y * y))
            for candidate in (-crossing, crossing):
                if x0 < candidate < x1:
                    breaks[n_breaks] = candidate
                    n_breaks += 1

    # Small fixed-size insertion sort keeps this helper Numba-friendly.
    for i in range(1, n_breaks):
        value = breaks[i]
        j = i - 1
        while j >= 0 and breaks[j] > value:
            breaks[j + 1] = breaks[j]
            j -= 1
        breaks[j + 1] = value

    area = 0.0
    for i in range(n_breaks - 1):
        left = breaks[i]
        right = breaks[i + 1]
        if right - left <= 1e-15:
            continue
        middle = 0.5 * (left + right)
        height = np.sqrt(max(0.0, 1.0 - middle * middle))
        upper = min(y1, height)
        lower = max(y0, -height)
        if upper <= lower:
            continue

        coefficient = 0.0
        constant = 0.0
        if height < y1:
            coefficient += 1.0
        else:
            constant += y1
        if -height > y0:
            coefficient += 1.0
        else:
            constant -= y0
        area += constant * (right - left)
        area += coefficient * (_unit_circle_primitive(right)
                               - _unit_circle_primitive(left))

    rectangle_area = (x1 - x0) * (y1 - y0)
    return min(rectangle_area, max(0.0, area))


@njit(cache=True, nogil=True, inline="always")
def _spot_cell_overlap(u0, u1, v0, v1, center_u, center_v,
                       half_u, half_v, use_ellipse):
    """Area shared by one detector cell and an ellipse/rectangle spot."""
    if use_ellipse:
        normalized_area = _unit_circle_rectangle_overlap(
            (u0 - center_u) / half_u,
            (u1 - center_u) / half_u,
            (v0 - center_v) / half_v,
            (v1 - center_v) / half_v,
        )
        return normalized_area * half_u * half_v

    overlap_u = min(u1, center_u + half_u) - max(u0, center_u - half_u)
    overlap_v = min(v1, center_v + half_v) - max(v0, center_v - half_v)
    if overlap_u <= 0.0 or overlap_v <= 0.0:
        return 0.0
    return overlap_u * overlap_v


@njit(cache=True, nogil=True, inline="always")
def _local_etendue_density(q_u, q_v, center_u, center_v, zoom_rate,
                           axial_distance, source_offset_x, source_offset_y):
    """Etendue density on the detector for one ray through the finite Eye."""
    # Image u is camera Y and image v is camera X.  Inverting
    # q = q_center + zoom_rate * a maps the detector location back to the
    # corresponding position a inside the Eye.
    eye_offset_x = (q_v - center_v) / zoom_rate
    eye_offset_y = (q_u - center_u) / zoom_rate
    dx = source_offset_x - eye_offset_x
    dy = source_offset_y - eye_offset_y
    distance2 = axial_distance * axial_distance + dx * dx + dy * dy
    return (axial_distance
            / (4.0 * np.pi * zoom_rate * zoom_rate
               * distance2 * np.sqrt(distance2)))


@njit(cache=True, nogil=True, inline="always")
def _rectangle_density_average(u0, u1, v0, v1, center_u, center_v,
                               zoom_rate, axial_distance,
                               source_offset_x, source_offset_y):
    """Two-point Gauss average of local etendue density on a rectangle."""
    midpoint_u = 0.5 * (u0 + u1)
    midpoint_v = 0.5 * (v0 + v1)
    offset_u = 0.5 * (u1 - u0) / np.sqrt(3.0)
    offset_v = 0.5 * (v1 - v0) / np.sqrt(3.0)
    total = 0.0
    for sign_u in (-1.0, 1.0):
        for sign_v in (-1.0, 1.0):
            total += _local_etendue_density(
                midpoint_u + sign_u * offset_u,
                midpoint_v + sign_v * offset_v,
                center_u, center_v, zoom_rate, axial_distance,
                source_offset_x, source_offset_y,
            )
    return 0.25 * total


@njit(cache=True, nogil=True)
def _spot_cell_local_etendue(u0, u1, v0, v1, center_u, center_v,
                             half_u, half_v, use_ellipse, overlap_area,
                             zoom_rate, axial_distance,
                             source_offset_x, source_offset_y):
    """Integrate local-ray etendue over one spot/cell intersection.

    Exact overlap areas are retained.  The slowly varying angular density is
    integrated with low-order deterministic quadrature: Gauss quadrature for
    rectangles, disk quadrature when a complete ellipse lies in one cell, and
    a bounded midpoint rule only on clipped ellipse boundary cells.
    """
    if overlap_area <= 0.0:
        return 0.0

    clipped_u0 = max(u0, center_u - half_u)
    clipped_u1 = min(u1, center_u + half_u)
    clipped_v0 = max(v0, center_v - half_v)
    clipped_v1 = min(v1, center_v + half_v)

    if not use_ellipse:
        average = _rectangle_density_average(
            clipped_u0, clipped_u1, clipped_v0, clipped_v1,
            center_u, center_v, zoom_rate, axial_distance,
            source_offset_x, source_offset_y,
        )
        return overlap_area * average

    # A complete ellipse in one detector cell is common when detector pixels
    # are large.  Integrate it independently of detector subpixel resolution.
    if (center_u - half_u >= u0 and center_u + half_u <= u1 and
            center_v - half_v >= v0 and center_v + half_v <= v1):
        # Uniform ellipse area is uniform in t=r**2 and theta.  Two Gauss
        # points in t and eight equally spaced angles capture the leading
        # finite-aperture correction without detector-grid sampling noise.
        total = 0.0
        inverse_sqrt_three = 1.0 / np.sqrt(3.0)
        for t_sign in (-1.0, 1.0):
            t = 0.5 * (1.0 + t_sign * inverse_sqrt_three)
            radius = np.sqrt(t)
            for angle_index in range(8):
                angle = 2.0 * np.pi * angle_index / 8.0
                q_u = center_u + half_u * radius * np.cos(angle)
                q_v = center_v + half_v * radius * np.sin(angle)
                total += _local_etendue_density(
                    q_u, q_v, center_u, center_v, zoom_rate,
                    axial_distance, source_offset_x, source_offset_y,
                )
        return overlap_area * total / 16.0

    # If the whole detector cell is inside the ellipse, its intersection is a
    # rectangle and the Gauss rule above applies directly.
    all_corners_inside = True
    for corner_u in (u0, u1):
        for corner_v in (v0, v1):
            normalized = (((corner_u - center_u) / half_u) ** 2
                          + ((corner_v - center_v) / half_v) ** 2)
            if normalized > 1.0:
                all_corners_inside = False
    if all_corners_inside:
        average = _rectangle_density_average(
            u0, u1, v0, v1, center_u, center_v, zoom_rate,
            axial_distance, source_offset_x, source_offset_y,
        )
        return overlap_area * average

    # Only boundary intersections require masked quadrature.  The exact area
    # remains supplied by _spot_cell_overlap; these samples estimate only the
    # mean of the smooth angular density over that area.
    density_sum = 0.0
    sample_count = 0
    for sample_u_index in range(4):
        q_u = clipped_u0 + (sample_u_index + 0.5) * (clipped_u1 - clipped_u0) / 4.0
        for sample_v_index in range(4):
            q_v = clipped_v0 + (sample_v_index + 0.5) * (clipped_v1 - clipped_v0) / 4.0
            normalized = (((q_u - center_u) / half_u) ** 2
                          + ((q_v - center_v) / half_v) ** 2)
            if normalized <= 1.0:
                density_sum += _local_etendue_density(
                    q_u, q_v, center_u, center_v, zoom_rate,
                    axial_distance, source_offset_x, source_offset_y,
                )
                sample_count += 1

    if sample_count == 0:
        # A very thin intersection can fall between all midpoint samples.
        # The closest point in the clipped rectangle is still inside the
        # ellipse whenever the exact overlap is positive (up to tangency).
        q_u = min(clipped_u1, max(clipped_u0, center_u))
        q_v = min(clipped_v1, max(clipped_v0, center_v))
        average = _local_etendue_density(
            q_u, q_v, center_u, center_v, zoom_rate,
            axial_distance, source_offset_x, source_offset_y,
        )
    else:
        average = density_sum / sample_count
    return overlap_area * average


@njit(cache=True, nogil=True)
def _rasterize_spots(u_axis, v_axis, cell_u, cell_v,
                     u_center, v_center, half_u, half_v,
                     i_min, i_max, j_min, j_max, valid, v_subpixels,
                     use_ellipse, zoom_rate, axial_distance,
                     source_offset_x, source_offset_y):
    """Rasterize spots into cell indices, areas, and local etendue weights."""
    counts = np.zeros(u_center.size, dtype=np.int64)
    for valid_index in range(valid.size):
        ray = valid[valid_index]
        count = 0
        for i in range(i_min[ray], i_max[ray] + 1):
            u0 = u_axis[i] - 0.5 * cell_u
            u1 = u0 + cell_u
            for j in range(j_min[ray], j_max[ray] + 1):
                v0 = v_axis[j] - 0.5 * cell_v
                v1 = v0 + cell_v
                overlap = _spot_cell_overlap(
                    u0, u1, v0, v1, u_center[ray], v_center[ray],
                    half_u[ray], half_v[ray], use_ellipse,
                )
                if overlap > 0.0:
                    count += 1
        counts[ray] = count

    offsets = np.empty(counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    for ray in range(counts.size):
        offsets[ray + 1] = offsets[ray] + counts[ray]
    pixel_indices = np.empty(offsets[-1], dtype=np.int32)
    etendue_weights = np.empty(offsets[-1], dtype=np.float32)

    for valid_index in range(valid.size):
        ray = valid[valid_index]
        output_index = offsets[ray]
        for i in range(i_min[ray], i_max[ray] + 1):
            u0 = u_axis[i] - 0.5 * cell_u
            u1 = u0 + cell_u
            for j in range(j_min[ray], j_max[ray] + 1):
                v0 = v_axis[j] - 0.5 * cell_v
                v1 = v0 + cell_v
                overlap = _spot_cell_overlap(
                    u0, u1, v0, v1, u_center[ray], v_center[ray],
                    half_u[ray], half_v[ray], use_ellipse,
                )
                if overlap > 0.0:
                    pixel_indices[output_index] = i * v_subpixels + j
                    etendue_weights[output_index] = _spot_cell_local_etendue(
                        u0, u1, v0, v1, u_center[ray], v_center[ray],
                        half_u[ray], half_v[ray], use_ellipse, overlap,
                        zoom_rate[ray], axial_distance[ray],
                        source_offset_x[ray], source_offset_y[ray],
                    )
                    output_index += 1
    return pixel_indices, etendue_weights, counts

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

        self._frozen = False

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
        self._subpixel_u_axis = None
        self._subpixel_v_axis = None
        self._transform_matrix = None

        self.subpixel_resolution = subpixel_resolution

    def __eq__(self, other):
        """bool: Check if two screens share identical discretisation parameters."""
        if isinstance(other, Screen):
            for k in self.__dict__.keys():
                if k == "_frozen":
                    continue
                if isinstance(self.__dict__[k], sparse.spmatrix):
                    if not np.all(self.__dict__[k].data == other.__dict__[k].data):
                        return False
                else:
                    if not np.all(self.__dict__[k] == other.__dict__[k]):
                        return False
            return True

        else:
            return False

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
        self._ensure_mutable()
        # set the subpixel resolution
        if not isinstance(subpixel_resolution, int) or subpixel_resolution < 1:
            raise ValueError("subpixel_resolution must be integer and larger than 1")
        self._subpixel_resolution = subpixel_resolution
        self._set_variables()

    @property
    def frozen(self):
        """bool: Whether this screen's geometry and discretisation are immutable."""
        return self._frozen

    def _ensure_mutable(self):
        if self._frozen:
            raise RuntimeError("Screen geometry is frozen because its Camera is registered in a World")

    def freeze(self):
        """Freeze screen geometry, cached grids, and sparse mappings."""
        if not self._frozen:
            for value in self.__dict__.values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
                elif sparse.issparse(value):
                    value.data.setflags(write=False)
                    value.indices.setflags(write=False)
                    value.indptr.setflags(write=False)
            self._frozen = True
        return self

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

    def _set_variables(self):
        """None: Populate cached geometry derived from the current subpixel resolution."""
        # set the subpixel shape, subpixel size and subpixel position
        self._subpixel_shape = self._pixel_shape * self._subpixel_resolution
        self._N_subpixel = self._subpixel_shape[0] * self._subpixel_shape[1]
        self._subpixel_size = self._screen_size / self._subpixel_shape
        self._subpixel_u_axis = np.linspace(
            self._subpixel_size[0] * 0.5,
            self._screen_size[0] - self._subpixel_size[0] * 0.5,
            self._subpixel_shape[0],
        )
        self._subpixel_v_axis = np.linspace(
            self._subpixel_size[1] * 0.5,
            self._screen_size[1] - self._subpixel_size[1] * 0.5,
            self._subpixel_shape[1],
        )
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

    def ray2image_grid(self, eye: Eye, rays: Rays, verbose=0,
                       etendue_per_subpixel=None):
        """Integrate finite-Eye ray footprints on the detector grid.

        Parameters
        ----------
        eye : Eye
            Eye projecting the rays to the screen.
        rays : Rays
            Source rays in camera coordinates.
        verbose : int, default=0
            Verbosity level.
        etendue_per_subpixel : ndarray, optional
            Deprecated compatibility argument.  Local etendue now depends on
            both the source ray and the position inside the finite Eye, so a
            detector-only cache is no longer used by this calculation.
        Returns
        -------
        scipy.sparse.csr_matrix
            Local-etendue weights, shape ``(N_subpixel, n_rays)``. Rows are
            detector subpixels and columns are input rays.

        Notes
        -----
        Ellipse/cell overlap area is analytic, so spots smaller than a cell do
        not disappear. Rectangle density uses 2-by-2 Gauss quadrature. An
        ellipse wholly contained in one cell uses 2 radial by 8 angular
        quadrature; clipped ellipse boundary cells use a 4-by-4 masked
        midpoint approximation for the local density average. The latter has
        no strict error guarantee, and detector subpixel refinement can affect
        local PSF accuracy. A spot clipped by the screen boundary contributes
        less total signal.

        For a finite Eye, every detector integration point is mapped back to
        a position in the Eye. The source-to-detector Jacobian and local
        solid-angle-normalized density are therefore source- and
        Eye-position-dependent rather than a detector-only cached factor.
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
        u_axis = self._subpixel_u_axis
        v_axis = self._subpixel_v_axis
        # Cell edges, rather than center extrema, are the physical screen
        # bounds. This also lets a spot smaller than one cell be retained.
        u_min, u_max = 0.0, float(self._screen_size[0])
        v_min, v_max = 0.0, float(self._screen_size[1])
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

        inside_screen = ((uv[:, 0] + half[:, 0] > u_min) &
                         (uv[:, 0] - half[:, 0] < u_max) &
                         (uv[:, 1] + half[:, 1] > v_min) &
                         (uv[:, 1] - half[:, 1] < v_max))

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

        i_min = np.zeros(rays.n, dtype=np.int32)
        i_max = np.zeros(rays.n, dtype=np.int32)
        j_min = np.zeros(rays.n, dtype=np.int32)
        j_max = np.zeros(rays.n, dtype=np.int32)
        i_min[valid] = np.clip(np.floor(u_low[valid] / du), 0, U_sub - 1).astype(np.int32)
        i_max[valid] = np.clip(np.ceil(u_high[valid] / du) - 1, 0, U_sub - 1).astype(np.int32)
        j_min[valid] = np.clip(np.floor(v_low[valid] / dv), 0, V_sub - 1).astype(np.int32)
        j_max[valid] = np.clip(np.ceil(v_high[valid] / dv) - 1, 0, V_sub - 1).astype(np.int32)

        # Source offset from the Eye centre follows from the central projected
        # ray: q0-eye = -f * source_offset / Z.  The rasterizer maps every
        # detector integration point back into the finite Eye and evaluates
        # its own local source-to-Eye ray geometry.
        source_offset = -(rays.Z[:, None] / eye.focal_length) \
            * (rays.XY - eye.position[None, :2])
        pixel_indices, local_etendue, counts = _rasterize_spots(
            u_axis, v_axis, du, dv, u_center, v_center, a_u, a_v,
            i_min, i_max, j_min, j_max, valid, V_sub, use_ellipse,
            rays.zoom_rate, rays.Z, source_offset[:, 0], source_offset[:, 1],
        )
        indptr = np.concatenate([np.array([0], dtype=np.int64),
                                 np.cumsum(counts, dtype=np.int64)])
        if etendue_per_subpixel is not None:
            etendue_per_subpixel = np.asarray(etendue_per_subpixel, dtype=np.float32)
            if etendue_per_subpixel.shape != (self.N_subpixel,):
                raise ValueError(
                    f"etendue_per_subpixel must have shape {(self.N_subpixel,)}"
                )
        # ``local_etendue`` already includes exact overlap area and the local
        # source-to-Eye solid-angle density for each spot/cell intersection.
        mat = sparse.csc_matrix(
            (local_etendue, pixel_indices, indptr),
            shape=(self.N_subpixel, rays.n),
        ).tocsr()
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
        return self._transform_matrix.dot(subpixel_image)

    def show_image(self, image: np.ndarray = None, ax: plt.Axes = None,
                   block: bool = True, pixel_image: bool = False, pm: bool = False,
                   colorbar: bool = True, masked: bool = False, show: bool = False, **kwargs) -> plt.Axes:
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
        show : bool, optional
            If True, plt.show() is called, by default False
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

        if show:
            plt.show(block=block)
        return ax

    def print_settings(self) -> None:
        """None: Display key screen discretisation parameters."""
        print("screen_shape: {}".format(self._screen_shape))
        print("pixel_shape: {}".format(self._pixel_shape))
        print("subpixel_resolution: {}".format(self._subpixel_resolution))
        print("pixel_size: {}".format(self._pixel_size))
        print("subpixel_size: {}".format(self._subpixel_size))
