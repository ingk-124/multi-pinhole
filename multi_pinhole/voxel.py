from itertools import chain

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

from utils.my_stdio import *

# TODO: add docstring, type hints, and tests(<- additional, help me copilot!)
# TODO: refactor variable names

COORDINATE_TYPES = ["cartesian", "torus", "cylindrical", "spherical"]
COORDINATE_PARAMETER_KEYS = {"cartesian": [["width", "depth", "height"],
                                           ["X", "Y", "Z"]],
                             "torus": [["major_radius", "minor_radius"],
                                       ["R_0", "a"]],
                             "cylindrical": [["radius", "height"],
                                             ["a", "h"]],
                             "spherical": [["radius"],
                                           ["a"]]}  # {coordinate_type: [keys, aliases]}


def cartesian_coordinates(width: float, depth: float, height: float):
    """Return the normalized coordinates for cartesian coordinates

    Parameters
    ----------
    width: float
        The length along the x-axis (width)
    depth: float
        The length along the y-axis (depth)
    height: float
        The length along the z-axis (height)

    Returns
    -------
    normalized_coordinates: function
        The normalized coordinates function
    """
    para = np.abs([width / 2, depth / 2, height / 2])

    def normalized_coordinates(points: np.ndarray):
        """Return the normalized coordinates (cartesian coordinates)

        Parameters
        ----------
        points: np.ndarray
            The points in the world coordinate system (n_points, 3)

        Returns
        -------
        normalized_points: np.ndarray
            The normalized points (n_points, 3)

        Notes
        -----
        The points are normalized by the length of the axes as:
            x = x / X
            y = y / Y
            z = z / Z
        If necessary, check the lengths of the axes.
        (self.coordinate_parameters["X"], self.coordinate_parameters["Y"], self.coordinate_parameters["Z"])
        """
        return points / para[None, :]

    return normalized_coordinates


def torus_coordinates(major_radius: float, minor_radius: float):
    """Return the normalized coordinates for torus coordinates

    Parameters
    ----------
    major_radius: float
        The major radius of the torus
    minor_radius: float
        The minor radius of the torus

    Returns
    -------
    normalized_coordinates: function
        The normalized coordinates function

    Notes
    -----
    Define torus coordinate as
        x = (R_0 + a * r * cos(theta)) * cos(phi),
        y = (R_0 + a * r * cos(theta)) * sin(phi),
        z = a * r * sin(theta),
    where a=minor_radius and R_0=major_radius.
    """
    R_0 = major_radius
    a = minor_radius

    def normalized_coordinates(points: np.ndarray):
        """Return the normalized coordinates (torus coordinates)

        Parameters
        ----------
        points: np.ndarray
            The points in the world coordinate system (n_points, 3)

        Returns
        -------
        normalized_points: np.ndarray
            The normalized points (n_points, 3)

        Notes
        -----
        The normalized coordinates are calculated by the following equations:
            R = sqrt(x^2 + y^2)
            r = sqrt((R - R_0)^2 + z^2) / a
            theta = arctan(z / R)
            phi = arctan(y / x)
        """
        R = np.linalg.norm(points[:, :2], axis=1)
        r = np.linalg.norm([R - R_0, points[:, 2]], axis=0) / a
        theta = np.arctan2(points[:, 2], R_0 - R)
        phi = np.arctan2(points[:, 1], points[:, 0])
        return np.stack([r, theta, phi], axis=1)

    return normalized_coordinates


def cylindrical_coordinates(radius: float, height: float):
    """Return the normalized coordinates for cylindrical coordinates

    Parameters
    ----------
    radius: float
        The radius of the cylinder
    height: float
        The height of the cylinder

    Returns
    -------
    normalized_coordinates: function
        The normalized coordinates function
    """
    a = radius
    h = height

    def normalized_coordinates(points: np.ndarray):
        """Return the normalized coordinates (cylindrical coordinates)

        Parameters
        ----------
        points: np.ndarray
            The points in the world coordinate system (n_points, 3)

        Returns
        -------
        normalized_points: np.ndarray
            The normalized points (n_points, 3)

        Notes
        -----
        The normalized coordinates are calculated by the following equations:
            r = sqrt(x^2 + y^2) / a
            theta = arctan(y / x)
            z = z / (h / 2)
        """
        r = np.linalg.norm(points[:, :2], axis=1) / a
        theta = np.arctan2(points[:, 1], points[:, 0])
        z = points[:, 2] / (h / 2)
        return np.stack([r, theta, z], axis=1)

    return normalized_coordinates


def spherical_coordinates(radius: float):
    """Return the normalized coordinates for spherical coordinates

    Parameters
    ----------
    radius: float
        The radius of the sphere

    Returns
    -------
    normalized_coordinates: function
        The normalized coordinates function
    """
    a = radius

    def normalized_coordinates(points: np.ndarray):
        """Return the normalized coordinates (spherical coordinates)

        Parameters
        ----------
        points: np.ndarray
            The points in the world coordinate system (n_points, 3)

        Returns
        -------
        normalized_points: np.ndarray
            The normalized points (n_points, 3)

        Notes
        -----
        The normalized coordinates are calculated by the following equations:
            r = sqrt(x^2 + y^2 + z^2) / radius
            theta = arccos(z / r)
            phi = arctan(y / x)
        """
        r = np.linalg.norm(points, axis=1) / a
        theta = np.arccos(points[:, 2] / r)
        phi = np.arctan2(points[:, 1], points[:, 0])
        return np.stack([r, theta, phi], axis=1)

    return normalized_coordinates


def interpolate_matrix_from_vertices(res=None):
    """Build an interpolation matrix from voxel vertex contributions.

    Parameters
    ----------
    res : int | tuple[int, int, int] | None, optional
        Resolution of the sub-grid in ``(x_res, y_res, z_res)`` form. When
        ``None`` the sub-grid is treated as ``(1, 1, 1)``. If a single integer is
        provided, it is broadcast to all three axes.

    Returns
    -------
    numpy.ndarray | scipy.sparse.csr_matrix
        Dense interpolation matrix of shape ``(N_sub_voxel, 8)`` when
        ``res`` is ``None``; otherwise a CSR sparse matrix where
        ``N_sub_voxel = x_res * y_res * z_res``.
    """
    if res is None:
        return np.full((1, 8), 1 / 8)
    else:
        res = np.broadcast_to(res, 3)
        N_sub_voxel = np.prod(res)
        matrix = np.zeros((N_sub_voxel, 8))

        for i in range(N_sub_voxel):
            i_x, i_y, i_z = np.unravel_index(i, res)
            a, b, c = (i_x + 0.5) / res[0], (i_y + 0.5) / res[1], (i_z + 0.5) / res[2]

            matrix[i] = np.array([(1 - a) * (1 - b) * (1 - c),
                                  (1 - a) * (1 - b) * c,
                                  (1 - a) * b * (1 - c),
                                  (1 - a) * b * c,
                                  a * (1 - b) * (1 - c),
                                  a * (1 - b) * c,
                                  a * b * (1 - c),
                                  a * b * c])
        # The matrix is dense, but it is converted to a sparse matrix.
        # Because it is used for dot product with a sparse matrix in Voxel.sub_voxel_interpolator.
        return sparse.csr_matrix(matrix)



def shifted_torus(r, theta, phi, delta):
    R = r * np.cos(theta) + delta
    Z = r * np.sin(theta)
    r_shifted = np.sqrt(R ** 2 + Z ** 2)
    theta_shifted = np.arctan2(Z, R)
    return r_shifted, theta_shifted, phi


def helical_displacement(r, theta, phi, m_, n_, phi_0, d, r_1, xi_0):
    r_ = r * np.exp(m_ * theta * 1j)
    xi = xi_0 * np.exp(-(r / r_1) ** d) * np.exp(n_ * (phi - phi_0) * 1j)
    r_new_complex = r_ - xi
    r_new = np.abs(r_new_complex)
    theta = np.angle(r_new_complex)
    phi = phi

    return r_new, theta, phi


def hollow(r, A, p, q, h, w):
    f1 = (1 - r ** p) ** q
    f2 = np.exp(-(r / w) ** 2)
    return A * (f1 - h * f2)

def helical_axis(r, theta, phi, m_, n_, r_a, phi_0):
    psi = n_/m_ * phi + phi_0
    dx = r_a * np.cos(psi)
    dy = r_a * np.sin(psi)
    _x = r * np.cos(theta)
    _y = r * np.sin(theta)
    r_new = np.sqrt((_x - dx) ** 2 + (_y - dy) ** 2)
    return r_new


def emission_profile(r, theta, phi, allow_negative=False, **params):
    m_ = params.get("m_", 1)
    n_ = params.get("n_", -1)
    delta = params.get("delta", 0)
    phi_0 = params.get("phi_0", 0)
    d = params.get("d", 2)
    r_1 = params.get("r_1", 0.5)
    xi_0 = params.get("xi_0", 0.1)
    A = params.get("A", 1)
    p = params.get("p", 2)
    q = params.get("q", 3)
    h = params.get("h", 0)
    w = params.get("w", 0.5)

    r_shifted, theta_shifted, phi_shifted = shifted_torus(r, theta, phi, delta)
    r_new, theta_new, phi_new = helical_displacement(r_shifted, theta_shifted, phi_shifted,
                                                     m_=m_, n_=n_, phi_0=phi_0, d=d, r_1=r_1, xi_0=xi_0)
    y = hollow(r_new, A=A, p=p, q=q, h=h, w=w)
    if not allow_negative:
        y = np.maximum(y, 0)
    return y


class Voxel:
    """
    Voxel class

    Attributes
    ----------
    ranges : ((float, float), (float, float), (float, float))
        axis ranges
    N_grid : int
        number of grid points (N_grid = (N_x + 1) * (N_y + 1) * (N_z + 1))
    grid_shape : (int, int, int)
        grid shape (N_x + 1, N_y + 1, N_z + 1)
    grid : np.ndarray (N_grid, 3)
        grid points
        The grid points are arranged in the order z->y->x.
        ex. [x0, y0, z0], [x0, y0, z1], ..., [x0, y0, zN_z], ..., [x0, yN_y, zN_z], ..., [xN_x, yN_y, zN_z]
    N_voxel : int
        number of voxels (N_voxel = N_x * N_y * N_z)
    voxel_shape : (int, int, int)
        voxel shape (N_x, N_y, N_z)
    voxel_indices : np.ndarray (N_voxel, 3)
        voxel indices
        The grid points are arranged in the order z->y->x (same as grid).
        ex. [0, 0, 0], [0, 0, 1], ..., [0, 0, N_z], ..., [0, N_y, N_z], ..., [N_x, N_y, N_z]
    vertices_indices : np.ndarray (N_voxel, 8)
        voxel vertices indices
    vertices : np.ndarray (N_voxel, 8, 3)
        voxel vertices
    d : np.ndarray (N_voxel, 3)
        voxel edge length
    gravity_center : np.ndarray (N_voxel, 3)
        voxel gravity center
    volume : np.ndarray (N_voxel,)
        voxel volume
    axes : list[np.ndarray, np.ndarray, np.ndarray]
        axes [x_axis, y_axis, z_axis]
    x_axis : np.ndarray (N_x + 1,)
        x axis for grid
    y_axis : np.ndarray (N_y + 1,)
        y axis for grid
    z_axis : np.ndarray (N_z + 1,)
        z axis for grid
    vx : np.ndarray (N_voxel, 8)
        x coordinates of voxel vertices
    vy : np.ndarray (N_voxel, 8)
        y coordinates of voxel vertices
    vz : np.ndarray (N_voxel, 8)
        z coordinates of voxel vertices
    N : int
        alias of N_voxel
    shape : (int, int, int)
        alias of voxel_shape

    Methods
    -------
    reset_axes()
        reset axes with ranges and voxel_shape already set
    uniform_axes(ranges, shape)
        set uniform axes with ranges and shape
    show_info()
        show voxel information
    """

    def __init__(self,
                 x_axis: np.ndarray = None,
                 y_axis: np.ndarray = None,
                 z_axis: np.ndarray = None,
                 coordinate_type: str = None,
                 rotation: Rotation | np.ndarray = None,
                 coordinate_parameters: dict = None,
                 sub_voxel_resolution: int | tuple[int, int, int] = None):
        """
        Initialize a voxel object.

        Parameters
        ----------
        x_axis : np.ndarray (N_x + 1,)
            x axis for grid
        y_axis : np.ndarray (N_y + 1,)
            y axis for grid
        z_axis : np.ndarray (N_z + 1,)
            z axis for grid
        coordinate_type : str
            The type of coordinates.
            The supported types are "cartesian", "torus", "cylindrical", and "spherical".
        coordinate_parameters : dict
            The parameters for the coordinates. If it is None, all parameters are set to 1.

            - cartesian: {"X": float, "Y": float, "Z": float}
            - torus: {"major_radius": float, "minor_radius": float} or {"R_0": float, "a": float}
            - cylindrical: {"radius": float, "height": float} or {"a": float, "h": float}
            - spherical: {"radius": float} or {"a": float}
        """

        # initialize axes
        x_axis = np.array([]) if x_axis is None else np.array(x_axis)
        y_axis = np.array([]) if y_axis is None else np.array(y_axis)
        z_axis = np.array([]) if z_axis is None else np.array(z_axis)
        axes = [x_axis, y_axis, z_axis]

        # initialize all attributes with None
        self._axes = None
        self._ranges = None

        # grid
        self._N_grid = 0
        self._grid_shape = None

        # voxel
        self._N_voxel = 0
        self._voxel_shape = None
        self._voxel_indices = None

        # vertices
        self._vertices_indices = None
        self._vertices = None

        # voxel properties
        # self._d = None
        # self._gravity_center = None
        # self._volume = None

        # interpolation
        self._vertices_interpolator = None
        self._res = (1, 1, 1)
        self._sub_voxel_matrix = interpolate_matrix_from_vertices(self._res)

        # coordinate
        self._rotation_matrix = None
        self._normalized_coordinates = None
        self._world = None
        self._coordinate_type = None
        self._coordinate_parameters = {}
        coordinate_parameters = {} if coordinate_parameters is None else coordinate_parameters
        # set attributes
        self.set_coordinate(coordinate_type=coordinate_type, rotation=rotation,
                            **coordinate_parameters)
        self.axes = axes
        self.res = sub_voxel_resolution
        self._voxel2vertices = None
        self.update()

    def __repr__(self):
        return f"Voxel(ranges={self.ranges}, N_voxel={self.N_voxel}, voxel_shape={self.voxel_shape}, " \
               f"coordinate_type={self.coordinate_type}, coordinate_parameters={self.coordinate_parameters})"

    def __getitem__(self, key):
        return self.vertices[key]

    def __eq__(self, other):
        if isinstance(other, Voxel) and (other.grid_shape == self.grid_shape):
            return all([np.all(axis1 == axis2) for axis1, axis2 in zip(self.axes, other.axes)])
        else:
            return False

    @property
    def ranges(self):
        """
        get axis ranges
        Returns
        -------
        ((float, float), (float, float), (float, float))
        """
        return self._ranges

    @property
    def N_grid(self):
        """
        get vertex number
        Returns
        -------
        int
        """
        return self._N_grid

    @property
    def grid_shape(self):
        """
        get vertex shape
        Returns
        -------
        (int, int, int)
        """
        return self._grid_shape

    @property
    def grid(self):
        """
        get vertex points
        Returns
        -------
        np.ndarray (N_grid, 3)
        """
        return np.stack(
            [axis[_] for axis, _ in zip(self.axes, np.unravel_index(np.arange(self.N_grid), self.grid_shape))], axis=-1)

    @property
    def N_voxel(self):
        """
        get voxel number
        Returns
        -------
        int
        """
        return self._N_voxel

    @property
    def voxel_shape(self):
        """
        get voxel shape
        Returns
        -------
        (int, int, int)
        """
        return self._voxel_shape

    @property
    def vertices_indices(self):
        """
        get vertices indices
        Returns
        -------
        np.ndarray (N_voxel, 8)
        """
        return self._vertices_indices

    @property
    def vertices_indices_3d(self):
        """
        get vertices indices in (N_voxel, 8, 3) form
        """
        ii, jj, kk = np.unravel_index(self._vertices_indices, self.grid_shape)  # each shape is (N_voxel, 8)
        return np.stack([ii, jj, kk], axis=-1)  # shape is (N_voxel, 8, 3)

    @property
    def vertices(self):
        """
        get vertices
        Returns
        -------
        np.ndarray (N_voxel, 8, 3)
        """
        ii, jj, kk = np.unravel_index(self._vertices_indices, self.grid_shape)  # each shape is (N_voxel, 8)
        return np.stack([self.x_axis[ii], self.y_axis[jj], self.z_axis[kk]], axis=-1)  # shape is (N_voxel, 8, 3)

    @property
    def gravity_center(self):
        """
        get gravity center

        Returns
        -------
        np.ndarray (N_voxel, 3)
        """
        # return self._gravity_center
        # calculate from cx, cy, cz_axis
        xxx, yyy, zzz = np.meshgrid(self.cx_axis, self.cy_axis, self.cz_axis, indexing="ij")
        return np.stack([xxx.ravel(), yyy.ravel(), zzz.ravel()], axis=1)

    def get_gravity_center(self, n=None):
        """
        get gravity center

        Parameters
        ----------
        n : int
            number of points to return

        Returns
        -------
        np.ndarray (n, 3)
        """
        # return self.gravity_center[n]  # original <- this requires all gravity centers to be calculated
        # calculate only for the requested indices
        indices = self.get_voxel_position(n)
        cx = self.cx_axis[indices[:, 0]]
        cy = self.cy_axis[indices[:, 1]]
        cz = self.cz_axis[indices[:, 2]]
        return np.stack([cx, cy, cz], axis=1)

    @property
    def volume(self):
        """
        get voxel volume

        Returns
        -------
        np.ndarray (N_voxel,)
        """
        return (self.dx_axis[:, None, None] * self.dy_axis[None, :, None] * self.dz_axis[None, None, :]).ravel()

    @property
    def coordinate_type(self):
        return self._coordinate_type

    @property
    def res(self):
        return self._res

    @res.setter
    def res(self, res: int | tuple[int, int, int] | np.ndarray = None):
        """
        set resolution of sub voxels

        Parameters
        ----------
        res : int or (int, int, int) or np.ndarray
            resolution of sub voxels (x_res, y_res, z_res)
            if res is None, res is not changed
        """

        if res is None:
            return
        else:
            try:
                _res = tuple(np.broadcast_to(res, 3))
            except ValueError:
                raise ValueError(f"{res=} must be int or (int, int, int)")
            if self._res != _res:
                self._res = _res
                self._sub_voxel_matrix = interpolate_matrix_from_vertices(self._res)

    def set_coordinate(self, coordinate_type: str = None, rotation: Rotation | np.ndarray = None,
                       show: bool = False, **coordinate_parameters):
        """
        set coordinate type and parameters

        Parameters
        ----------
        coordinate_type : str
            The type of coordinates.
            The supported types are "cartesian", "torus", "cylindrical", and "spherical".
        rotation : Rotation | np.ndarray
            The Rotation object or rotation matrix (3, 3).
        show : bool
            show info or not
        coordinate_parameters : dict
            The parameters for the coordinates. If it is None, all parameters are set to 1.

            - cartesian: {"width": float, "depth": float, "height": float} (alias: {"X": float, "Y": float, "Z": float})
            - torus: {"major_radius": float, "minor_radius": float} (alias: {"R_0": float, "a": float})
            - cylindrical: {"radius": float, "height": float} (alias: {"a": float, "h": float})
            - spherical: {"radius": float} (alias: {"a": float})

        Returns
        -------
        self : Voxel
        """
        if coordinate_type is None:
            self._coordinate_type = "cartesian"
        elif coordinate_type in COORDINATE_TYPES:
            if self._coordinate_type is None:
                self._coordinate_type = coordinate_type
            elif self._coordinate_type != coordinate_type:
                self._coordinate_type = coordinate_type
                self._coordinate_parameters = {}
        else:
            raise ValueError("The coordinate type is not supported.")

        if rotation is None:
            self._rotation_matrix = np.eye(3)
        elif isinstance(rotation, Rotation):
            self._rotation_matrix = rotation.as_matrix()
        elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
            self._rotation_matrix = rotation

        coordinate_keys = COORDINATE_PARAMETER_KEYS[self._coordinate_type]

        if coordinate_parameters is None:
            coordinate_parameters = {}
        if isinstance(coordinate_parameters, dict):
            for k in coordinate_parameters.keys():
                if all([k not in keys for keys in coordinate_keys]):
                    error_msg = (f"The key '{k}' is not supported in {self._coordinate_type} coordinates. "
                                 f"Coordinate type is set to {self._coordinate_type}, "
                                 f"so the keys must be {coordinate_keys[0]} (alias: {coordinate_keys[1]})")
                    raise KeyError(error_msg)
            for key, alias in zip(coordinate_keys[0], coordinate_keys[1]):
                # if both key and alias are in coordinate_parameters, key is used
                # <=> alias is used only when key is not in coordinate_parameters
                self._coordinate_parameters[key] = coordinate_parameters.get(key, coordinate_parameters.get(alias, 1))
        else:
            error_msg = (f"Coordinate parameter is must be a dict. \n"
                         f"Coordinate type is set to {self._coordinate_type}, "
                         f"so the keys must be {coordinate_keys[0]} (alias: {coordinate_keys[1]})")
            raise TypeError(error_msg)

        my_print(f"coordinate type: {self._coordinate_type}, coordinate parameters: {self._coordinate_parameters}",
                 show=show)
        return self

    @property
    def coordinate_parameters(self):
        return self._coordinate_parameters

    def normalized_coordinates(self, points=None):
        """

        Parameters
        ----------
        points: np.ndarray
            The points in the world coordinate system (n_points, 3)

        Returns
        -------
        normalized_points: np.ndarray
            The normalized points (n_points, 3)
        """
        if points is None:
            return self._normalized_coordinates(self.gravity_center.dot(self._rotation_matrix.T))
        else:
            return self._normalized_coordinates(points.dot(self._rotation_matrix.T))

    @property
    def axes(self):
        """
        get axes [x_axis, y_axis, z_axis]
        Returns
        -------
        list[np.ndarray, np.ndarray, np.ndarray]
        """
        return self._axes

    @axes.setter
    def axes(self, axes):
        self._axes = [np.array(axis) for axis in axes]

    @property
    def x_axis(self):
        return self.axes[0]

    @property
    def y_axis(self):
        return self.axes[1]

    @property
    def z_axis(self):
        return self.axes[2]

    @property
    def vx(self):
        """
        get x coordinates of vertices
        Returns
        -------
        np.ndarray (N_voxel, 8)
        """
        return self._vertices[..., 0]

    @property
    def vy(self):
        """
        get y coordinates of vertices
        Returns
        -------
        np.ndarray (N_voxel, 8)
        """
        return self._vertices[..., 1]

    @property
    def vz(self):
        """
        get z coordinates of vertices
        Returns
        -------
        np.ndarray (N_voxel, 8)
        """
        return self._vertices[..., 2]

    @property
    def grid_3d(self):
        """
        get grid points as 3d array
        Returns
        -------
        np.ndarray (3, N_x + 1, N_y + 1, N_z + 1)
        """
        return self._grid.T.reshape((3, *self.grid_shape))

    @property
    def voxel_indices_3d(self):
        """
        get voxel indices as 3d array
        Returns
        -------
        np.ndarray (N_x, N_y, N_z)
        """
        return self._voxel_indices.T.reshape((3, *self.voxel_shape))

    @property
    def shape(self):
        """
        alias of voxel_shape
        Returns
        -------
        (int, int, int)
        """
        return self.voxel_shape

    @property
    def N(self):
        """
        alias of N_voxel
        Returns
        -------
        int
        """
        return self.N_voxel

    @property
    def voxel2vertices(self):
        """
        get mapping from voxel index to vertex indices

        Returns
        -------
        np.ndarray (N_voxel, 8)
        """
        return self._voxel2vertices

    def set_world(self, world_obj):
        self._world = world_obj

    @staticmethod
    def uniform_voxel(ranges, shape, **kwargs):
        """
        create Voxel with uniform axes

        Parameters
        ----------
        ranges : ((float, float), (float, float), (float, float))
            axis ranges
        shape : (int, int, int)
            voxel shape
        kwargs : dict
            other arguments of Voxel

        Returns
        -------
        voxel : Voxel
        """
        axes = [np.round(np.linspace(start, end, num + 1), 6) for (start, end), num in zip(ranges, shape)]
        voxel = Voxel(*axes, **kwargs)
        return voxel

    def update(self):
        """
        Update attributes.
            - Avoids allocating the full `grid` / `vertices` unless explicitly requested.
            - Computes voxel centers, cell sizes and volumes in a vectorized way.
            - Builds `vertices_indices` with a fully vectorized formula.

        Parameters
        ----------
        build_grid : bool, default False
            If True, materialize `self._grid` ((N_grid,3)). Otherwise keep it None.
        build_vertices : bool, default False
            If True, materialize `self._vertices` ((N_voxel,8,3)). Otherwise keep it None.
        """

        # Basic shapes
        # grid
        self._grid_shape = tuple(len(axis) for axis in self.axes)  # (N_x+1, N_y+1, N_z+1)
        self._N_grid = int(np.prod(self._grid_shape))
        if self._N_grid == 0:
            return

        # voxel
        self._voxel_shape = tuple(max(0, N__ - 1) for N__ in self._grid_shape)  # (N_x, N_y, N_z)
        self._N_voxel = int(np.prod(self._voxel_shape))

        # Axis ranges
        self._ranges = tuple((float(np.min(axis)), float(np.max(axis))) for axis in self.axes)

        # Precompute per-axis diffs and centers (1D)
        x, y, z = self.axes

        # Vectorized voxel indices (i,j,k) and vertices_indices
        if self._N_voxel == 0:
            # degenerate
            self._voxel_indices = np.empty((0, 3), dtype=np.uint64)
            self._vertices_indices = np.empty((0, 8), dtype=np.uint64)
            self._vertices = None
        else:
            Vx, Vy, Vz = self._voxel_shape
            Gx, Gy, Gz = self._grid_shape

            ii, jj, kk = np.meshgrid(np.arange(Vx), np.arange(Vy), np.arange(Vz), indexing="ij")
            self._voxel_indices = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3).astype(np.uint64)  # (N_voxel, 3)

            # voxel_numbers = np.arange(self._N_voxel)  # (N_voxel,)
            # i_indices, j_indices, k_indices = self.get_voxel_position(voxel_numbers).T  # (N_voxel,) x3
            # vertex linear index base (grid is in z->y->x order: n = k + Nz*(j + Ny*i))
            # base = k_indices + Nz * (j_indices + Ny * i_indices)  # (N_voxel,)
            base = self._voxel_indices @ np.array([Gz * Gy, Gz, 1])  # (N_voxel,)
            # fixed offsets for the 8 vertex corners
            offset = np.array([0, 1, Gz, Gz + 1, Gz * Gy,
                               Gz * Gy + 1, Gz * Gy + Gz, Gz * Gy + Gz + 1])
            self._vertices_indices = np.array([base[:, None] + offset[None, :]],
                                              dtype=np.uint64).squeeze()  # (N_voxel,8)

            # Per-voxel cell sizes (N_voxel, 3) without building full 3D stacks
            # Use meshgrid on 1D diffs and 1D centers to create compact 3D then flatten.
            # DX, DY, DZ = np.meshgrid(dx, dy, dz, indexing="ij")
            # self._d = np.stack([DX, DY, DZ], axis=-1).reshape(-1, 3)  # (N_voxel,3)
            #
            # CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing="ij")
            # self._gravity_center = np.stack([CX, CY, CZ], axis=-1).reshape(-1, 3)  # (N_voxel,3)
            #
            # # Volumes (N_voxel,)
            # self._volume = (DX * DY * DZ).reshape(-1)

        # normalized coordinate function refresh (same as update())
        if self._coordinate_type == "cartesian":
            self._normalized_coordinates = cartesian_coordinates(**self._coordinate_parameters)
        elif self._coordinate_type == "torus":
            self._normalized_coordinates = torus_coordinates(**self._coordinate_parameters)
        elif self._coordinate_type == "cylindrical":
            self._normalized_coordinates = cylindrical_coordinates(**self._coordinate_parameters)
        elif self._coordinate_type == "spherical":
            self._normalized_coordinates = spherical_coordinates(**self._coordinate_parameters)

        return

    def get_voxel_position(self, n: int | np.ndarray):
        """
        Get voxel position (i, j, k) from voxel number n.

        Parameters
        ----------
        n : int or np.ndarray
            voxel number

        Returns
        -------
        np.ndarray
            voxel position (i, j, k)
        """
        n = self._type_check_n_voxel(n)
        # N_x, N_y, N_z = self.voxel_shape
        # i = n // (N_y * N_z)
        # j = (n % (N_y * N_z)) // N_z
        # k = n % N_z
        # return np.stack((i, j, k), axis=-1)  # (n_voxel, 3) or (3,)
        # return np.array(np.unravel_index(n, self.voxel_shape)).T  # (n_voxel, 3) or (3,)
        return self._voxel_indices[n]  # (n_voxel, 3)

    def get_voxel_number(self, i, j, k):
        """

        Parameters
        ----------
        i : int
            index along x
        j : int
            index along y
        k : int
            index along z

        Returns
        -------
        int
            voxel number
        """
        N_x, N_y, N_z = self.voxel_shape
        if not (0 <= i < N_x and 0 <= j < N_y and 0 <= k < N_z):
            raise IndexError(f"Index out of range: {(i, j, k)} for voxel_shape {self.voxel_shape}")
        n = k + N_z * (j + N_y * i)
        return n

    # -------- Lightweight helper properties (computed from 1D data) --------------
    @property
    def dx_axis(self):
        """1D cell size along x (length N_x)."""
        if self.axes is None or len(self.axes[0]) == 0:
            return np.array([])
        return np.diff(self.axes[0])

    @property
    def dy_axis(self):
        """1D cell size along y (length N_y)."""
        if self.axes is None or len(self.axes[1]) == 0:
            return np.array([])
        return np.diff(self.axes[1])

    @property
    def dz_axis(self):
        """1D cell size along z (length N_z)."""
        if self.axes is None or len(self.axes[2]) == 0:
            return np.array([])
        return np.diff(self.axes[2])

    @property
    def cx_axis(self):
        """1D centers along x (length N_x)."""
        if self.axes is None or len(self.axes[0]) == 0:
            return np.array([])
        x = self.axes[0]
        return 0.5 * (x[:-1] + x[1:])

    @property
    def cy_axis(self):
        """1D centers along y (length N_y)."""
        if self.axes is None or len(self.axes[1]) == 0:
            return np.array([])
        y = self.axes[1]
        return 0.5 * (y[:-1] + y[1:])

    @property
    def cz_axis(self):
        """1D centers along z (length N_z)."""
        if self.axes is None or len(self.axes[2]) == 0:
            return np.array([])
        z = self.axes[2]
        return 0.5 * (z[:-1] + z[1:])

    def show_info(self):
        print(f"{self.__class__.__name__} info:")
        print(f"ranges: {self.ranges}")
        print(f"x_axis: {self.axes[0]}")
        print(f"y_axis: {self.axes[1]}")
        print(f"z_axis: {self.axes[2]}")
        print(f"grid_shape: {self.grid_shape} (N_grid={self.N_grid})")
        print(f"voxel_shape: {self.voxel_shape} (N_voxel={self.N_voxel})")
        # print(f"{self.ranges}")
        # print(f"{self.N_grid=}")
        # print(f"{self.grid_shape=}")
        # print(f"{self.N_voxel=}")
        # print(f"{self.voxel_shape=}")

    def _type_check_n_voxel(self, n=None):
        """
        type check for voxel number n

        Parameters
        ----------
        n : int, list[int], slice, np.ndarray, optional (default=None)
            voxel numbers

        Returns
        -------
        n : np.ndarray
            voxel numbers as np.ndarray
        """

        if n is None:
            n = np.arange(self.N_voxel)
        else:
            if isinstance(n, int):
                n = np.array([n])
            elif isinstance(n, list):
                n = np.array(n)
            elif isinstance(n, slice):
                n = np.arange(self.N_voxel)[n]
            elif isinstance(n, np.ndarray):
                if n.ndim == 1:
                    if n.dtype == bool:
                        n = np.arange(self.N_voxel)[n]
                    elif np.issubdtype(n.dtype, np.integer):
                        n = n
                    else:
                        raise TypeError(f"n must be int or boolean np.ndarray")
                else:
                    raise TypeError(f"n must be 1d np.ndarray")
            else:
                raise TypeError(f"n must be int, slice, list, or np.ndarray")

            if np.any((n < 0) | (n >= self.N_voxel)):
                raise IndexError(f"voxel number out of range: n should satisfy 0 <= n < {self.N_voxel}")
        return n.astype(int)

    def get_sub_voxel(self, n=None, res=None, verbose=0):
        """
        get sub voxels with resolution `res`
        Parameters
        ----------
        n : int, list[int], slice, optional (default=None)
            voxel numbers (if vertices is specified, n is ignored)
        res : int or (int, int, int), optional (default=3)
            resolution of sub voxels (x_res, y_res, z_res)

        Returns
        -------
        sub_voxel : list[Voxel]
            sub voxels with resolution `res`
        """

        self.res = res
        n = self._type_check_n_voxel(n)

        i, j, k = self.get_voxel_position(n).T.astype(int)
        x0 = self.x_axis[i]
        x1 = self.x_axis[i + 1]
        y0 = self.y_axis[j]
        y1 = self.y_axis[j + 1]
        z0 = self.z_axis[k]
        z1 = self.z_axis[k + 1]

        if isinstance(n, int):
            sub_voxels = Voxel(np.linspace(x0, x1, self.res[0] + 1),
                               np.linspace(y0, y1, self.res[1] + 1),
                               np.linspace(z0, z1, self.res[2] + 1))
        else:
            sub_voxels = [Voxel(np.linspace(x0_, x1_, self.res[0] + 1),
                                np.linspace(y0_, y1_, self.res[1] + 1),
                                np.linspace(z0_, z1_, self.res[2] + 1))
                          for x0_, x1_, y0_, y1_, z0_, z1_ in my_zip(x0, x1, y0, y1, z0, z1, disable=verbose <= 0)]
        return sub_voxels

    def sub_voxel_interpolator(self, n=None, res=None, verbose=0):
        """
        get n-th sub voxel interpolator with resolution `res`

        Parameters
        ----------
        n : int or list[int]
            voxel number(s)
        res : int or (int, int, int), optional (default=3)
            resolution of sub voxels (x_res, y_res, z_res)
        verbose : int, default=0
            verbosity level

        Returns
        -------
        interpolator : scipy.sparse.csr_matrix (N_sub_voxel, N_voxel)
            interpolator for n-th sub voxel (N_sub_grid = x_res * y_res * z_res)
        """
        self.res = res
        n = self._type_check_n_voxel(n)
        # self.set_voxel2vertices(exist_ok=True, n_jobs=0, verbose=0)
        if self.voxel2vertices is None:
            raise RuntimeError("voxel2vertices is not calculated. "
                               "Please run set_voxel2vertices() before calling sub_voxel_interpolator().")

        # not necessary to check n because n is verified in set_voxel2vertices
        # the return value of set_voxel2vertices is a list of lil_matrix (length = len(n) or 1 (when n is int))
        # interpolator = [self._sub_voxel_matrix @ interpolate_vertices for interpolate_vertices in
        #                 self.set_voxel2vertices(n)]
        # voxel2vertices = self.set_voxel2vertices(verbose=verbose)  # (N_grid, N_voxel) or list of such matrices

        def _append_interpolator_matrix(vi):
            selector = sparse.coo_matrix((np.ones(8, dtype=bool), (np.arange(8), vi)),
                                         shape=(8, self.N_grid), dtype=bool).tocsr()  # (8, N_grid)
            return self._sub_voxel_matrix @ (selector @ self.voxel2vertices)

        interpolator = [_append_interpolator_matrix(self._vertices_indices[_n]) for _n in
                        my_tqdm(n, desc="Creating sub-voxel interpolator", disable=verbose <= 0)]

        return interpolator if len(interpolator) > 1 else interpolator[0]

    def get_random_point(self, n, N=1, rng=None):
        """
        get random points in n-th voxel

        Parameters
        ----------
        n : int
            voxel number
        N : int, default=1
            number of points
        rng : np.random._generator.Generator or float, default=None
            random number generator or seed

        Returns
        -------
        points : np.ndarray (N, 3)
            random points
        """
        if rng is None:
            rng = np.random.default_rng()
        else:
            if isinstance(rng, float):
                rng = np.random.default_rng(rng)
            elif isinstance(rng, np.random.Generator):
                pass
            else:
                raise TypeError(f"{rng=} must be float (seed) or np.random._generator.Generator")
        return rng.uniform(self.vertices[n][0], self.vertices[n][-1], (N, 3))

    def set_voxel2vertices(self, exist_ok=True, n_jobs=-2, verbose=0):
        """
        interpolate values to vertices

        Parameters
        ----------
        exist_ok : bool, default=True
            if True and voxel2vertices is already calculated, do nothing
        verbose : int, default=0
            verbosity level
        n_jobs : int, default=-2
            number of parallel jobs (-1 means all CPUs, -2 means all but one)

        Returns
        -------
        scipy.sparse.csr_matrix (N_grid, N_voxel)
            interpolation matrix from voxel values to grid values
        """
        if exist_ok and self._voxel2vertices is not None:
            return None

        cols = np.repeat(np.arange(self.N_voxel, dtype=np.int64), 8)  # (N_voxel * 8,)
        rows = self.vertices_indices.reshape(-1).astype(np.int64)  # (N_voxel * 8,)
        included_vertices = sparse.coo_matrix((np.ones_like(cols, dtype=bool), (rows, cols)),
                                              shape=(self.N_grid, self.N_voxel),
                                              dtype=bool).tocsr()  # (N_grid, N_voxel)

        def _func(virtual_vertices, grid):
            d_virtual = np.asarray([np.ptp(vv) for vv in virtual_vertices.T])
            para = (grid - virtual_vertices[0])[d_virtual.nonzero()] / d_virtual[d_virtual.nonzero()]  #
            data = [float(x) for x in range(0)]
            if len(para) == 3:
                # if i-th grid point is included in 8 voxels (inside the virtual voxel)
                a, b, c = para
                data.extend([(1 - a) * (1 - b) * (1 - c), (1 - a) * (1 - b) * c,
                             (1 - a) * b * (1 - c), (1 - a) * b * c,
                             a * (1 - b) * (1 - c), a * (1 - b) * c,
                             a * b * (1 - c), a * b * c])
            elif len(para) == 2:
                # if i-th grid point is included in 4 voxels (on the surface of grid)
                a, b = para
                data.extend([(1 - a) * (1 - b), (1 - a) * b,
                             a * (1 - b), a * b])
            elif len(para) == 1:
                # if i-th grid point is included in 2 voxels (on the edge of grid)
                a, = para
                data.extend([1 - a, a])
            else:
                # if i-th grid point is included in 1 voxel (on the vertex of grid)
                data.extend([1])
            return data

        #
        # vertices = self._vertices_indices[n]
        # matrix_list = []
        # for nth_vertices in vertices:
        #     matrix = sparse.lil_matrix((8, self.N_voxel))
        #     for i, v in enumerate(nth_vertices):
        #         included_voxel = np.where(np.any(self.vertices_indices == v, axis=1))[0]
        #         matrix[i, included_voxel] = _func(self.gravity_center[included_voxel], self.grid[v])
        #     matrix_list.append(matrix.tocsr())
        # return matrix_list
        # data = []
        # rows = []
        # cols = []
        # for v in my_range(self.N_grid, verbose=verbose, desc="Interpolating vertices"):
        #     voxel_indices = included_vertices.indices[included_vertices.indptr[v]:included_vertices.indptr[v + 1]]
        #     if len(voxel_indices) == 0:
        #         continue
        #     gc = self.get_gravity_center(voxel_indices)
        #     i, j, k = np.unravel_index(v, self.grid_shape)
        #     x, y, z = self.x_axis[i], self.y_axis[j], self.z_axis[k]
        #     data.extend(_func(gc, np.array([x, y, z])))
        #     rows.extend([v] * len(voxel_indices))
        #     cols.extend(voxel_indices.tolist())

        # parallel version
        def _process_vertex(v):
            voxel_indices = included_vertices.indices[included_vertices.indptr[v]:included_vertices.indptr[v + 1]]
            if len(voxel_indices) == 0:
                return [], [], []
            gc = self.get_gravity_center(voxel_indices)
            i, j, k = np.unravel_index(v, self.grid_shape)
            x, y, z = self.x_axis[i], self.y_axis[j], self.z_axis[k]
            data_ = _func(gc, np.array([x, y, z]))
            rows_ = [v] * len(voxel_indices)
            cols_ = voxel_indices.tolist()
            return data_, rows_, cols_

        # results = Parallel(n_jobs=n_jobs, verbose=0, backend="threading")(
        #     [delayed(_process_vertex)(v) for v in my_range(self.N_grid,
        #                                                    desc="Interpolating vertices", disable=verbose <= 0)])
        # no parallel
        results = [_process_vertex(v) for v in my_range(self.N_grid,
                                                        desc="Interpolating vertices", disable=verbose <= 0)]
        data_list, rows_list, cols_list = zip(*results)
        data = list(chain.from_iterable(data_list))
        rows = list(chain.from_iterable(rows_list))
        cols = list(chain.from_iterable(cols_list))

        self._voxel2vertices = sparse.coo_matrix((data, (rows, cols)), shape=(self.N_grid, self.N_voxel)).tocsr()
        return None


if __name__ == "__main__":
    # obj = Voxel(*[[10, 20, 40, 50, 80], [10, 15, 25, 40, 60], [10, 25, 40, 60, 85, 100]])
    obj = Voxel(coordinate_type="torus", coordinate_parameters={"major_radius": 100, "minor_radius": 50})
    # obj = obj.uniform_axes(ranges=[[-750, 750], [-750, 750], [-250, 250]], shape=[5, 5, 5], show_info=True)
    obj = obj.uniform_voxel(ranges=[[-1, 1], [-1, 1], [-1, 1]], shape=[49, 49, 49])
    print(dir(obj))
    _ = obj.sub_voxel_interpolator(n=0, res=5)
    obj.get_sub_voxel(5)
    obj.get_sub_voxel(np.array([0, 1, 2, 3, 4, 5]))
    # obj.show_info()

    # start = time.time()
    # obj.set_voxel2vertices()
    # print(time.time() - start)
