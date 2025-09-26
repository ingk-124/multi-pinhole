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
    """
    get interpolation matrix from values at vertices

    Parameters
    ----------
    res : int or (int, int, int), default=3
        resolution of sub grid (x_res, y_res, z_res)

    Returns
    -------
    matrix : np.ndarray (N_sub_voxel, 8) (N_sub_voxel = x_res * y_res * z_res)
        interpolation matrix
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


def shifted_torus(r, theta, cx, cy):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    A = 1 - cx ** 2 - cy ** 2
    B = cx * x + cy * y - 1
    C = 1 - x ** 2 - y ** 2
    c = (-B - np.sqrt(B ** 2 - A * C)) / A
    return c


def helical_island(r, theta, phi, a_i, w_i, psi_0=0, m_i=1, n_i=0, alpha=4):
    r_i = r - a_i
    psi = m_i * theta - n_i * phi + psi_0
    z_1 = np.exp(-((r_i / w_i) ** 2))
    z_2 = np.cos(psi)
    z_3 = 1 - (1 - 2 * r) ** alpha
    return z_1 * z_2 * z_3


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
        self._grid = None

        # voxel
        self._N_voxel = 0
        self._voxel_shape = None
        self._voxel_indices = None

        # vertices
        self._vertices_indices = None
        self._vertices = None

        # voxel properties
        self._d = None
        self._gravity_center = None
        self._volume = None

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

        # set attributes
        self.axes = axes
        self.set_coordinate(coordinate_type=coordinate_type, rotation=rotation, **self.coordinate_parameters)
        self.res = sub_voxel_resolution

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
        return self._grid

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
    def voxel_indices(self):
        return self._voxel_indices

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
    def vertices(self):
        """
        get vertices
        Returns
        -------
        np.ndarray (N_voxel, 8, 3)
        """
        return self._vertices

    @property
    def d(self):
        """
        get voxel edge length
        Returns
        -------
        np.ndarray (N, 3)
        """
        return self._d

    @property
    def gravity_center(self):
        """
        get gravity center
        Returns
        -------
        np.ndarray (N_voxel, 3)
        """
        return self._gravity_center

    @property
    def volume(self):
        return self._volume

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
        self.update()
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
            return self._normalized_coordinates(self._gravity_center.dot(self._rotation_matrix.T))
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
        self.update()

    @property
    def x_axis(self):
        return self.axes[0]

    @x_axis.setter
    def x_axis(self, x_axis):
        self._axes[0] = np.array(x_axis)
        self.update()

    @property
    def y_axis(self):
        return self.axes[1]

    @y_axis.setter
    def y_axis(self, y_axis):
        self._axes[1] = np.array(y_axis)
        self.update()

    @property
    def z_axis(self):
        return self.axes[2]

    @z_axis.setter
    def z_axis(self, z_axis):
        self._axes[2] = np.array(z_axis)
        self.update()

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

    def set_world(self, world_obj):
        self._world = world_obj

    def reset_axes(self):
        """
        reset axes with ranges and N_voxel
        """
        self.uniform_axes(self.ranges, self.voxel_shape)

    def uniform_axes(self, ranges, shape, show_info=False):
        """
        set uniform axes with ranges and divs

        Parameters
        ----------
        ranges : ((float, float), (float, float), (float, float))
            axis ranges
        shape : (int, int, int)
            voxel shape
        show_info : bool
            show info or not
        Returns
        -------
        self
        """
        axes, steps = zip(
            *[np.linspace(start, end, num + 1, retstep=True) for (start, end), num in zip(ranges, shape)])
        # for i, (axis, step, div) in enumerate(zip(axes, steps, divs)):
        #     print(f"axis {i}: {div=}")
        #     print(f"{step=}, {axis=}")
        self.axes = axes
        if show_info:
            self.show_info()

        return self

    def update(self):
        """
        update all attributes

        if 0 is in grid_shape i.e. N_grid == 0, then do nothing
        """

        # vertex points
        self._grid_shape = tuple(len(axis) for axis in self.axes)  # (N_x + 1, N_y + 1, N_z + 1)
        self._N_grid = np.prod(self._grid_shape)  # (N_x + 1) * (N_y + 1) * (N_z + 1)
        if self._N_grid == 0:
            return

        # increment coordinates in the order of z->y->x
        self._grid = np.stack(np.meshgrid(*self.axes, indexing="ij")).reshape((3, -1)).T  # (N_grid, 3)
        # ((N_x + 1) * (N_y + 1) * (N_z + 1), 3)

        # voxel number
        # voxels are 1 less than grid in each axis
        self._voxel_shape = tuple(len(axis) - 1 for axis in self.axes)  # (N_x, N_y, N_z)
        self._N_voxel = np.prod(self._voxel_shape)  # N_x * N_y * N_z

        # axis ranges
        self._ranges = tuple((np.min(axis), np.max(axis)) for axis in self.axes)

        # 3d voxel index (i,j,k)
        self._voxel_indices = np.stack(np.meshgrid(*[np.arange(s) for s in self._voxel_shape],
                                                   indexing="ij")).reshape((3, -1)).T  # (N_voxel, 3)

        # get 8 indices of vertices defining each voxel (k->j->i [[0,0,0],[0,0,1],...,[1,1,1]])
        vertex_index = np.mgrid[0:2, 0:2, 0:2].reshape((3, -1)).T  # (8, 3)

        # (i,j,k) -> n: n = k + N_z * (j + N_y * i)
        self._vertices_indices = np.array([[k + self._grid_shape[2] * (j + self._grid_shape[1] * i)
                                            for i, j, k in vertex_index + ijk] for ijk in
                                           my_tqdm(self._voxel_indices, verbose=1)])  # (N_voxel, 8)
        self._vertices = self._grid[self._vertices_indices]  # (N_voxel, 8, 3)

        # voxel edge length
        diff = [np.diff(axis) for axis in self.axes]
        self._d = np.stack(np.meshgrid(*diff, indexing="ij")).reshape((3, -1)).T  # (N_grid, 3)

        # gravity center
        self._gravity_center = np.mean(self._vertices, axis=1)

        # volume
        self._volume = np.prod(self._d, axis=1)

        if self._coordinate_type == "cartesian":
            self._normalized_coordinates = cartesian_coordinates(**self._coordinate_parameters)
        elif self._coordinate_type == "torus":
            self._normalized_coordinates = torus_coordinates(**self._coordinate_parameters)
        elif self._coordinate_type == "cylindrical":
            self._normalized_coordinates = cylindrical_coordinates(**self._coordinate_parameters)
        elif self._coordinate_type == "spherical":
            self._normalized_coordinates = spherical_coordinates(**self._coordinate_parameters)

        return

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

    def get_sub_voxel(self, vertices=None, n=None, res=None):
        """
        get sub voxels with resolution `res`
        Parameters
        ----------
        vertices : np.ndarray (8, 3) or (n_voxel, 8, 3), optional (default=None)
            vertices of voxels (n_voxel is the number of voxels)
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
        if vertices is None:
            if n is None:
                vertices = self.vertices
            else:
                if isinstance(n, int):
                    n = [n]
                elif isinstance(n, slice) or isinstance(n, list):
                    pass
                elif isinstance(n, np.ndarray):
                    if n.ndim == 1 and n.dtype in [bool, int]:
                        pass
                    else:
                        raise TypeError(f"{n=} must be int or bool np.ndarray")
                else:
                    raise TypeError(f"{n=} must be int, slice, list, or np.ndarray")
                vertices = self.vertices[n]
        else:
            vertices = np.array(vertices, ndmin=3)
            if vertices.shape[1:] != (8, 3):
                raise ValueError(f"vertices must be (8, 3) or (n_voxel, 8, 3)")

        sub_axes = [[np.linspace(x, y, r + 1) for x, y, r in zip(vox[0], vox[-1], self._res)] for vox in vertices]
        sub_voxels = [Voxel(*axes) for axes in sub_axes]
        if len(sub_voxels) == 1:
            return sub_voxels[0]
        else:
            return sub_voxels

    def sub_voxel_interpolator(self, n, res=None):
        """
        get n-th sub voxel interpolator with resolution `res`

        Parameters
        ----------
        n : int or list[int]
            voxel number(s)
        res : int or (int, int, int), optional (default=3)
            resolution of sub voxels (x_res, y_res, z_res)

        Returns
        -------
        interpolator : scipy.sparse.csr_matrix (N_sub_voxel, N_voxel)
            interpolator for n-th sub voxel (N_sub_grid = x_res * y_res * z_res)
        """
        self.res = res
        # not necessary to check n because n is verified in _interpolate_vertices
        # the return value of _interpolate_vertices is a list of lil_matrix (length = len(n) or 1 (when n is int))
        interpolator = [self._sub_voxel_matrix @ interpolate_vertices for interpolate_vertices in
                        self._interpolate_vertices(n)]
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

    # TODO: make calc_coordinate in Voxel class
    def _interpolate_vertices(self, n):
        """
        interpolate values to vertices

        Parameters
        ----------
        n : int or list[int] or np.ndarray (bool, int)
            voxel number

        Returns
        -------
        matrix_list : list[scipy.sparse.csr_matrix]
            interpolation matrix list
            if n is int, the length of matrix_list is 1
        """

        # type check for n
        if isinstance(n, int):
            n = [n]
        elif isinstance(n, slice) or isinstance(n, list):
            pass
        elif isinstance(n, np.ndarray):
            if n.ndim == 1 and n.dtype in [bool, int]:
                pass
            else:
                raise TypeError(f"{n=} must be int or bool np.ndarray")
        else:
            raise TypeError(f"{n=} must be int, slice, list, or np.ndarray")

        def _func(virtual_vertices, grid):
            d_virtual = np.asarray([np.ptp(vv) for vv in virtual_vertices.T])
            para = (grid - virtual_vertices[0])[d_virtual.nonzero()] / d_virtual[d_virtual.nonzero()]
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

        vertices = self._vertices_indices[n]
        matrix_list = []
        for nth_vertices in vertices:
            matrix = sparse.lil_matrix((8, self.N_voxel))
            for i, v in enumerate(nth_vertices):
                included_voxel = np.where(np.any(self.vertices_indices == v, axis=1))[0]
                matrix[i, included_voxel] = _func(self.gravity_center[included_voxel], self.grid[v])
            matrix_list.append(matrix.tocsr())
        return matrix_list


if __name__ == "__main__":
    # obj = Voxel(*[[10, 20, 40, 50, 80], [10, 15, 25, 40, 60], [10, 25, 40, 60, 85, 100]])
    obj = Voxel(coordinate_type="torus", coordinate_parameters={"major_radius": 100, "minor_radius": 50})
    obj = obj.uniform_axes(ranges=[[-750, 750], [-750, 750], [-250, 250]], shape=[5, 5, 5], show_info=True)
    print(dir(obj))
    _ = obj.sub_voxel_interpolator(n=0)
    # obj.show_info()

    # start = time.time()
    # obj._interpolate_vertices()
    # print(time.time() - start)
