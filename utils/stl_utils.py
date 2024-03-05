# Numpy-style docstring
import time
from numbers import Number
from typing import Tuple, Callable, Union, List

import numpy as np
import plotly.io as pio
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits import mplot3d
from plotly import graph_objects as go
from scipy import sparse
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
from stl import mesh
from tqdm.auto import trange

from utils.my_stdio import my_print

pio.renderers.default = "firefox"

# type aliases
# any length vector like object (accepts numpy.ndarray, list, tuple) (including 0D, 1D, 2D, 3D, etc.)

VectorLike = Union[np.ndarray, List[Number], Tuple[Number], Number]
# 2D vector like object (accepts numpy.ndarray, list, tuple)
Vector2DLike = Union[np.ndarray, List[Number], Tuple[Number, Number]]
# 3D vector like object (accepts numpy.ndarray, list, tuple)
Vector3DLike = Union[np.ndarray, List[Number], Tuple[Number, Number, Number]]
# Matrix like object (2D array) (accepts numpy.ndarray, list of list, tuple of tuple, etc.)
MatrixLike = Union[np.ndarray, List[List[Number]], Tuple[List[Number]], Tuple[List[Number]], Tuple[Tuple[Number]]]


def shape_check(shape: str, size: Union[Number, Vector2DLike], ok_shapes: dict = None) -> Tuple[str, Vector2DLike]:
    """Check the shape and size of the object.

    Parameters
    ----------
    shape : str (circle, ellipse, rectangle)
        The shape of the object.
    size : Union[Number, Vector2DLike]
        The size of the object.
    ok_shapes : dict, optional
        The acceptable shapes, by default {'circle':1, 'ellipse':2,'rectangle':2}

    Returns
    -------
    Tuple[str, Vector2DLike]
        The shape and size of the object.
    """
    if ok_shapes is None:
        ok_shapes = {'circle': 1, 'ellipse': 2, 'rectangle': 2}

    if shape not in ok_shapes.keys():
        raise ValueError(f"shape must be one of {ok_shapes.keys()}")

    if isinstance(size, Number):
        size = [size, ] * ok_shapes[shape]
    elif type(size) in [np.ndarray, list, tuple] and len(set(size)) == ok_shapes[shape]:
        pass
    else:
        raise ValueError(f"size must be a number or a list of length {ok_shapes[shape]} (shape={shape})")

    if shape == 'circle':
        size = [size[0], size[0]]

    return shape, np.array(size)


def generate_aperture_stl(shape: str, size: Union[Number, Vector2DLike], resolution: Union[Number, Vector2DLike],
                          max_size: Union[Number, Vector2DLike] = None) -> mesh.Mesh:
    """Generate an STL model of the aperture.

    Return a 2D STL model of the aperture. The vertexes include 10x10 grid points and edge points of the aperture.

    Parameters
    ----------
    shape : str
        The shape of the object. (circle, ellipse, rectangle)
    size : Vector2DLike, float
        The size of the object.
        radius for circle, [semi-major radius, semi-minor radius] for ellipse, [width, height] for rectangle.
    resolution : int
        The resolution of edge points of the aperture.
    max_size : Vector2DLike, float, optional (default is None)
        The maximum size of the aperture
        If max_size is None, the maximum size is set to be 1.5 times the size of the aperture.

    Returns
    -------
    stl_obj : mesh.Mesh
        The STL model of the aperture.
    """

    # check the shape and size of the object
    shape, size = shape_check(shape, size)
    if max_size is None:
        max_size = 1.5 * size
    max_size = [max_size, ] * 2 if isinstance(max_size, Number) else np.array(max_size)
    x_arr, y_arr = np.linspace(-max_size[0], max_size[0], 10), np.linspace(-max_size[1], max_size[1], 10)

    if shape == 'circle' or shape == 'ellipse':
        def condition(x, y):
            return (x / size[0]) ** 2 + (y / size[1]) ** 2 <= 1

        t = np.linspace(0, 2 * np.pi, resolution)
        edge_points = np.array([size[0] * np.cos(t), size[1] * np.sin(t)]).T

    elif shape == 'rectangle':
        def condition(x, y):
            return np.all([np.abs(x) <= size[0] / 2, np.abs(y) <= size[1] / 2], axis=0)

        x_edge = np.linspace(-size[0] / 2, size[0] / 2, resolution)
        y_edge = np.linspace(-size[1] / 2, size[1] / 2, resolution)
        e_1 = np.array([x_edge, np.ones_like(x_edge) * size[1] / 2]).T
        e_2 = np.array([np.ones_like(y_edge) * size[0] / 2, y_edge]).T
        e_3 = np.array([x_edge, np.ones_like(x_edge) * -size[1] / 2]).T
        e_4 = np.array([np.ones_like(y_edge) * -size[0] / 2, y_edge]).T
        edge_points = np.concatenate([e_1, e_2, e_3, e_4])
    else:
        raise ValueError(f"shape must be one of ['circle', 'ellipse', 'rectangle']")

    outer_points = np.array(np.meshgrid(x_arr, y_arr, indexing='ij')).reshape(2, -1).T
    outer_points = outer_points[~condition(outer_points[:, 0] / 1.5, outer_points[:, 1] / 1.5)]
    points = np.concatenate([outer_points, edge_points])

    return make_2D_surface(points, condition)


def rotate_model(model: mesh.Mesh, order: str = 'xyz', angles: Vector3DLike = (0, 0, 0), matrix: np.ndarray = None,
                 origin: Vector3DLike = (0, 0, 0), degrees: bool = True) -> mesh.Mesh:
    """Rotate the model by given Euler angles using

    Parameters
    ----------
    model : mesh.Mesh
        The model to be rotated.
    order : str, optional, default "xyz"
        The order of the rotation, by default "xyz".
    angles : Vector3DLike, float
        The angles to be rotated.
    matrix : np.ndarray, optional, default None
        The rotation matrix, by default None. If this argument is specified, the angles and order are ignored.
    origin : Vector3DLike, float, optional, default (0, 0, 0)
        The origin of the rotation, by default (0, 0, 0).
    degrees : bool, optional, default True
        Whether the angles are in degrees, by default True.
    
    Returns
    -------
    mesh.Mesh
        The rotated model.
    """
    origin = np.array(origin)
    rotation_matrix = Rotation.from_euler(order, angles, degrees=degrees).as_matrix() if (
            matrix is None) else np.array(matrix)

    # copy the model
    model = copy_model(model)
    model.translate(-origin)
    model.rotate_using_matrix(rotation_matrix)
    model.translate(origin)
    return model


def copy_model(model: mesh.Mesh, translate: np.ndarray = (0, 0, 0), rotation_matrix: np.ndarray = None) -> mesh.Mesh:
    """Copy the model.

    Parameters
    ----------
    model : mesh.Mesh
        The model to be copied.
    translate : np.ndarray, optional, default (0, 0, 0)
        The translation vector, by default (0, 0, 0).
    rotation_matrix : np.ndarray, optional, default None
        The rotation matrix, by default None.

    Returns
    -------
    mesh.Mesh
        The copied model.
    """
    model = mesh.Mesh(model.data.copy())
    model.translate(translate)
    if rotation_matrix is not None:
        model.rotate_using_matrix(rotation_matrix)
    return model


def check_intersection(triangle: np.ndarray, start: np.ndarray, end_points: np.ndarray) -> np.ndarray:
    """Check if a line segment intersects with a mesh.

    This function checks if a line segment intersects with a triangle using the Möller–Trumbore intersection algorithm.
    The line segment is defined by the start point and end point. The triangle is defined by three vectors.

    The line segment is checked to intersect with the triangle by solving the following equation:
    e_1 = b - a, e_2 = c - a, d = end - start, n = e_1 x e_2, r = start - a
    Line eq. defined by the line segment: R(t) = start + t * d
    Plane eq. defined by the triangle: T(u, v) = a + u * e_1 + v * e_2
    Intersection eq.: R(t) = T(u, v)
                    <=> start + t * d = a + u * e_1 + v * e_2
                    <=> t * (-d) + u * e_1 + v * e_2 = r
                    <=> (-d, e_1, e_2) * (t, u, v)^T = r  ... (1)

    If n dot d = 0, the line segment is parallel to the triangle, not intersecting with the triangle.
    If n dot d != 0, solve the equation (1) for t, u, v and check the following conditions:
        1. 0 < t <= 1
        2. u >= 0, v >= 0, u + v <= 1
    If all the conditions are satisfied, the line segment intersects with the triangle.

    Parameters
    ----------
    triangle : numpy.ndarray (shape: (3, 3))
        Three vectors of a triangle.
    start : numpy.ndarray (shape: (3, ))
        Start point of the line segment.
    end_points : numpy.ndarray (shape: (n, 3))
        End points of the line segment. n is the number of line segments.

    Returns
    -------
    condition : numpy.ndarray (shape: (n, ))
        The condition whether the line segment intersects with the triangle.
    """
    if end_points.shape[0] == 0:
        return np.array([], dtype=bool)

    PARALLEL_THRESHOLD = 1e-6
    a, b, c = triangle  # three vectors of a triangle (a, b, c: (3,))
    e_1, e_2 = b - a, c - a  # two edges of a triangle (e_1, e_2: (3,))
    d_ = np.subtract(end_points, start)  # direction of the line segment (d: (n, 3))
    n = np.cross(e_1, e_2)  # normal vector of the triangle (n: (3,))
    r = np.subtract(start, a)  # vector from the start point to a (r: (3,))

    # solve the equation and check the conditions
    def tuv(d_):
        if np.abs(np.dot(n, d_)) > PARALLEL_THRESHOLD:
            t, u, v = np.linalg.solve(np.array([-d_, e_1, e_2]).T, r)
            if (0 < t <= 1) and (u >= 0) and (v >= 0) and (u + v <= 1):
                return True
        return False

    return np.apply_along_axis(func1d=tuv, axis=1, arr=d_)  # (n, ) boolean array


def delta_cone(mesh_obj: mesh.Mesh, start: np.ndarray) -> np.ndarray:
    """
    Calculate the condition whether the grid points are inside the cone defined by the mesh and the origin.

    Define the m-th triangle of the mesh as abc, the origin as o, the grid point as p.
    Vectors oa, ob, oc, and n are defined as a - o, b - o, c - o, and ob x oa, respectively.
    The plane oab satisfies n dot (x - oa) = 0. (where x is a point on the plane)
    The necessary condition for p to be inside the cone is that p is on the same side of the plane oab as c.
    It is equivalent to sign(n dot (p - oa)) = sign(n dot (c - oa)) <=> (n dot (p - o)) * (n dot (c - o)) >= 0.
    The condition that p is inside the cone is AND of the above conditions for three sides (oab, obc, oca) of the cone.

    Parameters
    ----------
    mesh_obj : mesh.Mesh object
        STL mesh object with M triangles.
    start : numpy.ndarray of shape (3, )
        The origin of the light.

    Returns
    -------
    n_oab : numpy.ndarray (shape: (M, 3))
        The normal vectors of the plane oab.
    n_obc : numpy.ndarray (shape: (M, 3))
        The normal vectors of the plane obc.
    n_oca : numpy.ndarray (shape: (M, 3))
        The normal vectors of the plane oca.
    zero_volume : numpy.ndarray of shape (M, )
        The condition whether the triangle has zero volume.

    """

    # check if the start point is on the plane defined by the mesh
    zero_volume = ~np.isclose(0., np.einsum('ij,ij->i', mesh_obj.normals, start - mesh_obj.v0))  # (M,)

    # origin: np.ndarray of shape (3, )
    # vector oa, ob, oc: np.ndarray of shape (M,3) (from origin to a, b, c)
    oa, ob, oc = mesh_obj.v0 - start, mesh_obj.v1 - start, mesh_obj.v2 - start
    # vectors p: np.ndarray of shape (N, 3) (from origin to grid points)
    # p = grid_points - start

    # plane oab: (b - a) x (origin - a) dot (b - a) = 0
    n_oab = np.cross(oa, ob)  # normal vectors of plane oab (M, 3)
    # n_oab_dot = np.einsum('ij,ij->i', oc - oa, n_oab)  # (M,)
    n_oab *= np.sign(np.einsum('ij,ij->i', oc - oa, n_oab))[:, None]  # normal vectors with sign (M, 3)
    # p_oab = np.einsum('ij,kj->ik', p, n_oab) * n_oab_dot >= 0  # (N, M) <- too large. Use sparse matrix & list comp.
    # p_oab = sparse.hstack([sparse.csr_matrix((p @ n) >= 0).T for n in n_oab], format='csr')  # (N, M)

    # plane obc: (c - b) x (origin - b) dot (c - b) = 0
    n_obc = np.cross(ob, oc)  # normal vectors of plane obc (M, 3)
    # n_obc_dot = np.einsum('ij,ij->i', oa - ob, n_obc)  # (M,)
    n_obc *= np.sign(np.einsum('ij,ij->i', oa - ob, n_obc))[:, None]  # normal vectors with sign (M, 3)
    # p_obc = np.einsum('ij,kj->ik', p, n_obc) * n_obc_dot >= 0  # (N, M)
    # p_obc = sparse.hstack([sparse.csr_matrix((p @ n) >= 0).T for n in n_obc], format='csr')  # (N, M)

    # plane oca: (a - c) x (origin - c) dot (a - c) = 0
    n_oca = np.cross(oc, oa)  # normal vectors of plane oca (M, 3)
    # n_oca_dot = np.einsum('ij,ij->i', ob - oc, n_oca)  # (M,)
    n_oca *= np.sign(np.einsum('ij,ij->i', ob - oc, n_oca))[:, None]  # normal vectors with sign (M, 3)
    # p_oca = np.einsum('ij,kj->ik', p, n_oca) * n_oca_dot >= 0  # (N, M)
    # p_oca = sparse.hstack([sparse.csr_matrix((p @ n) >= 0).T for n in n_oca], format='csr')  # (N, M)

    # return AND of the above conditions
    # cone = p_oab & p_obc & p_oca  # (N, M)
    # cone = p_oab.multiply(p_obc).multiply(p_oca)  # (N, M)
    # inside_cone = sparse.hstack([sparse.csr_matrix((p @ n_oab_ >= 0) & (p @ n_obc_ >= 0) & (p @ n_oca_ >= 0) & v).T for
    #                              n_oab_, n_obc_, n_oca_, v in
    #                              zip(n_oab, n_obc, n_oca, zero_volume)], format="csr")  # (N, M)
    # return inside_cone
    return n_oab, n_obc, n_oca, zero_volume


def check_distance(mesh_obj: mesh.Mesh, start: np.ndarray, grid_points: np.ndarray,
                   inside_cone: sparse.csr_matrix) -> np.ndarray:
    """
    Check if the grid points are farther than all vertices of the mesh.

    Parameters
    ----------
    mesh_obj : mesh.Mesh object
        STL mesh object with M triangles.
    start : numpy.ndarray (shape: (3, ))
        The start point of the ray.
    grid_points : numpy.ndarray (shape: (N, 3))
        The grid points to be checked.
    inside_cone : sparse.csr_matrix (shape: (N, M))
        The condition whether the grid points are inside the cone defined by the mesh and the origin.

    Returns
    -------
    farther_points
    """

    # calculate bounding sphere
    r_mesh_max = np.linalg.norm(mesh_obj.vectors - start, axis=-1).max(axis=-1, initial=0)  # (M, 3, 3) -> (M, 3)
    r_grid = np.linalg.norm(grid_points - start, axis=-1)  # (N, 3) -> (N, )

    # check if the grid is farther than the bounding sphere for each mesh
    # farther_points = np.array([*np.frompyfunc(lambda x: x > r_mesh_max, 1, 1)(r_grid)])  # (N, M) <- too large
    # rewrite the above line using sparse matrix
    farther_points = sparse.hstack([sparse.csr_matrix(r_grid > _max).T for _max in r_mesh_max])  # (N, M)

    # return farther_points
    return r_mesh_max


def check_visible(mesh_obj: mesh.Mesh, start: np.ndarray, grid_points: np.ndarray,
                  verbose: int = 0) -> np.ndarray:
    """
    Check if the grid points are visible from the start point.

    This function provides a visibility of the grid points from the start point.
    Three conditions are checked to determine the visibility:
    C_1. Whether the grid points are inside the cone defined by the mesh and the origin.
    C_2. Whether the grid points are farther than all vertices of the mesh.
    C_3. Whether the line segment between the start point and the grid point intersects with the mesh.

    The shadow related to the mesh is defined as C_1 & C_2 for each mesh and OR of the shadows for all meshes.
    C_3 is not evaluated for the grid points being outside `check_list` in which the grid points are invisible from
    the start point.

    Parameters
    ----------
    mesh_obj : mesh.Mesh object
        STL mesh object with M triangles.
    start : numpy.ndarray (shape: (3, ))
        The start point of the ray.
    grid_points : numpy.ndarray (shape: (N, 3))
        The grid points to be checked.
    verbose : int, optional, default 0
        The verbosity level.

    Returns
    -------
    visible : numpy.ndarray (shape: (N,)) (dtype=bool)
        The condition of the grid points. If the n-th grid point is visible from the start point, the value at n is True.
    """
    if verbose > 0:
        _trange = trange
    else:
        _trange = range
    N = grid_points.shape[0]
    M = mesh_obj.vectors.shape[0]
    p = grid_points - start  # (N, 3)
    # calculate bounding sphere
    r_grid = np.linalg.norm(p, axis=-1)  # (N, )
    r_mesh_max = np.linalg.norm(mesh_obj.vectors - start, axis=-1).max(axis=-1, initial=0)  # (M, 3, 3) -> (M, 3)

    my_print(f"{N=}, {M=}", show=verbose > 0)

    # inside_cone = delta_cone(mesh_obj, start, grid_points)  # list of sparse.csr_matrix of shape (N, 1) (len = M)
    n_oab, n_obc, n_oca, zero_volume = delta_cone(mesh_obj, start)
    my_print("delta_cone done", show=verbose > 0)

    # check if the grid is farther than the bounding sphere for each mesh
    shadow = np.zeros(N, dtype=bool)  # (N, )
    for m in _trange(M):
        shadow += (p @ n_oab[m] >= 0) & (p @ n_obc[m] >= 0) & (p @ n_oca[m] >= 0) & zero_volume[m] & (
                r_grid > r_mesh_max[m])
    my_print("shadow_check done", show=verbose > 0)

    time.sleep(0.05)
    intersection = shadow.copy()  # (N, )
    # grid points which we can see from the start point
    for m in _trange(M):
        # check if the grid is not in the shadow of any mesh and inside the bounding sphere of m-th mesh
        # not in shadow of any mesh & inside the bounding sphere of m-th mesh -> True
        check_list = (p @ n_oab[m] >= 0) & (p @ n_obc[m] >= 0) & (p @ n_oca[m] >= 0) & zero_volume[m] & ~shadow
        # If any grid is inside the bounding sphere of m-th mesh, check intersection
        # if np.any(check_list):
        intersection[check_list] += check_intersection(mesh_obj.vectors[m], start, grid_points[check_list])
    time.sleep(0.05)
    my_print("intersection check done", show=verbose > 0)

    # (True if the grid points are not visible from the start point)
    visible = np.logical_not(intersection)  # (N, )

    return visible


def stl2mesh3d(stl_mesh: mesh.Mesh):
    """
    Convert a stl mesh to a Plotly mesh3d object.

    Parameters
    ----------
    stl_mesh

    Returns
    -------
    vertices : numpy.ndarray (shape: (N, 3))
        The unique vertices of the mesh.
    (I, J, K) : tuple of numpy.ndarray (shape: (M, ))
        The indices of the vertices to define the triangles.
    """
    # stl_mesh is read by numpy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points)
    # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
    p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3 * k for k in range(p)])
    J = np.take(ixr, [3 * k + 1 for k in range(p)])
    K = np.take(ixr, [3 * k + 2 for k in range(p)])
    return vertices, (I, J, K)


def plotly_show_stl(stl_mesh: mesh.Mesh, fig: go.Figure = None, color: str = 'lightblue',
                    opacity: float = 0.5, show_edges: bool = False, show_fig: bool = True,
                    linearg: dict = None, **kwargs):
    """
    Plot a stl mesh using Plotly.

    Parameters
    ----------
    fig : go.Figure
        The figure to be updated.
    stl_mesh : mesh.Mesh
        The stl mesh to be plotted.
    color : str
        The color of the mesh.
    opacity : float
        The opacity of the mesh.
    show_edges : bool
        Whether to show the edges of the mesh.
    show_fig : bool
        Whether to show the figure.
    linearg : dict
        The arguments to be passed to go.Scatter3d for the edges.
    kwargs : dict
        Other arguments to be passed to go.Layout.

    Returns
    -------
    fig : go.Figure
        The updated figure.
    """
    if fig is None:
        fig = go.Figure()
    if linearg is None:
        linearg = dict(color='black', width=1)

    vertices, (I, J, K) = stl2mesh3d(stl_mesh)
    mesh3d = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                       i=I, j=J, k=K,
                       color=color, opacity=opacity, showscale=False)
    fig.add_trace(mesh3d)
    if show_edges:
        Xe = []
        Ye = []
        Ze = []
        for T in stl_mesh.vectors:
            Xe.extend([T[k % 3][0] for k in range(4)] + [None])
            Ye.extend([T[k % 3][1] for k in range(4)] + [None])
            Ze.extend([T[k % 3][2] for k in range(4)] + [None])
        edges = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', line=linearg)
        fig.add_trace(edges)
    # update axes ranges and aspect ratio of 3d plot
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.update_layout(**kwargs)

    if show_fig:
        fig.show()
    return fig


def plotly_show_axes(R: Rotation, origin: np.ndarray = np.zeros(3),
                     fig: go.Figure = None, show_fig: bool = True, axis_length: float = 1.0, cone_scale: float = 0.2,
                     name: str = 'axes', **kwargs):
    """
    Plot the axes of a rotation matrix.
    Parameters
    ----------
    R: Rotation
        The rotation matrix.
    origin: np.ndarray
        The origin of the axes.
    fig: go.Figure
        The figure to be updated.
    show_fig: bool
        Whether to show the figure.
    axis_length:
        The length of the axes.
    cone_scale:
        The scale of the cones.
    name: str
        The name of the axes.
    kwargs: dict
        Other arguments to be passed to go.Layout.

    Returns
    -------
    fig: go.Figure
        The updated figure.
    """
    if fig is None:
        fig = go.Figure()

    # get the axes
    if isinstance(R, Rotation):
        X, Y, Z = R.as_matrix() * axis_length
    elif isinstance(R, np.ndarray):
        X, Y, Z = R * axis_length
    else:
        raise TypeError(f"R must be a Rotation or a numpy.ndarray")
    # plot the axes
    line_x = go.Scatter3d(x=[origin[0], origin[0] + X[0]],
                          y=[origin[1], origin[1] + X[1]],
                          z=[origin[2], origin[2] + X[2]],
                          mode='lines', line=dict(color='red', width=4), name=None)
    line_y = go.Scatter3d(x=[origin[0], origin[0] + Y[0]],
                          y=[origin[1], origin[1] + Y[1]],
                          z=[origin[2], origin[2] + Y[2]],
                          mode='lines', line=dict(color='green', width=4), name=None)
    line_z = go.Scatter3d(x=[origin[0], origin[0] + Z[0]],
                          y=[origin[1], origin[1] + Z[1]],
                          z=[origin[2], origin[2] + Z[2]],
                          mode='lines', line=dict(color='blue', width=4), name=None)
    cone_x = go.Cone(x=[origin[0] + X[0]], y=[origin[1] + X[1]], z=[origin[2] + X[2]],
                     u=[X[0] * cone_scale], v=[X[1] * cone_scale], w=[X[2] * cone_scale], anchor="tail",
                     colorscale=[[0, 'red'], [1, 'red']], showscale=False, name=None)
    cone_y = go.Cone(x=[origin[0] + Y[0]], y=[origin[1] + Y[1]], z=[origin[2] + Y[2]],
                     u=[Y[0] * cone_scale], v=[Y[1] * cone_scale], w=[Y[2] * cone_scale], anchor="tail",
                     colorscale=[[0, 'green'], [1, 'green']], showscale=False, name=None)
    cone_z = go.Cone(x=[origin[0] + Z[0]], y=[origin[1] + Z[1]], z=[origin[2] + Z[2]],
                     u=[Z[0] * cone_scale], v=[Z[1] * cone_scale], w=[Z[2] * cone_scale], anchor="tail",
                     colorscale=[[0, 'blue'], [1, 'blue']], showscale=False, name=None)
    # annotations of the axes
    labels = go.Scatter3d(x=[origin[0] + X[0], origin[0] + Y[0], origin[0] + Z[0]],
                          y=[origin[1] + X[1], origin[1] + Y[1], origin[1] + Z[1]],
                          z=[origin[2] + X[2], origin[2] + Y[2], origin[2] + Z[2]],
                          mode='text', text=[f"{name}_{ax}" for ax in "xyz"],
                          textposition="middle center", name=None)
    fig.add_traces([line_x, cone_x, line_y, cone_y, line_z, cone_z, labels]
                   ).update_layout(scene=dict(aspectmode='data'))
    fig.update_layout(**kwargs)

    if show_fig:
        fig.show()

    return fig


def show_stl(model, ax=None, fsz=10, elev=30, azim=30, facecolors="lightblue", edgecolors="k", lw=0.1,
             x_lim=None, y_lim=None, z_lim=None, modify_axes=False, full_model=True, show_origin=False, show_fig=False,
             **kwargs):
    """Show the STL model using matplotlib.

    Parameters
    ----------
    model : mesh.Mesh
        The STL model to be shown.
    ax : matplotlib.axes.Axes, optional, default None
        The axes to be used.
    fsz : float, optional, default 10
        The font size of the axes.
    elev : float, optional, default 30
        The elevation of the view.
    azim : float, optional, default 30
        The azimuth of the view.
    facecolors : str, optional, default "lightblue"
        The face color of the model.
    edgecolors : str, optional, default None
        The edge color of the model.
    lw : float, optional, default 0.1
        The line width of the model.
    x_lim : tuple, optional, default None
        The x limits of the axes.
    y_lim : tuple, optional, default None
        The y limits of the axes.
    z_lim : tuple, optional, default None
        The z limits of the axes.
    modify_axes : bool, optional, default False
        Whether to modify the axes.
    full_model : bool, optional, default True
        Whether to show the full model.
    show_origin: bool, optional, default False
        Whether to show the origin.
    show_fig : bool, optional, default False
        Whether to show the figure.
    kwargs : dict
        Other arguments to be passed to Poly3DCollection.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes used.

    Notes
    -----
    The axes limits are set to the range of the model if `modify_axes` is True.
     modify_axes |  full_model   |   *_lim    |       axes   |   model
        False    |     False     |    None    | ->    axis   |  limited
        False    |     False     |  not None  | ->   *_lim   |  limited
        False    |     True      |    None    | ->    axis   |   full
        False    |     True      |  not None  | ->   *_lim   |   full
        True     |     False     |    None    | ->   model   |   full
        True     |     False     |  not None  | ->   *_lim   |  limited
        True     |     True      |    None    | ->   model   |   full
        True     |     True      |  not None  | ->   *_lim   |   full
    """
    if ax is None:
        ax = plt.subplot(projection='3d')
    # Get the x, y, z ranges of the object
    model.update_min()
    model.update_max()
    mx_lim, my_lim, mz_lim = zip(model.min_, model.max_)

    x_lim = (mx_lim if modify_axes else ax.get_xlim()) if x_lim is None else x_lim
    y_lim = (my_lim if modify_axes else ax.get_ylim()) if y_lim is None else y_lim
    z_lim = (mz_lim if modify_axes else ax.get_zlim()) if z_lim is None else z_lim

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_box_aspect((np.ptp(x_lim), np.ptp(y_lim), np.ptp(z_lim)))

    if show_origin:
        ax.scatter(0, 0, 0, color='k', s=50, zorder=1)
        ax.quiver(0, 0, 0, x_lim[1] * 1.2, 0, 0,
                  color='k', linewidth=1, zorder=-1, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, y_lim[1] * 1.2, 0, color='k', linewidth=1, zorder=-1, arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, z_lim[1] * 1.2, color='k', linewidth=1, zorder=-1, arrow_length_ratio=0.1)
        ax.text(x_lim[1] * 1.25, 0, 0, 'x', fontsize=fsz)
        ax.text(0, y_lim[1] * 1.25, 0, 'y', fontsize=fsz)
        ax.text(0, 0, z_lim[1] * 1.25, 'z', fontsize=fsz)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    ls = LightSource(azdeg=225.0, altdeg=45.0)
    if full_model:
        print("full_model")
        vectors = model.vectors
    else:
        # Get the vectors that are inside the limits
        # If all the vertices of a triangle are inside the limits, show the triangle
        condition = np.all((model.vectors >= np.array([x_lim[0], y_lim[0], z_lim[0]])[None, None, :]) &
                           (model.vectors <= np.array([x_lim[1], y_lim[1], z_lim[1]])[None, None, :]), axis=(1, 2))
        vectors = model.vectors[condition]

    mesh_data = mplot3d.art3d.Poly3DCollection(vectors, lightsource=ls, shade=True, zsort='min', zorder=1,
                                               facecolors=facecolors, edgecolors=edgecolors, lw=lw, **kwargs)

    # Create a new plot
    ax.add_collection3d(mesh_data)

    # Modify the viewing angle
    ax.view_init(elev=elev, azim=azim)

    if show_fig:
        ax.figure.show()

    return ax


def make_stl(vertices: np.ndarray, faces: np.ndarray) -> mesh.Mesh:
    """
    Make stl data from vertices and faces

    Parameters
    ----------
    vertices : np.ndarray
        vertices of mesh
    faces : np.ndarray
        faces of mesh

    Returns
    -------
    stl_data : mesh.Mesh
        stl data
    """

    stl_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_data.vectors[i][j] = vertices[f[j], :]

    stl_data.update_normals()
    return stl_data


def meshed_surface(para_1: np.ndarray, para_2: np.ndarray,
                   func: Callable, **func_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make meshed surface

    Parameters
    ----------
    para_1 : np.ndarray
        (N, ) array of parameters
    para_2 : np.ndarray
        (N, ) array of parameters
    func : function
        get 3D coordinates from 2D parameters

    Returns
    -------
    vertices : np.ndarray
        (N, 3) array of vertices
    faces : np.ndarray
        (N, 3) array of faces
    """
    params = np.array(np.meshgrid(para_1, para_2)).T.reshape(-1, 2)
    tri = Delaunay(params)
    x, y, z = func(params[:, 0], params[:, 1], **func_kwargs)
    vertices = np.array([x, y, z]).T
    faces = tri.simplices
    return vertices, faces


def torus(theta: np.ndarray, phi: np.ndarray, a: float = 1, R: float = 2):
    """
    Return 3D coordinates on torus surface

    Parameters
    ----------
    theta : np.ndarray
        angle of toroidal direction
    phi : np.ndarray
        angle of poloidal direction
    a : float
        minor radius of torus
    R : float
        major radius of torus

    Returns
    -------
    x : np.ndarray
        x coordinates
    y : np.ndarray
        y coordinates
    z : np.ndarray
        z coordinates
    """
    x = (R + a * np.cos(theta)) * np.cos(phi)
    y = (R + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)
    return x, y, z


def sphere(theta: np.ndarray, phi: np.ndarray, r: float = 1):
    """
    Return 3D coordinates on sphere surface

    Parameters
    ----------
    theta : np.ndarray
        angle of toroidal direction
    phi : np.ndarray
        angle of poloidal direction
    r : float
        radius of sphere

    Returns
    -------
    x : np.ndarray
        x coordinates
    y : np.ndarray
        y coordinates
    z : np.ndarray
        z coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def make_2D_surface(points: np.ndarray, condition: Callable) -> mesh.Mesh:
    """
    Make unclosed surface from vertices
    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of points
    condition : Callable
        condition is a function that returns True if points are not included in the surface
        condition(x, y) -> bool

    Returns
    -------
    stl_data : mesh.Mesh
        unclosed stl data (z = 0)
    """

    tri = Delaunay(points)
    gravity = np.mean(points[tri.simplices], axis=1)
    faces = tri.simplices[~condition(*gravity.T)]

    stl_data = make_stl(np.hstack([points, np.zeros((points.shape[0], 1))]), faces)
    return stl_data


if __name__ == '__main__':
    vertices = np.array([[1, 10, -10],
                         [1, -10, 10],
                         [1, -10, -10]]) + [5, 3, 13]
    faces = np.array([[0, 2, 1]])
    model = make_stl(vertices, faces)

    eye_position = np.array([20, 0, 0])
    points = np.stack(np.meshgrid(np.linspace(-15, 5, 51),
                                  np.linspace(-10, 10, 51),
                                  6, indexing="ij")).reshape((3, -1)).T
    res = check_visible(model, eye_position, points)
    axes = plt.subplots(2, 3, subplot_kw=dict(projection='3d', proj_type='ortho'),
                        figsize=(10, 5))[1].ravel()
    for i, [ax, cond, title] in enumerate(
            zip(axes, res, ["inside_cone", "farther_points", "shadow", "check_list", "intersection", "visible"])):
        # print(cond.squeeze().astype(int))
        ax.set_title(title)
        ax.scatter(*eye_position, color="g", s=100, marker="*")
        ax.scatter(*points[cond.squeeze()].T, c="r", s=10, label="true")
        ax.scatter(*points[~cond.squeeze()].T, c="b", s=10, label="false")
        ax.legend()
        show_stl(model, modify_axes=True, ax=ax, elev=30, azim=60)
        # plot plane defined by the mesh and eye position
    axes[0].figure.tight_layout()
    axes[0].figure.show()

    # # model = make_stl(*meshed_surface(np.linspace(0, 2 * np.pi, 20), np.linspace(0, 2 * np.pi, 20), torus, a=1, R=2))
    # model = mesh.Mesh.from_file("../cube.stl")
    # model.x = model.x - np.max(model.x)
    # model.y = model.y - np.max(model.y)
    # model.z = model.z - np.max(model.z)
    # model_rot1 = copy_model(model)
    # model_rot2 = copy_model(model)
    # model_rot3 = copy_model(model)
    #
    # # Rotate the model
    # rotate_model(model_rot1, "XYZ", (45, 0, 0), origin=[model.x.min(), model.y.min(), model.z.min()])
    # rotate_model(model_rot2, "XYZ", (45, -90, 0), origin=[model.x.min(), model.y.min(), model.z.min()])
    # rotate_model(model_rot3, "XYZ", (45, -90, 90), origin=[model.x.min(), model.y.min(), model.z.min()])
    # # model_rot1.translate([0, 20, 0])
    # # model_rot2.translate([0, 40, 0])
    # # model_rot3.translate([0, 60, 0])
    #
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # # show_stl(ax, model, modify_axes=True, alpha=0.4, edgecolors='0.2', lw=0.1, facecolors="w")
    # # show_stl(ax, model_shift, modify_axes=True, alpha=0.4, edgecolors='0.2', lw=0.1, facecolors="w")
    # # ax.set_ylim(-1.5, 3.5)
    # # plt.show()
    #
    # # fig = plotly_show_stl(model, show_edges=True, show_fig=False, opacity=1, color='lightpink')
    # fig = go.Figure()
    # # plot axis (3 arrows representing XYZ axis)
    # fig = plotly_show_axes(Rotation.identity(),
    #                        origin=[0, 0, 0], show_fig=False,
    #                        fig=fig, axis_length=5, name="origin")
    # fig = plotly_show_axes(Rotation.from_euler("xyz", (45, 0, 0), degrees=True),
    #                        origin=[10, 0, 0], show_fig=False,
    #                        fig=fig, axis_length=5, name="rot1")
    # fig = plotly_show_axes(Rotation.from_euler("xyz", (45, -90, 0), degrees=True),
    #                        origin=[20, 0, 0], show_fig=False,
    #                        fig=fig, axis_length=5, name="rot2")
    # fig = plotly_show_axes(Rotation.from_euler("xyz", (45, -90, 90), degrees=True),
    #                        origin=[30, 0, 0], show_fig=False,
    #                        fig=fig, axis_length=5, name="rot3")
    # # fig = plotly_show_stl(model_rot1, show_edges=True, show_fig=False, opacity=1, fig=fig, color='lightgreen')
    # # fig = plotly_show_stl(model_rot2, show_edges=True, show_fig=False, opacity=1, fig=fig, color='lightblue')
    # # fig = plotly_show_stl(model_rot3, show_edges=True, show_fig=False, opacity=1, fig=fig, color='lightyellow')
    #
    # fig.show()
