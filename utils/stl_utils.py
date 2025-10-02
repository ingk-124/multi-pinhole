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

from utils.my_stdio import *

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


# MARK: - STL utilities
def shape_check(shape: str, size: Union[Number, Vector2DLike], ok_shapes: dict = None) -> Tuple[str, Vector2DLike]:
    """Normalize aperture ``shape`` and ``size`` specifications.

    Parameters
    ----------
    shape : str
        Desired aperture shape. One of ``{"circle", "ellipse", "rectangle", "square"}``.
    size : Union[Number, Vector2DLike]
        Size specification. Circles/squares accept a scalar radius/edge length or a
        two-element sequence of equal values. Ellipses/rectangles accept either a
        scalar (expanded to both axes) or a two-element ``(height, width)`` pair.
    ok_shapes : dict, optional
        Deprecated argument that is ignored during validation.

    Returns
    -------
    Tuple[str, np.ndarray]
        Normalized ``(shape, size)`` pair where ``size`` is a two-element array
        containing positive ``(height, width)`` values.
    """
    # 許可形状
    allowed = {'circle', 'ellipse', 'rectangle', 'square'}
    if ok_shapes is None:
        ok_shapes = {'circle': 1, 'ellipse': 2, 'rectangle': 2, 'square': 1}
    if shape not in allowed:
        raise ValueError(f"shape must be one of {sorted(allowed)}")

    # size を 1D 配列化
    if isinstance(size, Number):
        arr = np.array([float(size)], dtype=float)
    else:
        arr = np.array(size, dtype=float).ravel()

    # 形状ごとの許容と展開
    if shape in ('circle', 'square'):
        if arr.size == 1:
            s = float(arr[0])
            size_hw = np.array([s, s], dtype=float)
        elif arr.size == 2:
            # 2要素の場合は等しいことを要求（厳密一致だと浮動小数で落ちるので isclose）
            if not np.isclose(arr[0], arr[1]):
                raise ValueError(f"{shape} expects equal sides: got {arr.tolist()}")
            s = float(arr[0])
            size_hw = np.array([s, s], dtype=float)
        else:
            raise ValueError(f"{shape} size must be a number or a 2-element sequence")
    elif shape in ('ellipse', 'rectangle'):
        if arr.size == 1:
            s = float(arr[0])
            size_hw = np.array([s, s], dtype=float)  # 正方形/正円相当も許容
        elif arr.size == 2:
            size_hw = arr.astype(float, copy=False)
        else:
            raise ValueError(f"{shape} size must be a number or a 2-element sequence")
    else:
        # 念のため（上の allowed で弾いているので通常ここには来ない）
        raise ValueError(f"Unsupported shape: {shape}")

    # 数値チェック（正・有限）
    if not np.all(np.isfinite(size_hw)) or np.any(size_hw <= 0):
        raise ValueError(f"size must be positive finite numbers: got {size_hw.tolist()}")

    return shape, size_hw


def generate_aperture_stl(shape: str, size: Union[Number, Vector2DLike], resolution: Union[Number, Vector2DLike],
                          max_size: Union[Number, Vector2DLike] = None) -> mesh.Mesh:
    """Create a planar STL mesh representing an aperture outline.

    Parameters
    ----------
    shape : str
        Aperture type (``"circle"``, ``"ellipse"`` or ``"rectangle"``).
    size : Union[Number, Vector2DLike]
        Characteristic size of the aperture. Circles use a radius, ellipses use
        ``(semi-major, semi-minor)`` axes and rectangles use ``(width, height)``.
    resolution : Union[Number, Vector2DLike]
        Number of samples used to discretize the aperture boundary.
    max_size : Union[Number, Vector2DLike], optional
        Extent of the surrounding grid used to generate supporting vertices.
        Defaults to ``1.5`` times the normalized aperture size.

    Returns
    -------
    mesh.Mesh
        Mesh containing the 2D aperture surface in the ``z = 0`` plane.
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
    """Rotate an STL mesh either by Euler angles or by an explicit matrix.

    Parameters
    ----------
    model : mesh.Mesh
        Source mesh to rotate.
    order : str, optional
        Euler angle order used when ``matrix`` is not provided, by default ``"xyz"``.
    angles : Vector3DLike, optional
        Euler angles describing the rotation applied around ``origin``.
    matrix : np.ndarray, optional
        Precomputed rotation matrix. When supplied, ``order`` and ``angles`` are ignored.
    origin : Vector3DLike, optional
        Rotation pivot point expressed in model coordinates.
    degrees : bool, optional
        If ``True`` (default) Euler ``angles`` are interpreted in degrees, otherwise radians.

    Returns
    -------
    mesh.Mesh
        Deep copy of ``model`` rotated in place.
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
    """Return a transformed deep copy of ``model``.

    Parameters
    ----------
    model : mesh.Mesh
        Mesh to duplicate.
    translate : np.ndarray, optional
        Translation applied to the copy before returning.
    rotation_matrix : np.ndarray, optional
        Optional rotation applied after translation using ``mesh.Mesh.rotate_using_matrix``.

    Returns
    -------
    mesh.Mesh
        Independent copy of the original mesh with requested transforms applied.
    """
    model = mesh.Mesh(model.data.copy())
    model.translate(translate)
    if rotation_matrix is not None:
        model.rotate_using_matrix(rotation_matrix)
    return model


# MARK: - Visibility calculation utilities
def check_intersection(triangle: np.ndarray, start_point: np.ndarray, end_points: np.ndarray,
                       behind_start_included: float | bool = False, eps: float = 1e-6) -> np.ndarray:
    """Test intersection between a triangle and one or more line segments.

    Parameters
    ----------
    triangle : np.ndarray, shape (3, 3)
        Triangle vertices ordered as ``(a, b, c)``.
    start_point : np.ndarray, shape (3,)
        Shared origin of the tested line segments.
    end_points : np.ndarray, shape (N, 3)
        Endpoints of the segments to test.
    behind_start_included : float or bool, optional
        If ``True``, intersections with ``t <= 1`` are reported regardless of sign.
        A numeric value allows visibility checks to include a bounded region
        behind ``start_point``.
    eps : float, optional
        Numerical tolerance applied to barycentric and ``t`` comparisons.

    Returns
    -------
    np.ndarray, shape (N,)
        Boolean mask indicating which segments intersect the triangle.

    Notes
    -----
    Implements the Möller–Trumbore intersection algorithm. The triangle edges
    are defined as ``e₁ = b - a`` and ``e₂ = c - a`` while each segment uses the
    direction ``d = end - start`` and the offset ``r = start - a``. Solving the
    coupled system ``start + t·d = a + u·e₁ + v·e₂`` yields the barycentric
    coordinates ``(u, v)`` and parametric distance ``t`` via

    ``u = (r · (d × e₂)) / det``, ``v = (r · (e₁ × d)) / det`` and
    ``t = (r · (e₁ × e₂)) / det`` where ``det = -d · (e₁ × e₂)``. When
    ``det ≠ 0`` the segment intersects the triangle iff ``u ≥ 0``, ``v ≥ 0``,
    ``u + v ≤ 1`` and ``0 < t ≤ 1`` (relaxed to ``t ≤ 1`` when
    ``behind_start_included`` is truthy). Parallel segments with ``det = 0`` are
    rejected.
    """

    if end_points.shape[0] == 0:
        return np.array([], dtype=bool)

    N = end_points.shape[0]

    # basic vectors
    a, b, c = triangle  # three vectors of a triangle (a, b, c: (3,))
    e_1, e_2 = b - a, c - a  # two edges of a triangle (e_1, e_2: (3,))
    d_ = np.subtract(end_points, start_point)  # direction of the line segment (d: (N, 3))
    r = np.subtract(start_point, a)  # vector from the start point to a (r: (3,))
    n_0 = np.cross(e_1, e_2)  # normal vector of the triangle (n_0: (3,))
    det = -np.einsum('ij,j->i', d_, n_0)  # (N,) array

    dtype = det.dtype
    eps = dtype.type(eps)

    non_parallel = np.abs(det) > eps  # (N,) boolean array (non-parallel condition)

    if not np.any(non_parallel):
        return np.zeros_like(N, dtype=bool)

    inv_det = np.zeros_like(det)
    inv_det[non_parallel] = 1.0 / det[non_parallel]

    n_2 = np.cross(d_, e_2)  # (N, 3)
    u = np.einsum('i,ji,j->j', r, n_2, inv_det)  # (N,) array
    ok = non_parallel & (u >= -eps)  # (N,) boolean array (u condition)
    if not np.any(ok):
        return np.zeros_like(N, dtype=bool)

    n_1 = np.cross(e_1, d_)  # (N, 3)
    v = np.einsum('i,ji,j->j', r, n_1, inv_det)  # (N,) array
    ok &= (v >= -eps) & (u + v <= 1 + eps)  # (N,) boolean array (v condition)
    if not np.any(ok):
        return np.zeros_like(N, dtype=bool)

    # t = np.einsum('ij,j->i', r, n_0) * inv_det  # (N,) array
    t = np.einsum('i,i,j->j', r, n_0, inv_det)  # (N,) array

    if isinstance(behind_start_included, bool):
        t_min = np.full(N, -np.inf if behind_start_included else 0, dtype=dtype)
    elif isinstance(behind_start_included, Number):
        # d_norm = np.linalg.norm(d_, axis=1)
        z = d_[..., 2]
        f = dtype.type(behind_start_included)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_min = np.where(z > eps, f / z, np.inf).astype(dtype, copy=False)
    else:
        raise ValueError("behind_start_included must be bool or float (default: False)")
    ok &= (t > t_min - eps) & (t <= 1 + eps)  # (N,) boolean array (t condition)
    return ok


def delta_cone(mesh_obj: mesh.Mesh, start_point: np.ndarray) -> np.ndarray:
    """Compute cone plane normals for each triangle in ``mesh_obj``.

    Parameters
    ----------
    mesh_obj : mesh.Mesh
        STL mesh with ``M`` triangular faces.
    start_point : np.ndarray, shape (3,)
        Point of origin used to form cones with each triangle.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Signed normals for the planes ``oab``, ``obc`` and ``oca`` along with a
        boolean mask identifying triangles with non-zero volume.

    Notes
    -----
    Each triangle ``abc`` defines a cone with apex ``o = start_point``. The
    helper constructs vectors ``oa``, ``ob`` and ``oc`` together with oriented
    plane normals ``n_oab = oa × ob``, ``n_obc = ob × oc`` and ``n_oca = oc × oa``.
    A grid point ``p`` lies inside the cone when it resides on the same side of
    all three planes as the opposite vertex, i.e.

    ``sign(n_oab · (p - o)) = sign(n_oab · (c - o))`` and similarly for the
    ``obc`` and ``oca`` planes. The returned normals are signed so the checks can
    be performed with simple dot-product comparisons.
    """

    # check if the start point is on the plane defined by the mesh
    zero_volume = ~np.isclose(0., np.einsum('ij,ij->i', mesh_obj.normals, start_point - mesh_obj.v0))  # (M,)

    # origin: np.ndarray of shape (3, )
    # vector oa, ob, oc: np.ndarray of shape (M,3) (from origin to a, b, c)
    oa, ob, oc = mesh_obj.v0 - start_point, mesh_obj.v1 - start_point, mesh_obj.v2 - start_point
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


def delta_cone_prepare(triangles: np.ndarray, start_point: np.ndarray, eps: float = 1e-6):
    """Generate oriented cone planes for a batch of triangles.

    Parameters
    ----------
    triangles : np.ndarray, shape (M, 3, 3)
        Vertices of ``M`` triangles.
    start_point : np.ndarray, shape (3,)
        Cone origin shared across triangles.
    eps : float, optional
        Tolerance used to filter degenerate triangles.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Array of signed plane normals with shape ``(M, 3, 3)`` and a boolean
        validity mask indicating non-zero volume triangles.

    Notes
    -----
    The triangle vertices are translated so the apex ``o`` sits at the origin
    (``a = A - o``, ``b = B - o``, ``c = C - o``). The helper rejects
    near-degenerate triangles by checking ``|a · (b × c)| > eps`` before
    assembling the oriented plane normals ``(a × b)``, ``(b × c)`` and ``(c × a)``.
    The normals are flipped so each points toward the third vertex, enabling the
    half-space tests described in :func:`delta_cone`.
    """
    dtype = triangles.dtype
    eps = dtype.type(eps)

    a = triangles[:, 0, :] - start_point  # (M, 3)
    b = triangles[:, 1, :] - start_point  # (M, 3)
    c = triangles[:, 2, :] - start_point  # (M, 3)

    # ignore zero volume triangles
    volume = np.einsum('ij,ij->i', a, np.cross(b, c))  # (M,)
    # valid = np.linalg.norm(tri_n, axis=1) > eps  # (M,)
    valid = np.abs(volume) > eps  # (M,)
    # calculate normal vectors of planes oab, obc, oca with correct sign (the third point side is positive)
    n_oab = np.cross(a, b)  # (M, 3)
    s = np.sign(np.einsum('ij,ij->i', c, n_oab))  # (M,)
    n_oab *= s[:, None]  # (M, 3)

    n_obc = np.cross(b, c)  # (M, 3)
    s = np.sign(np.einsum('ij,ij->i', a, n_obc))  # (M,)
    n_obc *= s[:, None]  # (M, 3)

    n_oca = np.cross(c, a)  # (M, 3)
    s = np.sign(np.einsum('ij,ij->i', b, n_oca))  # (M,)
    n_oca *= s[:, None]  # (M, 3)

    planes = np.stack([n_oab, n_obc, n_oca], axis=1)  # (M, 3, 3) (triangles, n_*, xyz)
    return planes, valid


def delta_cone_apply(triangles: np.ndarray, start_point: np.ndarray, end_points: np.ndarray,
                     eps: float = 1e-6, allow_behind: bool = False,
                     batch_size: int = 65536, verbose: int = 0
                     ) -> sparse.csr_matrix:
    """Test which points fall inside the cones defined by ``triangles``.

    Parameters
    ----------
    triangles : np.ndarray, shape (M, 3, 3)
        Triangle vertices forming each cone.
    start_point : np.ndarray, shape (3,)
        Apex of all cones.
    end_points : np.ndarray, shape (N, 3)
        Candidate points to evaluate.
    eps : float, optional
        Numerical tolerance forwarded to :func:`delta_cone_prepare`.
    allow_behind : bool, optional
        When ``True`` the cones also include the opposite side of each plane.
    batch_size : int, optional
        Number of points processed per batch to limit memory usage.
    verbose : int, optional
        Controls progress reporting via :func:`my_range`.

    Returns
    -------
    tuple[sparse.csr_matrix, np.ndarray]
        Sparse mask of shape ``(M, N)`` marking points inside each cone and a
        boolean array describing which triangles were valid.

    Notes
    -----
    For each batch of candidate points the helper evaluates the signed distance
    to the oriented planes prepared by :func:`delta_cone_prepare`. Points that
    satisfy ``p · n ≥ -eps`` for all three plane normals are inside the cone. If
    ``allow_behind`` is ``True`` the opposite half-spaces ``p · n ≤ eps`` are
    also accepted, allowing cones to extend behind the apex. The collected mask
    is emitted as a sparse matrix with rows aligned to the original triangle
    indices and columns to the source point order.
    """
    dtype = triangles.dtype
    eps = dtype.type(eps)
    N = end_points.shape[0]
    M = triangles.shape[0]

    if N == 0 or M == 0:
        return sparse.csr_matrix((M, N), dtype=bool)

    # preparation
    planes, valid = delta_cone_prepare(triangles, start_point, eps=eps)
    valid_idx = np.where(valid)[0]
    n_0 = planes[:, 0, :]  # (M, 3)
    n_1 = planes[:, 1, :]  # (M, 3)
    n_2 = planes[:, 2, :]  # (M, 3)

    # all mesh are zero volume -> return empty
    if not np.any(valid):
        return sparse.csr_matrix((M, N), dtype=bool), valid

    rows = []
    cols = []

    max_mem = 750 * 1024 * 1024  # 750 MB
    item_size = np.dtype(dtype).itemsize
    max_batch = int(max_mem // (valid.sum() * item_size))
    batch_size = min(batch_size, max(1, max_batch))
    end_points = end_points - start_point  # (N, 3)

    # X: points
    # P: planes
    # X @ n_oab.T >= 0 -> X exists above the plane oab (same side as c)
    for s in my_range(0, N, batch_size, verbose=verbose):
        t = min(s + batch_size, N)
        # D = np.einsum('bj,mij->bmi', end_points[s:t],
        #               planes[valid])  # (batch_size, M_valid, 3)
        # ok_front = (D >= -eps).all(axis=2)
        # ok_back = (D <= eps).all(axis=2)  # behind -> all of X@n_*.T <= 0

        D_0 = end_points[s:t] @ n_0[valid_idx].T  # (batch_size, M_valid)
        ok_0 = D_0 >= -eps
        del D_0
        D_1 = end_points[s:t] @ n_1[valid_idx].T  # (batch_size, M_valid)
        ok_1 = D_1 >= -eps
        del D_1
        D_2 = end_points[s:t] @ n_2[valid_idx].T  # (batch_size, M_valid)
        ok_2 = D_2 >= -eps
        del D_2

        if allow_behind:
            ok_front = ok_0 & ok_1 & ok_2
            ok_back = (~ok_0) & (~ok_1) & (~ok_2)
            inside = ok_front | ok_back
        else:
            ok_front = ok_0 & ok_1 & ok_2
            inside = ok_front

        c, r = np.nonzero(inside)  # c: grid point index, r: triangle index
        rows.append(valid_idx[r])  # valid_idx[r]: original triangle index
        cols.append(c + s)  # c + s: original grid point index

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones_like(rows, dtype=bool)
    return sparse.csr_matrix((data, (rows, cols)), shape=(M, N), dtype=bool), valid


def delta_cone_apply_test():
    triangles = np.array([[[-1, -1, 1], [-1, 2, 1], [2, -1, 1]], ], dtype=np.float32)
    start_point = np.array([0, 0, 0], dtype=np.float32)
    points = np.meshgrid(np.linspace(-2, 2, 25),
                         np.linspace(-2, 2, 25),
                         np.linspace(-2, 2, 25), indexing='ij')
    points = np.array(points).reshape(3, -1).T.astype(np.float32)

    cone, valid = delta_cone_apply(triangles, start_point, points, allow_behind=True, verbose=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=triangles[0, :, 0], y=triangles[0, :, 1], z=triangles[0, :, 2],
                               mode='lines+markers', name='Triangle 1', line=dict(color='blue')))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[1],
                               mode='markers', name='start', marker=dict(color='black')))
    fig.add_trace(go.Scatter3d(x=points[..., 0], y=points[..., 1], z=points[..., 2],
                               mode='markers', name='start', marker=dict(color='black', size=1)))
    for i in range(triangles.shape[0]):
        inside_points = points[cone.getrow(i).nonzero()[1]]
        fig.add_trace(go.Scatter3d(x=inside_points[:, 0], y=inside_points[:, 1], z=inside_points[:, 2],
                                   mode='markers', name=f'Inside Points {i + 1}', marker=dict(size=5)))
    fig.show()
    return fig


def check_visible(mesh_obj, start: np.ndarray, grid_points: np.ndarray, verbose: int = 0,
                  behind_start_included: float | bool = False, dtype: type = np.float32,
                  batch_points: int = 65536) -> np.ndarray:
    """Determine which ``grid_points`` are visible from ``start``.

    Parameters
    ----------
    mesh_obj : mesh.Mesh
        Scene geometry represented by ``M`` triangles.
    start : np.ndarray, shape (3,)
        Ray origin for the visibility test.
    grid_points : np.ndarray, shape (N, 3)
        Sample points to classify as visible or occluded.
    verbose : int, optional
        Verbosity level forwarded to helper utilities.
    behind_start_included : float or bool, optional
        Controls whether intersections behind ``start`` are considered blocking.
    dtype : type, optional
        Numpy dtype used for intermediate calculations. Defaults to ``np.float32``.
    batch_points : int, optional
        Number of candidate points processed per batch during cone evaluation.

    Returns
    -------
    np.ndarray, shape (N,), dtype=bool
        Mask with ``True`` for points that remain visible.

    Notes
    -----
    Visibility is determined by two coupled conditions:

    ``C₁``
        Whether a point resides inside the viewing cone induced by each mesh
        triangle. This is evaluated with :func:`delta_cone_apply`, which returns
        a sparse mask of candidate points per triangle. Points outside every
        cone remain visible without further checks.
    ``C₂``
        Whether the segment from ``start`` to the candidate point intersects the
        triangle. For candidates inside a cone the function calls
        :func:`check_intersection`, which applies the Möller–Trumbore algorithm.

    A point becomes occluded as soon as any triangle satisfies ``C₁`` *and*
    ``C₂``. Otherwise the point is marked visible.
    """

    N = grid_points.shape[0]
    M = mesh_obj.vectors.shape[0]
    if N == 0:
        return np.zeros(0, dtype=bool)
    if M == 0:
        return np.ones(N, dtype=bool)

    triangles = mesh_obj.vectors.astype(dtype, copy=False)  # (M, 3, 3)
    start = start.astype(dtype, copy=False)  # (3,)
    grid_points = grid_points.astype(dtype, copy=False)  # (N, 3)
    my_print(f"{N=}, {M=}", show=verbose > 0)

    allow_behind = isinstance(behind_start_included, Number) or behind_start_included is True
    my_print("delta_cone_apply start", show=verbose > 0)
    start_time = time.time()
    cand, valid = delta_cone_apply(triangles, start, grid_points,
                                   allow_behind=allow_behind,
                                   batch_size=batch_points, verbose=verbose,
                                   eps=1e-6)  # (M, N)
    time.sleep(0.1)
    my_print(f"delta_cone_apply done in {time.time() - start_time:.3f} sec", show=verbose > 0)
    visible = np.ones(N, dtype=bool)
    my_print("check_intersection start", show=verbose > 0)
    start_time = time.time()
    for i in my_range(M, verbose=verbose):
        if valid[i]:
            inside_grid_points = cand.getrow(i).nonzero()[1]  # 点 i が三角形 j のコーン内にあるインデックス
            if inside_grid_points.size > 0:
                intersected = check_intersection(triangles[i], start, grid_points[inside_grid_points],
                                                 behind_start_included=behind_start_included, eps=1e-6)
                visible[inside_grid_points[intersected]] = False
    my_print(f"check_intersection done in {time.time() - start_time:.3f} sec", show=verbose > 0)

    return visible


def check_visible_test():
    vertices = np.array([[-1, -1, 1], [-1, 2, 1], [2, -1, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1]])
    model = make_stl(vertices, faces)
    triangles = model.vectors
    start_point = np.array([0, 0, 0.5], dtype=np.float32)
    points = np.meshgrid(np.linspace(-2, 2, 25),
                         np.linspace(-2, 2, 25),
                         np.linspace(-2, 2, 25), indexing='ij')
    points = np.array(points).reshape(3, -1).T.astype(np.float32)
    cone, valid = delta_cone_apply(triangles, start_point, points, allow_behind=True, verbose=1)
    visible = check_visible(model,
                            start_point, points, verbose=1)
    visible2 = check_visible(model,
                             start_point, points, verbose=1, behind_start_included=True)
    # 1 x 3 figures
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[1],
                               mode='markers', name='start', marker=dict(color='black')))
    fig.add_trace(go.Scatter3d(x=points[..., 0], y=points[..., 1], z=points[..., 2],
                               mode='markers', name='start', marker=dict(color='black', size=1)))
    inside_points = points[cone.getrow(0).nonzero()[1]]
    fig.add_trace(go.Scatter3d(x=inside_points[:, 0], y=inside_points[:, 1], z=inside_points[:, 2],
                               mode='markers', name=f'Inside Points 1', marker=dict(size=2)))
    invisible_points = points[~visible]
    fig.add_trace(go.Scatter3d(x=invisible_points[:, 0], y=invisible_points[:, 1], z=invisible_points[:, 2],
                               mode='markers', name=f'Invisible Points (no mesh)', marker=dict(size=4, color='red')))
    invisible_points2 = points[~visible2]
    fig.add_trace(go.Scatter3d(x=invisible_points2[:, 0], y=invisible_points2[:, 1], z=invisible_points2[:, 2],
                               mode='markers', name=f'Invisible Points (with mesh)',
                               marker=dict(size=3, color='green')))
    fig.show()
    return fig


def check_visible_old(mesh_obj: mesh.Mesh, start: np.ndarray, grid_points: np.ndarray,
                      verbose: int = 0, behind_start_included: bool = False) -> np.ndarray:
    """Legacy visibility algorithm retained for regression testing.

    Parameters
    ----------
    mesh_obj : mesh.Mesh
        STL mesh containing ``M`` triangles.
    start : np.ndarray, shape (3,)
        Viewpoint or ray origin.
    grid_points : np.ndarray, shape (N, 3)
        Locations to test for visibility.
    verbose : int, optional
        Verbosity level forwarded to progress utilities.
    behind_start_included : bool, optional
        When ``True`` any points located behind ``start`` are treated as occluded.

    Returns
    -------
    np.ndarray, shape (N,), dtype=bool
        Boolean visibility mask comparable to :func:`check_visible`.
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
        intersection[check_list] += check_intersection(mesh_obj.vectors[m], start, grid_points[check_list],
                                                       behind_start_included)
    time.sleep(0.05)
    my_print("intersection check done", show=verbose > 0)

    # (True if the grid points are not visible from the start point)
    visible = np.logical_not(intersection)  # (N, )

    return visible


# MARK: STL visualization utilities
def stl2mesh3d(stl_mesh: mesh.Mesh):
    """Convert ``numpy-stl`` triangle data into a Plotly ``Mesh3d`` representation.

    Returns
    -------
    tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]
        Unique vertex coordinates and corresponding face index arrays ``(i, j, k)``.
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
    """Render ``stl_mesh`` into a Plotly figure with optional edge overlays.

    Parameters
    ----------
    stl_mesh : mesh.Mesh
        Mesh to visualize.
    fig : go.Figure, optional
        Figure receiving the traces. A new figure is created when omitted.
    color : str, optional
        Fill color for the ``Mesh3d`` trace.
    opacity : float, optional
        Face opacity supplied to Plotly.
    show_edges : bool, optional
        When ``True`` an additional ``Scatter3d`` trace highlights triangle edges.
    show_fig : bool, optional
        Whether to call :meth:`go.Figure.show` before returning.
    linearg : dict, optional
        Keyword arguments passed to the edge trace.
    **kwargs
        Additional layout updates applied to ``fig``.

    Returns
    -------
    go.Figure
        Figure instance containing the rendered mesh.
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
    """Plot a triad of axes derived from ``R`` in a Plotly figure.

    Parameters
    ----------
    R : Rotation | np.ndarray
        Rotation instance or 3×3 matrix providing axis directions.
    origin : np.ndarray, optional
        Anchor position of the axes in world coordinates.
    fig : go.Figure, optional
        Figure to update. A new figure is created when omitted.
    show_fig : bool, optional
        Display the figure via :meth:`go.Figure.show`.
    axis_length : float, optional
        Length of each axis line segment.
    cone_scale : float, optional
        Scale applied to the arrowhead cones.
    name : str, optional
        Prefix used for axis labels in the annotation trace.
    **kwargs
        Additional layout options forwarded to :meth:`go.Figure.update_layout`.

    Returns
    -------
    go.Figure
        Figure containing the axes visualization.
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
    """Visualize an STL ``model`` with Matplotlib's 3D toolkit.

    Parameters
    ----------
    model : mesh.Mesh
        Mesh object to render.
    ax : matplotlib.axes.Axes, optional
        Target axes. A new 3D subplot is created when omitted.
    fsz : float, optional
        Font size used for axis annotations.
    elev : float, optional
        Elevation angle passed to :meth:`Axes3D.view_init`.
    azim : float, optional
        Azimuth angle for the view initialization.
    facecolors : str, optional
        Fill color forwarded to :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`.
    edgecolors : str, optional
        Edge color used for rendered polygons.
    lw : float, optional
        Line width applied to polygon edges.
    x_lim, y_lim, z_lim : tuple[float, float], optional
        Axis limits. When ``None`` the existing axis limits are reused unless
        ``modify_axes`` is ``True``.
    modify_axes : bool, optional
        Override existing axes limits with model bounds when ``True``.
    full_model : bool, optional
        Render the entire mesh regardless of limits when ``True``.
    show_origin : bool, optional
        Add axis arrows and labels at the origin for reference.
    show_fig : bool, optional
        Display the containing figure via :meth:`matplotlib.figure.Figure.show`.
    **kwargs
        Additional keyword arguments passed to :class:`Poly3DCollection`.

    Returns
    -------
    matplotlib.axes.Axes
        Axes instance containing the rendered mesh.
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


# MARK: STL generation utilities
def make_stl(vertices: np.ndarray, faces: np.ndarray) -> mesh.Mesh:
    """Create an ``numpy-stl`` mesh from vertex and face arrays.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions with shape ``(N, 3)``.
    faces : np.ndarray
        Triangle indices referencing ``vertices``.

    Returns
    -------
    mesh.Mesh
        Mesh instance with normals computed.
    """

    stl_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_data.vectors[i][j] = vertices[f[j], :]

    stl_data.update_normals()
    return stl_data


def meshed_surface(para_1: np.ndarray, para_2: np.ndarray,
                   func: Callable, **func_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``func`` across a parameter grid and triangulate the resulting surface.

    Parameters
    ----------
    para_1, para_2 : np.ndarray
        Parameter vectors defining the rectangular sampling grid.
    func : Callable
        Callable mapping ``(u, v)`` inputs to ``(x, y, z)`` coordinates.
    **func_kwargs
        Additional keyword arguments forwarded to ``func``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the sampled vertices and Delaunay faces.
    """
    params = np.array(np.meshgrid(para_1, para_2)).T.reshape(-1, 2)
    tri = Delaunay(params)
    x, y, z = func(params[:, 0], params[:, 1], **func_kwargs)
    vertices = np.array([x, y, z]).T
    faces = tri.simplices
    return vertices, faces


def torus(theta: np.ndarray, phi: np.ndarray, a: float = 1, R: float = 2):
    """Evaluate torus surface coordinates for grids of ``theta`` and ``phi``.

    Parameters
    ----------
    theta : np.ndarray
        Toroidal angle values.
    phi : np.ndarray
        Poloidal angle values.
    a : float, optional
        Minor radius of the torus.
    R : float, optional
        Major radius of the torus.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays of ``x``, ``y`` and ``z`` coordinates.
    """
    x = (R + a * np.cos(theta)) * np.cos(phi)
    y = (R + a * np.cos(theta)) * np.sin(phi)
    z = a * np.sin(theta)
    return x, y, z


def sphere(theta: np.ndarray, phi: np.ndarray, r: float = 1):
    """Evaluate spherical surface coordinates for the supplied angles.

    Parameters
    ----------
    theta : np.ndarray
        Polar (inclination) angle values.
    phi : np.ndarray
        Azimuthal angle values.
    r : float, optional
        Sphere radius.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays of ``x``, ``y`` and ``z`` coordinates.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def make_2D_surface(points: np.ndarray, condition: Callable) -> mesh.Mesh:
    """Triangulate planar ``points`` and construct a filtered STL mesh.

    Parameters
    ----------
    points : np.ndarray
        Planar coordinates with shape ``(N, 2)``.
    condition : Callable
        Predicate evaluated on triangle centroids. Triangles are retained when
        the predicate returns ``False``.

    Returns
    -------
    mesh.Mesh
        Unclosed STL mesh constrained to the ``z = 0`` plane.
    """

    tri = Delaunay(points)
    gravity = np.mean(points[tri.simplices], axis=1)
    faces = tri.simplices[~condition(*gravity.T)]

    stl_data = make_stl(np.hstack([points, np.zeros((points.shape[0], 1))]), faces)
    return stl_data


if __name__ == '__main__':
    # delta_cone_apply_test()
    check_visible_test()

    vertices = np.array([[5, 3, -3],
                         [5, -3, 3],
                         [5, -3, -3]])
    faces = np.array([[0, 2, 1]])
    model = make_stl(vertices, faces)

    eye_position = np.array([0, 0, 0])
    points = np.stack(np.meshgrid(10,
                                  np.linspace(-15, 15, 51),
                                  np.linspace(-15, 15, 51),
                                  indexing="ij")).reshape((3, -1)).T

    cond = check_visible(model, eye_position, points)
    fig = go.Figure()
    plotly_show_stl(model, fig=fig, show_fig=False)
    fig.add_trace(go.Scatter3d(x=[eye_position[0]],
                               y=[eye_position[1]],
                               z=[eye_position[2]], mode="markers",
                               marker=dict(size=3, color="green")))
    x, y, z = points[cond.squeeze()].T
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers",
                               marker=dict(size=3, color="red")))
    x, y, z = points[~cond.squeeze()].T
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers",
                               marker=dict(size=3, color="blue")))
    for x, y, z in vertices:
        a = x - eye_position[0]
        yy = y * 10 / a
        zz = z * 10 / a
        fig.add_trace(go.Scatter3d(x=[0, 10],
                                   y=[0, yy],
                                   z=[0, zz], mode="lines",
                                   marker=dict(size=3, color="black")))
    fig.show()

    vertices = np.array([[-5, 3, -3],
                         [-5, -3, 3],
                         [-5, -3, -3],
                         [-5, 3, 3],
                         [-10, 3, -3],
                         [-10, -3, 3],
                         [-10, -3, -3],
                         [-10, 3, 3]])
    # faces = np.array([[0, 2, 1], [0, 1, 3]])
    faces = np.array([[0, 1, 2], [4, 5, 7]])
    # faces = np.array([[0, 1, 3]])
    # model = make_stl(vertices, faces)
    model = generate_aperture_stl(shape="circle", size=3.6, resolution=40, max_size=10)
    model.translate([0, 0, 5])
    # model.rotate([0, 1, 0], np.pi/2)
    eye_position = np.array([0, 0, 10])
    points = np.stack(np.meshgrid(np.linspace(-15, 15, 51),
                                  np.linspace(0, 15, 51),
                                  np.linspace(5, 15, 5),
                                  indexing="ij")).reshape((3, -1)).T

    cond = check_visible(model, eye_position, points, behind_start_included=-7)
    # cond = check_visible(model, eye_position, points, behind_start_included=True)
    fig = go.Figure()
    plotly_show_stl(model, fig=fig, show_fig=False)
    fig.add_trace(go.Scatter3d(x=[eye_position[0]],
                               y=[eye_position[1]],
                               z=[eye_position[2]], mode="markers",
                               marker=dict(size=3, color="green")))
    x, y, z = points[cond.squeeze()].T
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers",
                               marker=dict(size=2, color="red")))
    x, y, z = points[~cond.squeeze()].T
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers",
                               marker=dict(size=2, color="blue")))
    # for x, y, z in vertices:
    #     a = x - eye_position[0]
    #     yy = y * 10 / a
    #     zz = z * 10 / a
    #     fig.add_trace(go.Scatter3d(x=[x, 10],
    #                                y=[y, yy],
    #                                z=[z, zz], mode="lines",
    #                                marker=dict(size=3, color="black")))
    fig.show()
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
