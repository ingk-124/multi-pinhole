"""Scene orchestration and cache ownership for voxel-camera simulations.

``World`` owns scene objects and their visibility/projection caches. Private
numerical helpers perform calculations without owning scene state. See
``docs/world.md`` for workflow and persistence details.
"""

import gc
import os
import time
from collections.abc import Hashable, Mapping
from concurrent.futures import FIRST_COMPLETED, wait
from concurrent.futures.thread import ThreadPoolExecutor
from types import MappingProxyType
from typing import Literal, Tuple, List

import dill
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from stl import mesh

from .core import Aperture, Camera, Eye, Screen
from ._visibility import (
    calculate_point_visibility,
    calculate_visible_vertex_mask,
    classify_visible_voxels,
)
from .projection import (
    EyeProjectionWorkEstimate,
    PointSourceResolutionEstimate,
    ProjectionWorkEstimate,
    make_optical_binning,
    select_circumsphere_resolution,
)
from ._projection_matrix import (
    build_optical_projection_matrix,
    sum_eye_projections,
)
from .voxel import Voxel
from .utils import stl_utils
from .utils import type_check_and_list
from .utils.my_stdio import *


# Increment when a serialized projection representation or its numerical
# meaning becomes incompatible with an older cached matrix. Legacy pickles do
# not have this attribute and are invalidated when loaded.
PROJECTION_CACHE_SCHEMA_VERSION = 3


def type_list(obj, type_):
    """Normalize an input object into a list of a specific type.

    Note
    ----
    This helper is not currently called anywhere in this module; setters on
    :class:`World` (cameras, walls, ``inside_vertices``-related indices) use
    :func:`multi_pinhole.utils.type_check_and_list` instead, which offers the
    same normalization plus an optional ``default`` fallback for ``None``.

    Parameters
    ----------
    obj : Any
        Value that should either be an instance of ``type_`` or an iterable of
        ``type_`` instances.
    type_ : type
        Expected element type that each entry in ``obj`` must satisfy.

    Returns
    -------
    list[type_]
        List containing the objects of type ``type_`` extracted from ``obj``.

    Raises
    ------
    TypeError
        Raised when ``obj`` is neither an instance of ``type_`` nor a list of
        compatible objects.
    """

    if isinstance(obj, type_):
        return [obj]
    elif isinstance(obj, list):
        for _ in obj:
            if not isinstance(_, type_):
                raise TypeError(f"obj should be a list of {type_}, not {type(_)}")
        return obj
    else:
        raise TypeError(f"obj should be a list of {type_} or a {type_}, not {type(obj)}")


def _blocks_lengths(arr_blocks) -> np.ndarray:
    """Return lengths of 0th dimensions for an iterable of arrays.

    Parameters
    ----------
    arr_blocks : Iterable[np.ndarray | None]
        Collection of array blocks where ``None`` indicates an empty block.

    Returns
    -------
    np.ndarray
        Array of block lengths with entries of zero when the block is ``None``.
    """

    return np.array([0 if blk is None else blk.shape[0] for blk in arr_blocks], dtype=np.int64)


def _slice_blocks(
        pts_blocks: List[np.ndarray],
        S_blocks: List[sparse.csr_matrix],
        start: int,
        stop: int,
        n_vox: int
) -> Tuple[np.ndarray, sparse.csr_matrix]:
    """Slice concatenated block data within a global index range.

    Parameters
    ----------
    pts_blocks : list[np.ndarray]
        Point blocks whose rows correspond to individual entries in the global
        concatenated array.
    S_blocks : list[scipy.sparse.csr_matrix]
        Sparse matrix blocks aligned with ``pts_blocks`` rows.
    start : int
        Inclusive starting index of the global slice.
    stop : int
        Exclusive ending index of the global slice.
    n_vox : int
        Number of voxel columns used to size the returned sparse matrix when
        the slice is empty.

    Returns
    -------
    tuple[np.ndarray, scipy.sparse.csr_matrix]
        Tuple containing the stacked point array and sparse matrix slice for
        the requested index window.
    """
    out_pts, out_S = [], []
    if start >= stop:
        return np.empty((0, 3)), sparse.csr_matrix((0, n_vox))
    lens = _blocks_lengths(pts_blocks)
    csum = np.cumsum(np.concatenate([[0], lens]))
    b = int(np.searchsorted(csum, start, side="right") - 1)
    pos = start
    while pos < stop and b < len(pts_blocks):
        L = lens[b]
        if L == 0:
            b += 1
            continue
        blk_start, blk_stop = csum[b], csum[b + 1]
        s = max(0, pos - blk_start)
        t = min(L, stop - blk_start)
        if s < t:
            out_pts.append(pts_blocks[b][s:t])
            out_S.append(S_blocks[b][s:t, :])
            pos = blk_start + t
        else:
            pos = blk_stop
        b += 1
    pts_chunk = np.concatenate(out_pts, axis=0) if out_pts else np.empty((0, 3))
    S_chunk = sparse.vstack(out_S, format="csr") if out_S else sparse.csr_matrix((0, n_vox))
    return pts_chunk, S_chunk


class World:
    """Scene container binding a voxel grid, cameras, and optional walls.

    ``World`` maintains the bookkeeping needed to simulate imaging of a
    voxelized volume by one or more multi-pinhole cameras: it normalizes
    camera/wall inputs, tracks which voxel vertices are "inside" the modeled
    volume, caches per-camera visibility results, and builds the sparse
    projection matrices used to render synthetic images. See ``__init__``
    below for the full parameter reference.
    """

    def __init__(self,
                 voxel: Voxel = None,
                 cameras: Mapping[Hashable, Camera] | list[Camera] | Camera = None,
                 walls: list[mesh.Mesh] | mesh.Mesh = None,
                 inside_func: callable = None,
                 verbose: int = 1
                 ):
        """Instantiate the world container and initialize linked components.

        Parameters
        ----------
        voxel : Voxel, optional
            Voxel volume description used to compute visibility and projection
            matrices. A new :class:`~multi_pinhole.Voxel` instance is created
            when ``None`` (default).
        cameras : Camera, list[Camera], or dict[Hashable, Camera], optional
            Cameras registered with the world. Lists receive integer keys from
            ``range(len(cameras))``; dictionaries retain their supplied keys.
            A single Camera receives key ``0``.
        walls : mesh.Mesh or list[mesh.Mesh], optional
            STL meshes representing obstructions or environmental walls. Each
            mesh is registered to support visibility checks. ``None`` (default)
            skips wall initialization.
        inside_func : callable, optional
            Callable accepting three ``numpy.ndarray`` coordinate arrays
            ``(X, Y, Z)`` and returning a boolean mask identifying voxels that
            belong inside the world volume. When provided, the function is
            executed immediately to seed :attr:`inside_vertices`.
        verbose : int, optional
            Verbosity level controlling informational output. Defaults to ``1``.

        Notes
        -----
        Cameras and walls are normalized through :func:`type_check_and_list`
        ensuring that downstream operations receive homogeneous lists of
        objects.
        """

        self._voxel = voxel if voxel is not None else Voxel()
        self._voxel.set_world(self)

        self._cameras = self._normalize_cameras(cameras)
        for _ in self._cameras.values():
            _.set_world(self)

        self._walls = []
        self._wall_ranges = None
        self.walls = walls

        self._inside_function = None
        self._inside_kwargs = {}
        self._inside_vertices = None
        self._visible_vertices = {i: None for i in self._cameras.keys()}
        self._visible_voxels = {i: None for i in self._cameras.keys()}
        self._projection = {i: [None] * len(self._cameras[i].eyes) for i in self._cameras.keys()}
        self._projection_settings = {i: [None] * len(self._cameras[i].eyes) for i in self._cameras.keys()}
        self._P_matrix = {i: None for i in self._cameras.keys()}
        self._projection_cache_schema_version = PROJECTION_CACHE_SCHEMA_VERSION
        self.verbose = verbose

        if inside_func is not None:
            self.set_inside_vertices(inside_func)

    @staticmethod
    def _normalize_cameras(cameras):
        """Normalize Camera, list, or keyed dict input to a new dictionary."""
        if cameras is None:
            return {}
        if isinstance(cameras, Camera):
            return {0: cameras}
        if isinstance(cameras, Mapping):
            normalized = dict(cameras)
        elif isinstance(cameras, (list, tuple)):
            normalized = dict(enumerate(cameras))
        else:
            raise TypeError("cameras must be a Camera, list/tuple of Camera, dict of Camera, or None")

        if not all(isinstance(camera, Camera) for camera in normalized.values()):
            raise TypeError("every cameras value must be a Camera")
        if len({id(camera) for camera in normalized.values()}) != len(normalized):
            raise ValueError("the same Camera instance cannot be registered twice in one World")
        return normalized

    def __repr__(self):
        """Represent the world with its core components for debugging.

        Returns
        -------
        str
            Readable summary containing the voxel object, registered cameras,
            and wall meshes.
        """

        return f"World(voxel={self.voxel}, CAMERA={self.cameras}, walls={self.walls})"

    def __copy__(self):
        """Create a shallow copy retaining voxel and camera references.

        Returns
        -------
        World
            New world sharing the original voxel and camera objects.
        """

        return World(cameras=self.cameras, voxel=self.voxel)

    def camera_info(self):
        """Summarize currently registered cameras.

        Returns
        -------
        str
            Multiline description enumerating each camera and its
            :func:`__repr__` output.
        """

        txt = "Camera Info:\n"
        for i, camera in self._cameras.items():
            txt += f" Camera {i}:\n"
            txt += camera.__repr__() + "\n"
        txt = txt.rstrip("\n")
        return txt

    def voxel_info(self):
        """Describe the voxel grid configuration.

        Returns
        -------
        str
            Representation of the :class:`~multi_pinhole.Voxel` instance
            associated with the world.
        """

        return self.voxel.__repr__()

    def save_world(self, filename):
        """Persist the world instance to disk.

        Parameters
        ----------
        filename : str or os.PathLike[str]
            Destination path where the serialized world should be written.
        """
        self._ensure_projection_cache_schema()
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load_world(filename):
        """Deserialize a world instance from disk.

        Parameters
        ----------
        filename : str or os.PathLike[str]
            Path to a pickled :class:`World` instance created by
            :meth:`save_world`.

        Returns
        -------
        World
            Restored world populated with the serialized state.
        """
        with open(filename, "rb") as f:
            loaded_world = dill.load(f)
        if not isinstance(loaded_world, World):
            raise TypeError("serialized object is not a World")
        loaded_world._ensure_projection_cache_schema()
        return loaded_world

    @property
    def cameras(self):
        """Mapping[Hashable, Camera]: Read-only stable-key camera mapping."""
        return MappingProxyType(self._cameras)

    @property
    def voxel(self):
        """Voxel: Primary voxel model used for visibility and projection."""
        return self._voxel

    @property
    def walls(self):
        """list[mesh.Mesh]: Collection of STL meshes representing walls."""
        return self._walls

    @property
    def wall_ranges(self):
        """zip[tuple[float, float]]: Axis-aligned bounds for registered walls."""
        if not self.walls:
            return None
        _ = [(w.update_min(), w.update_max()) for w in self.walls]
        self._wall_ranges = zip(np.min([w.min_ for w in self._walls], axis=0),
                                np.max([w.max_ for w in self._walls], axis=0))
        return self._wall_ranges

    @property
    def visible_voxels(self):
        """dict[int, np.ndarray]: Visibility state of each voxel per camera."""
        return self._visible_voxels

    @property
    def P_matrix(self):
        """dict[int, sparse.csr_matrix]: Cached voxel-to-pixel projection matrices."""
        return self._P_matrix

    @property
    def projection(self):
        """dict[int, list[sparse.csr_matrix]]: Pixel-space projection matrices per eye."""
        return self._projection

    @property
    def projection_cache_schema_version(self) -> int:
        """Version governing compatibility of serialized projection caches."""
        return self._projection_cache_schema_version

    def _projection_operator(self, camera_idx: Hashable, eye_idx: int | None):
        """Return one cached projection matrix without triggering construction."""
        if camera_idx not in self._cameras:
            raise KeyError(f"camera key {camera_idx!r} is not registered")
        if eye_idx is None:
            operator = self._P_matrix[camera_idx]
            description = f"camera {camera_idx!r}"
        else:
            if not isinstance(eye_idx, (int, np.integer)):
                raise TypeError("eye_idx must be an integer or None")
            if eye_idx < 0 or eye_idx >= len(self._cameras[camera_idx].eyes):
                raise IndexError(
                    f"eye_idx must be in [0, {len(self._cameras[camera_idx].eyes) - 1}]"
                )
            operator = self._projection[camera_idx][int(eye_idx)]
            description = f"camera {camera_idx!r}, eye {int(eye_idx)}"
        if operator is None:
            raise RuntimeError(
                f"projection matrix for {description} is not constructed; "
                "call set_projection_matrix() first"
            )
        return operator

    def project(self, emission, camera_idx: Hashable, eye_idx: int | None = None):
        """Apply a cached voxel-to-pixel projection matrix.

        Parameters
        ----------
        emission : array-like, shape (N_voxel,)
            Emission value at every voxel center.
        camera_idx : Hashable
            Key of the camera to project. The camera's Eye contributions are
            summed when ``eye_idx`` is ``None``.
        eye_idx : int, optional
            Select one Eye instead of the camera-summed projection.

        Returns
        -------
        numpy.ndarray, shape (N_pixel,)
            Pixel-space image.

        Notes
        -----
        This method never constructs a projection implicitly. Call
        :meth:`set_projection_matrix` before projecting.
        """
        emission = np.asarray(emission)
        if emission.ndim != 1 or emission.shape[0] != self.voxel.N:
            raise ValueError(
                f"emission must have shape {(self.voxel.N,)}, got {emission.shape}"
            )
        operator = self._projection_operator(camera_idx, eye_idx)
        return np.asarray(operator @ emission).reshape(-1)

    def backproject(self, image, camera_idx: Hashable, eye_idx: int | None = None):
        """Apply the transpose of a cached voxel-to-pixel projection matrix.

        Parameters
        ----------
        image : array-like, shape (N_pixel,)
            Pixel-space values to map back to voxel space.
        camera_idx : Hashable
            Key of the camera whose transpose is applied. The camera-summed
            projection is used when ``eye_idx`` is ``None``.
        eye_idx : int, optional
            Select one Eye instead of the camera-summed projection.

        Returns
        -------
        numpy.ndarray, shape (N_voxel,)
            Result of the discrete adjoint operation ``P.T @ image``.

        Notes
        -----
        Backprojection is the linear adjoint, not an inverse reconstruction.
        This method never constructs a projection implicitly.
        """
        operator = self._projection_operator(camera_idx, eye_idx)
        image = np.asarray(image)
        expected_shape = (operator.shape[0],)
        if image.ndim != 1 or image.shape != expected_shape:
            raise ValueError(
                f"image must have shape {expected_shape}, got {image.shape}"
            )
        return np.asarray(operator.T @ image).reshape(-1)

    @property
    def inside_vertices(self):
        """np.ndarray: Boolean mask indicating which voxel vertices lie inside the world."""
        if self._inside_vertices is None:
            return np.ones(self.voxel.N_grid, dtype=bool)
        else:
            return self._inside_vertices

    def _invalidate_visibility_cache(self) -> None:
        """Clear cached visibility and projection data for every camera.

        Resets :attr:`_visible_vertices`, :attr:`_visible_voxels`,
        :attr:`_projection`, and :attr:`_P_matrix` to ``None`` placeholders so
        that the next visibility/projection query recomputes from scratch.
        Called whenever the inside-vertex mask changes.
        """
        self._visible_vertices = {i: None for i in self._cameras.keys()}
        self._visible_voxels = {i: None for i in self._cameras.keys()}
        self._invalidate_projection_cache()

    def _invalidate_projection_cache(self) -> None:
        """Clear projection data while retaining valid visibility data."""
        self._projection = {i: [None] * len(self._cameras[i].eyes) for i in self._cameras.keys()}
        self._projection_settings = {i: [None] * len(self._cameras[i].eyes) for i in self._cameras.keys()}
        self._P_matrix = {i: None for i in self._cameras.keys()}

    def _ensure_projection_cache_schema(self) -> None:
        """Invalidate absent or incompatible serialized projection caches."""
        cached_version = getattr(self, "_projection_cache_schema_version", None)
        if cached_version != PROJECTION_CACHE_SCHEMA_VERSION:
            self._invalidate_projection_cache()
            self._projection_cache_schema_version = PROJECTION_CACHE_SCHEMA_VERSION

    @inside_vertices.setter
    def inside_vertices(self, inside_vertices: np.ndarray):
        """Validate and store the inside-vertex mask.

        Parameters
        ----------
        inside_vertices : numpy.ndarray
            Boolean array of length ``voxel.N_grid`` marking which vertices are
            considered inside the world volume.

        Raises
        ------
        TypeError
            Raised when ``inside_vertices`` is not a ``numpy.ndarray``.
        ValueError
            Raised when the array size does not match ``voxel.N_grid``.
        """
        if not isinstance(inside_vertices, np.ndarray):
            raise TypeError(f"inside_voxels should be a np.ndarray, not {type(inside_vertices)}")
        if inside_vertices.size != self.voxel.N_grid:
            raise ValueError("inside_voxels should be a np.ndarray with the length of the number of grid points.")
        self._inside_function = None
        self._inside_kwargs = {}
        self._inside_vertices = inside_vertices.astype(bool)
        self._invalidate_visibility_cache()

    @cameras.setter
    def cameras(self, cameras: Mapping[Hashable, Camera] | list[Camera] | Camera):
        """Register or replace cameras managed by the world.

        Parameters
        ----------
        cameras : Camera, list[Camera], or dict[Hashable, Camera]
            Camera instances that should replace the current set.

        Notes
        -----
        When an incoming camera already exists in the current mapping, its
        cached visibility and projection matrices are reused to avoid
        recalculation.
        """
        cameras = self._normalize_cameras(cameras)
        old_cameras = self._cameras
        old_indices = {id(camera): index for index, camera in old_cameras.items()}
        camera_dict = {}
        visible_vertices = {}
        visible_voxels = {}
        projection = {}
        projection_settings = {}
        P_matrix = {}
        for camera_key, camera in cameras.items():
            if id(camera) in old_indices:
                ind = old_indices[id(camera)]
                # keep the previous visible voxels, projection, and P matrix
                visible_vertices[camera_key] = self._visible_vertices[ind]
                visible_voxels[camera_key] = self._visible_voxels[ind]
                projection[camera_key] = self._projection[ind]
                projection_settings[camera_key] = self._projection_settings[ind]
                P_matrix[camera_key] = self._P_matrix[ind]
            else:
                # if the camera is not in the original camera list,
                # set the visible voxels, projection, and P matrix to None
                visible_vertices[camera_key] = None
                visible_voxels[camera_key] = None
                projection[camera_key] = [None] * len(camera.eyes)
                projection_settings[camera_key] = [None] * len(camera.eyes)
                P_matrix[camera_key] = None
            camera.set_world(self)
            camera_dict[camera_key] = camera

        retained_ids = {id(camera) for camera in cameras}
        for old_camera in old_cameras.values():
            if id(old_camera) not in retained_ids:
                old_camera.unset_world(self)

        self._cameras = camera_dict
        self._visible_vertices = visible_vertices
        self._visible_voxels = visible_voxels
        self._projection = projection
        self._projection_settings = projection_settings
        self._P_matrix = P_matrix
        if all([_ is None for _ in visible_voxels.values()]):
            print("Notice: All cameras are updated.")
        elif None in visible_voxels.values():
            reused_index = [repr(key) for key in visible_voxels if visible_voxels[key] is not None]
            xth = ", ".join(reused_index[:-1]) + " and " + reused_index[-1] if \
                len(reused_index) > 1 else f"{reused_index[0]}"
            print(f"Notice: {xth} camera{'s are' if len(reused_index) > 1 else ' is'} reused.")
        else:
            print("Notice: All cameras are reused.")
        print(self.camera_info())

    def add_camera(self, camera_key: Hashable, new_camera: Camera):
        """Register one Camera under an explicit stable key.

        Parameters
        ----------
        camera_key : Hashable
            Stable key used by all per-camera caches.
        new_camera : Camera
            Camera object to register.

        Notes
        -----
        Newly inserted cameras start without cached visibility or projection
        data until recomputation is triggered.
        """
        if not isinstance(camera_key, Hashable):
            raise TypeError("camera_key must be hashable")
        if camera_key in self._cameras:
            raise KeyError(f"camera key {camera_key!r} is already registered")
        if not isinstance(new_camera, Camera):
            raise TypeError("new_camera must be a Camera")
        if any(new_camera is existing for existing in self._cameras.values()):
            raise ValueError("the same Camera instance cannot be registered twice in one World")

        new_camera.set_world(self)
        self._cameras[camera_key] = new_camera
        self._visible_vertices[camera_key] = None
        self._visible_voxels[camera_key] = None
        self._projection[camera_key] = [None] * len(new_camera.eyes)
        self._projection_settings[camera_key] = [None] * len(new_camera.eyes)
        self._P_matrix[camera_key] = None
        print(self.camera_info())

    def remove_camera(self, camera_key):
        """Remove one camera without renumbering any remaining keys.

        Parameters
        ----------
        camera_key : Hashable
            Key of the Camera to remove.

        Notes
        -----
        Keys are stable: removing one Camera does not change any other key.
        A future explicit key-reset helper may be added if compact integer
        keys are needed.
        """
        if camera_key not in self._cameras:
            raise KeyError(f"camera key {camera_key!r} is not registered")
        removed_camera = self._cameras.pop(camera_key)
        self._visible_vertices.pop(camera_key)
        self._visible_voxels.pop(camera_key)
        self._projection.pop(camera_key)
        self._projection_settings.pop(camera_key)
        self._P_matrix.pop(camera_key)
        removed_camera.unset_world(self)
        print(self.camera_info())

    def change_camera(self, camera_key: Hashable, camera: Camera):
        """Replace the Camera at a stable key.

        Parameters
        ----------
        camera_key : Hashable
            Existing key whose Camera should be replaced.
        camera : Camera
            New Camera instance assigned to the key.

        Raises
        ------
        KeyError
            Raised when ``camera_key`` is not registered.
        """
        if camera_key not in self._cameras:
            raise KeyError(f"camera key {camera_key!r} is not registered")
        if not isinstance(camera, Camera):
            raise TypeError("camera must be a Camera")
        if any(camera is existing for existing_key, existing in self._cameras.items()
               if existing_key != camera_key):
            raise ValueError("the same Camera instance cannot be registered twice in one World")

        old_camera = self._cameras[camera_key]
        camera.set_world(self)
        self._cameras[camera_key] = camera
        self._visible_vertices[camera_key] = None
        self._visible_voxels[camera_key] = None
        self._projection[camera_key] = [None] * len(camera.eyes)
        self._projection_settings[camera_key] = [None] * len(camera.eyes)
        self._P_matrix[camera_key] = None
        if old_camera is not camera:
            old_camera.unset_world(self)

        print(self.camera_info())

    @voxel.setter
    def voxel(self, voxel):
        """Assign a new voxel model and invalidate cached projections.

        Parameters
        ----------
        voxel : Voxel
            Voxel object to associate with the world.

        Raises
        ------
        TypeError
            Raised when ``voxel`` is not an instance of :class:`Voxel`.
        """
        if not isinstance(voxel, Voxel):
            raise TypeError(f"voxel should be a Voxel, not {type(voxel)}")
        if voxel != self._voxel:
            my_print("Voxel is updated.", show=self.verbose > 0)
            self._voxel = voxel
            self._visible_voxels = {i: None for i in self._cameras.keys()}
            self._invalidate_projection_cache()

    @walls.setter
    def walls(self, walls: list[mesh.Mesh] | mesh.Mesh):
        """Update the STL meshes that define world boundaries.

        Parameters
        ----------
        walls : mesh.Mesh or list[mesh.Mesh]
            Mesh objects describing occluding geometry.
        """
        walls = type_check_and_list(walls, mesh.Mesh)
        if walls != self._walls:
            self._walls = walls
            self._visible_voxels = {i: None for i in self._cameras.keys()}
            if self._walls:
                for wall in self._walls:
                    wall.update_min()
                    wall.update_max()
                self._wall_ranges = zip(np.min([_.min_ for _ in self._walls], axis=0),
                                        np.max([_.max_ for _ in self._walls], axis=0))

    def set_inside_vertices(self, function: callable, **kwargs) -> None:
        """Derive the inside-vertex mask via a user-provided function.

        Parameters
        ----------
        function : callable
            Callable accepting the voxel grid coordinates ``(X, Y, Z)`` as
            ``numpy.ndarray`` arguments and returning a boolean array marking
            interior vertices.
        **kwargs : Any
            Additional keyword arguments forwarded to ``function``.

        Raises
        ------
        TypeError
            Raised when ``function`` is not callable.
        ValueError
            Raised when the returned mask does not match ``voxel.N_grid``.

        Notes
        -----
        The resulting boolean array is stored in :attr:`inside_vertices` and
        used to restrict visibility calculations to in-bounds voxels.
        """
        if not callable(function):
            raise TypeError(f"function should be a callable, not {type(function)}")
        inside_vertices = function(*self.voxel.grid.T, **kwargs).astype(bool)
        if inside_vertices.size != self.voxel.N_grid:
            raise ValueError("The return value of the function should be a boolean array with the length of the number "
                             "of grid points.")
        else:
            self._inside_function = function
            self._inside_kwargs = dict(kwargs)
            self._inside_vertices = inside_vertices
            self._invalidate_visibility_cache()
            my_print(f"Inside vertices are updated. (N_inside_vertices: {np.sum(inside_vertices)})",
                     show=self.verbose > 0)

    def _inside_points(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the stored inside mask at arbitrary points when available."""
        if self._inside_function is None:
            return np.ones(points.shape[0], dtype=bool)
        return self._inside_function(*points.T, **self._inside_kwargs).astype(bool)

    def find_visible_points(self, points: np.ndarray, camera_idx: Hashable, eye_idx: int = None,
                            verbose: int = 1) -> np.ndarray:
        """Determine point visibility for a specific camera and eye selection.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape ``(N_points, 3)`` containing world-coordinate points
            whose visibility should be evaluated.
        camera_idx : Hashable
            Key of the camera used for visibility testing.
        eye_idx : int or list[int], optional
            Specific eye indices within the camera that should be considered.
            ``None`` (default) evaluates all eyes.
        verbose : int, optional
            Verbosity level controlling logging output. Defaults to ``1``.

        Returns
        -------
        numpy.ndarray
            Boolean matrix of shape ``(N_eye, N_points)`` indicating whether
            each point is visible from the selected eyes.

        Raises
        ------
        ValueError
            Raised when ``camera_idx`` or ``eye_idx`` reference non-existent
            entries.
        """

        if camera_idx not in self._cameras.keys():
            raise KeyError(f"camera key {camera_idx!r} is not registered")
        _camera = self._cameras[camera_idx]
        walls_in_camera = [stl_utils.copy_model(wall, -_camera.camera_position, _camera.rotation_matrix.T) for
                           wall in self.walls]
        camera_points = _camera.world2camera(points)  # (N_points, 3)

        eye_idx = range(len(_camera.eyes)) if eye_idx is None else type_check_and_list(eye_idx, int)
        if any([_e >= len(_camera.eyes) or _e < 0 for _e in eye_idx]):
            raise ValueError(f"eye_idx should be in the range of [0, {len(_camera.eyes) - 1}]")

        return calculate_point_visibility(
            camera_points=camera_points,
            eyes=_camera.eyes,
            eye_indices=eye_idx,
            apertures=_camera.apertures,
            walls_in_camera=walls_in_camera,
            verbose=verbose,
        )  # (N_eye, N_points)

    def _find_visible_vertices(self, force: bool = False, verbose: int = None,
                               camera_idx: Hashable = None) -> None:
        """Compute visibility masks for voxel vertices per camera.

        Parameters
        ----------
        force : bool, optional
            When ``True`` (default ``False``), recompute visibility even if
            cached data with matching shape exists.
        verbose : int, optional
            Verbosity level overriding :attr:`verbose`. ``None`` preserves the
            world default.
        camera_idx : Hashable, optional
            Specific camera key to update. ``None`` (default) processes all
            cameras.

        Notes
        -----
        Visibility results are stored in :attr:`_visible_vertices` as boolean
        arrays shaped ``(N_eye, N_vertices)``.
        """
        verbose = self.verbose if verbose is None else verbose
        if camera_idx is not None:
            if camera_idx not in self._cameras.keys():
                raise KeyError(f"camera key {camera_idx!r} is not registered")
            cameras_to_process = [camera_idx]
        else:
            cameras_to_process = list(self._cameras.keys())
        for c_ in cameras_to_process:
            camera = self._cameras[c_]
            if force:
                my_print("Force to calculate visible vertices.", show=verbose > 0)
                self._visible_vertices[c_] = None
            if self._visible_vertices[c_] is not None:
                if self._visible_vertices[c_].shape == (len(camera.eyes), self.voxel.N_grid):
                    my_print(f"Visible vertices for camera {c_!r} is already calculated.",
                             show=verbose > 0)
                    continue
                else:
                    my_print(f"Visible vertices for camera {c_!r} has wrong shape. "
                             f"Recalculating...", show=verbose > 0)
            else:
                my_print(f"Finding visible vertices for camera {c_!r}...", show=verbose > 0)
            if np.sum(self.inside_vertices) == 0:
                my_print("No inside vertices. Skip calculating visible vertices.", show=verbose > 0)
                inside_visibility = np.ones((len(camera.eyes), 0), dtype=bool)
            else:
                inside_visibility = self.find_visible_points(
                    self.voxel.grid[self.inside_vertices],
                    camera_idx=c_,
                    verbose=verbose,
                )  # (N_eye, N_inside_points)
            visible_vertices = calculate_visible_vertex_mask(
                self.inside_vertices,
                inside_visibility,
            )
            self._visible_vertices[c_] = visible_vertices
            my_print(f"Visible vertices for camera {c_!r} is calculated.", show=verbose > 0)

    def find_visible_voxels(self, force: bool = False, verbose: int | None = None):
        """Evaluate voxel visibility states for each camera.

        Parameters
        ----------
        force : bool, optional
            When ``True`` (default ``False``), forces recomputation of vertex
            visibility before aggregating voxel visibility.
        verbose : int, optional
            Verbosity level overriding :attr:`verbose`. ``None`` preserves the
            world default.

        Notes
        -----
        Results are stored in :attr:`_visible_voxels` as integer arrays with
        shape ``(N_eye, N_voxels)``. Values are interpreted as:

        * ``0`` – not visible
        * ``1`` – partially visible (some vertices visible)
        * ``2`` – fully visible (all vertices visible)
        """

        verbose = self.verbose if verbose is None else verbose
        my_print("Finding visible voxels...", show=verbose > 0)
        visible_voxels = {}
        for c_ in self._cameras.keys():
            cached = self._visible_voxels.get(c_)
            expected_shape = (len(self._cameras[c_].eyes), self.voxel.N)
            if not force and cached is not None and cached.shape == expected_shape:
                visible_voxels[c_] = cached
                my_print(f"Visible voxels for camera {c_!r} is already calculated.",
                         show=verbose > 1)
                continue
            self._find_visible_vertices(force=force, verbose=verbose, camera_idx=c_)
            visible_voxels[c_] = classify_visible_voxels(
                self._visible_vertices[c_],
                self.voxel.vertices_indices,
            )
            my_print(f"Visible voxels for camera {c_!r} is calculated.", show=verbose > 0)
        self._visible_voxels = visible_voxels
        my_print("Finding visible voxels is done.", show=verbose > 0)

    def _calc_voxel_image_for_eye_optical(
            self, camera_idx: Hashable, eye_idx: int,
            full_voxels: np.ndarray, partial_voxels: np.ndarray,
            full_subvoxel_res, partial_subvoxel_res,
            max_nnz: int, max_working_memory: int,
            optical_bin_width_pixels, verbose: int):
        """Build ordinary sparse P using visible-voxel optical bins."""
        return build_optical_projection_matrix(
            voxel=self.voxel,
            camera=self.cameras[camera_idx],
            eye_idx=eye_idx,
            full_voxels=full_voxels,
            partial_voxels=partial_voxels,
            full_subvoxel_res=full_subvoxel_res,
            partial_subvoxel_res=partial_subvoxel_res,
            max_nnz=max_nnz,
            max_working_memory=max_working_memory,
            optical_bin_width_pixels=optical_bin_width_pixels,
            verbose=verbose,
            point_visibility=lambda points: self.find_visible_points(
                points, camera_idx=camera_idx, eye_idx=eye_idx, verbose=0,
            ),
            inside_points=self._inside_points,
            make_binning=make_optical_binning,
        )

    def estimate_source_resolution(
            self, camera_idx: Hashable = 0, eye_idx: int = 0,
            voxel_indices=None, max_resolution=4,
            point_source_threshold: float = 1.0 / 8.0,
            detector_grid: str = "psf",
            batch_size: int = 100_000) -> PointSourceResolutionEstimate:
        """Recommend source resolution using a local perspective heuristic.

        This diagnostic does not alter the voxel grid or a cached projection.
        It compares a voxel-center local-perspective circumsphere scale with
        the selected detector/PSF scale. The same threshold gives an uncapped
        near-cubic axis-wise resolution, clipped by ``max_resolution``.

        Parameters
        ----------
        camera_idx : Hashable, optional
            Camera key. Defaults to zero.
        eye_idx : int, optional
            Eye index within the camera. Defaults to zero.
        voxel_indices : array-like of int, optional
            Voxels to evaluate. ``None`` evaluates the complete grid.
        max_resolution : int or (int, int, int), optional
            Axis-wise ceiling for the ideal resolution.
        point_source_threshold : float, optional
            Maximum projected circumsphere diameter in reference-scale units
            for selecting res=1. Defaults to 1/8.
        detector_grid : {"psf", "subpixel", "pixel"}, optional
            Scale used to normalize projected source displacement. ``"psf"``
            uses the larger of detector subpixel pitch and the local projected
            Eye footprint, so a finite pinhole PSF can dominate the required
            source quadrature. The other modes use detector pitch alone.
        batch_size : int, optional
            Maximum voxel count whose six chord endpoints are materialized at
            once. This bounds diagnostic memory for large grids.

        Returns
        -------
        PointSourceResolutionEstimate
            Diagnostics in the same order as ``voxel_indices``.

        Notes
        -----
        This is not a rigorous upper bound for projection over the complete
        finite voxel. Large voxels near an Eye can be underestimated. The
        point-source threshold is a sampling policy, not an image-error
        tolerance; ``ideal_resolution`` likewise carries no error guarantee.
        Invalid geometry uses ``max_resolution`` as a fallback, and capped
        axes may not reach the heuristic recommendation.
        """
        if camera_idx not in self.cameras:
            raise KeyError(f"unknown camera key: {camera_idx!r}")
        camera = self.cameras[camera_idx]
        if not 0 <= eye_idx < len(camera.eyes):
            raise IndexError("eye_idx is out of range")
        if detector_grid not in {"psf", "subpixel", "pixel"}:
            raise ValueError("detector_grid must be 'psf', 'subpixel', or 'pixel'")
        if isinstance(batch_size, (bool, np.bool_)) or int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        batch_size = int(batch_size)

        if voxel_indices is None:
            voxel_indices = np.arange(self.voxel.N, dtype=np.int64)
        else:
            voxel_indices = self.voxel._type_check_n_voxel(voxel_indices)

        resolution_parts = []
        ratio_parts = []
        diameter_parts = []
        reference_parts = []
        point_source_parts = []
        ideal_parts = []
        capped_parts = []
        valid_parts = []
        for start in range(0, voxel_indices.size, batch_size):
            selected = voxel_indices[start:start + batch_size]
            centers = self.voxel.get_gravity_center(selected)
            points_camera = camera.world2camera(centers)
            eye = camera.eyes[eye_idx]
            points_in_eye = eye.camera2eye(points_camera)
            if detector_grid == "pixel":
                reference_size = float(np.min(camera.screen.pixel_size))
            elif detector_grid == "subpixel":
                reference_size = float(np.min(camera.screen.subpixel_size))
            else:
                axial_distance = points_in_eye[:, 2]
                zoom_rate = np.full(axial_distance.shape, np.nan)
                in_front = axial_distance > 0.0
                zoom_rate[in_front] = (
                    1.0 + eye.focal_length / axial_distance[in_front]
                )
                spot_size_uv = (
                    eye.eye_size[::-1][None, :] *
                    np.abs(zoom_rate[:, None])
                )
                # Invalid/behind-Eye samples fail the geometry test below; use
                # detector pitch here so the diagnostic scale stays finite.
                spot_size_uv = np.where(
                    np.isfinite(spot_size_uv), spot_size_uv, 0.0,
                )
                reference_size = np.min(np.maximum(
                    camera.screen.subpixel_size[None, :], spot_size_uv,
                ), axis=1)
            estimate = select_circumsphere_resolution(
                points_in_eye, self.voxel.get_edge_lengths(selected),
                focal_length=eye.focal_length,
                reference_size=reference_size,
                fallback_resolution=max_resolution,
                point_source_threshold=point_source_threshold,
            )
            resolution_parts.append(estimate.resolution)
            ratio_parts.append(estimate.ratio)
            diameter_parts.append(estimate.projected_diameter)
            reference_parts.append(estimate.reference_size)
            point_source_parts.append(estimate.point_source)
            ideal_parts.append(estimate.ideal_resolution)
            capped_parts.append(estimate.capped)
            valid_parts.append(estimate.valid)

        empty_shape = (0, 3)
        return PointSourceResolutionEstimate(
            resolution=(np.concatenate(resolution_parts) if resolution_parts else
                        np.empty(empty_shape, dtype=np.int64)),
            ratio=(np.concatenate(ratio_parts) if ratio_parts else
                   np.empty(0, dtype=float)),
            projected_diameter=(np.concatenate(diameter_parts) if diameter_parts else
                                np.empty(0, dtype=float)),
            reference_size=(np.concatenate(reference_parts) if reference_parts else
                            np.empty(0, dtype=float)),
            point_source=(np.concatenate(point_source_parts) if point_source_parts else
                          np.empty(0, dtype=bool)),
            ideal_resolution=(np.concatenate(ideal_parts) if ideal_parts else
                              np.empty(empty_shape, dtype=float)),
            capped=(np.concatenate(capped_parts) if capped_parts else
                    np.empty(empty_shape, dtype=bool)),
            valid=(np.concatenate(valid_parts) if valid_parts else
                   np.empty(0, dtype=bool)),
        )

    @staticmethod
    def _normalize_projection_resolution(value, default=None) -> tuple[int, int, int]:
        """Normalize one fixed source resolution without mutating the Voxel."""
        value = default if value is None else value
        if value is None:
            raise ValueError("resolution must be specified")
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("resolution must contain positive integers")
        try:
            resolution = np.asarray(np.broadcast_to(value, 3), dtype=float)
        except ValueError as exc:
            raise ValueError("resolution must be an integer or length-3 sequence") from exc
        if (not np.all(np.isfinite(resolution)) or np.any(resolution < 1) or
                np.any(resolution != np.floor(resolution))):
            raise ValueError("resolution must contain positive integers")
        return tuple(int(item) for item in resolution)

    @classmethod
    def _resolve_projection_resolutions(
            cls, res, res_mode: str, partial_res,
            ) -> tuple[tuple[int, int, int] | None,
                       tuple[int, int, int], bool]:
        """Validate public source-resolution settings in one place.

        ``fixed`` uses ``res`` directly, ``auto`` clips each geometric ideal
        resolution by ``res``, and ``ideal`` is the explicit uncapped escape
        hatch. The latter requires both ``res=None`` and a fixed
        ``partial_res`` so uncapped work cannot be requested accidentally and
        discontinuous partial visibility never inherits an undefined value.
        """
        if res_mode not in {"fixed", "auto", "ideal"}:
            raise ValueError("res_mode must be 'fixed', 'auto', or 'ideal'")
        if res_mode == "ideal":
            if res is not None:
                raise ValueError("res_mode='ideal' requires res=None")
            if partial_res is None:
                raise ValueError(
                    "res_mode='ideal' requires an explicit partial_res",
                )
            return None, cls._normalize_projection_resolution(partial_res), True
        if res is None:
            raise ValueError(f"res_mode={res_mode!r} requires a finite res")
        full_resolution = cls._normalize_projection_resolution(res)
        partial_resolution = cls._normalize_projection_resolution(
            partial_res, full_resolution,
        )
        return full_resolution, partial_resolution, res_mode == "auto"

    @classmethod
    def _projection_cache_key(
            cls, res, res_mode: str, partial_res,
            chunk_strategy: str, optical_bin_width_pixels,
            point_source_threshold: float) -> tuple:
        """Return the numerical settings that determine one eye projection.

        Execution-only controls such as parallelism and working-memory limits
        are deliberately excluded: they may change chunk boundaries, but not
        the resulting quadrature policy.
        """
        full_resolution, partial_resolution, adaptive = (
            cls._resolve_projection_resolutions(res, res_mode, partial_res)
        )
        if chunk_strategy not in {"voxel", "optical"}:
            raise ValueError("chunk_strategy must be 'voxel' or 'optical'")
        optical_width = None
        if chunk_strategy == "optical":
            try:
                width = np.asarray(
                    np.broadcast_to(optical_bin_width_pixels, 2), dtype=float,
                )
            except ValueError as exc:
                raise ValueError(
                    "optical_bin_width_pixels must be scalar or length 2",
                ) from exc
            optical_width = tuple(float(item) for item in width)
        return (
            res_mode, full_resolution, partial_resolution,
            float(point_source_threshold) if adaptive else None,
            chunk_strategy, optical_width,
        )

    def preflight_projection(
            self, res: int | tuple[int, int, int] | None,
            res_mode: Literal["fixed", "auto", "ideal"] = "fixed",
            partial_res: int | tuple[int, int, int] | None = None,
            point_source_threshold: float = 1.0 / 8.0,
            force_visibility: bool = False,
            verbose: int = 0) -> ProjectionWorkEstimate:
        """Estimate projection source-sample work without constructing ``P``.

        Parameters
        ----------
        res : int or tuple of int or None
            Full-voxel resolution. Required for ``"fixed"`` and ``"auto"``;
            use ``None`` only with uncapped ``"ideal"``.
        res_mode : {"fixed", "auto", "ideal"}, default="fixed"
            Fully-visible source sampling policy.
        partial_res : int or tuple of int, optional
            Partial-voxel resolution. If ``None``, use ``res``. An explicit
            value is required for ``res_mode="ideal"``.
        point_source_threshold : float, default=1/8
            Dimensionless sampling threshold, not an image-error tolerance.
        force_visibility : bool, default=False
            Recompute rather than reuse voxel visibility.
        verbose : int, default=0
            Verbosity level.

        Returns
        -------
        ProjectionWorkEstimate
            Per-Eye sample counts and heuristic resolution diagnostics.

        Notes
        -----
        Parameters match :meth:`set_projection_matrix` for source quadrature.
        ``fixed`` uses the mandatory ``res`` directly. ``auto`` selects a
        geometric ideal per fully-visible voxel and clips it axis-wise by
        ``res``. The explicitly requested ``ideal`` mode is uncapped and
        therefore requires ``res=None`` plus a fixed ``partial_res``.

        Fully-visible scheduled counts are exact for this sampling policy.
        Partial-voxel samples are reported before point visibility and inside
        masks, so those counts are upper bounds. Visibility is cached
        by :meth:`find_visible_voxels` and can be reused by the subsequent
        projection calculation. This method does not build or modify a
        projection matrix. ``ideal`` means uncapped heuristic work, not a
        numerical error guarantee.
        """
        full_resolution, partial_resolution, adaptive = (
            self._resolve_projection_resolutions(
                res, res_mode, partial_res,
            )
        )
        partial_cost = int(np.prod(partial_resolution))

        self.find_visible_voxels(force=force_visibility, verbose=verbose)
        rows = []
        for camera_key, camera in self.cameras.items():
            for eye_index in range(len(camera.eyes)):
                state = self.visible_voxels[camera_key][eye_index]
                full = np.flatnonzero(state == 2)
                partial = np.flatnonzero(state == 1)
                estimate = None
                if adaptive:
                    if full.size:
                        estimate = self.estimate_source_resolution(
                            camera_key, eye_index, full,
                            max_resolution=full_resolution,
                            point_source_threshold=point_source_threshold,
                            detector_grid="psf",
                        )
                        resolutions = estimate.resolution
                    else:
                        resolutions = np.empty((0, 3), dtype=np.int64)
                else:
                    resolutions = np.broadcast_to(
                        full_resolution, (full.size, 3),
                    )

                if full.size:
                    unique, counts = np.unique(
                        resolutions, axis=0, return_counts=True,
                    )
                    buckets = tuple(
                        (tuple(int(item) for item in resolution), int(count))
                        for resolution, count in zip(unique, counts)
                    )
                    full_samples = sum(
                        int(np.prod(resolution)) * count
                        for resolution, count in buckets
                    )
                else:
                    buckets = ()
                    full_samples = 0

                if estimate is not None:
                    valid_ideal = estimate.ideal_resolution[estimate.valid]
                    if valid_ideal.size:
                        percentiles = np.percentile(
                            valid_ideal, (50, 95, 100), axis=0,
                        )
                        ideal_p50, ideal_p95, ideal_max = (
                            tuple(float(item) for item in row)
                            for row in percentiles
                        )
                    else:
                        ideal_p50 = ideal_p95 = ideal_max = None
                    point_source_voxels = int(estimate.point_source.sum())
                    capped_axes = int(estimate.capped.sum())
                    invalid_voxels = int((~estimate.valid).sum())
                else:
                    ideal_p50 = ideal_p95 = ideal_max = None
                    point_source_voxels = capped_axes = invalid_voxels = 0

                rows.append(EyeProjectionWorkEstimate(
                    camera_key=camera_key,
                    eye_index=eye_index,
                    full_voxels=int(full.size),
                    partial_voxels=int(partial.size),
                    full_samples=int(full_samples),
                    partial_samples_upper_bound=int(partial.size) * partial_cost,
                    full_resolution_buckets=buckets,
                    partial_resolution=partial_resolution,
                    ideal_p50=ideal_p50,
                    ideal_p95=ideal_p95,
                    ideal_max=ideal_max,
                    point_source_voxels=point_source_voxels,
                    capped_axes=capped_axes,
                    invalid_voxels=invalid_voxels,
                ))
        return ProjectionWorkEstimate(eyes=tuple(rows))

    def _calc_voxel_image_for_eye(self, camera_idx: Hashable, eye_idx: int,
                                  res: int | tuple[int, int, int] | None,
                                  res_mode: Literal["fixed", "auto", "ideal"] = "fixed",
                                  n_jobs: int = -2,
                                  verbose: int = None, max_nnz: int = 100_000_000,
                                  partial_res: int | tuple[int, int, int] = None,
                                  max_working_memory: int = 1_000_000_000,
                                  chunk_strategy: str = "voxel",
                                  optical_bin_width_pixels=1.0,
                                  point_source_threshold: float = 1.0 / 8.0):
        """Construct voxel-to-image projection for a specific camera eye.

        Parameters
        ----------
        camera_idx : Hashable
            Key of the camera whose projection should be calculated.
        eye_idx : int
            Eye index within the selected camera.
        res : int, (int, int, int), or None
            Required fixed resolution or adaptive ceiling. Must be ``None``
            only when ``res_mode='ideal'``.
        res_mode : {"fixed", "auto", "ideal"}, optional
            Source-resolution policy for fully-visible voxels.
        n_jobs : int, optional
            Parallelism level for interpolation and visibility tasks. Negative
            values are interpreted relative to available CPU cores. Defaults to
            ``-2``.
        verbose : int, optional
            Verbosity level overriding :attr:`verbose`. ``None`` preserves the
            default.
        max_nnz : int, optional
            Upper bound on the allowed number of non-zero elements in the
            projection matrix before chunking is used to control memory usage.
        partial_res : int or (int, int, int), optional
            Sub-voxel resolution used only for partially visible voxels.
            ``None`` reuses ``res``.
        max_working_memory : int, optional
            Approximate byte budget for transient data held by in-flight
            projection chunks. Defaults to one billion bytes.
        chunk_strategy : {"voxel", "optical"}, optional
            ``"voxel"`` uses the established contiguous-voxel chunks.
            ``"optical"`` is an experimental, uncompressed validation path
            that reorders visible voxel centers by detector-pitch bins before
            expanding each work chunk into sub-voxel samples.
        optical_bin_width_pixels : float or (float, float), optional
            Optical bin width in detector-pixel pitches for the experimental
            optical strategy.
        point_source_threshold : float, optional
            Maximum projected circumsphere diameter in local finite-Eye PSF
            units for selecting res=1. Defaults to 1/8.

        Notes
        -----
        The resulting pixel-space sparse matrix is stored in
        ``self._projection[camera_idx][eye_idx]``. Subpixel images exist only
        as transient numerical-integration data.
        """
        n_jobs = os.cpu_count() + 1 + n_jobs if n_jobs < 0 else n_jobs
        n_jobs = max(1, n_jobs)
        if max_working_memory <= 0:
            raise ValueError("max_working_memory must be positive")
        if chunk_strategy not in {"voxel", "optical"}:
            raise ValueError("chunk_strategy must be 'voxel' or 'optical'")
        full_subvoxel_res, partial_subvoxel_res, adaptive = (
            self._resolve_projection_resolutions(
                res, res_mode, partial_res,
            )
        )
        if adaptive and chunk_strategy != "voxel":
            raise ValueError(
                "res_mode='auto' and 'ideal' require chunk_strategy='voxel'",
            )
        verbose = self.verbose if verbose is None else verbose
        timing_start = time.perf_counter()
        _camera = self.cameras[camera_idx]
        screen = _camera.screen
        N_vox = self.voxel.N
        # check visible voxels (0->invisible, 1->partially visible, 2->fully visible)
        self.find_visible_voxels(verbose=verbose)
        visibility_elapsed = time.perf_counter() - timing_start
        vis_flag = self.visible_voxels[camera_idx][eye_idx]  # (N_vox, )
        partial_voxels = np.flatnonzero(vis_flag == 1)
        full_voxels = np.flatnonzero(vis_flag == 2)
        full_res = sparse.coo_matrix((screen.N_pixel, N_vox))
        partial_res = sparse.coo_matrix((screen.N_pixel, N_vox))

        # No visible voxels
        if partial_voxels.size + full_voxels.size == 0:
            my_print("No visible voxels. Setting projection matrix to zero matrix.", show=verbose > 0)
            self._projection[camera_idx][eye_idx] = sparse.csr_matrix((screen.N_pixel, N_vox))
            return

        if chunk_strategy == "optical":
            optical_start = time.perf_counter()
            self._projection[camera_idx][eye_idx] = self._calc_voxel_image_for_eye_optical(
                camera_idx=camera_idx, eye_idx=eye_idx,
                full_voxels=full_voxels, partial_voxels=partial_voxels,
                full_subvoxel_res=full_subvoxel_res,
                partial_subvoxel_res=partial_subvoxel_res,
                max_nnz=max_nnz, max_working_memory=max_working_memory,
                optical_bin_width_pixels=optical_bin_width_pixels,
                verbose=verbose,
            )
            my_print(
                f"Optical projection matrix for camera {camera_idx!r}, "
                f"eye {eye_idx + 1}/{len(_camera.eyes)} calculated in "
                f"{time.perf_counter() - optical_start:.3f}s after visibility",
                show=verbose > 0,
            )
            return

        def _full_vox_proc(sv_gc, S):
            """Project sub-voxel centers for fully-visible voxels and integrate to voxel resolution.

            ``sv_gc`` are sub-voxel gravity centers and ``S`` is the matching
            sub-voxel interpolator/integration matrix from
            :func:`_sub_voxel_interpolator_matrix`; visibility checks are
            skipped since these voxels are already known to be fully visible.

            Returns
            -------
            tuple[np.ndarray, np.ndarray, np.ndarray]
                COO ``(data, row, col)`` triplet of the resulting
                ``(N_pixel, N_vox)`` sparse contribution.
            """
            I_subpixel = _camera.calc_image_vec(
                eye_idx, points=sv_gc, verbose=0, check_visibility=False,
            )
            res = (screen.transform_matrix @ (I_subpixel @ S)).tocoo()
            del I_subpixel, sv_gc
            return res.data, res.row, res.col

        def _partial_vox_proc(sv_gc, S, mask):
            """Project only the visible sub-voxel centers for partially-visible voxels.

            Same as :func:`_full_vox_proc`, but restricted to the sub-voxel
            samples where ``mask`` (per-sample visibility) is ``True``; ``S``
            is sliced accordingly before integration.

            Returns
            -------
            tuple[np.ndarray, np.ndarray, np.ndarray]
                COO ``(data, row, col)`` triplet, empty when ``mask`` has no
                ``True`` entries.
            """
            if not np.any(mask):
                return np.array([]), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            I_subpixel = _camera.calc_image_vec(
                eye_idx, points=sv_gc[mask], verbose=0, check_visibility=False,
            )
            S = S[mask, :]
            res = (screen.transform_matrix @ (I_subpixel @ S)).tocoo()
            del I_subpixel, sv_gc, mask
            return res.data, res.row, res.col

        def _sub_voxel_interpolator_matrix(voxel_indices, subvoxel_res, points=None):
            """Build direct center-to-sub-voxel interpolation for a chunk."""
            return self.voxel._build_source_quadrature_matrix(
                n=voxel_indices, res=subvoxel_res, points=points,
            )

        def _sparse_nbytes(matrix):
            matrix = matrix.tocsr()
            return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes

        def _triplet_nbytes(data, row, col):
            return data.nbytes + row.nbytes + col.nbytes

        result_buffer_limit = max(1, min(128 * 2 ** 20, max_working_memory // 4))

        def _estimate_batch_size(sample_count, sample_points, sample_image,
                                 sample_interpolator, sample_result, total_voxels):
            """Estimate a chunk size from transient bytes and legacy nnz cap."""
            transient_bytes = (
                sample_points.nbytes
                + _sparse_nbytes(sample_image)
                + _sparse_nbytes(sample_interpolator)
                + _sparse_nbytes(sample_result)
            )
            bytes_per_voxel = max(1.0, transient_bytes / sample_count)
            in_flight = 1 if n_jobs == 1 else 2 * n_jobs
            bytes_per_chunk = max(1, max_working_memory // in_flight)
            memory_batch = max(1, int(bytes_per_chunk // bytes_per_voxel))

            nnz_per_voxel = sample_image.nnz / sample_count
            if nnz_per_voxel == 0:
                nnz_per_voxel = screen.N_subpixel * 0.01
            nnz_batch = max(1, int(np.ceil(max_nnz / nnz_per_voxel) / n_jobs))
            # A scene that fits in one memory-sized chunk would otherwise use
            # only one worker even when n_jobs > 1. Keep up to two waves of
            # chunks available for load balancing without increasing the
            # existing in-flight memory bound.
            parallel_batch = (total_voxels if n_jobs == 1 else
                              max(1, int(np.ceil(total_voxels / (2 * n_jobs)))))
            batch = min(memory_batch, nnz_batch, parallel_batch, total_voxels)
            my_print(
                f"Estimated transient memory={bytes_per_voxel / 2 ** 20:.3f} MiB/voxel, "
                f"chunk budget={bytes_per_chunk / 2 ** 20:.1f} MiB",
                show=verbose > 1,
            )
            return max(1, int(batch))

        def _process_parallel_chunks(chunks, process_chunk, desc):
            """Process a bounded set of chunks and consolidate results as they finish.

            Drains ``futures`` as they complete (via
            ``concurrent.futures.as_completed``), periodically consolidating
            buffered COO triplets into ``res`` to bound peak memory usage,
            and shows a progress bar labeled ``desc`` when verbose.

            Parameters
            ----------
            chunks : iterable[slice]
                Chunk slices to process.
            process_chunk : callable
                Callable accepting a chunk and returning a COO triplet.
            desc : str
                Progress-bar label.

            Returns
            -------
            scipy.sparse.coo_matrix
                Sum of all future results, shape ``(screen.N_pixel, N_vox)``.
            """
            res = sparse.csr_matrix((screen.N_pixel, N_vox))
            data_buf, row_buf, col_buf = [], [], []
            buffer_nbytes = 0
            chunk_iter = iter(chunks)
            max_in_flight = 2 * n_jobs

            def _flush_buffer():
                nonlocal res, data_buf, row_buf, col_buf, buffer_nbytes
                if not data_buf:
                    return
                block = sparse.coo_matrix(
                    (np.concatenate(data_buf),
                     (np.concatenate(row_buf), np.concatenate(col_buf))),
                    shape=(screen.N_pixel, N_vox),
                ).tocsr()
                res += block
                data_buf, row_buf, col_buf = [], [], []
                buffer_nbytes = 0

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                pending = set()

                def _submit_next():
                    try:
                        chunk = next(chunk_iter)
                    except StopIteration:
                        return False
                    pending.add(executor.submit(process_chunk, chunk))
                    return True

                while len(pending) < max_in_flight and _submit_next():
                    pass

                with tqdm(desc=desc, total=len(chunks), disable=verbose <= 0) as pbar:
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            data, row, col = fut.result()
                            data_buf.append(data)
                            row_buf.append(row)
                            col_buf.append(col)
                            buffer_nbytes += _triplet_nbytes(data, row, col)
                            if buffer_nbytes >= result_buffer_limit:
                                _flush_buffer()
                            pbar.update()
                            _submit_next()

            _flush_buffer()

            return res.tocoo()

        def _process_full_group(group_voxels, group_resolution):
            """Project one bucket whose voxels share an axis-wise resolution."""
            group_result = sparse.coo_matrix((screen.N_pixel, N_vox))
            sample_n = np.random.choice(
                group_voxels, size=min(group_voxels.size, 20), replace=False,
            )
            sample_gc = self.voxel.get_sub_voxel_centers(
                n=sample_n, res=group_resolution,
            )
            sample_I = _camera.calc_image_vec(
                eye_idx, points=sample_gc, verbose=0, check_visibility=False,
            )
            sample_S = _sub_voxel_interpolator_matrix(
                sample_n, group_resolution, points=sample_gc,
            )
            sample_result = screen.transform_matrix @ (sample_I @ sample_S)
            batch_size = _estimate_batch_size(
                sample_n.size, sample_gc, sample_I, sample_S, sample_result,
                group_voxels.size,
            )
            del sample_I, sample_S, sample_result
            chunks = [
                slice(start, min(start + batch_size, group_voxels.size))
                for start in range(0, group_voxels.size, batch_size)
            ]
            description = f"Processing full voxels res={tuple(group_resolution)}"
            my_print(
                f"{description} in {len(chunks)} chunks "
                f"(n_step={batch_size}, full_size={group_voxels.size})",
                show=verbose > 0,
            )

            if n_jobs == 1:
                data_buf, row_buf, col_buf = [], [], []
                buffer_nbytes = 0
                for chunk in my_tqdm(chunks, desc=description,
                                     disable=verbose <= 0):
                    voxel_indices = group_voxels[chunk]
                    sv_gc = self.voxel.get_sub_voxel_centers(
                        n=voxel_indices, res=group_resolution,
                    )
                    S = _sub_voxel_interpolator_matrix(
                        voxel_indices, group_resolution, points=sv_gc,
                    )
                    data, row, col = _full_vox_proc(sv_gc, S)
                    data_buf.append(data)
                    row_buf.append(row)
                    col_buf.append(col)
                    buffer_nbytes += _triplet_nbytes(data, row, col)
                    if buffer_nbytes >= result_buffer_limit:
                        group_result = group_result + sparse.coo_matrix(
                            (np.concatenate(data_buf),
                             (np.concatenate(row_buf), np.concatenate(col_buf))),
                            shape=(screen.N_pixel, N_vox),
                        )
                        data_buf, row_buf, col_buf = [], [], []
                        buffer_nbytes = 0
                if data_buf:
                    group_result = group_result + sparse.coo_matrix(
                        (np.concatenate(data_buf),
                         (np.concatenate(row_buf), np.concatenate(col_buf))),
                        shape=(screen.N_pixel, N_vox),
                    )
            else:
                def _process_full_chunk(chunk):
                    voxel_indices = group_voxels[chunk]
                    sv_gc = self.voxel.get_sub_voxel_centers(
                        n=voxel_indices, res=group_resolution,
                    )
                    S = _sub_voxel_interpolator_matrix(
                        voxel_indices, group_resolution, points=sv_gc,
                    )
                    return _full_vox_proc(sv_gc, S)

                group_result = _process_parallel_chunks(
                    chunks, _process_full_chunk, desc=description,
                )
            return group_result

        if full_voxels.size == 0:
            my_print("Skipping full voxels processing.", show=verbose > 0)
        else:
            full_start = time.perf_counter()
            if adaptive:
                estimate = self.estimate_source_resolution(
                    camera_idx, eye_idx, full_voxels,
                    max_resolution=full_subvoxel_res,
                    point_source_threshold=point_source_threshold,
                    detector_grid="psf",
                )
                group_resolutions, inverse = np.unique(
                    estimate.resolution, axis=0, return_inverse=True,
                )
                full_groups = [
                    (full_voxels[inverse == group_index], tuple(resolution))
                    for group_index, resolution in enumerate(group_resolutions)
                ]
                my_print(
                    f"Adaptive full-voxel resolution selected "
                    f"{len(full_groups)} buckets; point-source voxels="
                    f"{estimate.point_source.sum()}/{estimate.point_source.size}; "
                    f"capped axes={estimate.capped.sum()}",
                    show=verbose > 0,
                )
            else:
                full_groups = [(full_voxels, full_subvoxel_res)]

            for group_voxels, group_resolution in full_groups:
                full_res = full_res + _process_full_group(
                    group_voxels, group_resolution,
                )

            my_print("Full voxels processed.", show=verbose > 0)
        full_elapsed = 0.0 if full_voxels.size == 0 else time.perf_counter() - full_start

        if partial_voxels.size == 0:
            my_print("Skipping partial voxels processing.", show=verbose > 0)
        else:
            partial_start = time.perf_counter()
            # random sample voxels to estimate n_step
            sample_mask = None
            for _ in range(10):  # avoid all non-visible samples without risking an infinite loop
                sample_n = np.random.choice(partial_voxels, size=min(partial_voxels.size, 20), replace=False)
                sample_gc = self.voxel.get_sub_voxel_centers(n=sample_n, res=partial_subvoxel_res)
                sample_mask = self.find_visible_points(sample_gc, camera_idx=camera_idx,
                                                       eye_idx=eye_idx, verbose=0).squeeze()
                sample_mask = sample_mask & self._inside_points(sample_gc)
                if np.any(sample_mask):
                    break

            if sample_mask is not None and np.any(sample_mask):
                sample_I = _camera.calc_image_vec(eye_idx, points=sample_gc[sample_mask], verbose=0,
                                                  check_visibility=False)  # (N_subpixel, num_visible)
                sample_S = _sub_voxel_interpolator_matrix(sample_n, partial_subvoxel_res,
                                                          points=sample_gc)
                sample_S = sample_S[sample_mask, :]
            else:
                sample_I = sparse.csr_matrix((screen.N_subpixel, 0))
                sample_S = sparse.csr_matrix((0, N_vox))
            sample_result = screen.transform_matrix @ (sample_I @ sample_S)
            batch_size = _estimate_batch_size(
                sample_n.size, sample_gc, sample_I, sample_S, sample_result,
                partial_voxels.size,
            )
            del sample_I, sample_S, sample_result
            _chunks = [slice(_i, min(_i + batch_size, partial_voxels.size)) for _i in
                       range(0, partial_voxels.size, batch_size)]
            my_print(f"Processing partial voxels in {len(_chunks)} chunks "
                     f"(n_step={batch_size}, partial_size={partial_voxels.size})",
                     show=verbose > 0)

            if n_jobs == 1:
                # initialize result matrix for partial voxels
                data_buf, row_buf, col_buf = [], [], []
                buffer_nbytes = 0
                for i, _slice in enumerate(my_tqdm(_chunks, desc="Processing partial voxels", disable=verbose <= 0)):
                    sv_gc = self.voxel.get_sub_voxel_centers(n=partial_voxels[_slice], res=partial_subvoxel_res)
                    mask = self.find_visible_points(sv_gc, camera_idx=camera_idx,
                                                    eye_idx=eye_idx, verbose=0).squeeze()
                    mask = mask & self._inside_points(sv_gc)
                    S = _sub_voxel_interpolator_matrix(partial_voxels[_slice], partial_subvoxel_res,
                                                       points=sv_gc)
                    data, row, col = _partial_vox_proc(sv_gc, S, mask)
                    data_buf.append(data)
                    row_buf.append(row)
                    col_buf.append(col)
                    buffer_nbytes += _triplet_nbytes(data, row, col)

                    if buffer_nbytes >= result_buffer_limit:
                        sum_of_buf = sparse.coo_matrix((np.concatenate(data_buf),
                                                        (np.concatenate(row_buf), np.concatenate(col_buf))),
                                                       shape=(screen.N_pixel, N_vox))
                        partial_res = partial_res + sum_of_buf
                        # clear buffer
                        data_buf, row_buf, col_buf = [], [], []
                        buffer_nbytes = 0
                # summarize remaining buffer
                if data_buf:
                    sum_of_buf = sparse.coo_matrix((np.concatenate(data_buf),
                                                    (np.concatenate(row_buf), np.concatenate(col_buf))),
                                                   shape=(screen.N_pixel, N_vox))
                    partial_res = partial_res + sum_of_buf
                    del data_buf, row_buf, col_buf
            else:
                def _process_partial_chunk(_slice):
                    voxel_indices = partial_voxels[_slice]
                    sv_gc = self.voxel.get_sub_voxel_centers(n=voxel_indices, res=partial_subvoxel_res)
                    mask = self.find_visible_points(sv_gc, camera_idx=camera_idx,
                                                    eye_idx=eye_idx, verbose=0).squeeze()
                    mask = mask & self._inside_points(sv_gc)
                    S = _sub_voxel_interpolator_matrix(voxel_indices, partial_subvoxel_res,
                                                       points=sv_gc)
                    return _partial_vox_proc(sv_gc, S, mask)

                partial_res = _process_parallel_chunks(_chunks, _process_partial_chunk,
                                                       desc="Processing partial voxels")

            my_print(f"Partial voxels processed.", show=verbose > 0)
        partial_elapsed = 0.0 if partial_voxels.size == 0 else time.perf_counter() - partial_start

        assembly_start = time.perf_counter()
        self._projection[camera_idx][eye_idx] = (full_res + partial_res).tocsr()
        del full_res, partial_res
        assembly_elapsed = time.perf_counter() - assembly_start
        total_elapsed = time.perf_counter() - timing_start
        my_print(
            "Projection timing: "
            f"visibility={visibility_elapsed:.3f}s, full={full_elapsed:.3f}s, "
            f"partial={partial_elapsed:.3f}s, assembly={assembly_elapsed:.3f}s, "
            f"total={total_elapsed:.3f}s",
            show=verbose > 1,
        )
        my_print(f"Projection matrix for camera {camera_idx!r}, "
                 f"eye {eye_idx + 1}/{len(_camera.eyes)} is calculated.", show=verbose > 0)
        return

    def trace_line(self, points, camera_idx: Hashable = 0, eye_idx: int = 0, coord_type: str = "XY"):
        """Project world-coordinate points onto a camera screen.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape ``(N, 3)`` containing world-coordinate points.
        camera_idx : Hashable, optional
            Key of the camera used for projection. Defaults to ``0``.
        eye_idx : int, optional
            Eye index within the camera providing the rays. Defaults to ``0``.
        coord_type : {'XY', 'UV'}, optional
            Coordinate frame for the return value. ``'XY'`` (default) produces
            camera-plane coordinates, whereas ``'UV'`` converts to screen pixel
            indices.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(N, 2)`` containing projected coordinates.

        Raises
        ------
        ValueError
            Raised when ``coord_type`` is unsupported.
        """
        points_in_camera = self.cameras[camera_idx].world2camera(points)
        rays = self.cameras[camera_idx].eyes[eye_idx].calc_rays(points_in_camera)
        if coord_type == "XY":
            return rays.XY
        elif coord_type == "UV":
            return self.cameras[camera_idx].screen.xy2uv(rays.XY)
        else:
            raise ValueError("coord_type should be 'XY' or 'UV'")

    def set_projection_matrix(
            self, res: int | tuple[int, int, int] | None,
            res_mode: Literal["fixed", "auto", "ideal"] = "fixed",
            verbose: int = 1, parallel: int = -1,
                              partial_res: int | tuple[int, int, int] = None,
                              force: bool = False,
                              max_working_memory: int = 1_000_000_000,
                              chunk_strategy: str = "voxel",
                              optical_bin_width_pixels=1.0,
                              point_source_threshold: float = 1.0 / 8.0):
        """Populate voxel-to-screen projection matrices for all cameras.

        Parameters
        ----------
        res : int, (int, int, int), or None
            Mandatory source resolution. ``fixed`` uses it directly and
            ``auto`` treats it as an axis-wise ceiling. Pass ``None`` only
            with the explicitly uncapped ``res_mode='ideal'``.
        res_mode : {"fixed", "auto", "ideal"}, default="fixed"
            Fully-visible source-resolution policy.
        verbose : int, default=1
            Verbosity level controlling progress logging.
        parallel : int, default=-1
            Degree of parallelism forwarded to
            :meth:`_calc_voxel_image_for_eye`. Negative values are interpreted
            relative to available CPU cores.
        partial_res : int or (int, int, int), optional
            Sub-voxel resolution used only for partially visible voxels.
            ``None`` reuses ``res``.
        force : bool, optional
            When ``True`` (default ``False``) forces recalculation even if
            cached matrices already exist with matching numerical settings
            and shapes.
        max_working_memory : int, optional
            Approximate byte budget for transient in-flight chunk data.
            Defaults to one billion bytes.
        chunk_strategy : {"voxel", "optical"}, optional
            Projection work ordering.  The default preserves contiguous voxel
            chunks.  ``"optical"`` enables the experimental, uncompressed
            optical-bin validation path.
        optical_bin_width_pixels : float or (float, float), optional
            Optical-bin width in detector-pixel pitches when
            ``chunk_strategy="optical"``. Defaults to one pixel.
        point_source_threshold : float, optional
            Maximum projected circumsphere diameter in local finite-Eye PSF
            units for selecting res=1. Defaults to 1/8; it is not a numerical
            error tolerance.

        Notes
        -----
        Aggregated matrices are stored in :attr:`_P_matrix` keyed by camera
        index, while per-eye pixel-space matrices remain in
        :attr:`_projection`. Subpixels are transient integration samples.
        """
        cache_key = self._projection_cache_key(
            res, res_mode, partial_res, chunk_strategy,
            optical_bin_width_pixels, point_source_threshold,
        )
        my_print("Calculating projection matrix", show=verbose > 0)
        indices = list(self.cameras.keys())
        for _c in indices:
            flag = False
            for _e in range(len(self.cameras[_c].eyes)):
                my_print(f"Processing camera {_c!r}, eye {_e + 1}/{len(self.cameras[_c].eyes)}",
                         show=verbose > 0)
                if force or \
                        (self._projection_settings[_c][_e] != cache_key) or \
                        (self._projection[_c][_e] is None) or \
                        (self._projection[_c][_e].shape != (self.cameras[_c].screen.N_pixel, self.voxel.N)):
                    my_print(f"Calculating projection matrix for camera {_c!r}, "
                             f"eye {_e + 1}/{len(self.cameras[_c].eyes)}", show=verbose > 0)
                    self._calc_voxel_image_for_eye(camera_idx=_c, eye_idx=_e,
                                                   res=res, res_mode=res_mode,
                                                   n_jobs=parallel,
                                                   verbose=verbose, max_nnz=100_000_000,
                                                   partial_res=partial_res,
                                                   max_working_memory=max_working_memory,
                                                   chunk_strategy=chunk_strategy,
                                                   optical_bin_width_pixels=optical_bin_width_pixels,
                                                   point_source_threshold=point_source_threshold)
                    # Record settings only after successful construction so a
                    # failed calculation can never make a stale P look valid.
                    self._projection_settings[_c][_e] = cache_key
                    flag = True
                else:
                    my_print(f"Projection matrix for camera {_c!r}, "
                             f"eye {_e + 1}/{len(self.cameras[_c].eyes)} is already calculated.", show=verbose > 0)
            if flag or (self._P_matrix[_c] is None) or \
                    (self._P_matrix[_c].shape != (self.cameras[_c].screen.N_pixel, self.voxel.N)):
                # at least one eye is recalculated
                # Per-eye matrices already share physical pixel coordinates.
                self._P_matrix[_c] = sum_eye_projections(
                    self._projection[_c],
                    (self.cameras[_c].screen.N_pixel, self.voxel.N),
                )
                my_print(f"Projection matrix for camera {_c!r} is calculated.", show=verbose > 0)
            else:
                my_print(f"Projection matrix for camera {_c!r} is already calculated.",
                         show=verbose > 0)

        my_print("Projection matrices are set.", show=verbose > 0)

    def draw_camera_orientation(self, ax=None, show_fig: bool = False, x_lim=None, y_lim=None, z_lim=None,
                                 **kwargs):
        """Visualize cameras, voxel bounds, and optional walls in 3D.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object used for plotting. ``None`` creates a new 3D axis.
        show_fig : bool, optional
            When ``True`` (default ``False``), immediately displays the figure.
        x_lim : tuple[float, float], optional
            Explicit x-axis limits. ``None`` auto-scales based on scene bounds.
        y_lim : tuple[float, float], optional
            Explicit y-axis limits. ``None`` auto-scales based on scene bounds.
        z_lim : tuple[float, float], optional
            Explicit z-axis limits. ``None`` auto-scales based on scene bounds.
        **kwargs : Any
            Additional keyword arguments forwarded to :func:`stl_utils.show_stl`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the rendered world context.
        """
        if ax is None:
            ax = plt.subplot(projection="3d", proj_type="ortho")

        vx_lim, vy_lim, vz_lim = self.voxel.ranges
        wx_lim, wy_lim, wz_lim = self.voxel.ranges if self.wall_ranges is None else self.wall_ranges

        x_lim = (min(vx_lim[0], wx_lim[0],
                     *[camera.camera_position[0] for camera in self.cameras.values()]) * 1.1,
                 max(vx_lim[1], wx_lim[1],
                     *[camera.camera_position[0] for camera in self.cameras.values()]) * 1.1) if x_lim is None else x_lim
        y_lim = (min(vy_lim[0], wy_lim[0],
                     *[camera.camera_position[1] for camera in self.cameras.values()]) * 1.1,
                 max(vy_lim[1], wy_lim[1],
                     *[camera.camera_position[1] for camera in self.cameras.values()]) * 1.1) if y_lim is None else y_lim
        z_lim = (min(vz_lim[0], wz_lim[0],
                     *[camera.camera_position[2] for camera in self.cameras.values()]) * 1.1,
                 max(vz_lim[1], wz_lim[1],
                     *[camera.camera_position[2] for camera in self.cameras.values()]) * 1.1) if z_lim is None else z_lim

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)

        ax.set_box_aspect((np.ptp(ax.get_xlim()),
                           np.ptp(ax.get_ylim()),
                           np.ptp(ax.get_zlim())))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        for camera in self.cameras.values():
            camera.draw_camera_orientation(ax=ax)

        for wall in self.walls:
            stl_utils.show_stl(wall, ax=ax, show_fig=False, **kwargs)

        if show_fig:
            ax.figure.show()

        return ax
