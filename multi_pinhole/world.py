"""World container that ties voxels, cameras, and walls into a scene.

:class:`World` is the top-level orchestration object: it owns a
:class:`~multi_pinhole.voxel.Voxel` grid, one or more
:class:`~multi_pinhole.core.Camera` instances, and optional STL ``walls``
that occlude visibility. It computes and caches per-camera, per-eye
visibility masks (:meth:`World.find_visible_voxels`) and sparse
voxel-to-screen projection matrices (:meth:`World.set_projection_matrix`),
and provides persistence (:meth:`World.save_world` /
:meth:`World.load_world`) and 3D visualization
(:meth:`World.draw_camera_orientation`) helpers.

See ``docs/world.md`` for a narrative overview of the module.
"""

import gc
import os
import time
from collections.abc import Hashable, Mapping
from concurrent.futures import FIRST_COMPLETED, wait
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import replace
from types import MappingProxyType
from typing import Tuple, List

import dill
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from stl import mesh

from .core import Aperture, Camera, Eye, Screen
from .projection import (
    HybridProjectionOperator,
    build_projection_block,
    combine_projection_operators,
    make_optical_binning,
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

        # Voxel
        self._voxel = voxel if voxel is not None else Voxel()
        self._voxel.set_world(self)

        # Cameras
        self._cameras = self._normalize_cameras(cameras)
        for _ in self._cameras.values():
            _.set_world(self)

        # Walls
        self._walls = []
        self._wall_ranges = None
        self.walls = walls

        # initialize other attributes
        self._inside_function = None
        self._inside_kwargs = {}
        self._inside_vertices = None
        self._visible_vertices = {i: None for i in self._cameras.keys()}
        self._visible_voxels = {i: None for i in self._cameras.keys()}
        self._projection = {i: [None] * len(self._cameras[i].eyes) for i in self._cameras.keys()}
        self._P_matrix = {i: None for i in self._cameras.keys()}
        self._projection_cache_schema_version = PROJECTION_CACHE_SCHEMA_VERSION
        self.verbose = verbose

        # set inside vertices if inside_func is provided
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

    # MARK: Save and Load
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

    # MARK: Properties
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

    # TODO: この辺をうまく活用してinside_verticesを制御する
    # future use
    # @property
    # def coordinate_type(self):
    #     return self._coordinate_type

    # @property
    # def coordinate_parameters(self):
    #     return self._coordinate_parameters

    # @property
    # def normalized_coordinates(self):
    #     return (lambda x: x) if self._normalized_coordinates is None else self._normalized_coordinates

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
        """dict[int, list[sparse.csr_matrix]]: Subpixel projection matrices per eye."""
        return self._projection

    @property
    def projection_cache_schema_version(self) -> int:
        """Version governing compatibility of serialized projection caches."""
        return self._projection_cache_schema_version

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
        self._P_matrix = {i: None for i in self._cameras.keys()}

    def _ensure_projection_cache_schema(self) -> None:
        """Invalidate absent or incompatible serialized projection caches."""
        cached_version = getattr(self, "_projection_cache_schema_version", None)
        if cached_version != PROJECTION_CACHE_SCHEMA_VERSION:
            self._invalidate_projection_cache()
            self._projection_cache_schema_version = PROJECTION_CACHE_SCHEMA_VERSION

    # MARK: Setters
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
        P_matrix = {}
        for camera_key, camera in cameras.items():
            if id(camera) in old_indices:
                ind = old_indices[id(camera)]
                # keep the previous visible voxels, projection, and P matrix
                visible_vertices[camera_key] = self._visible_vertices[ind]
                visible_voxels[camera_key] = self._visible_voxels[ind]
                projection[camera_key] = self._projection[ind]
                P_matrix[camera_key] = self._P_matrix[ind]
            else:
                # if the camera is not in the original camera list,
                # set the visible voxels, projection, and P matrix to None
                visible_vertices[camera_key] = None
                visible_voxels[camera_key] = None
                projection[camera_key] = [None] * len(camera.eyes)
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
        self._P_matrix = P_matrix
        # self._P_matrix = None
        if all([_ is None for _ in visible_voxels.values()]):
            print("Notice: All cameras are updated.")
        elif None in visible_voxels.values():
            # Notice: *th, *th, and *th cameras are reused.
            reused_index = [repr(key) for key in visible_voxels if visible_voxels[key] is not None]
            xth = ", ".join(reused_index[:-1]) + " and " + reused_index[-1] if \
                len(reused_index) > 1 else f"{reused_index[0]}"
            print(f"Notice: {xth} camera{'s are' if len(reused_index) > 1 else ' is'} reused.")
        else:
            # All cameras are reused.
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
        self._P_matrix.pop(camera_key)
        removed_camera.unset_world(self)
        # TODO: Add an explicit reset_camera_keys() helper if compact integer
        # keys are ever needed. Automatic renumbering is intentionally avoided.
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
            self._projection = {i: [None] * len(self._cameras[i].eyes) for i in self._cameras.keys()}
            self._P_matrix = {i: None for i in self._cameras.keys()}

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

    # MARK: Methods
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

        visible = np.zeros((len(eye_idx), points.shape[0]), dtype=bool)  # (N_eye, N_points)

        for i, _e in enumerate(eye_idx):
            _eye = _camera.eyes[_e]
            # check if the voxel is behind the camera (N_points, )
            visible[i] = camera_points[:, 2] >= _eye.position[-1]
            # check if the voxel is in front of the camera (N_points, )
            my_print(f"checking visible points for eye {_e + 1}/{len(_camera.eyes)}",
                     show=verbose > 0)
            my_print("-" * 15, show=verbose > 0)
            # get conditions for each aperture and wall
            my_print(f"--- checking for apertures ---", show=verbose > 0)
            aperture_visible = []
            for a, aperture in enumerate(_camera.apertures):
                if aperture.stl_model is None:
                    aperture.set_model()
                aperture_visible.append(stl_utils.check_visible(mesh_obj=aperture.stl_model,
                                                                start=_eye.position,
                                                                grid_points=camera_points,
                                                                verbose=verbose,
                                                                behind_start_included=True))  # (N_points, )
                my_print(f"{a + 1}/{len(_camera.apertures)} done", show=verbose > 0)
                my_print("-" * 15, show=verbose > 0)
            if aperture_visible:
                # Apertures are blocking surfaces, matching Camera.calc_image_vec():
                # intersecting any aperture makes the ray invisible.
                visible[i] &= np.all(aperture_visible, axis=0)

            my_print("--- checking for walls ---", show=verbose > 0)
            for w, wall_in_camera in enumerate(walls_in_camera):
                visible[i] *= stl_utils.check_visible(mesh_obj=wall_in_camera,
                                                      start=_eye.position,
                                                      grid_points=camera_points,
                                                      verbose=verbose)  # (N_points, )
                my_print(f"{w + 1}/{len(walls_in_camera)} done", show=verbose > 0)
                my_print("-" * 15, show=verbose > 0)
        return visible  # (N_eye, N_points)

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
            visible_vertices = np.zeros((len(camera.eyes), self.voxel.N_grid), dtype=bool)
            visible_vertices[:, self.inside_vertices] = True
            if np.sum(self.inside_vertices) == 0:
                my_print("No inside vertices. Skip calculating visible vertices.", show=verbose > 0)
            else:
                visible_vertices[:, self.inside_vertices] \
                    = self.find_visible_points(self.voxel.grid[self.inside_vertices],
                                               camera_idx=c_,
                                               verbose=verbose)  # (N_eye, N_inside_points)
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
            conditions_any = np.any(self._visible_vertices[c_][:, self.voxel.vertices_indices], axis=-1).astype(int)
            conditions_all = np.all(self._visible_vertices[c_][:, self.voxel.vertices_indices], axis=-1).astype(int)
            visible_voxels[c_] = conditions_any + conditions_all
            my_print(f"Visible voxels for camera {c_!r} is calculated.", show=verbose > 0)
        self._visible_voxels = visible_voxels
        my_print("Finding visible voxels is done.", show=verbose > 0)

    def _calc_voxel_image_for_eye_optical(
            self, camera_idx: Hashable, eye_idx: int,
            full_voxels: np.ndarray, partial_voxels: np.ndarray,
            full_subvoxel_res, partial_subvoxel_res,
            max_nnz: int, max_working_memory: int,
            optical_bin_width_pixels, verbose: int,
            projection_representation: str = "sparse",
            psf_tolerance: float = 0.0,
            psf_metric: str = "relative_l2",
            psf_grouping: str = "recursive",
            max_factorized_byte_fraction: float | None = 0.8):
        """Build sparse or hybrid projection using visible-voxel optical bins.

        This experimental path bins visible voxel centers, packs complete bins
        into work chunks, and only then expands those voxels into sub-voxel
        samples. Subpixels are transient quadrature samples: their images are
        binned to detector pixels before either sparse assembly or hybrid PSF
        grouping. Hybrid mode groups pixel-space PSFs independently inside
        each optical scope and stores either a direct block or ``Q @ A``.
        """
        camera = self.cameras[camera_idx]
        screen = camera.screen
        n_vox = self.voxel.N
        index_start = time.perf_counter()
        full_resolution = tuple(int(r) for r in np.broadcast_to(full_subvoxel_res, 3))
        partial_resolution = tuple(int(r) for r in np.broadcast_to(partial_subvoxel_res, 3))
        full_cost = int(np.prod(full_resolution))
        partial_cost = int(np.prod(partial_resolution))
        visible_voxels = np.concatenate((full_voxels, partial_voxels))
        sample_costs = np.concatenate((
            np.full(full_voxels.size, full_cost, dtype=np.int64),
            np.full(partial_voxels.size, partial_cost, dtype=np.int64),
        ))
        voxel_centers = self.voxel.get_gravity_center(visible_voxels)
        if visible_voxels.size == 0:
            return sparse.csr_matrix((screen.N_pixel, n_vox))

        def _sparse_nbytes(matrix):
            matrix = matrix.tocsr()
            return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes

        if full_voxels.size:
            sample_voxels = full_voxels[:min(full_voxels.size, 20)]
            sample_resolution = full_resolution
        else:
            sample_voxels = partial_voxels[:min(partial_voxels.size, 20)]
            sample_resolution = partial_resolution
        sample_points = self.voxel.get_sub_voxel_centers(
            sample_voxels, res=sample_resolution,
        )
        sample_S = self.voxel.sub_voxel_interpolator_from_centers(
            sample_voxels, res=sample_resolution, points=sample_points,
        )
        sample_I_subpixel = camera.calc_image_vec(
            eye_idx, points=sample_points, verbose=0, check_visibility=False,
        )
        sample_I = screen.transform_matrix @ sample_I_subpixel
        sample_result = sample_I @ sample_S
        sample_count = sample_points.shape[0]
        transient_bytes = (
            sample_points.nbytes + _sparse_nbytes(sample_I) + _sparse_nbytes(sample_S)
            + _sparse_nbytes(sample_result)
        )
        bytes_per_sample = max(1.0, transient_bytes / sample_count)
        memory_samples = max(1, int((max_working_memory // 2) // bytes_per_sample))
        nnz_per_sample = sample_I.nnz / sample_count
        if nnz_per_sample == 0.0:
            nnz_per_sample = screen.N_pixel * 0.01
        nnz_samples = max(1, int(np.ceil(max_nnz / nnz_per_sample)))
        max_samples = min(memory_samples, nnz_samples, int(sample_costs.sum()))
        del sample_I_subpixel, sample_I, sample_S, sample_result

        binning = make_optical_binning(
            camera, eye_idx, voxel_centers,
            bin_width_pixels=optical_bin_width_pixels,
            max_scope_samples=max_samples,
            sample_costs=sample_costs,
        )
        work_chunks = binning.work_chunks(max_samples=max_samples)
        index_elapsed = time.perf_counter() - index_start
        my_print(
            f"Processing {visible_voxels.size} visible voxels "
            f"({sample_costs.sum()} sub-voxel samples before partial masking) in "
            f"{binning.n_scopes} optical scopes and {len(work_chunks)} work chunks "
            f"(max_samples={max_samples})",
            show=verbose > 0,
        )

        if projection_representation == "hybrid":
            work_offsets = binning.work_offsets(max_samples=max_samples)
            scope_offsets = binning.scope_offsets
            operators = []
            compression_stats = []
            exact_direct_blocks = []
            enforce_final_byte_fraction = (
                psf_tolerance > 0.0
                and max_factorized_byte_fraction is not None
                and max_factorized_byte_fraction > 0.0
            )
            scope_number = 0

            def _expand_scope(packed_visible_indices):
                """Expand one scope without running point visibility yet.

                ``packed_visible_indices`` index the concatenated
                ``visible_voxels = [full_voxels, partial_voxels]`` array.  The
                full/partial point arrays and their S rows stay paired, while
                S columns retain the original global voxel numbering. Partial
                points are visibility-tested in one work-chunk batch below.
                """
                owners = visible_voxels[packed_visible_indices]
                is_full = packed_visible_indices < full_voxels.size
                full_points = np.empty((0, 3), dtype=float)
                partial_points = np.empty((0, 3), dtype=float)
                full_S = sparse.csr_matrix((0, n_vox))
                partial_S = sparse.csr_matrix((0, n_vox))

                full_owners = owners[is_full]
                if full_owners.size:
                    full_points = self.voxel.get_sub_voxel_centers(
                        full_owners, res=full_resolution,
                    )
                    full_S = self.voxel.sub_voxel_interpolator_from_centers(
                        full_owners, res=full_resolution, points=full_points,
                    )

                partial_owners = owners[~is_full]
                if partial_owners.size:
                    partial_points = self.voxel.get_sub_voxel_centers(
                        partial_owners, res=partial_resolution,
                    )
                    partial_S = self.voxel.sub_voxel_interpolator_from_centers(
                        partial_owners, res=partial_resolution,
                        points=partial_points,
                    )
                return full_points, full_S, partial_points, partial_S

            for work_start, work_stop in zip(work_offsets[:-1], work_offsets[1:]):
                scope_points = []
                scope_interpolators = []
                expanded_scope_offsets = [0]
                expanded_scopes = []

                # work offsets always coincide with scope boundaries.  Keep a
                # monotonic scope cursor so packed positions are never confused
                # with values stored in binning.order.
                while scope_number < binning.n_scopes and \
                        scope_offsets[scope_number] < work_stop:
                    scope_start = int(scope_offsets[scope_number])
                    scope_stop = int(scope_offsets[scope_number + 1])
                    if scope_start < work_start or scope_stop > work_stop:
                        raise RuntimeError("optical work chunk split a compression scope")
                    packed_visible_indices = binning.order[scope_start:scope_stop]
                    expanded_scopes.append(_expand_scope(packed_visible_indices))
                    scope_number += 1

                # Visibility of partial subvoxels is expensive for an STL
                # wall. Evaluate all partial points from this work chunk in one
                # call, then split the boolean result back by scope-local row
                # counts. This preserves scope boundaries without thousands of
                # tiny visibility calls.
                partial_lengths = [scope[2].shape[0] for scope in expanded_scopes]
                if sum(partial_lengths):
                    partial_points_work = np.concatenate(
                        [scope[2] for scope in expanded_scopes if scope[2].size],
                        axis=0,
                    )
                    partial_visible_work = self.find_visible_points(
                        partial_points_work, camera_idx=camera_idx,
                        eye_idx=eye_idx, verbose=0,
                    ).reshape(-1)
                    partial_visible_work &= self._inside_points(partial_points_work)
                else:
                    partial_visible_work = np.empty(0, dtype=bool)

                partial_start = 0
                for (full_points, full_S, partial_points, partial_S), partial_length \
                        in zip(expanded_scopes, partial_lengths):
                    point_parts = []
                    interpolator_parts = []
                    if full_points.size:
                        point_parts.append(full_points)
                        interpolator_parts.append(full_S)
                    if partial_length:
                        partial_stop = partial_start + partial_length
                        visible_mask = partial_visible_work[partial_start:partial_stop]
                        partial_start = partial_stop
                        if np.any(visible_mask):
                            point_parts.append(partial_points[visible_mask])
                            interpolator_parts.append(partial_S[visible_mask])
                    if not point_parts:
                        continue
                    points_part = np.concatenate(point_parts, axis=0)
                    S_part = sparse.vstack(interpolator_parts, format="csr")
                    scope_points.append(points_part)
                    scope_interpolators.append(S_part)
                    expanded_scope_offsets.append(
                        expanded_scope_offsets[-1] + points_part.shape[0]
                    )

                if not scope_points:
                    continue
                points_work = np.concatenate(scope_points, axis=0)
                S_work = sparse.vstack(scope_interpolators, format="csr")
                I_subpixel_work = camera.calc_image_vec(
                    eye_idx, points=points_work, verbose=0,
                    check_visibility=False,
                )
                # Subpixels are integration samples, not persistent detector
                # coordinates. Grouping after this exact linear binning keeps
                # area-integration accuracy while reducing the PSF row count.
                I_work = screen.transform_matrix @ I_subpixel_work
                del I_subpixel_work
                # Keep one exact work-chunk block transiently. Scope-local CSR
                # payloads are not additive because final summation shares
                # row pointers and can merge duplicate entries. This block is
                # used for the final, representation-level byte safety gate.
                if enforce_final_byte_fraction:
                    exact_direct_blocks.append((I_work @ S_work).tocsr())
                work_scope_operators = []

                # I and S use the same expanded subvoxel row/column order.
                # Each slice below is one independent optical scope even when
                # several scopes shared the same expensive PSF calculation.
                for sample_start, sample_stop in zip(
                        expanded_scope_offsets[:-1], expanded_scope_offsets[1:]):
                    operator, stats = build_projection_block(
                        I_work[:, sample_start:sample_stop],
                        S_work[sample_start:sample_stop],
                        tolerance=psf_tolerance,
                        metric=psf_metric,
                        algorithm=psf_grouping,
                        max_factorized_byte_fraction=max_factorized_byte_fraction,
                        max_scope_dense_bytes=max(1, max_working_memory // 4),
                    )
                    work_scope_operators.append(operator)
                    compression_stats.append(stats)

                # Consolidate now so each completed work chunk retains only
                # one global detector-row pointer array. Keeping one CSR Q and
                # direct matrix per tiny optical scope would make indptr
                # overhead scale with the number of scopes.
                if work_scope_operators:
                    operators.append(
                        combine_projection_operators(work_scope_operators)
                    )

            if not operators:
                return HybridProjectionOperator.empty(
                    (screen.N_pixel, n_vox),
                )
            result = combine_projection_operators(operators)
            if enforce_final_byte_fraction:
                exact_direct = sum(
                    exact_direct_blocks,
                    sparse.csr_matrix((screen.N_pixel, n_vox)),
                ).tocsr()
                exact_direct.sum_duplicates()
                exact_direct.eliminate_zeros()
                exact_direct_nbytes = _sparse_nbytes(exact_direct)
            if enforce_final_byte_fraction and result.storage_nbytes > \
                    max_factorized_byte_fraction * exact_direct_nbytes:
                # A scope-local byte win need not survive final CSR
                # consolidation. Fall back globally so the requested storage
                # saving is enforced on the actual retained representation.
                compression_stats = [
                    replace(
                        stats,
                        used_factorization=False,
                        direct_nbytes=stats.candidate_direct_nbytes,
                        q_nbytes=0,
                        a_nbytes=0,
                        global_direct_fallback=True,
                    )
                    for stats in compression_stats
                ]
                result = HybridProjectionOperator(
                    exact_direct,
                    sparse.csr_matrix((screen.N_pixel, 0)),
                    sparse.csr_matrix((0, n_vox)),
                    compression_stats=compression_stats,
                )
            n_active = sum(stats.n_active_samples for stats in compression_stats)
            n_groups = sum(
                stats.n_groups for stats in compression_stats
                if stats.used_factorization
            )
            n_factorized_scopes = sum(
                stats.used_factorization for stats in compression_stats
            )
            my_print(
                f"Hybrid projection: {n_factorized_scopes}/"
                f"{len(compression_stats)} scopes factorized, "
                f"active_samples={n_active}, stored_groups={n_groups}, "
                f"storage={result.storage_nbytes / 2 ** 20:.2f} MiB",
                show=verbose > 0,
            )
            return result

        projection_start = time.perf_counter()
        result = sparse.csr_matrix((screen.N_pixel, n_vox))
        data_buf, row_buf, col_buf = [], [], []
        buffer_nbytes = 0
        result_buffer_limit = max(1, min(128 * 2 ** 20, max_working_memory // 4))

        def _flush():
            nonlocal result, data_buf, row_buf, col_buf, buffer_nbytes
            if not data_buf:
                return
            block = sparse.coo_matrix(
                (np.concatenate(data_buf),
                 (np.concatenate(row_buf), np.concatenate(col_buf))),
                shape=result.shape,
            ).tocsr()
            result += block
            data_buf, row_buf, col_buf = [], [], []
            buffer_nbytes = 0

        def _project_voxels(owners, resolution, check_point_visibility):
            if owners.size == 0:
                return None
            points = self.voxel.get_sub_voxel_centers(owners, res=resolution)
            S = self.voxel.sub_voxel_interpolator_from_centers(
                owners, res=resolution, points=points,
            )
            if check_point_visibility:
                mask = self.find_visible_points(
                    points, camera_idx=camera_idx, eye_idx=eye_idx, verbose=0,
                ).reshape(-1)
                mask &= self._inside_points(points)
                if not np.any(mask):
                    return None
                points = points[mask]
                S = S[mask]
            I_subpixel = camera.calc_image_vec(
                eye_idx, points=points, verbose=0, check_visibility=False,
            )
            # In the uncompressed path, integrate subvoxel columns before
            # detector binning. This keeps the transient multiplication small
            # while still returning only the persistent pixel-space matrix.
            return (screen.transform_matrix @ (I_subpixel @ S)).tocoo()

        for chunk in my_tqdm(work_chunks, desc="Processing optical work chunks",
                             disable=verbose <= 0):
            ordered_voxels = visible_voxels[chunk]
            is_full = chunk < full_voxels.size
            chunk_results = (
                _project_voxels(ordered_voxels[is_full], full_resolution, False),
                _project_voxels(ordered_voxels[~is_full], partial_resolution, True),
            )
            for P_chunk in chunk_results:
                if P_chunk is None:
                    continue
                data_buf.append(P_chunk.data)
                row_buf.append(P_chunk.row)
                col_buf.append(P_chunk.col)
                buffer_nbytes += (P_chunk.data.nbytes + P_chunk.row.nbytes
                                  + P_chunk.col.nbytes)
            if buffer_nbytes >= result_buffer_limit:
                _flush()
        _flush()
        projection_elapsed = time.perf_counter() - projection_start
        my_print(
            f"Optical sparse-P timing: index={index_elapsed:.3f}s, "
            f"projection={projection_elapsed:.3f}s",
            show=verbose > 1,
        )
        return result

    def _calc_voxel_image_for_eye(self, camera_idx: Hashable, eye_idx: int, res: int, n_jobs: int = -2,
                                  verbose: int = None, max_nnz: int = 100_000_000,
                                  partial_res: int | tuple[int, int, int] = None,
                                  max_working_memory: int = 1_000_000_000,
                                  chunk_strategy: str = "voxel",
                                  optical_bin_width_pixels=1.0,
                                  projection_representation: str = "sparse",
                                  psf_tolerance: float = 0.0,
                                  psf_metric: str = "relative_l2",
                                  psf_grouping: str = "recursive",
                                  max_factorized_byte_fraction: float | None = 0.8):
        """Construct voxel-to-image projection for a specific camera eye.

        Parameters
        ----------
        camera_idx : Hashable
            Key of the camera whose projection should be calculated.
        eye_idx : int
            Eye index within the selected camera.
        res : int
            Sub-voxel resolution used when generating interpolation matrices.
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
        projection_representation : {"sparse", "hybrid"}, optional
            ``"hybrid"`` stores direct scopes together with factorized
            ``Q @ A`` scopes. Hybrid representation requires optical chunks.

        Notes
        -----
        The resulting pixel-space matrix/operator is stored in
        ``self._projection[camera_idx][eye_idx]``. Subpixel images exist only
        as transient numerical-integration data.
        """
        n_jobs = os.cpu_count() + 1 + n_jobs if n_jobs < 0 else n_jobs
        n_jobs = max(1, n_jobs)
        if max_working_memory <= 0:
            raise ValueError("max_working_memory must be positive")
        if chunk_strategy not in {"voxel", "optical"}:
            raise ValueError("chunk_strategy must be 'voxel' or 'optical'")
        if projection_representation not in {"sparse", "hybrid"}:
            raise ValueError("projection_representation must be 'sparse' or 'hybrid'")
        if projection_representation == "hybrid" and chunk_strategy != "optical":
            raise ValueError("hybrid projection requires chunk_strategy='optical'")
        if projection_representation == "hybrid":
            if not np.isfinite(psf_tolerance) or psf_tolerance < 0.0:
                raise ValueError("psf_tolerance must be finite and non-negative")
            if psf_metric not in {"relative_l2", "l1"}:
                raise ValueError("psf_metric must be 'relative_l2' or 'l1'")
            if psf_grouping not in {"recursive", "leader"}:
                raise ValueError("psf_grouping must be 'recursive' or 'leader'")
            if max_factorized_byte_fraction is not None and \
                    not 0.0 <= max_factorized_byte_fraction <= 1.0:
                raise ValueError(
                    "max_factorized_byte_fraction must be in [0, 1] or None"
                )
        verbose = self.verbose if verbose is None else verbose
        timing_start = time.perf_counter()
        _camera = self.cameras[camera_idx]
        screen = _camera.screen
        N_vox = self.voxel.N
        self.voxel.res = res
        full_subvoxel_res = self.voxel.res
        self.voxel.res = partial_res
        partial_subvoxel_res = self.voxel.res

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
            if projection_representation == "hybrid":
                self._projection[camera_idx][eye_idx] = HybridProjectionOperator.empty(
                    (screen.N_pixel, N_vox),
                )
            else:
                self._projection[camera_idx][eye_idx] = sparse.csr_matrix(
                    (screen.N_pixel, N_vox),
                )
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
                projection_representation=projection_representation,
                psf_tolerance=psf_tolerance,
                psf_metric=psf_metric,
                psf_grouping=psf_grouping,
                max_factorized_byte_fraction=max_factorized_byte_fraction,
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
                eye_idx, points=sv_gc[mask], verbose=0,
                check_visibility=False,
            )
            S = S[mask, :]
            res = (screen.transform_matrix @ (I_subpixel @ S)).tocoo()
            del I_subpixel, sv_gc, mask
            return res.data, res.row, res.col

        def _sub_voxel_interpolator_matrix(voxel_indices, subvoxel_res, points=None):
            """Build direct center-to-sub-voxel interpolation for a chunk."""
            return self.voxel.sub_voxel_interpolator_from_centers(
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
            batch = min(memory_batch, nnz_batch, total_voxels)
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

        if full_voxels.size == 0:
            my_print("Skipping full voxels processing.", show=verbose > 0)
        else:
            full_start = time.perf_counter()
            # random sample voxels to estimate n_step
            sample_n = np.random.choice(full_voxels, size=min(full_voxels.size, 20), replace=False)
            sample_gc = self.voxel.get_sub_voxel_centers(n=sample_n, res=full_subvoxel_res)
            sample_I = _camera.calc_image_vec(
                eye_idx, points=sample_gc, verbose=0, check_visibility=False,
            )
            sample_S = _sub_voxel_interpolator_matrix(sample_n, full_subvoxel_res,
                                                      points=sample_gc)
            sample_result = screen.transform_matrix @ (sample_I @ sample_S)
            batch_size = _estimate_batch_size(
                sample_n.size, sample_gc, sample_I, sample_S, sample_result,
                full_voxels.size,
            )
            del sample_I, sample_S, sample_result
            _chunks = [slice(_i, min(_i + batch_size, full_voxels.size)) for _i in
                       range(0, full_voxels.size, batch_size)]
            my_print(f"Processing full voxels in {len(_chunks)} chunks "
                     f"(n_step={batch_size}, full_size={full_voxels.size})",
                     show=verbose > 0)

            if n_jobs == 1:
                # initialize result matrix for full voxels
                data_buf, row_buf, col_buf = [], [], []
                buffer_nbytes = 0
                for i, _slice in enumerate(my_tqdm(_chunks, desc="Processing full voxels", disable=verbose <= 0)):
                    sv_gc = self.voxel.get_sub_voxel_centers(n=full_voxels[_slice], res=full_subvoxel_res)
                    S = _sub_voxel_interpolator_matrix(full_voxels[_slice], full_subvoxel_res,
                                                       points=sv_gc)
                    data, row, col = _full_vox_proc(sv_gc, S)
                    data_buf.append(data)
                    row_buf.append(row)
                    col_buf.append(col)
                    buffer_nbytes += _triplet_nbytes(data, row, col)

                    if buffer_nbytes >= result_buffer_limit:
                        sum_of_buf = sparse.coo_matrix((np.concatenate(data_buf),
                                                        (np.concatenate(row_buf), np.concatenate(col_buf))),
                                                       shape=(screen.N_pixel, N_vox))
                        full_res = full_res + sum_of_buf
                        # clear buffer
                        data_buf, row_buf, col_buf = [], [], []
                        buffer_nbytes = 0
                # summarize remaining buffer
                if data_buf:
                    sum_of_buf = sparse.coo_matrix((np.concatenate(data_buf),
                                                    (np.concatenate(row_buf), np.concatenate(col_buf))),
                                                   shape=(screen.N_pixel, N_vox))
                    full_res = full_res + sum_of_buf
                    del data_buf, row_buf, col_buf
            else:
                def _process_full_chunk(_slice):
                    voxel_indices = full_voxels[_slice]
                    sv_gc = self.voxel.get_sub_voxel_centers(n=voxel_indices, res=full_subvoxel_res)
                    S = _sub_voxel_interpolator_matrix(voxel_indices, full_subvoxel_res,
                                                       points=sv_gc)
                    return _full_vox_proc(sv_gc, S)

                full_res = _process_parallel_chunks(_chunks, _process_full_chunk,
                                                    desc="Processing full voxels")

            my_print(f"Full voxels processed.", show=verbose > 0)
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
                sample_I = _camera.calc_image_vec(
                    eye_idx, points=sample_gc[sample_mask], verbose=0,
                    check_visibility=False,
                )
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

    def set_projection_matrix(self, res: int = None, verbose: int = 1, parallel: int = -1,
                              partial_res: int | tuple[int, int, int] = None,
                              force: bool = False,
                              max_working_memory: int = 1_000_000_000,
                              chunk_strategy: str = "voxel",
                              optical_bin_width_pixels=1.0,
                              projection_representation: str = "sparse",
                              psf_tolerance: float = 0.0,
                              psf_metric: str = "relative_l2",
                              psf_grouping: str = "recursive",
                              max_factorized_byte_fraction: float | None = 0.8):
        """Populate voxel-to-screen projection matrices for all cameras.

        Parameters
        ----------
        res : int, optional
            Sub-voxel resolution used when recomputing projection matrices.
            ``None`` preserves the voxel default.
        verbose : int, optional
            Verbosity level controlling progress logging. Defaults to ``1``.
        parallel : int, optional
            Degree of parallelism forwarded to
            :meth:`_calc_voxel_image_for_eye`. Negative values are interpreted
            relative to available CPU cores. Defaults to ``-1``.
        partial_res : int or (int, int, int), optional
            Sub-voxel resolution used only for partially visible voxels.
            ``None`` reuses ``res``.
        force : bool, optional
            When ``True`` (default ``False``) forces recalculation even if
            cached matrices already exist with matching shapes.
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
        projection_representation : {"sparse", "hybrid"}, optional
            Result representation. ``"hybrid"`` requires optical chunks and
            stores a mixture of direct CSR blocks and ``Q @ A`` blocks.
        psf_tolerance : float, optional
            Maximum normalized-PSF grouping distance. Zero explicitly
            bypasses grouping and stores direct blocks.
        psf_metric : {"relative_l2", "l1"}, optional
            Distance used to accept a PSF group.
        psf_grouping : {"recursive", "leader"}, optional
            PSF grouping algorithm.
        max_factorized_byte_fraction : float or None, optional
            Use ``Q @ A`` only when its exact CSR payload is at most this
            fraction of the exact direct block payload. Defaults to ``0.8``.
            ``None`` always retains a nonempty factorization; zero bypasses
            grouping.

        Notes
        -----
        Aggregated matrices are stored in :attr:`_P_matrix` keyed by camera
        index. Per-eye pixel-space matrices/operators remain in
        :attr:`_projection`; subpixels are transient integration samples.
        """
        my_print("Calculating projection matrix", show=verbose > 0)
        indices = list(self.cameras.keys())
        for _c in indices:
            flag = False
            for _e in range(len(self.cameras[_c].eyes)):
                my_print(f"Processing camera {_c!r}, eye {_e + 1}/{len(self.cameras[_c].eyes)}",
                         show=verbose > 0)
                # The experimental optical ordering is intentionally rebuilt;
                # cache keys do not encode projection strategy yet.
                cached_projection = self._projection[_c][_e]
                representation_matches = (
                    isinstance(cached_projection, HybridProjectionOperator)
                    == (projection_representation == "hybrid")
                )
                if force or chunk_strategy == "optical" or not representation_matches or \
                        (cached_projection is None) or \
                        (self._projection[_c][_e].shape != (self.cameras[_c].screen.N_pixel, self.voxel.N)):
                    my_print(f"Calculating projection matrix for camera {_c!r}, "
                             f"eye {_e + 1}/{len(self.cameras[_c].eyes)}", show=verbose > 0)
                    self._calc_voxel_image_for_eye(camera_idx=_c, eye_idx=_e, res=res, n_jobs=parallel,
                                                   verbose=verbose, max_nnz=100_000_000,
                                                   partial_res=partial_res,
                                                   max_working_memory=max_working_memory,
                                                   chunk_strategy=chunk_strategy,
                                                   optical_bin_width_pixels=optical_bin_width_pixels,
                                                   projection_representation=projection_representation,
                                                   psf_tolerance=psf_tolerance,
                                                   psf_metric=psf_metric,
                                                   psf_grouping=psf_grouping,
                                                   max_factorized_byte_fraction=max_factorized_byte_fraction)
                    flag = True
                else:
                    my_print(f"Projection matrix for camera {_c!r}, "
                             f"eye {_e + 1}/{len(self.cameras[_c].eyes)} is already calculated.", show=verbose > 0)
            if flag or (self._P_matrix[_c] is None) or \
                    (self._P_matrix[_c].shape != (self.cameras[_c].screen.N_pixel, self.voxel.N)):
                # at least one eye is recalculated
                if projection_representation == "hybrid":
                    # Each eye is already in the same pixel/global-voxel
                    # coordinates, so aggregation is a direct operator sum.
                    self._P_matrix[_c] = combine_projection_operators(
                        self._projection[_c],
                    )
                else:
                    self._P_matrix[_c] = sum(
                        self._projection[_c],
                        sparse.csr_matrix(
                            (self.cameras[_c].screen.N_pixel, self.voxel.N),
                        ),
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
        # set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # draw camera coordinate system in the world coordinate system
        # x, y, z axes in the world coordinate system is the same as axes in the figure
        # draw camera position
        for camera in self.cameras.values():
            camera.draw_camera_orientation(ax=ax)

        for wall in self.walls:
            stl_utils.show_stl(wall, ax=ax, show_fig=False, **kwargs)

        if show_fig:
            ax.figure.show()

        return ax


if __name__ == '__main__':

    camera_1 = Camera(eyes=[Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.25, focal_length=20,
                                position=[0, 0]),
                            Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.25, focal_length=20,
                                position=[-5, 5]),
                            # Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.25, focal_length=15,
                            #     position=[5, -5])
                            ],
                      screen=Screen(screen_shape="rectangle", screen_size=[15, 30], pixel_shape=(25, 50),
                                    subpixel_resolution=15),
                      apertures=[
                          Aperture(shape="circle", size=10, position=[0, 0, 80]).set_model(resolution=40,
                                                                                           max_size=200),
                          # Aperture(shape="rectangle", size=(100, 10), position=[0, 0, 80]).set_model(
                          #     resolution=40, max_size=100)
                      ],
                      camera_position=[670, 670, 0],
                      # camera_position=[0, 300, 0],
                      ).set_rotation_euler("xyz",
                                            (-90, 45, 180),
                                            # (-90, 0, 180),
                                            degrees=True)
    camera_2 = Camera(eyes=[Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.25, focal_length=20,
                                position=[0, 0]),
                            Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.25, focal_length=20,
                                position=[-5, 5]),
                            # Eye(eye_type="pinhole", eye_shape="circle", eye_size=0.25, focal_length=15,
                            #     position=[5, -5])
                            ],
                      screen=Screen(screen_shape="circle", screen_size=[17, 17], pixel_shape=(32, 32),
                                    subpixel_resolution=15),
                      apertures=[
                          Aperture(shape="circle", size=10, position=[0, 0, 80]).set_model(resolution=40,
                                                                                           max_size=200),
                          # Aperture(shape="rectangle", size=(100, 10), position=[0, 0, 80]).set_model(
                          #     resolution=40, max_size=100)
                      ],
                      camera_position=[670, -500, 0],
                      # camera_position=[0, 300, 0],
                      ).set_rotation_euler("xz",
                                            (90, -90),
                                            degrees=True)
    # wall = mesh.Mesh.from_file("../relax.stl")
    wall = mesh.Mesh.from_file("../RELAX_chamber.stl")
    vox = Voxel(x_axis=np.hstack([np.linspace(-800, 0, 8, endpoint=False), np.linspace(0, 800, 17, endpoint=True)]),
                y_axis=np.linspace(-800, 800, 33),
                z_axis=np.linspace(-300, 300, 13))
    # vox = Voxel().uniform_axes(ranges=[[-800, 800], [-800, 800], [-300, 300]],
    #                            shape=[11, 11, 11], show_info=True)
    world = World(cameras=[camera_1, camera_2],
                  # voxel=Voxel().uniform_axes([[-780, 780], [-780, 780], [-270, 270]], (53, 19, 53)),
                  voxel=vox,
                  walls=wall
                  )
    # world.voxel.uniform_axes([[-40, 40], [-40, 40], [-40, 40]], (5, 5, 5))
    # points = world.voxel.gravity_center
    # points = points[points[:, 2] < 250]
    print("set world")
    print(world)
    vox.set_coordinate("torus", major_radius=500, minor_radius=250)
    # vox.set_coordinate("cylindrical", radius=50, height=100,
    #                    rotation=Rotation.from_euler("yz", (-0, -90), degrees=True)
    #                    )
    # r, theta, Z = vox.normalized_coordinates().T
    r, theta, phi = vox.normalized_coordinates().T
    inside = r < 1

    world.inside_voxels = inside
    world.set_projection_matrix(res=5, verbose=1, parallel=3)

    # f = torus_fourier_bessel(1, 1, 0)(r, theta, phi).real * (r < 1)
    # fig = multi_volume_rendering([r * (r < 1), theta * (r < 1), f], vox.gravity_center,
    #                              opacity=0.2,
    #                              surface_count=11,  # keep low for big data
    #                              # colorscale='Plasma',
    #                              # isomin=-1.5, isomax=1.5,
    #                              # where f around 0 -> opacity=0.1, f around maximum or minimum -> opacity=1
    #                              opacityscale=[[0, 0.9], [0.25, 1], [0.5, 0.], [0.75, 1], [1, 0.9]],
    #                              # opacityscale='extreme',
    #                              # slices_z=dict(show=True, locations=[0.]),
    #                              # slices_y=dict(show=True, locations=[0.]),
    #                              # slices_x=dict(show=True, locations=[0.]),
    #                              surface=dict(fill=1, pattern='all'),
    #                              caps=dict(x_show=False, y_show=False, z_show=False),
    #                              colorscale="RdBu_r",
    #
    #                              )
    # fig.show()

    visible = world.visible_voxels[0]
    visible_vertices = stl_utils.check_visible(mesh_obj=camera_1.apertures[0].stl_model,
                                               start=camera_1.eyes[0].position,
                                               grid_points=camera_1.world2camera(world.voxel.grid))

    print("--- camera orientation ---")
    ax = plt.subplot(projection="3d", proj_type="ortho")
    # ax.scatter(*world.voxel.grid[visible_vertices].T, s=10, c="g", alpha=1, ec="k", lw=0.1)
    # ax.scatter(*world.voxel.grid[~visible_vertices].T, s=10, c="gray", alpha=0.1)

    ax.scatter(*world.voxel.gravity_center[visible[0]].T, s=10, c="g", alpha=1, ec="k", lw=0.1)
    ax.scatter(*world.voxel.gravity_center[~visible[0]].T, s=10, c="gray", alpha=0.1)
    # ax.scatter(*world.voxel.gravity_center[~np.any(visible, axis=0)].T, s=10, c="gray", alpha=0.1)
    camera_2.draw_camera_orientation(ax=ax)
    # stl_utils.show_stl(obj, ax, facecolors="lightblue", edgecolors="k", alpha=0.5, linewidth=0.05)
    ax.set_title("Rotation", fontsize='xx-large')
    # ax.view_init(elev=60)
    ax.view_init(elev=30, azim=30)
    ax.figure.tight_layout()
    plt.show()
    # del ax

    # obj.translate(my_camera.apertures[1].position)
    # obj = stl_utils.rotate_model(obj, matrix=np.linalg.inv(my_camera.rotation_matrix))
    #

    ax = plt.subplot(projection="3d", proj_type="ortho")
    ax.scatter(*camera_1.world2camera(world.voxel.gravity_center[visible[0]]).T, s=10,
               zdir="x", alpha=1, c="g", ec="k", lw=0.1)
    # ax.scatter(*my_camera.world2camera(world.voxel.gravity_center[visible[1]]).T, s=10,
    #            zdir="x", alpha=0.1)
    ax.scatter(*camera_1.world2camera(world.voxel.gravity_center[~visible[0]]).T, s=10,
               zdir="x", alpha=0.1, c="gray")
    Xlim, Ylim, Zlim = zip(np.min(camera_1.world2camera(world.voxel.grid), axis=0) * 1.1,
                           np.max(camera_1.world2camera(world.voxel.grid), axis=0) * 1.1)
    camera_1.draw_optical_system(ax=ax, X_lim=Xlim, Y_lim=Ylim, Z_lim=Zlim)
    # stl_utils.show_stl(ax, my_camera.apertures[0].stl_model, facecolors="orange", edgecolors="k", alpha=0.5,
    #                    linewidth=1)

    # stl_utils.show_stl(obj, ax, facecolors="orange", edgecolors="k", alpha=0.5, linewidth=0.05)
    #
    # obj.translate(my_camera.apertures[0].position)
    # obj = stl_utils.rotate_model(obj, "x", 10, degrees=True)
    # obj = stl_utils.rotate_model(obj, matrix=[[0, 1, 0],
    #                                           [0, 0, 1],
    #                                           [1, 0, 0]])
    # stl_utils.show_stl(obj, ax, facecolors="lightblue", edgecolors="k", alpha=0.8, linewidth=0.05, modify_axes=False)
    ax.set_xlabel(ax.get_xlabel() + "(x)")
    ax.set_ylabel(ax.get_ylabel() + "(y)")
    ax.set_zlabel(ax.get_zlabel() + "(z)")
    ax.view_init(elev=30, azim=30)
    # axis limits
    # ax.set_xlim(-10, 210)
    # ax.set_ylim(100, -100)
    # ax.set_zlim(-100, 100)
    # ax.set_box_aspect((ax.get_xlim()[1] - ax.get_xlim()[0],
    #                    ax.get_ylim()[1] - ax.get_ylim()[0],
    #                    ax.get_zlim()[1] - ax.get_zlim()[0]))

    # ax.view_init(azim=180, elev=0)
    ax.figure.show()
    del ax
    plt.close("all")

    ax = world.draw_camera_orientation(
        # x_lim=(-760, 760), y_lim=(-760, 760), z_lim=(-260, 260),
        show_fig=1, elev=60, azim=-30, alpha=0)
    ax.scatter(*world.voxel.gravity_center[world.visible_voxels[0].sum(axis=0)].T, s=10, c="r", alpha=1,
               ec="k", lw=0.1, zorder=0.1)
    sub_voxels = world.voxel.get_sub_voxel(n=world.visible_voxels[0][0], res=(5, 5, 5))
    points = np.concatenate([sub_voxel.gravity_center for sub_voxel in sub_voxels])
    ax.scatter(*points.T, s=1, c="g", alpha=0.5, ec="k", lw=0.1, zorder=0.1)
    camera_grid = world.cameras[0].world2camera(vox.grid)
    cond = np.any(camera_grid[vox.vertices_indices, 2] > 0, axis=1)
    # ax.scatter(*vox.gravity_center[cond].T, s=50, c="b", alpha=1, ec="k", lw=0.1, zorder=0.1)
    # ax.scatter(*new_voxel.gravity_center[~visible].T, s=10, c="gray", alpha=0.1, zorder=0.1)
    ax.set_title("Camera orientations")
    ax.figure.show()

    # fig = stl_utils.plotly_show_stl(world.walls[0], show_fig=0, show_edges=1)
    # fig.add_trace(go.Volume(x=world.voxel.gravity_center[:, 0],
    #                         y=world.voxel.gravity_center[:, 1],
    #                         z=world.voxel.gravity_center[:, 2],
    #                         value=sum(world.visible_voxels),
    #                         slices_z=dict(show=True, locations=[0]),
    #                         isomin=0.1, surface_count=15, colorscale="jet"))

    # sub_voxels = world.voxel.get_sub_voxel(n=world.visible_voxels[0], res=2)
    # points = np.concatenate([sub_voxel.gravity_center for sub_voxel in sub_voxels])
    # image_array, visible = world.calc_image_vec(my_camera.eyes[0], points=points, verbose=1, parallel=True)
    # voxel_index = np.tile(np.arange(world.voxel.N_voxel)[world.visible_voxels[0]],
    #                       (sub_voxels[0].N_voxel, 1)).T.ravel()[visible.nonzero()[0]]
    # im_vec_list = [image_array[:, voxel_index == _].sum(axis=1) for _ in np.arange(world.voxel.N_voxel)]

    camera_1.screen.show_image(camera_1.screen.cosine(camera_1.eyes[0]) ** 4)
    s_time = time.time()
    im_vec_list_camera = world.im_vec_list_camera
    im_vec_list = im_vec_list_camera[0]
    print(f"calc_all_voxel_image: {time.time() - s_time} s")
    #
    #
    #
    camera_1.screen.show_image(sum(im_vec_list).sum(axis=1), block=False)
    ax = plt.subplot()
    camera_1.screen.show_image(camera_1.screen.subpixel_to_pixel(sum(im_vec_list).sum(axis=1)), block=False,
                               ax=ax)
    for i, eye in enumerate(camera_1.eyes):
        c = camera_1.screen.xy2uv(eye.calc_rays(camera_1.apertures[0].position).XY)
        r = eye.focal_length / (camera_1.apertures[0].position[-1] - eye.focal_length) * camera_1.apertures[
            0].size
        ax.plot(np.cos(np.linspace(0, 2 * np.pi)) * r[0] + c[0, 1],
                np.sin(np.linspace(0, 2 * np.pi)) * r[1] + c[0, 0], c="r")
        ax.scatter(*camera_1.screen.xy2uv(eye.position[:2]), ec="k", c="w", marker="*", s=50)
        ax.scatter(c[0, 1], c[0, 0], c="k", marker="x", s=50)

    ax.figure.show()

    plt.close("all")

    x_ = camera_1.screen.pixel_position.reshape(*camera_1.screen.pixel_shape, -1)[
        camera_1.screen.pixel_shape[0] // 2, :, 1]
    cosine = camera_1.screen.subpixel_to_pixel(camera_1.screen.cosine(camera_1.eyes[0])).reshape(
        camera_1.screen.pixel_shape)[
                 camera_1.screen.pixel_shape[0] // 2] / camera_1.screen.subpixel_resolution ** 2
    ax = plt.subplot()
    intensity1d = np.asarray(np.asarray(camera_1.screen.subpixel_to_pixel(im_vec_list[0]).sum(axis=1)).reshape(
        camera_1.screen.pixel_shape)[camera_1.screen.pixel_shape[0] // 2])
    ax.plot(x_, intensity1d / intensity1d.max())
    ax.plot(x_, cosine)
    ax.plot(x_, cosine ** 2)
    ax.plot(x_, cosine ** 3)
    ax.plot(x_, cosine ** 4)
    ax.figure.show()

    fig, axes = plt.subplots()
    camera_1.screen.show_image(sum(im_vec_list) @ world.voxel.gravity_center[..., 0],
                               block=False, ax=axes, cmap="RdBu_r", pm=True, pixel_image=True)

    random_generator = np.random.default_rng(seed=1234)

    voxel_val = random_generator.random(world.voxel.N_voxel)

    # my_camera.screen.show_image(im_vec_list[1].sum(axis=1), block=False)
    # my_camera.screen.show_image(sum(im_vec_list).sum(axis=1), block=False)
    #
    # print(world.voxel.N_voxel)
    # points = world.voxel.gravity_center[visible[0]]
    # print(points.shape)
    # im_vec = world.calc_image_vec(world.cameras.eyes[0], points=points, verbose=1)
    # print(im_vec)
    # print(im_vec.shape)
    # image = im_vec.sum(axis=1)
    # rgb_image = np.array([image, image, image]).T
    # world.cameras.screen.show_image(image, block=False)
    # world.cameras.screen.show_image(world.cameras.screen.subpixel_to_pixel(image), block=False)
    # world.cameras.screen.show_image(rgb_image / rgb_image.max(), block=False)
    print("Done")

    # points = np.stack(np.meshgrid(np.linspace(-50, 0, 21),
    #                               np.linspace(0, 100, 21),
    #                               np.linspace(0, 200, 21),
    #                               indexing="ij")).reshape((3, -1)).T
    # aperture = my_camera.apertures[0]
    # res = stl_utils.check_visible(aperture.stl_model, my_camera.eyes[0].position,
    #                               points, full_result=True)
    # axes = plt.subplots(2, 3, subplot_kw=dict(projection='3d', proj_type='ortho'),
    #                     figsize=(10, 5))[1].ravel()
    # for i, [ax, cond, title] in enumerate(
    #         zip(axes, res, ["inside_cone", "farther_points", "shadow", "check_list", "intersection", "visible"])):
    #     cond_ = np.any(cond, axis=1) if cond.ndim == 2 else cond
    #     ax.set_title(title)
    #     ax.scatter(*my_camera.eyes[0].position, color="g", s=100, marker="*")
    #     ax.scatter(*points[cond_.squeeze()].T, c="r", s=10, label="true", alpha=0.1)
    #     ax.scatter(*points[~cond_.squeeze()].T, c="b", s=10, label="false", alpha=0.1)
    #     ax.legend()
    #     stl_utils.show_stl(aperture.stl_model, modify_axes=True, ax=ax, elev=0, azim=-90, edgecolors="k", lw=0.5)
    #     # plot plane defined by the mesh and eye position
    # axes[0].figure.tight_layout()
    # axes[0].figure.show()

    # vertices_visible = stl_utils.check_visible(my_camera.apertures[1].stl_model, my_camera.eyes[0].position,
    #                                            my_camera.world2camera(world.voxel.gravity_center))
    # ax = plt.subplot(projection="3d", proj_type="ortho")
    # ax.scatter(*my_camera.world2camera(world.voxel.gravity_center).T, c=vertices_visible, zorder=2)
    # ax.scatter(*my_camera.eyes[0].position, c="r", zorder=3)
    # stl_utils.show_stl(my_camera.apertures[1].stl_model, ax=ax, modify_axes=True, edgecolors="k", lw=0.1)
    # ax.view_init(azim=-90, elev=0)
    # ax.figure.show()

    # fig = stl_utils.plotly_show_stl(mesh.Mesh.from_file("../Stanford_Bunny.stl"), color="lightblue", opacity=0.8,
    #                                 show_edges=True, show_fig=False)
    # # orthographic
    # fig.update_layout(scene=dict(camera=dict(eye=dict(x=0, y=0, z=0))))

    # res = stl_utils.check_visible(mesh_obj=my_camera.apertures[0].stl_model, start=my_camera.eyes[0].position,
    #                               grid_points=my_camera.world2camera(points))
