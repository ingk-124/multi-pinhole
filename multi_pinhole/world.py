import gc
import os
import time
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Tuple, List

import dill
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from stl import mesh

from multi_pinhole import Camera, Voxel, Eye, Screen, Aperture
from utils import stl_utils
from utils import type_check_and_list
from utils.my_stdio import *


def type_list(obj, type_):
    """Normalize an input object into a list of a specific type.

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
    def __init__(self,
                 voxel: Voxel = None,
                 cameras: list[Camera] | Camera = None,
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
        cameras : Camera or list[Camera], optional
            Camera instances that should be registered with the world. Any
            single instance is promoted to a list. When omitted the world starts
            without cameras.
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
        # TODO: use dictionary to store cameras and other attributes related to cameras
        # self._cameras = type_check_and_list(cameras, Camera)
        self._cameras = {i: cam for i, cam in enumerate(type_check_and_list(cameras, Camera))}
        for _ in self._cameras.values():
            _.set_world(self)

        # Walls
        self._walls = []
        self._wall_ranges = None
        self.walls = walls

        # initialize other attributes
        self._inside_vertices = None
        self._visible_vertices = {i: None for i in self._cameras.keys()}
        self._visible_voxels = {i: None for i in self._cameras.keys()}
        self._projection = {i: [None] * len(self._cameras[i].eyes) for i in self._cameras.keys()}
        self._P_matrix = {i: None for i in self._cameras.keys()}
        self.verbose = verbose

        # set inside vertices if inside_func is provided
        if inside_func is not None:
            self.set_inside_vertices(inside_func)

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
        return loaded_world

    # MARK: Properties
    @property
    def cameras(self):
        """dict[int, Camera]: Mapping of camera indices to instances."""
        return self._cameras

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
    def inside_vertices(self):
        """np.ndarray: Boolean mask indicating which voxel vertices lie inside the world."""
        if self._inside_vertices is None:
            return np.ones(self.voxel.N_grid, dtype=bool)
        else:
            return self._inside_vertices

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
        self._inside_vertices = inside_vertices.astype(bool)

    @cameras.setter
    def cameras(self, cameras: list[Camera] | Camera):
        """Register or replace cameras managed by the world.

        Parameters
        ----------
        cameras : Camera or list[Camera]
            Camera instances that should replace the current set.

        Notes
        -----
        When an incoming camera already exists in the current mapping, its
        cached visibility and projection matrices are reused to avoid
        recalculation.
        """
        cameras = type_check_and_list(cameras, Camera)
        camera_dict = {}
        visible_voxels = {}
        projection = {}
        P_matrix = {}
        for c_, camera in enumerate(cameras):
            if camera in self._cameras.values():
                # get the index of the camera in the original camera list
                ind = list(self._cameras.values()).index(camera)
                # keep the previous visible voxels, projection, and P matrix
                visible_voxels[c_] = self._visible_voxels[ind]
                projection[c_] = self._projection[ind]
                P_matrix[c_] = self._P_matrix[ind]
            else:
                # if the camera is not in the original camera list,
                # set the visible voxels, projection, and P matrix to None
                visible_voxels[c_] = None
                projection[c_] = None
                P_matrix[c_] = None
            camera_dict[c_] = camera

        self._cameras = camera_dict
        self._visible_voxels = visible_voxels
        self._projection = projection
        self._P_matrix = P_matrix
        # self._P_matrix = None
        if all([_ is None for _ in visible_voxels.values()]):
            print("Notice: All cameras are updated.")
        elif None in visible_voxels.values():
            # Notice: *th, *th, and *th cameras are reused.
            reused_index = [f"{i + 1}th" for i in visible_voxels.keys() if visible_voxels[i] is not None]
            xth = ", ".join(reused_index[:-1]) + " and " + reused_index[-1] if \
                len(reused_index) > 1 else f"{reused_index[0]}"
            print(f"Notice: {xth} camera{'s are' if len(reused_index) > 1 else ' is'} reused.")
        else:
            # All cameras are reused.
            print("Notice: All cameras are reused.")
        print(self.camera_info())

    def add_camera(self, new_camera: Camera | list[Camera]):
        """Append additional cameras to the world.

        Parameters
        ----------
        new_camera : Camera or list[Camera]
            Camera objects that should be appended to the existing mapping.

        Notes
        -----
        Newly inserted cameras start without cached visibility or projection
        data until recomputation is triggered.
        """
        new_camera = type_check_and_list(new_camera, Camera)
        current_len = len(self._cameras)
        for i, cam in enumerate(new_camera):
            self._cameras[current_len + i] = cam
            self._visible_voxels[current_len + i] = None
            self._projection[current_len + i] = [None] * len(cam.eyes)
            self._P_matrix[current_len + i] = None
        print(self.camera_info())

    def remove_camera(self, index: int | list[int]):
        """Remove one or more cameras by index.

        Parameters
        ----------
        index : int or list[int]
            Camera indices that should be deleted from the world mapping.

        Notes
        -----
        After removal, camera-related dictionaries are re-indexed so that
        indices remain contiguous and zero-based.
        """
        index = type_check_and_list(index, int)
        for i in sorted(index, reverse=True):
            if i in self._cameras:
                del self._cameras[i]
                del self._visible_voxels[i]
                del self._projection[i]
                del self._P_matrix[i]
        # re-index the cameras and related attributes
        self._cameras = {new_i: cam for new_i, cam in enumerate(self._cameras.values())}
        self._visible_voxels = {new_i: self._visible_voxels[old_i] for new_i, old_i in
                                enumerate(sorted(self._visible_voxels.keys()))}
        self._projection = {new_i: self._projection[old_i] for new_i, old_i in
                            enumerate(sorted(self._projection.keys()))}
        self._P_matrix = {new_i: self._P_matrix[old_i] for new_i, old_i in enumerate(sorted(self._P_matrix.keys()))}
        print(self.camera_info())

    def change_camera(self, index: int | list[int], camera: Camera | list[Camera]):
        """Replace cameras at the provided indices.

        Parameters
        ----------
        index : int or list[int]
            Positions within the current camera mapping that should be
            updated.
        camera : Camera or list[Camera]
            New camera instances to assign to the corresponding indices.

        Raises
        ------
        ValueError
            Raised when ``index`` and ``camera`` lengths differ.
        """
        index = type_check_and_list(index, int)
        camera = type_check_and_list(camera, Camera)
        if len(index) != len(camera):
            raise ValueError("The length of index and camera should be the same.")

        for i, cam in zip(index, camera):
            if i in self._cameras:
                self._cameras[i] = cam
                self._visible_voxels[i] = None
                self._projection[i] = [None] * len(cam.eyes)
                self._P_matrix[i] = None

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
            self._inside_vertices = inside_vertices
            print(f"Inside vertices are updated. (N_inside_vertices: {np.sum(inside_vertices)})")

    def find_visible_points(self, points: np.ndarray, camera_idx: int, eye_idx: int = None,
                            verbose: int = 1) -> np.ndarray:
        """Determine point visibility for a specific camera and eye selection.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape ``(N_points, 3)`` containing world-coordinate points
            whose visibility should be evaluated.
        camera_idx : int
            Index of the camera used for visibility testing.
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
            raise ValueError(f"The camera index {camera_idx} is not in the camera list.")
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
            for a, aperture in enumerate(_camera.apertures):
                if aperture.stl_model is None:
                    aperture.set_model()
                visible[i] *= stl_utils.check_visible(mesh_obj=aperture.stl_model,
                                                      start=_eye.position,
                                                      grid_points=camera_points,
                                                      verbose=verbose,
                                                      behind_start_included=True)  # (N_points, )
                my_print(f"{a + 1}/{len(_camera.apertures)} done", show=verbose > 0)
                my_print("-" * 15, show=verbose > 0)
                time.sleep(0.1)

            my_print("--- checking for walls ---", show=verbose > 0)
            for w, wall_in_camera in enumerate(walls_in_camera):
                visible[i] *= stl_utils.check_visible(mesh_obj=wall_in_camera,
                                                      start=_eye.position,
                                                      grid_points=camera_points,
                                                      verbose=verbose)  # (N_points, )
                my_print(f"{w + 1}/{len(walls_in_camera)} done", show=verbose > 0)
                my_print("-" * 15, show=verbose > 0)
                time.sleep(0.1)
        return visible  # (N_eye, N_points)

    def _find_visible_vertices(self, force: bool = False, verbose: int = None, camera_idx: int = None) -> None:
        """Compute visibility masks for voxel vertices per camera.

        Parameters
        ----------
        force : bool, optional
            When ``True`` (default ``False``), recompute visibility even if
            cached data with matching shape exists.
        verbose : int, optional
            Verbosity level overriding :attr:`verbose`. ``None`` preserves the
            world default.
        camera_idx : int, optional
            Specific camera index to update. ``None`` (default) processes all
            cameras.

        Notes
        -----
        Visibility results are stored in :attr:`_visible_vertices` as boolean
        arrays shaped ``(N_eye, N_vertices)``.
        """
        verbose = self.verbose if verbose is None else verbose
        if camera_idx is not None:
            if camera_idx not in self._cameras.keys():
                raise ValueError(f"The camera index {camera_idx} is not in the camera list.")
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
                    my_print(f"Visible vertices for camera {c_ + 1}/{len(self._cameras)} is already calculated.",
                             show=verbose > 0)
                    continue
                else:
                    my_print(f"Visible vertices for camera {c_ + 1}/{len(self._cameras)} has wrong shape. "
                             f"Recalculating...", show=verbose > 0)
            else:
                my_print(f"Finding visible vertices for camera {c_ + 1}/{len(self._cameras)}...", show=verbose > 0)
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
            my_print(f"Visible vertices for camera {c_ + 1}/{len(self._cameras)} is calculated.", show=verbose > 0)

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
        visible_voxels = {c_: None for c_ in self._cameras.keys()}
        for c_ in self._cameras.keys():
            self._find_visible_vertices(force=force, verbose=verbose, camera_idx=c_)
            conditions_any = np.any(self._visible_vertices[c_][:, self.voxel.vertices_indices], axis=-1).astype(int)
            conditions_all = np.all(self._visible_vertices[c_][:, self.voxel.vertices_indices], axis=-1).astype(int)
            visible_voxels[c_] = conditions_any + conditions_all
            my_print(f"Visible voxels for camera {c_ + 1}/{len(self._cameras)} is calculated.", show=verbose > 0)
        self._visible_voxels = visible_voxels
        my_print("Finding visible voxels is done.", show=verbose > 0)

    def _calc_voxel_image_for_eye(self, camera_idx: int, eye_idx: int, res: int, n_jobs: int = -2,
                                  verbose: int = None, max_nnz: int = 100_000_000):
        """Construct voxel-to-image projection for a specific camera eye.

        Parameters
        ----------
        camera_idx : int
            Index of the camera whose projection should be calculated.
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

        Notes
        -----
        The resulting sparse matrix is stored in
        ``self._projection[camera_idx][eye_idx]``.
        """
        n_jobs = os.cpu_count() + 1 + n_jobs if n_jobs < 0 else n_jobs
        n_jobs = max(1, n_jobs)
        verbose = self.verbose if verbose is None else verbose
        _camera = self.cameras[camera_idx]
        screen = _camera.screen
        N_vox = self.voxel.N
        self.voxel.res = res
        self.voxel.set_voxel2vertices(exist_ok=True, n_jobs=n_jobs, verbose=verbose)

        # check visible voxels (0->invisible, 1->partially visible, 2->fully visible)
        self.find_visible_voxels()
        vis_flag = self.visible_voxels[camera_idx][eye_idx]  # (N_vox, )
        partial_voxels = np.flatnonzero(vis_flag == 1)
        full_voxels = np.flatnonzero(vis_flag == 2)

        # No visible voxels
        if partial_voxels.size + full_voxels.size == 0:
            my_print("No visible voxels. Setting projection matrix to zero matrix.", show=verbose > 0)
            self._projection[camera_idx][eye_idx] = sparse.csr_matrix((screen.N_subpixel, N_vox))
            return

        def _full_vox_proc(sv_gc, S):
            I = _camera.calc_image_vec(eye_idx, points=sv_gc, verbose=0,
                                       check_visibility=False)  # (N_subpixel, num_vox * K)
            res = (I @ S).tocoo()  # (N_subpixel, N_vox)
            del I, sv_gc
            gc.collect()
            return res.data, res.row, res.col

        def _partial_vox_proc(sv_gc, S, mask):
            if not np.any(mask):
                return sparse.csr_matrix((screen.N_subpixel, N_vox))
            I = _camera.calc_image_vec(eye_idx, points=sv_gc[mask], verbose=0,
                                       check_visibility=False)  # (N_subpixel, num_visible*K)
            S = S[mask, :]
            res = (I @ S).tocoo()  # (N_subpixel, N_vox)
            del I, sv_gc, mask
            gc.collect()
            return res.data, res.row, res.col

        def _process_tasks(futures, desc):
            res = sparse.coo_matrix((screen.N_subpixel, N_vox))
            data_buf, row_buf, col_buf = [], [], []
            with tqdm(desc=desc, total=len(futures), disable=verbose <= 0) as pbar:
                for i, fut in enumerate(as_completed(futures)):
                    data, row, col = fut.result()
                    futures.remove(fut)  # free memory
                    data_buf.append(data)
                    row_buf.append(row)
                    col_buf.append(col)
                    # summarize every 10 chunks to reduce memory usage
                    if i % 10 == 0:
                        sum_of_buf = sparse.coo_matrix((np.concatenate(data_buf),
                                                        (np.concatenate(row_buf), np.concatenate(col_buf))),
                                                       shape=(screen.N_subpixel, N_vox))
                        res += sum_of_buf
                        # clear buffer
                        data_buf, row_buf, col_buf = [], [], []
                    pbar.update()

            # summarize remaining buffer
            if data_buf:
                sum_of_buf = sparse.coo_matrix((np.concatenate(data_buf),
                                                (np.concatenate(row_buf), np.concatenate(col_buf))),
                                               shape=(screen.N_subpixel, N_vox))
                res += sum_of_buf
                del data_buf, row_buf, col_buf
                gc.collect()

            return res

        if full_voxels.size == 0:
            my_print("Skipping full voxels processing.", show=verbose > 0)
        else:
            # random sample voxels to estimate n_step
            sample_n = np.random.choice(full_voxels, size=min(full_voxels.size, 20), replace=False)
            sample_gc = np.concatenate([sv.gravity_center for sv in self.voxel.get_sub_voxel(n=sample_n)],
                                       axis=0)  # (sample_size * K, 3)
            sample_I = _camera.calc_image_vec(eye_idx, points=sample_gc, verbose=0,
                                              check_visibility=False)  # (N_subpixel, sample_size * K)
            est_nnz = sample_I.nnz / sample_n.size  # average nnz per voxel
            if est_nnz == 0:
                density = 0.01
                est_nnz = screen.N_subpixel * density
            batch_size = int(np.clip(np.ceil(max_nnz / est_nnz) / n_jobs, 1, full_voxels.size) / 10)  * 10
            _chunks = [slice(_i, min(_i + batch_size, full_voxels.size)) for _i in
                       range(0, full_voxels.size, batch_size)]
            my_print(f"Processing full voxels in {len(_chunks)} chunks "
                     f"(n_step={batch_size}, full_size={full_voxels.size})",
                     show=verbose > 0)

            if n_jobs == 1:
                # initialize result matrix for full voxels
                full_res = sparse.coo_matrix((screen.N_subpixel, N_vox))

                data_buf, row_buf, col_buf = [], [], []
                for i, _slice in enumerate(my_tqdm(_chunks, desc="Processing full voxels", disable=verbose <= 0)):
                    sv_gc = np.concatenate(
                        [sv.gravity_center for sv in self.voxel.get_sub_voxel(n=full_voxels[_slice])],
                        axis=0)  # (num_vox * K, 3)
                    S = sparse.vstack(self.voxel.sub_voxel_interpolator(n=full_voxels[_slice], verbose=0)).tocsr()
                    data, row, col = _full_vox_proc(sv_gc, S)
                    data_buf.append(data)
                    row_buf.append(row)
                    col_buf.append(col)

                    # summarize every 10 chunks to reduce memory usage
                    if i % 10 == 0:
                        sum_of_buf = sparse.coo_matrix((np.concatenate(data_buf),
                                                        (np.concatenate(row_buf), np.concatenate(col_buf))),
                                                       shape=(screen.N_subpixel, N_vox))
                        full_res = full_res + sum_of_buf
                        # clear buffer
                        data_buf, row_buf, col_buf = [], [], []
                # summarize remaining buffer
                if data_buf.size > 0:
                    sum_of_buf = sparse.coo_matrix((data_buf, (row_buf, col_buf)),
                                                   shape=(screen.N_subpixel, N_vox))
                    full_res = full_res + sum_of_buf
                    del data_buf, row_buf, col_buf
                    gc.collect()
            else:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    futures = []
                    # submit tasks
                    for _slice in my_tqdm(_chunks, desc="Submitting full voxel tasks",
                                          disable=verbose <= 0, leave=False):
                        sv_gc = np.concatenate(
                            [sv.gravity_center for sv in self.voxel.get_sub_voxel(n=full_voxels[_slice])],
                            axis=0)  # (num_vox * K, 3)
                        S = sparse.vstack(self.voxel.sub_voxel_interpolator(n=full_voxels[_slice], verbose=0)).tocsr()
                        futures.append(executor.submit(_full_vox_proc, sv_gc, S))

                full_res = _process_tasks(futures, desc="Processing full voxels")

            my_print(f"Full voxels processed.", show=verbose > 0)

        if partial_voxels.size == 0:
            my_print("Skipping partial voxels processing.", show=verbose > 0)
        else:
            # random sample voxels to estimate n_step
            while True:  # to avoid all non-visible samples (just in case)
                sample_n = np.random.choice(partial_voxels, size=min(partial_voxels.size, 20), replace=False)
                sample_gc = np.concatenate([sv.gravity_center for sv in self.voxel.get_sub_voxel(n=sample_n)],
                                           axis=0)  # (sample_size * K, 3)
                sample_mask = self.find_visible_points(sample_gc, camera_idx=camera_idx,
                                                       eye_idx=eye_idx, verbose=0).squeeze()
                if np.any(sample_mask):
                    break

            sample_I = _camera.calc_image_vec(eye_idx, points=sample_gc[sample_mask], verbose=0,
                                              check_visibility=False)  # (N_subpixel, num_visible)
            est_nnz = sample_I.nnz / sample_n.size  # average nnz per voxel
            batch_size = int(np.clip(np.ceil(max_nnz / est_nnz) / n_jobs, 1, partial_voxels.size))
            _chunks = [slice(_i, min(_i + batch_size, partial_voxels.size)) for _i in
                       range(0, partial_voxels.size, batch_size)]
            my_print(f"Processing partial voxels in {len(_chunks)} chunks "
                     f"(n_step={batch_size}, partial_size={partial_voxels.size})",
                     show=verbose > 0)

            if n_jobs == 1:
                # initialize result matrix for partial voxels
                partial_res = sparse.coo_matrix((screen.N_subpixel, N_vox))
                data_buf, row_buf, col_buf = [], [], []
                for i, _slice in enumerate(my_tqdm(_chunks, desc="Processing partial voxels", disable=verbose <= 0)):
                    sv_gc = np.concatenate(
                        [sv.gravity_center for sv in self.voxel.get_sub_voxel(n=partial_voxels[_slice])],
                        axis=0)  # (num_vox * K, 3)
                    mask = self.find_visible_points(sv_gc, camera_idx=camera_idx,
                                                    eye_idx=eye_idx, verbose=0).squeeze()
                    S = sparse.vstack(self.voxel.sub_voxel_interpolator(n=partial_voxels[_slice], verbose=0)).tocsr()
                    data, row, col = _partial_vox_proc(sv_gc, S, mask)
                    data_buf.append(data)
                    row_buf.append(row)
                    col_buf.append(col)

                    # summarize every 10 chunks to reduce memory usage
                    if i % 10 == 0:
                        sum_of_buf = sparse.coo_matrix((np.concatenate(data_buf),
                                                        (np.concatenate(row_buf), np.concatenate(col_buf))),
                                                       shape=(screen.N_subpixel, N_vox))
                        partial_res = partial_res + sum_of_buf
                        # clear buffer
                        data_buf, row_buf, col_buf = [], [], []
                # summarize remaining buffer
                if data_buf.size > 0:
                    sum_of_buf = sparse.coo_matrix((data_buf, (row_buf, col_buf)),
                                                   shape=(screen.N_subpixel, N_vox))
                    partial_res = partial_res + sum_of_buf
                    del data_buf, row_buf, col_buf
                    gc.collect()
            else:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    futures = []
                    # submit tasks
                    for _slice in my_tqdm(_chunks, desc="Submitting partial voxel tasks",
                                          disable=verbose <= 0, leave=False):
                        sv_gc = np.concatenate(
                            [sv.gravity_center for sv in self.voxel.get_sub_voxel(n=partial_voxels[_slice])],
                            axis=0)  # (num_vox * K, 3)
                        mask = self.find_visible_points(sv_gc, camera_idx=camera_idx,
                                                        eye_idx=eye_idx, verbose=0).squeeze()
                        S = sparse.vstack(
                            self.voxel.sub_voxel_interpolator(n=partial_voxels[_slice], verbose=0)).tocsr()
                        futures.append(executor.submit(_partial_vox_proc, sv_gc, S, mask))

                partial_res = _process_tasks(futures, desc="Processing partial voxels")

            my_print(f"Partial voxels processed.", show=verbose > 0)

        self._projection[camera_idx][eye_idx] = (full_res + partial_res).tocsr()
        del full_res, partial_res
        gc.collect()
        my_print(f"Projection matrix for camera {camera_idx + 1}/{len(self.cameras)}, "
                 f"eye {eye_idx + 1}/{len(_camera.eyes)} is calculated.", show=verbose > 0)
        return

    def trace_line(self, points, camera_idx: int = 0, eye_idx: int = 0, coord_type: str = "XY"):
        """Project world-coordinate points onto a camera screen.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape ``(N, 3)`` containing world-coordinate points.
        camera_idx : int, optional
            Index of the camera used for projection. Defaults to ``0``.
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
                              force: bool = False):
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
        force : bool, optional
            When ``True`` (default ``False``) forces recalculation even if
            cached matrices already exist with matching shapes.

        Notes
        -----
        Aggregated matrices are stored in :attr:`_P_matrix` keyed by camera
        index, while per-eye subpixel matrices remain in :attr:`_projection`.
        """
        my_print("Calculating projection matrix", show=verbose > 0)
        indices = list(self.cameras.keys())
        for _c in indices:
            flag = False
            for _e in range(len(self.cameras[_c].eyes)):
                my_print(f"Processing camera {_c + 1}/{len(self.cameras)}, eye {_e + 1}/{len(self.cameras[_c].eyes)}",
                         show=verbose > 0)
                if force or (self._projection[_c][_e] is None) or \
                        (self._projection[_c][_e].shape != (self.cameras[_c].screen.N_subpixel, self.voxel.N)):
                    my_print(f"Calculating projection matrix for camera {_c + 1}/{len(self.cameras)}, "
                             f"eye {_e + 1}/{len(self.cameras[_c].eyes)}", show=verbose > 0)
                    self._calc_voxel_image_for_eye(camera_idx=_c, eye_idx=_e, res=res, n_jobs=parallel,
                                                   verbose=verbose, max_nnz=100_000_000)
                    flag = True
                else:
                    my_print(f"Projection matrix for camera {_c + 1}/{len(self.cameras)}, "
                             f"eye {_e + 1}/{len(self.cameras[_c].eyes)} is already calculated.", show=verbose > 0)
            if flag or (self._P_matrix[_c] is None) or \
                    (self._P_matrix[_c].shape != (self.cameras[_c].screen.N_pixel, self.voxel.N)):
                # at least one eye is recalculated
                # combine all projection matrices
                _proj = sum(self._projection[_c], sparse.csr_matrix((self.cameras[_c].screen.N_subpixel, self.voxel.N)))
                self._P_matrix[_c] = self.cameras[_c].screen.transform_matrix * _proj
                my_print(f"Projection matrix for camera {_c + 1}/{len(self.cameras)} is calculated.", show=verbose > 0)
            else:
                my_print(f"Projection matrix for camera {_c + 1}/{len(self.cameras)} is already calculated.",
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
                     *[camera.camera_position[0] for camera in self.cameras]) * 1.1,
                 max(vx_lim[1], wx_lim[1],
                     *[camera.camera_position[0] for camera in self.cameras]) * 1.1) if x_lim is None else x_lim
        y_lim = (min(vy_lim[0], wy_lim[0],
                     *[camera.camera_position[1] for camera in self.cameras]) * 1.1,
                 max(vy_lim[1], wy_lim[1],
                     *[camera.camera_position[1] for camera in self.cameras]) * 1.1) if y_lim is None else y_lim
        z_lim = (min(vz_lim[0], wz_lim[0],
                     *[camera.camera_position[2] for camera in self.cameras]) * 1.1,
                 max(vz_lim[1], wz_lim[1],
                     *[camera.camera_position[2] for camera in self.cameras]) * 1.1) if z_lim is None else z_lim

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
        for camera in self.cameras:
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
                      ).set_rotation_matrix("xyz",
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
                      ).set_rotation_matrix("xz",
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
