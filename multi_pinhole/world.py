import gc
import time
from typing import Tuple, List

import dill
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy import sparse
from stl import mesh
from tqdm.contrib import tzip

from multi_pinhole import Camera, Voxel, Eye, Screen, Aperture
from utils import stl_utils
from utils import type_check_and_list
from utils.my_stdio import my_print


def type_list(obj, type_):
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
    return np.array([0 if blk is None else blk.shape[0] for blk in arr_blocks], dtype=np.int64)


def _slice_blocks(
        pts_blocks: List[np.ndarray],
        S_blocks: List[sparse.csr_matrix],
        start: int,
        stop: int,
        n_vox: int
) -> Tuple[np.ndarray, sparse.csr_matrix]:
    """連結した (pts_blocks, S_blocks) をグローバル範囲 [start:stop) で部分スライス。"""
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
        """The world class

        Parameters
        ----------
        voxel: Voxel
            The voxel model
        cameras: Camera or list[Camera]
            The camera model
        walls: list[mesh.Mesh], optional (default is None)
            The walls in the world
        inside_func: callable, optional (default is None)
            A function that takes three parameters X, Y, Z and returns a boolean array

        Notes
        -----
        The walls should be a list of mesh.Mesh or a mesh.Mesh.
        The inside voxels should be a boolean array with the length of the number of voxels.
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
        self._projection = {i: [None]*len(self._cameras[i].eyes) for i in self._cameras.keys()}
        self._P_matrix = {i: None for i in self._cameras.keys()}
        self.verbose = verbose

        # set inside vertices if inside_func is provided
        if inside_func is not None:
            self.set_inside_vertices(inside_func)

    def __repr__(self):
        return f"World(voxel={self.voxel}, CAMERA={self.cameras}, walls={self.walls})"

    def __copy__(self):
        return World(cameras=self.cameras, voxel=self.voxel)

    def camera_info(self):
        """Print the summary of the cameras"""
        txt = "Camera Info:\n"
        for i, camera in self._cameras.items():
            txt += f" Camera {i}:\n"
            txt += camera.__repr__() + "\n"
        txt = txt.rstrip("\n")
        return txt

    def voxel_info(self):
        """Print the summary of the voxel"""
        return self.voxel.__repr__()

    # MARK: Save and Load
    def save_world(self, filename):
        """Save the world to a file

        Parameters
        ----------
        filename: str
            The filename to save the world
        """
        with open(filename, "wb") as f:
            dill.dump(self, f)

    def load_world(self, filename):
        """Load the world from a file

        Parameters
        ----------
        filename: str
            The filename to load the world
        """
        with open(filename, "rb") as f:
            loaded_world = dill.load(f)
        self.__dict__.update(loaded_world.__dict__)

    # MARK: Properties
    @property
    def cameras(self):
        return self._cameras

    @property
    def voxel(self):
        return self._voxel

    @property
    def walls(self):
        return self._walls

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
        if not self.walls:
            return None
        _ = [(w.update_min(), w.update_max()) for w in self.walls]
        self._wall_ranges = zip(np.min([w.min_ for w in self._walls], axis=0),
                                np.max([w.max_ for w in self._walls], axis=0))
        return self._wall_ranges

    @property
    def visible_voxels(self):
        return self._visible_voxels

    # @property
    # def P_matrix(self):
    # TODO: implement this property
    # if None in self._P_matrix_list:
    #     print("One or more P matrix is not calculated.")
    #     print("Please run set_projection_matrix method.")
    #     return
    # elif (self._P_matrix is None) or (
    #         self._P_matrix.shape[0] != sum([P.shape[0] for P in self._P_matrix_list])):
    #     self._P_matrix = sparse.vstack(self._P_matrix_list)
    # return self._P_matrix

    @property
    def P_matrix(self):
        return self._P_matrix

    @property
    def inside_vertices(self):
        if self._inside_vertices is None:
            return np.ones(self.voxel.N_grid, dtype=bool)
        else:
            return self._inside_vertices

    # MARK: Setters
    @inside_vertices.setter
    def inside_vertices(self, inside_vertices: np.ndarray):
        if not isinstance(inside_vertices, np.ndarray):
            raise TypeError(f"inside_voxels should be a np.ndarray, not {type(inside_vertices)}")
        if inside_vertices.size != self.voxel.N_grid:
            raise ValueError("inside_voxels should be a np.ndarray with the length of the number of grid points.")
        self._inside_vertices = inside_vertices.astype(bool)

    @cameras.setter
    def cameras(self, cameras: list[Camera] | Camera):
        """Set the cameras

        Parameters
        ----------
        cameras
            List of Camera objects

        Notes
        -----
        If the new camera is the same as the previous one, the visible voxels, projection, and P matrix are reused.
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
        """Add a camera

        Parameters
        ----------
        new_camera: Camera | list[Camera]
            The camera object to add

        Notes
        -----
        Added cameras are appended to the end of the camera list.
        """
        new_camera = type_check_and_list(new_camera, Camera)
        current_len = len(self._cameras)
        for i, cam in enumerate(new_camera):
            self._cameras[current_len + i] = cam
            self._visible_voxels[current_len + i] = None
            self._projection[current_len + i] = [None]*len(cam.eyes)
            self._P_matrix[current_len + i] = None
        print(self.camera_info())

    def remove_camera(self, index: int | list[int]):
        """Remove a camera

        Parameters
        ----------
        index: int
            The index of the camera to remove

        Notes
        -----
        After removing the camera(s), the keys of the dictionaries related to cameras are re-indexed.
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
        """Change cameras at the specified indices

        Parameters
        ----------
        index: int | list[int]
            The index of the camera to change
        camera: Camera | list[Camera]
            The new camera object
        """
        index = type_check_and_list(index, int)
        camera = type_check_and_list(camera, Camera)
        if len(index) != len(camera):
            raise ValueError("The length of index and camera should be the same.")

        for i, cam in zip(index, camera):
            if i in self._cameras:
                self._cameras[i] = cam
                self._visible_voxels[i] = None
                self._projection[i] = [None]*len(cam.eyes)
                self._P_matrix[i] = None

        print(self.camera_info())

    @voxel.setter
    def voxel(self, voxel):
        if not isinstance(voxel, Voxel):
            raise TypeError(f"voxel should be a Voxel, not {type(voxel)}")
        if voxel != self._voxel:
            my_print("Voxel is updated.", show=self.verbose > 0)
            self._voxel = voxel
            self._visible_voxels = {i: None for i in self._cameras.keys()}
            self._projection = {i: [None]*len(self._cameras[i].eyes) for i in self._cameras.keys()}
            self._P_matrix = {i: None for i in self._cameras.keys()}

    @walls.setter
    def walls(self, walls: list[mesh.Mesh] | mesh.Mesh):
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
        """
        Find the inside vertices using a function

        Parameters
        ----------
        function: callable
            A function that takes three parameters X, Y, Z and returns a boolean array
        kwargs: dict
            Additional parameters for the function

        Notes
        -----
        The function should take three parameters X, Y, Z, which are the vertices of the voxel grid.
        The return value should be a boolean array with the length of the number of vertices and True for inside vertices.
        The boolean array is stored in `self.inside_vertices`.
        """
        if not callable(function):
            raise TypeError(f"function should be a callable, not {type(function)}")
        inside_vertices = function(*self.voxel.grid.T, **kwargs).astype(bool)
        if self._inside_vertices.size != self.voxel.N_grid:
            raise ValueError("The return value of the function should be a boolean array with the length of the number "
                             "of grid points.")
        else:
            self._inside_vertices = inside_vertices
            print(f"Inside vertices are updated. (N_inside_vertices: {np.sum(inside_vertices)})")

    def _find_visible_points(self, points: np.ndarray, camera_idx: int, eye_idx: int = None,
                             verbose: int = 1) -> np.ndarray:
        """
        Return the conditions of visible vertices for each camera

        Parameters
        ----------
        points: np.ndarray (N_points, 3)
            The points to check visibility
        camera_idx: int
            The index of the camera to check visibility
        verbose: int, optional (default is 1)
            The verbose level

        Returns
        -------
        visible: np.ndarray (N_eye, N_points)
            The conditions of visible voxels for the camera.
            If lines from the eye to the point are not intersected with any mesh (aperture or wall),
            the points are defined as visible (True).
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

        for _e in eye_idx:
            _eye = _camera.eyes[_e]
            # check if the voxel is behind the camera (N_points, )
            visible[_e] = camera_points[:, 2] >= _eye.position[-1]
            # check if the voxel is in front of the camera (N_points, )
            my_print(f"checking visible points for eye {_e + 1}/{len(_camera.eyes)}",
                     show=verbose > 0)
            my_print("-" * 15, show=verbose > 0)
            # get conditions for each aperture and wall
            my_print(f"--- checking for apertures ---", show=verbose > 0)
            for a, aperture in enumerate(_camera.apertures):
                if aperture.stl_model is None:
                    aperture.set_model()
                visible[_e] *= stl_utils.check_visible(mesh_obj=aperture.stl_model,
                                                       start=_eye.position,
                                                       grid_points=camera_points,
                                                       verbose=verbose,
                                                       behind_start_included=True)  # (N_points, )
                my_print(f"{a + 1}/{len(_camera.apertures)} done", show=verbose > 0)
                my_print("-" * 15, show=verbose > 0)
                time.sleep(0.1)

            my_print("--- checking for walls ---", show=verbose > 0)
            for w, wall_in_camera in enumerate(walls_in_camera):
                visible[_e] *= stl_utils.check_visible(mesh_obj=wall_in_camera,
                                                       start=_eye.position,
                                                       grid_points=camera_points,
                                                       verbose=verbose)  # (N_points, )
                my_print(f"{w + 1}/{len(walls_in_camera)} done", show=verbose > 0)
                my_print("-" * 15, show=verbose > 0)
                time.sleep(0.1)
        return visible  # (N_eye, N_points)

    def _find_visible_vertices(self, force: bool = False, verbose: int = None, camera_idx: int = None) -> None:
        """Set the conditions of visible vertices for each camera

        Parameters
        ----------
        force: bool, optional (default is False)
            If True, the visible vertices are recalculated even if they are already calculated.
        verbose: int, optional (default is None)
            The verbose level

        Notes
        -----
        The conditions are stored in `self._visible_vertices` as a dictionary with the camera index as the key.
        The condition array has the shape of (N_eye, N_vertices).
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
                    = self._find_visible_points(self.voxel.grid[self.inside_vertices],
                                                camera_idx=c_,
                                                verbose=verbose)  # (N_eye, N_inside_points)
            self._visible_vertices[c_] = visible_vertices
            my_print(f"Visible vertices for camera {c_ + 1}/{len(self._cameras)} is calculated.", show=verbose > 0)

    def find_visible_voxels(self, force=False, verbose=None):
        """Return the conditions of visible voxels
        Parameters
        ----------
        force: bool, optional (default is False)
            If True, the visible vertices are recalculated even if they are already calculated.
        verbose: int, optional (default is None)
            The verbose level

        Notes
        -----
        The conditions are stored in `self._visible_voxels` as a list with the camera index as the key.
        The condition array has the shape of (N_eye, N_voxels).
        The value of the condition array is:
            0: not visible
            1: partially visible (some vertices are visible)
            2: fully visible (all vertices are visible)
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

    def _calc_all_voxel_image(self, camera: Camera,
                              res: int = None, verbose: int = 0, parallel: int = 4) -> list[sparse.csc_matrix]:
        """Calculate the image vector for each voxel (from eye to voxel center)

        Parameters
        ----------
        camera: Camera
            The camera object
        res: int
            The resolution of the sub-voxel
        verbose : int, optional (default is 0)
            verbose level for parallel calculation
        parallel : int, optional (default is 0)
            number of parallel processes for parallel calculation (0: no parallel calculation)

        Returns
        -------
        image_vec: sparse.csc_matrix
            The image vector on the screen
        """
        if verbose > 0:
            print("Calculating image vector for each voxel")
            _tzip = tzip
        else:
            _tzip = zip

        if camera in self.cameras:
            _c = self.cameras.index(camera)
            _visible_voxels = self.visible_voxels[_c]
        else:
            raise ValueError("The camera is not in the camera list. Please set the camera first.")

        _im_vec_list = []
        for _e, _eye in enumerate(camera.eyes):
            print(f"Calculating image vector for eye {_e + 1}/{len(camera.eyes)}"
                  f" (camera {_c + 1}/{len(self.cameras)})")

            voxel_index = np.argwhere(_visible_voxels[_e]).flatten()
            _sub_voxels = self.voxel.get_sub_voxel(n=voxel_index, res=res)
            interpolator_list = self.voxel.sub_voxel_interpolator(n=voxel_index, res=res)

            # @jit(nopython=True, cache=True)
            def _calc_image_vec(interpolator, sub_voxel, _visible):
                if _visible == 1:
                    index = self.find_visible_points(sub_voxel.gravity_center,
                                                     cameras=camera, eye_num=_e, verbose=0)[0][0]
                    return camera.calc_image_vec(_e, points=sub_voxel.gravity_center[index], parallel=parallel) @ \
                        interpolator[index]

                elif _visible == 2:
                    return camera.calc_image_vec(_e, points=sub_voxel.gravity_center,
                                                 parallel=parallel) @ interpolator
                else:
                    pass
                gc.collect()

            my_print(f"Total tasks: {len(_sub_voxels)}", show=verbose > 0)

            im_vec_sub_voxels = Parallel(n_jobs=parallel, verbose=verbose, backend="loky")(
                [delayed(_calc_image_vec)(interpolator, sub_voxel, _visible) for interpolator, sub_voxel, _visible
                 in zip(interpolator_list, _sub_voxels, self.visible_voxels[_c][_e][voxel_index])])

            _im_vec_list.append(sum(im_vec_sub_voxels))

        return _im_vec_list

    def _calc_voxel_image_for_eye(self, camera_idx: int, eye_idx: int, res: int, parallel: int = 0,
                                  target_work_per_batch: int = 5_000_000,
                                  sample_for_batch: int = 2000, verbose: int=None) -> sparse.csc_matrix:
        """
        Calculate the image vector for visible voxels for a specific eye of a camera
        Parameters
        ----------
        camera_idx: int
            The index of the camera
        eye_idx: int
            The index of the eye
        res: int
            The resolution of the sub-voxel
        parallel: int, optional (default is 0)
            number of parallel processes for parallel calculation (0: no parallel calculation)
        target_work_per_batch: int, optional (default is 5_000_000)
            The target work per batch for estimating the batch size
        sample_for_batch: int, optional (default is 2000)
            The number of samples for estimating the batch size
        verbose: int, optional (default is None)
            The verbose level
        Returns
        -------
        projection_matrix: sparse.csc_matrix
            The projection matrix for the specified eye of the camera

        Notes
        -----
        The projection matrix has the shape of (N_subpixel, N_voxel).
        For fully visible voxels, all sub-voxels are used for projection.
        For partially visible voxels, only the visible sub-voxels are used for projection.
        The projection matrix is calculated in batches to reduce memory usage.
        The batch size is estimated based on the number of non-zero elements in the image vector and the interpolator.
        """

        verbose = self.verbose if verbose is None else verbose
        _camera = self.cameras[camera_idx]
        screen = _camera.screen
        N_vox = self.voxel.N
        self.find_visible_voxels()

        # 可視フラグ: 0=不可視, 1=部分可視, 2=完全可視

        vis_flag = self.visible_voxels[camera_idx][eye_idx]  # (N_vox, )
        voxels_part = np.flatnonzero(vis_flag == 1)
        voxels_full = np.flatnonzero(vis_flag == 2)

        if voxels_part.size + voxels_full.size == 0:
            return sparse.csc_matrix((screen.N_subpixel, N_vox))

        # 取得順（FULL→PART）
        my_print(f"Getting sub-voxels and interpolators...", show=verbose > 0)
        start_time = time.time()
        vox_order = np.concatenate([voxels_full, voxels_part])
        sub_list = self.voxel.get_sub_voxel(n=vox_order, res=res)  # list of sub-voxel
        interp_list = self.voxel.sub_voxel_interpolator(n=vox_order, res=res)  # list of CSR (K × N_vox)
        my_print(f"Done. ({time.time() - start_time:.1f} sec)", show=verbose > 0)

        # voxel_id -> (centers(K,3), interpolator(K×N_vox))
        sub_map = {v: (sv.gravity_center, itp) for v, sv, itp in zip(vox_order, sub_list, interp_list)}

        # --- FULL: 全行採用 ---
        full_pts_blocks, full_S_blocks = [], []
        for v in voxels_full:
            centers, Ivn = sub_map[v]
            full_pts_blocks.append(centers)  # (K,3)
            full_S_blocks.append(Ivn)  # (K×N_vox)

        # --- PART: まとめて可視判定 → 可視行だけ採用 ---
        part_pts_blocks, part_S_blocks = [], []
        if voxels_part.size:
            centers_blocks = [sub_map[v][0] for v in voxels_part]
            sizes = np.array([c.shape[0] for c in centers_blocks], dtype=np.int64)
            if sizes.sum() > 0:
                centers_cat = np.concatenate(centers_blocks, axis=0)
                mask_cat = self._find_visible_points(centers_cat,
                                                     camera_idx=camera_idx, eye_idx=eye_idx, verbose=1).squeeze()

                # 分配
                off = 0
                for v, K in zip(voxels_part, sizes):
                    m = mask_cat[off:off + K]
                    if np.any(m):
                        centers, Ivn = sub_map[v]
                        part_pts_blocks.append(centers[m])
                        part_S_blocks.append(Ivn[m, :])
                    off += K

        my_print(f"Full voxels: {len(full_pts_blocks)}, Part voxels: {len(part_pts_blocks)}",
                 show=self.verbose > 0)
        # 以降、(pts_blocks, S_blocks) をチャンクで処理
        pts_blocks = full_pts_blocks + part_pts_blocks
        S_blocks = full_S_blocks + part_S_blocks
        total_pts = int(_blocks_lengths(pts_blocks).sum())
        if total_pts == 0:
            return sparse.csc_matrix((screen.N_subpixel, N_vox))

        # バッチサイズ見積り（投影の nnz/点 と 補間の nnz/行 から）
        def _estimate_batch_points() -> int:
            n_samp = min(total_pts, int(sample_for_batch))
            samp_pts, _ = _slice_blocks(pts_blocks, S_blocks, 0, n_samp, N_vox)
            if samp_pts.shape[0] == 0:
                return 1
            I_s = _camera.calc_image_vec(eye_idx, points=samp_pts, parallel=0)
            k = max(1.0, I_s.nnz / float(samp_pts.shape[0]))  # 1点あたりのスクリーン nnz
            # S 側（補間）の平均 nnz/行を簡易に推定：先頭ブロックの一部で十分
            S_head = S_blocks[0] if S_blocks else sparse.csr_matrix((0, N_vox))
            if S_head.shape[0] > 4096:
                S_head = S_head[:4096, :]
            r = max(1.0, float(np.mean(np.asarray(S_head.getnnz(axis=1)).ravel())) if S_head.shape[0] else 8.0)
            bs = int(max(1, target_work_per_batch / (k * r)))
            return int(min(bs, 100_000))

        batch_pts = _estimate_batch_points()

        # チャンクループ：投影 → 右掛けで集約
        result = sparse.csc_matrix((screen.N_subpixel, N_vox))
        # done = 0
        # while done < total_pts:
        #     s = done
        #     t = min(done + batch_pts, total_pts)
        #     pts_chunk, S_chunk = _slice_blocks(pts_blocks, S_blocks, s, t, N_vox)
        #     if pts_chunk.shape[0] == 0:
        #         break
        #     # 上位で並列化するなら parallel=0 に、逆にするならここで並列
        #     I_chunk = _camera.calc_image_vec(eye_idx, points=pts_chunk, parallel=parallel, verbose=verbose)
        #     result += I_chunk @ S_chunk
        #     done = t
        def _sub_proc(pts, S):
            if pts.shape[0] == 0:
                return sparse.csc_matrix((screen.N_subpixel, N_vox))
            return _camera.calc_image_vec(eye_idx, points=pts, parallel=0, verbose=0) @ S
        pts_chunks, S_chunks = zip(*[_slice_blocks(pts_blocks, S_blocks, s, min(s + batch_pts, total_pts), N_vox) for
                                     s in range(0, total_pts, batch_pts)])
        res = Parallel(n_jobs=parallel, verbose=verbose, backend="loky")(
            [delayed(_sub_proc)(pts, S) for pts, S in zip(pts_chunks, S_chunks)])

        result = sum(res, result)
        self._projection[camera_idx][eye_idx] = result

    def set_projection_matrix(self, res: int = None, verbose: int = 1, parallel: int = -1,
                              force: bool = False):
        """Get the projection matrix from the voxel to the screen

        Parameters
        ----------
        index: int | list[int], optional (default is None)
            The index of the camera
        res: int, optional (default is None)
            The resolution of the sub-voxel
        verbose: int, optional (default is 1)
            The verbose level
        parallel: int, optional (default is -1)
            The number of parallel processes
        force: bool, optional (default is False)
            Whether to force to calculate the projection matrix

        Returns
        -------
        P: np.ndarray
            The projection matrix (N_pixel, N_voxel)
        """
        my_print("Calculating projection matrix", show=verbose > 0)
        indices = list(self.cameras.keys())
        for _c in indices:
            camera = self.cameras[_c]
            for _e in range(len(camera.eyes)):
                if force or (self._projection[_c][_e] is None) or \
                        (self._projection[_c][_e].shape != (camera.screen.N_subpixel, self.voxel.N)):
                    my_print(f"Calculating projection matrix for camera {_c + 1}/{len(self.cameras)}, "
                             f"eye {_e + 1}/{len(camera.eyes)}", show=verbose > 0)
                    self._calc_voxel_image_for_eye(camera_idx=_c, eye_idx=_e, res=res, parallel=parallel,
                                                   verbose=verbose)
                else:
                    my_print(f"Projection matrix for camera {_c + 1}/{len(self.cameras)}, "
                             f"eye {_e + 1}/{len(camera.eyes)} is already calculated.", show=verbose > 0)
            # combine all projection matrices
            self._P_matrix[_c] = sum(self._projection[_c], sparse.csc_matrix(
                (camera.screen.N_subpixel, self.voxel.N)))
            my_print(f"Projection matrix for camera {_c + 1}/{len(self.cameras)} is calculated.", show=verbose > 0)
        my_print("Calculating projection matrix is done.", show=verbose > 0)
    def draw_camera_orientation(self, ax=None, show_fig=False, x_lim=None, y_lim=None, z_lim=None, **kwargs):
        """Draw the camera orientation

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes to draw the camera orientation
        show_fig: bool, optional (default is False)
            Whether to show the figure
        x_lim: tuple, optional (default is None)
            The x limits of the axes
        y_lim: tuple, optional (default is None)
            The y limits of the axes
        z_lim: tuple, optional (default is None)
            The z limits of the axes
        kwargs: dict
            The keyword arguments for show_stl
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
