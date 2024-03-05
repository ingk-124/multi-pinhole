import gc
import time

import dill
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy import sparse
from stl import mesh
from tqdm.contrib import tzip

from utils import stl_utils
from utils import type_check_and_list
from utils.my_stdio import my_print
from .core import Camera, Eye, Screen, Aperture
from .voxel import Voxel


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


class World:
    def __init__(self,
                 voxel: Voxel = None,
                 cameras: list[Camera] | Camera = None,
                 walls: list[mesh.Mesh] | mesh.Mesh = None,
                 inside_voxels: np.ndarray = None,
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
        inside_voxels: np.ndarray, optional (default is None)
            The inside voxels


        Notes
        -----
        The walls should be a list of mesh.Mesh or a mesh.Mesh.
        The inside voxels should be a boolean array with the length of the number of voxels.
        """

        self._voxel = voxel if voxel is not None else Voxel()
        self._voxel.set_world(self)

        self._cameras = type_check_and_list(cameras, Camera)

        for _ in self._cameras:
            _.set_world(self)

        self._walls = []
        self._wall_ranges = None
        self.walls = walls
        self._inside_voxels = True
        self._visible_voxels = None
        self._projection = [None] * len(self._cameras)
        self._P_matrix_list = [None] * len(self._cameras)
        self._P_matrix = None
        self.verbose = verbose

        self._normalized_coordinates = None

    def __repr__(self):
        return f"World(voxel={self.voxel}, CAMERA={self.cameras}, walls={self.walls})"

    def __copy__(self):
        return World(cameras=self.cameras, voxel=self.voxel)

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

    @property
    def cameras(self):
        return self._cameras

    @property
    def voxel(self):
        return self._voxel

    @property
    def walls(self):
        return self._walls

    @property
    def coordinate_type(self):
        return self._coordinate_type

    @property
    def coordinate_parameters(self):
        return self._coordinate_parameters

    @property
    def normalized_coordinates(self):
        return (lambda x: x) if self._normalized_coordinates is None else self._normalized_coordinates

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
        if self._visible_voxels is None or len(self._visible_voxels) != len(self._cameras):
            self.find_visible_voxels()
        return self._visible_voxels

    @property
    def P_matrix(self):
        if None in self._P_matrix_list:
            print("One or more P matrix is not calculated.")
            print("Please run set_projection_matrix method.")
            return
        elif (self._P_matrix is None) or (
                self._P_matrix.shape[0] != sum([P.shape[0] for P in self._P_matrix_list])):
            self._P_matrix = sparse.vstack(self._P_matrix_list)
        return self._P_matrix

    @property
    def P_matrix_list(self):
        return self._P_matrix_list

    @property
    def inside_voxels(self):
        return self._inside_voxels

    @inside_voxels.setter
    def inside_voxels(self, inside_voxels: np.ndarray):
        if not isinstance(inside_voxels, np.ndarray):
            raise TypeError(f"inside_voxels should be a np.ndarray, not {type(inside_voxels)}")
        if inside_voxels.shape != (self.voxel.N_voxel,):
            raise ValueError(f"inside_voxels should be a np.ndarray with the length of the number of voxels, "
                             f"not {inside_voxels.shape}")
        self._inside_voxels = inside_voxels
        self._visible_voxels = [vv * self._inside_voxels for vv in self.visible_voxels]

    @property
    def inside_vertices(self):
        return self._inside_vertices

    @inside_vertices.setter
    def inside_vertices(self, inside_vertices: np.ndarray):
        if not isinstance(inside_vertices, np.ndarray):
            raise TypeError(f"inside_voxels should be a np.ndarray, not {type(inside_vertices)}")
        if inside_vertices.shape != (self.voxel.N_voxel,):
            raise ValueError(f"inside_voxels should be a np.ndarray with the length of the number of voxels, "
                             f"not {inside_vertices.shape}")
        self._inside_voxels = inside_vertices
        self._visible_voxels = [vv * self._inside_voxels for vv in self.visible_voxels]

    @cameras.setter
    def cameras(self, cameras: list[Camera] | Camera):
        """Set the cameras

        Parameters
        ----------
        cameras
            The camera objects
        """
        cameras = type_check_and_list(cameras, Camera)
        camera_index = []
        visible_voxels = []
        projection = []
        P_matrix_list = []

        for c_, camera in enumerate(cameras):
            if camera in self._cameras:
                # get the index of the camera in the original camera list
                ind = self._cameras.index(camera)
                # keep the previous visible voxels, projection, and P matrix
                visible_voxels.append(self._visible_voxels[ind])
                projection.append(self._projection[ind])
                P_matrix_list.append(self._P_matrix_list[ind])
                camera_index.append(c_)
            else:
                # if the camera is not in the original camera list,
                # set the visible voxels, projection, and P matrix to None
                visible_voxels.append(None)
                projection.append(None)
                P_matrix_list.append(None)
                camera_index.append(None)

        self._cameras = cameras
        self._visible_voxels = visible_voxels
        self._projection = projection
        self._P_matrix_list = P_matrix_list
        self._P_matrix = None

        if all([_ is None for _ in camera_index]):
            print("Notice: All cameras are updated.")
        elif None in camera_index:
            # Notice: *th, *th, and *th cameras are reused.
            reused_index = [f"{i + 1}th" for i in camera_index if i is not None]
            xth = ", ".join(reused_index[:-1]) + " and " + reused_index[-1] if \
                len(reused_index) > 1 else f"{reused_index[0]}"
            print(f"Notice: {xth} camera{'s are' if len(reused_index) > 1 else ' is'} reused.")
        else:
            # All cameras are reused.
            print("Notice: All cameras are reused.")

    def add_camera(self, new_camera: Camera | list[Camera]):
        """Add a camera

        Parameters
        ----------
        new_camera: Camera | list[Camera]
            The camera object to add
        """
        new_camera = type_check_and_list(new_camera, Camera)
        self.cameras = self._cameras + new_camera

    def remove_camera(self, index: int | list[int]):
        """Remove a camera

        Parameters
        ----------
        index: int
            The index of the camera to remove
        """
        index = type_check_and_list(index, int)
        self.cameras = [c for i, c in enumerate(self._cameras) if i not in index]

    def change_camera(self, index: int | list[int],
                      camera: Camera | list[Camera]):
        """Change a camera

        Parameters
        ----------
        index: int
            The index of the camera to change
        camera: Camera
            The new camera object
        """
        index = type_check_and_list(index, int)
        camera = type_check_and_list(camera, Camera)

        self.cameras = [c if i not in index else camera.pop(0) for i, c in enumerate(self._cameras)]

    @voxel.setter
    def voxel(self, voxel):
        if not isinstance(voxel, Voxel):
            raise TypeError(f"voxel should be a Voxel, not {type(voxel)}")
        if voxel != self._voxel:
            my_print("Voxel is updated.", show=self.verbose > 0)
            self._voxel = voxel
            self._visible_voxels = [None] * len(self._cameras)
            self._projection = [None] * len(self._cameras)
            self._P_matrix_list = [None] * len(self._cameras)
            self._P_matrix = None

    @walls.setter
    def walls(self, walls: list[mesh.Mesh] | mesh.Mesh):
        walls = type_check_and_list(walls, mesh.Mesh)
        if walls != self._walls:
            self._walls = walls
            self._visible_voxels = [None] * len(self._cameras)
            if self._walls:
                for wall in self._walls:
                    wall.update_min()
                    wall.update_max()
                self._wall_ranges = zip(np.min([_.min_ for _ in self._walls], axis=0),
                                        np.max([_.max_ for _ in self._walls], axis=0))

    def set_inside_voxels_from_axis(self, axis: np.ndarray):
        """Set the inside voxels from the axis

        Parameters
        ----------
        axis: np.ndarray
            The axis of the inside voxels (n_points, 3)
        """

        def inside(start):
            """Check if the start point is inside the walls

            Parameters
            ----------
            start: np.ndarray
                The start point (3, )

            Returns
            -------
            inside: bool
                Whether the start point is inside the walls (N_voxel, )
            """
            inside_vertices = np.all(
                [stl_utils.check_visible(mesh_obj=wall, start=start, grid_points=self.voxel.grid, verbose=1) for wall in
                 self.walls], axis=0)[self.voxel.vertices_indices]
            return np.any(inside_vertices, axis=1)

        self._inside_voxels = np.any(np.apply_along_axis(inside, axis=1, arr=axis), axis=0)

    def find_visible_voxels(self, force=False, verbose=None):
        """Return the conditions of visible voxels

        Notes
        -----
        The condition is a boolean array with the length of the number of voxels.
        If more than one vertex of the voxel is visible, the voxel is defined as visible.
        """
        verbose = self.verbose if verbose is None else verbose
        if force:
            print("Force to calculate visible voxels.")
            self._visible_voxels = [None] * len(self.cameras)

        for c_, camera in enumerate(self.cameras):
            if self._visible_voxels[c_] is not None:
                print(f"Visible voxels for camera {c_ + 1}/{len(self.cameras)} is already calculated.")
                continue
            else:
                print(f"Finding visible voxels for camera {c_ + 1}/{len(self.cameras)}")
                visible_vertices = self.find_visible_points(self.voxel.grid, cameras=camera, verbose=verbose)[0]
                conditions_any = np.any(visible_vertices[:, self.voxel.vertices_indices], axis=-1).astype(int)
                conditions_all = np.all(visible_vertices[:, self.voxel.vertices_indices], axis=-1).astype(int)
                self._visible_voxels[c_] = conditions_any + conditions_all

    def find_visible_points(self, points: np.ndarray, cameras: list[Camera] | Camera = None,
                            eye_num: int | slice = None, verbose: int = 1) -> list[np.ndarray]:
        """Return the conditions of visible voxels

        Parameters
        ----------
        points: np.ndarray (N_points, 3)
            The points to check visibility
        cameras: list[Camera] | Camera, optional (default is None)
            The camera objects
        eye_num: int | slice, optional (default is None)
            The index of the eye
        verbose: int, optional (default is 1)
            The verbose level

        Returns
        -------
        visible: list[np.ndarray] (N_camera, N_eye, N_points)
            The conditions of visible voxels for each camera.
            If lines from the eye to the point are not intersected with any mesh (aperture or wall),
            the points are defined as visible (True).

        Notes
        -----
        The condition is a boolean array with the length of the number of voxels.
        If more than one vertex of the voxel is visible, the voxel is defined as visible.
        """
        visible_list = []
        if cameras is None:
            cameras = self.cameras
        else:
            cameras = type_list(cameras, Camera)

        if eye_num is None:
            eye_slice = slice(None)
        elif isinstance(eye_num, int):
            eye_slice = slice(eye_num, eye_num + 1)
        elif isinstance(eye_num, slice):
            eye_slice = eye_num
        else:
            raise TypeError(f"eye_num should be an int or a slice, not {type(eye_num)}")

        for _c, _camera in enumerate(cameras):
            walls_in_camera = [stl_utils.copy_model(wall, -_camera.camera_position, _camera.rotation_matrix.T) for
                               wall in self.walls]
            condition_list = []  # condition for each eye
            camera_grid = _camera.world2camera(points)
            for _e, _eye in enumerate(_camera.eyes[eye_slice]):
                visible = camera_grid[:, 2] > _eye.position[-1]
                # check if the voxel is in front of the camera (N_points, )

                my_print(f"checking visible points for eye {_e + 1}/{len(_camera.eyes)}",
                         show=verbose > 0)
                my_print("-" * 15, show=verbose > 0)
                # get conditions for each aperture and wall
                my_print(f"--- checking for apertures ---", show=verbose > 0)
                for a, aperture in enumerate(_camera.apertures):
                    if aperture.stl_model is None:
                        aperture.set_model()
                    visible *= stl_utils.check_visible(mesh_obj=aperture.stl_model, start=_eye.position,
                                                       grid_points=camera_grid, verbose=verbose)  # (N_points, )
                    my_print(f"{a + 1}/{len(_camera.apertures)} done", show=verbose > 0)
                    my_print("-" * 15, show=verbose > 0)

                my_print("--- checking for walls ---", show=verbose > 0)
                for w, wall_in_camera in enumerate(walls_in_camera):
                    visible *= stl_utils.check_visible(mesh_obj=wall_in_camera, start=_eye.position,
                                                       grid_points=camera_grid, verbose=verbose)  # (N_points, )
                    my_print(f"{w + 1}/{len(walls_in_camera)} done", show=verbose > 0)
                    my_print("-" * 15, show=verbose > 0)

                condition_list.append(visible)  # (N_eye, N_points)

            visible_list.append(np.array(condition_list, dtype=bool))  # (N_camera, N_eye, N_points)
        return visible_list

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

    def set_projection_matrix(self, index: int | list[int] = None,
                              res: int = None, verbose: int = 1, parallel: int = -1,
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
        index = type_check_and_list(index, int, default=range(len(self.cameras)))

        for i in index:
            camera = self.cameras[i]
            if self._projection[i] is None or force:
                im_vec_list = self._calc_all_voxel_image(camera=camera, res=res, verbose=verbose, parallel=parallel)
                self._projection[i] = sum(im_vec_list)
                self._P_matrix_list[i] = camera.screen.transform_matrix @ self._projection[i]
            else:
                my_print(f"Projection matrix for camera {i + 1}/{len(self.cameras)} is already calculated.",
                         show=verbose > 0)

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
        c = camera_1.screen.xy2uv(eye.calc_rays(camera_1.apertures[0].position).xy)
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
