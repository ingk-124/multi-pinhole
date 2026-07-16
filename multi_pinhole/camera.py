"""Camera composition, coordinate transforms, orientation, and drawing."""
from numbers import Number
from typing import List, Tuple, Union

import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from scipy import sparse
from scipy.spatial.transform import Rotation
from stl import mesh
from typing_extensions import Literal

from .aperture import Aperture
from .eye import Eye
from .rays import Rays
from .screen import Screen
from .utils import stl_utils
from .utils.my_stdio import my_tqdm

VectorLike = Union[np.ndarray, List[Number], Tuple[Number], Number]
Vector2DLike = Union[np.ndarray, List[Number], Tuple[Number, Number]]
Vector3DLike = Union[np.ndarray, List[Number], Tuple[Number, Number, Number]]
MatrixLike = Union[np.ndarray, List[List[Number]], Tuple[List[Number]], Tuple[List[Number]], Tuple[Tuple[Number]]]

class Camera:
    """A multi-pinhole camera combining eyes, apertures, and a screen.

    A ``Camera`` groups one or more :class:`Eye` instances (all sharing the
    same ``eye_type``), their :class:`Aperture` geometries, and a single
    :class:`Screen`, and positions/orients the whole assembly within the
    world coordinate system (see :meth:`world2camera`). Ray bundles are
    produced per eye via :meth:`calc_image_vec`, which projects world points
    through the requested eye, applies aperture visibility checks, and
    rasterizes the result onto the screen. See ``__init__`` below for the
    full parameter reference.
    """

    def __init__(self,
                 eyes: List[Eye],
                 apertures: Union[Aperture, List[Aperture]],
                 screen: Screen,
                 camera_position: Tuple[float, float, float],
                 rotation_matrix: np.ndarray = None,
                 camera_name: str = None):
        """Camera

        Parameters
        ----------
        eyes : List[Eye]
            list of eyes
        apertures : Aperture or List[Aperture]
            aperture or list of apertures
        screen : Screen
            screen
        camera_position : Tuple[float, float, float]
            camera position in the world coordinate system (origin of the camera coordinate system)
        rotation_matrix : np.ndarray, optional
            rotation matrix from the world coordinate system to the camera coordinate system, by default None
        camera_name : str, optional
            camera name, by default None

        Raises
        ------
        ValueError
            if every eye_type is not 'pinhole' or 'concave_lens'
            if Eyes are not located on the same plane when eye_type is 'concave_lens'

        """

        self._camera_name = camera_name if camera_name is not None else "camera"
        self._world = None
        self._worlds = []
        self._frozen = False
        self._eyes = list(eyes) if isinstance(eyes, (list, tuple)) else [eyes, ]
        if len({eye.eye_type for eye in self._eyes}) == 1:
            self._eye_type = self._eyes[0].eye_type
        else:
            raise ValueError("eye_type of all eyes should be the same")
        self._apertures = list(apertures) if isinstance(apertures, (list, tuple)) else [apertures, ]
        self._screen = screen
        self._camera_position = np.array(camera_position, dtype=float, copy=True)
        self._rotation_matrix = (np.eye(3) if rotation_matrix is None
                                 else np.array(rotation_matrix, dtype=float, copy=True))

    @classmethod
    def single_pinhole(cls,
                       focal_length: float,
                       eye_size: Union[float, Vector2DLike],
                       screen_size: Union[Number, Vector2DLike],
                       pixel_shape: Tuple[int, int],
                       apertures: Union[Aperture, List[Aperture]],
                       *,
                       eye_shape: Literal["circle", "ellipse", "rectangle"] = "circle",
                       screen_shape: Literal["circle", "ellipse", "square", "rectangle"] = "square",
                       subpixel_resolution: int = 1,
                       wavelength_range: Tuple[float, float] = (0.01, 0.1),
                       camera_name: str = None):
        """Create a single-pinhole camera in its local reference pose.

        The screen center is the camera origin, the eye is centered at
        ``(X, Y) = (0, 0)``, the world-space camera position is the origin,
        and the world-to-camera rotation is the identity matrix.  The
        resulting camera can be placed with :meth:`set_camera_position`,
        :meth:`set_rotation_euler`, and the translation methods.

        Parameters are forwarded to :class:`Eye` and :class:`Screen`; the
        aperture geometry remains explicit because it is specific to the
        physical camera being modeled.
        """
        eye = Eye(position=(0.0, 0.0),
                  focal_length=focal_length,
                  eye_type="pinhole",
                  eye_size=eye_size,
                  eye_shape=eye_shape,
                  wavelength_range=wavelength_range)
        screen = Screen(screen_shape=screen_shape,
                        screen_size=screen_size,
                        pixel_shape=pixel_shape,
                        subpixel_resolution=subpixel_resolution)
        return cls(eyes=eye,
                   apertures=apertures,
                   screen=screen,
                   camera_position=(0.0, 0.0, 0.0),
                   rotation_matrix=np.eye(3),
                   camera_name=camera_name)

    def __repr__(self):
        """str: Render a concise textual summary of the camera configuration."""
        return f"Camera(eye_type={self._eye_type}, camera_position={self._camera_position})"

    def __eq__(self, other):
        """bool: Equality comparison based on optics, sensors, and pose."""
        if not isinstance(other, Camera):
            return False
        else:
            for k in self.__dict__.keys():
                if k in ("_world", "_worlds", "_frozen"):
                    continue
                elif k == "_eyes":
                    if len(self.__dict__[k]) != len(other.__dict__[k]):
                        return False
                    elif not all([eye1 == eye2 for eye1, eye2 in zip(self.__dict__[k], other.__dict__[k])]):
                        return False
                elif k == "_apertures":
                    if len(self.__dict__[k]) != len(other.__dict__[k]):
                        return False
                    elif not all([aperture1 == aperture2 for aperture1, aperture2 in
                                  zip(self.__dict__[k], other.__dict__[k])]):
                        return False
                elif k == "_screen":
                    if self.__dict__[k] != other.__dict__[k]:
                        return False
                else:
                    if not np.all(self.__dict__[k] == other.__dict__[k]):
                        return False
            d1 = self.__dict__.copy()
            d1.pop("_world")
            d1.pop("_worlds")
            d1.pop("_frozen")
            d1.pop("_eyes")
            d1.pop("_apertures")
            d1.pop("_screen")

            d2 = other.__dict__.copy()
            d2.pop("_world")
            d2.pop("_worlds")
            d2.pop("_frozen")
            d2.pop("_eyes")
            d2.pop("_apertures")
            d2.pop("_screen")

            return all([np.all(v1 == v2) for (v1, v2) in zip(d1.values(), d2.values())]) and \
                all([eye1 == eye2 for eye1, eye2 in zip(self._eyes, other._eyes)]) and \
                all([aperture1 == aperture2 for aperture1, aperture2 in zip(self._apertures, other._apertures)]) and \
                self._screen == other._screen

    @property
    def eye_type(self):
        """str: Shared optical mode for all eyes (``"pinhole"`` or ``"concave_lens"``)."""
        return self._eye_type

    @property
    def eyes(self):
        """tuple[Eye, ...]: Eyes mounted on the camera, exposed read-only."""
        return tuple(self._eyes)

    @property
    def apertures(self):
        """tuple[Aperture, ...]: Aperture geometries, exposed read-only."""
        return tuple(self._apertures)

    @property
    def screen(self):
        """Screen: Display surface receiving projected rays."""
        return self._screen

    @property
    def camera_position(self):
        """numpy.ndarray: camera position in the world coordinate system."""
        return self._camera_position

    @property
    def camera_x(self):
        """numpy.ndarray: camera right direction in the world coordinate system."""
        # X-direction of the camera coordinate system in the world coordinate system
        return self.rotation_matrix.T @ np.array([1, 0, 0]).ravel()

    @property
    def camera_y(self):
        """numpy.ndarray: camera Y (image-down) direction in world coordinates."""
        # Y-direction of the camera coordinate system in the world coordinate system
        return self.rotation_matrix.T @ np.array([0, 1, 0]).ravel()

    @property
    def camera_z(self):
        """numpy.ndarray: camera look direction in the world coordinate system."""
        # Z-direction of the camera coordinate system in the world coordinate system
        return self.rotation_matrix.T @ np.array([0, 0, 1]).ravel()

    @property
    def rotation_matrix(self):
        """numpy.ndarray: rotation matrix from the world coordinate system to the camera coordinate system."""
        return self._rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, rotation_matrix):
        """None: Override the world-to-camera rotation matrix.

        Parameters
        ----------
        rotation_matrix : np.ndarray
            ``3×3`` orthonormal rotation transforming world vectors into camera coordinates.
        """
        self._ensure_mutable()
        self.set_rotation_matrix(rotation_matrix)

    @property
    def world(self):
        """World or None: Most recently registered world, if any."""
        return self._world

    @property
    def worlds(self):
        """tuple[World, ...]: Worlds currently sharing this frozen camera."""
        return tuple(self._worlds)

    @property
    def frozen(self):
        """bool: Whether this camera and its optical geometry are immutable."""
        return self._frozen

    def _ensure_mutable(self):
        if self._frozen:
            raise RuntimeError(
                "Camera geometry is frozen because it is registered in a World; "
                "create a new Camera and use World.change_camera()"
            )

    def freeze(self):
        """Freeze this camera and all Eye, Screen, and Aperture geometry."""
        if not self._frozen:
            for eye in self._eyes:
                eye.freeze()
            self._screen.freeze()
            for aperture in self._apertures:
                aperture.freeze()
            self._camera_position.setflags(write=False)
            self._rotation_matrix.setflags(write=False)
            self._frozen = True
        return self

    def set_world(self, world_obj):
        """None: Register the :class:`World` scene providing geometry and emitters.

        Parameters
        ----------
        world_obj : World
            World instance this camera observes.
        """
        if world_obj is None:
            self._worlds.clear()
            self._world = None
            return
        if not any(world is world_obj for world in self._worlds):
            self._worlds.append(world_obj)
        self._world = world_obj
        self.freeze()

    def unset_world(self, world_obj):
        """Detach one World while keeping the camera permanently frozen."""
        self._worlds = [world for world in self._worlds if world is not world_obj]
        self._world = self._worlds[-1] if self._worlds else None

    def set_camera_position(self, camera_position):
        """Set the camera origin to an absolute world-coordinate position.

        Parameters
        ----------
        camera_position : Tuple[float, float, float]
            camera position in the world coordinate system

        Returns
        -------
        Camera
            The camera instance for fluent-style chaining.
        """
        self._ensure_mutable()
        camera_position = np.array(camera_position, dtype=float, copy=True)
        if camera_position.shape != (3,):
            raise ValueError("camera_position must be a 3D vector")
        self._camera_position = camera_position
        return self

    def translate_world(self, offset):
        """Translate the camera by an offset expressed in world coordinates."""
        self._ensure_mutable()
        offset = np.asarray(offset)
        if offset.shape != (3,):
            raise ValueError("offset must be a 3D vector")
        self._camera_position = self._camera_position + offset
        return self

    def translate_camera(self, offset):
        """Translate the camera by an offset expressed in camera coordinates.

        This operation uses the current rotation, so it generally does not
        commute with changing the camera orientation.
        """
        self._ensure_mutable()
        offset = np.asarray(offset)
        if offset.shape != (3,):
            raise ValueError("offset must be a 3D vector")
        self._camera_position = self._camera_position + self.rotation_matrix.T @ offset
        return self

    def set_rotation_euler(self, order, angle, degrees=True):
        """Set the absolute world-to-camera rotation using Euler angles.

        Parameters
        ----------
        order : str
            order of rotation, e.g. 'xyz'
        angle : float
            angle of rotation in degree
        degrees : bool, optional
            if True, angle is in degree, else in radian, by default True

        Returns
        -------
        Camera
            The camera instance with the updated rotation matrix.
        """
        self._ensure_mutable()
        self._rotation_matrix = Rotation.from_euler(order, angle, degrees=degrees).as_matrix()
        return self

    def set_rotation_matrix(self, rotation_matrix):
        """Set the absolute world-to-camera rotation matrix."""
        self._ensure_mutable()
        matrix = np.array(rotation_matrix, dtype=float, copy=True)
        if matrix.shape != (3, 3):
            raise ValueError("rotation_matrix must have shape (3, 3)")
        if not np.allclose(matrix @ matrix.T, np.eye(3), rtol=0.0, atol=1e-10):
            raise ValueError("rotation_matrix must be orthonormal")
        if not np.isclose(np.linalg.det(matrix), 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("rotation_matrix must have determinant +1")
        self._rotation_matrix = matrix
        return self

    def set_orientation(self, look, *, right=None, down=None):
        """Set camera orientation from axes expressed in world coordinates.

        Parameters
        ----------
        look : Vector3DLike
            Camera Z direction (screen normal pointing toward the scene).
        right : Vector3DLike, optional
            Approximate camera X direction (image-right) in world coordinates.
        down : Vector3DLike, optional
            Approximate camera Y direction (image-down) in world coordinates.

        Notes
        -----
        Exactly one of ``right`` and ``down`` must be supplied.  The given
        lateral axis is projected onto the plane perpendicular to ``look``
        before the remaining axis is formed, which removes small CAD or
        floating-point non-orthogonality while preserving a right-handed
        camera frame.
        """
        if (right is None) == (down is None):
            raise ValueError("exactly one of right and down must be provided")

        def normalized(vector, name):
            vector = np.asarray(vector, dtype=float)
            if vector.shape != (3,):
                raise ValueError(f"{name} must be a 3D vector")
            norm = np.linalg.norm(vector)
            if not np.isfinite(norm) or norm <= 1e-12:
                raise ValueError(f"{name} must be a non-zero finite vector")
            return vector / norm

        z_axis = normalized(look, "look")
        if right is not None:
            x_candidate = np.asarray(right, dtype=float)
            if x_candidate.shape != (3,):
                raise ValueError("right must be a 3D vector")
            x_axis = normalized(x_candidate - np.dot(x_candidate, z_axis) * z_axis, "right")
            y_axis = np.cross(z_axis, x_axis)
        else:
            y_candidate = np.asarray(down, dtype=float)
            if y_candidate.shape != (3,):
                raise ValueError("down must be a 3D vector")
            y_axis = normalized(y_candidate - np.dot(y_candidate, z_axis) * z_axis, "down")
            x_axis = np.cross(y_axis, z_axis)

        return self.set_rotation_matrix(np.stack([x_axis, y_axis, z_axis]))

    def set_orientation_from_points(self, look_point, *, right_point=None, down_point=None):
        """Set orientation from world-coordinate points viewed from the camera.

        Parameters
        ----------
        look_point : Vector3DLike
            A world-coordinate point in the camera's look direction.
        right_point : Vector3DLike, optional
            A world-coordinate point in the camera's image-right direction.
        down_point : Vector3DLike, optional
            A world-coordinate point in the camera's image-down direction.

        Notes
        -----
        Exactly one of ``right_point`` and ``down_point`` must be supplied.
        Each point is converted to a direction by subtracting the current
        :attr:`camera_position`, then :meth:`set_orientation` performs the
        orthonormalization.  Set the camera position before calling this
        method.
        """
        if (right_point is None) == (down_point is None):
            raise ValueError("exactly one of right_point and down_point must be provided")

        def direction_to(point, name):
            point = np.asarray(point, dtype=float)
            if point.shape != (3,):
                raise ValueError(f"{name} must be a 3D point")
            return point - self.camera_position

        look = direction_to(look_point, "look_point")
        right = None if right_point is None else direction_to(right_point, "right_point")
        down = None if down_point is None else direction_to(down_point, "down_point")
        return self.set_orientation(look, right=right, down=down)

    def world2camera(self, points):
        """transform points from the world coordinate system to the camera coordinate system

        Parameters
        ----------
        points : numpy.ndarray
            points in the world coordinate system (shape: (n, 3))

        Returns
        -------
        numpy.ndarray
            points in the camera coordinate system (shape: (n, 3))
        """
        return (self.rotation_matrix @ (points - self.camera_position[None, :]).T).T

    def add_eye(self, eye):
        """Add an eye to the camera.

        Parameters
        ----------
        eye : Eye
            an eye object

        Raises
        ------
        ValueError
            if eye_type of the new eye is different from the other eyes
        """
        self._ensure_mutable()
        if self._eye_type is None:
            self._eye_type = eye.eye_type
        elif self._eye_type != eye.eye_type:
            raise ValueError("eye_type of the new eye is different from the other eyes")
        self._eyes.append(eye)

    def add_aperture(self, aperture):
        """Add an aperture to the camera.

        Parameters
        ----------
        aperture : Aperture
            an aperture object
        """
        self._ensure_mutable()
        self._apertures.append(aperture)

    def calc_image_vec(self, eye_num, points, verbose: int = 0, check_visibility: bool = True,
                       etendue_per_subpixel=None):
        """sparse.csr_matrix: Assemble ray hits into a sparse image vector.

        Parameters
        ----------
        eye_num : int
            index of the eyes
        points : numpy.ndarray
            points in the world coordinate system (shape: (n, 3))
        verbose : int, optional
            Verbosity level forwarded to progress reporters.
        check_visibility : bool, optional
            Whether to cull rays occluded by apertures before projection.
        etendue_per_subpixel : np.ndarray, optional
            Deprecated compatibility argument passed to
            :meth:`Screen.ray2image_grid`; local finite-Eye etendue is now
            evaluated from each source ray and Eye position.

        Returns
        -------
        sparse.csr_matrix
            Image mapping matrix ``(N_subpixel, n)`` describing which subpixels are reached by each ray.
            The intensity information must be calculated separately.
        """
        eye = self._eyes[eye_num]
        points_in_camera = self.world2camera(points)
        if check_visibility:
            # Apertures are treated as blocking surfaces: a ray is usable only
            # when it avoids every aperture mesh.
            visible_list = [stl_utils.check_visible(mesh_obj=aperture.stl_model,
                                                    start=eye.position,
                                                    grid_points=points_in_camera,
                                                    behind_start_included=True) for aperture in self._apertures]
            visible = np.all(visible_list, axis=0) if visible_list else np.ones(points_in_camera.shape[0], dtype=bool)
            rays = eye.calc_rays(points_in_camera, visible)
        else:
            rays = eye.calc_rays(points_in_camera)

        # print("ray2image start")
        # res1 = self.screen.ray2image(eye, rays, parallel=parallel)
        # res2 = self.screen.ray2image2(eye, rays, parallel=parallel)
        mat = self.screen.ray2image_grid(
            eye, rays, verbose=verbose,
            etendue_per_subpixel=etendue_per_subpixel,
        )
        return mat

    def draw_optical_system(self, ax=None, show_focal_length=True, show_aperture=True, show_screen=True,
                            X_lim=None, Y_lim=None, Z_lim=None):
        """matplotlib.axes.Axes: Visualise optical elements in a 3D Matplotlib scene.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target 3D axis; a new figure is created when omitted.
        show_focal_length : bool, optional
            Toggle rendering of eye focal length vectors.
        show_aperture : bool, optional
            Toggle drawing aperture planes and models.
        show_screen : bool, optional
            Toggle rendering of the imaging screen.
        X_lim : Tuple[float, float], optional
            Manual bounds for the x-axis of the plot.
        Y_lim : Tuple[float, float], optional
            Manual bounds for the y-axis of the plot.
        Z_lim : Tuple[float, float], optional
            Manual bounds for the z-axis of the plot.

        Returns
        -------
        matplotlib.axes.Axes
            Axis containing the rendered optical setup.
        """
        # draw the optical system
        # note: axes in the figure are not equal to axes in the camera coordinate system
        # x-axis in the figure is Z-axis in the camera coordinate system
        # y-axis in the figure is X-axis in the camera coordinate system
        # z-axis in the figure is -Y-axis in the camera coordinate system
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')

        # set axes limits
        X_lim = (min([*[eye.position[0] for eye in self._eyes],
                      *[aperture.stl_model.x.min() if aperture.stl_model else
                        -aperture.size[0] / 2 + aperture.position[0] for aperture in self._apertures],
                      -self._screen.screen_size[1] / 2]),
                 max([*[eye.position[0] for eye in self._eyes],
                      *[aperture.stl_model.x.max() if aperture.stl_model else
                        aperture.size[0] / 2 + aperture.position[0] for aperture in self._apertures],
                      self._screen.screen_size[1] / 2])) if X_lim is None else X_lim
        Y_lim = (min([*[eye.position[1] for eye in self._eyes],
                      *[aperture.stl_model.y.min() if aperture.stl_model else
                        -aperture.size[1] / 2 + aperture.position[1] for aperture in self._apertures],
                      -self._screen.screen_size[0] / 2]),
                 max([*[eye.position[1] for eye in self._eyes],
                      *[aperture.stl_model.y.max() if aperture.stl_model else
                        aperture.size[1] / 2 + aperture.position[1] for aperture in self._apertures],
                      self._screen.screen_size[0] / 2])) if Y_lim is None else Y_lim
        Z_lim = (0,
                 max(*[aperture.position[2] for aperture in self._apertures],
                     *[eye.position[2] for eye in self._eyes])) if Z_lim is None else Z_lim

        ax.set_xlim(-Z_lim[1] * 0.1, Z_lim[1] * 1.1)
        ax.set_ylim(1.1 * X_lim[0], 1.1 * X_lim[1])
        ax.set_zlim(1.1 * Y_lim[0], 1.1 * Y_lim[1])
        ax.invert_yaxis()
        ax.invert_zaxis()

        # set axes labels
        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        ax.set_zlabel("Y")

        ax.set_box_aspect((Z_lim[1] - Z_lim[0], X_lim[1] - X_lim[0], Y_lim[1] - Y_lim[0]))
        ax.set_title("Optical system")

        # draw the origin of the camera coordinate system
        ax.scatter(0, 0, 0, c="k", zdir="z")
        # draw eyes
        for eye in self._eyes:
            # draw eye position
            ax.scatter(eye.position[0], eye.position[1], eye.position[2], c="k", marker="x", s=50, zdir="x")
            # draw eye principal point
            ax.scatter(eye.principal_point[0], eye.principal_point[1], eye.principal_point[2], edgecolors="r",
                       facecolors="none", marker="o", s=50, zdir="x")
            if show_focal_length:
                # draw eye focal length
                ax.quiver(eye.principal_point[2], eye.principal_point[0], eye.principal_point[1],
                          -eye.focal_length, 0, 0, color="r")
            if self._eye_type == "pinhole":
                pass
            elif self._eye_type == "lens":
                # draw eye shape
                if eye.eye_shape == "rectangle":
                    # patch of rectangle
                    patch2d = Rectangle((eye.position[0] - eye.eye_size[0] / 2,
                                         eye.position[1] - eye.eye_size[1] / 2),
                                        eye.eye_size[0], eye.eye_size[1])
                else:
                    # patch of ellipse
                    patch2d = Ellipse((eye.position[0], eye.position[1]), eye.eye_size[0], eye.eye_size[1])
                # transform patch to 3D
                ax.add_collection3d(col=patch2d, zs=eye.position[2], zdir="x")
        # draw screen
        if show_screen:
            # draw screen shape
            # center of the screen is at the origin of the camera coordinate system
            if self._screen.screen_shape == "rectangle":
                # patch of rectangle
                patch2d = Rectangle(-np.array(self._screen.screen_size[::-1]) / 2, *self._screen.screen_size[::-1],
                                    facecolor="orange", edgecolor="k", alpha=0.5, linewidth=2)
            else:
                # patch of ellipse
                patch2d = Ellipse((0, 0), *self._screen.screen_size * 2,
                                  facecolor="orange", edgecolor="k", alpha=0.5, linewidth=2)
            # transform patch to 3D
            ax.add_patch(patch2d)
            art3d.pathpatch_2d_to_3d(patch2d, z=0, zdir="x")
        # draw apertures
        if show_aperture:
            for aperture in self._apertures:
                # draw aperture position
                ax.scatter(aperture.position[0], aperture.position[1], aperture.position[2], c="k", marker="x", s=100,
                           zdir="x")
                # show aperture stl model
                if aperture.stl_model is not None:
                    tmp_model = stl_utils.rotate_model(aperture.stl_model, matrix=[[0, 1, 0],
                                                                                   [0, 0, 1],
                                                                                   [1, 0, 0]])
                    stl_utils.show_stl(tmp_model, ax=ax, alpha=0.5, facecolors="orange", edgecolors="k", lw=0.5)
                else:
                    # draw aperture shape
                    if aperture.shape == "rectangle":
                        # patch of rectangle
                        patch2d = Rectangle((aperture.position[0] - aperture.size[0] / 2,
                                             aperture.position[1] - aperture.size[1] / 2),
                                            *aperture.size,
                                            facecolor="none", edgecolor="k", linewidth=2)
                    else:
                        # patch of ellipse
                        patch2d = Ellipse(aperture.position, *aperture.size * 2,
                                          facecolor="none", edgecolor="k", linewidth=2)
                    # TODO: aperture rotation (allow to set arbitrary normal vector) (future work)
                    # rotation matrix is not specified only one vector, so... how to rotate?
                    # ->
                    # transform patch to 3D
                    ax.add_patch(patch2d)
                    art3d.pathpatch_2d_to_3d(patch2d, z=aperture.position[2], zdir="x")

                    # draw camera coordinate system (length is 0.8*axes limit)
        _ = np.mean([ax.get_ylim()[0], ax.get_zlim()[0]]) * 0.8
        # X-axis in the camera coordinate system (y-axis in the figure)
        ax.quiver(0, 0, 0, 0, _, 0, color="r")
        # Y-axis in the camera coordinate system (-z-axis in the figure)
        ax.quiver(0, 0, 0, 0, 0, _, color="g")
        # Z-axis in the camera coordinate system (x-axis in the figure)
        ax.quiver(0, 0, 0, _, 0, 0, color="b")

        return ax

    # todo: make plotly version of draw_optical_system

    def draw_camera_orientation_plotly(self, fig=None, **kwargs):
        """go.Figure: Render camera axes within Plotly for interactive viewing.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure, optional
            Existing figure to append orientation glyphs to.
        **kwargs
            Additional keyword arguments forwarded to :func:`stl_utils.plotly_show_axes`.

        Returns
        -------
        plotly.graph_objects.Figure
            Figure augmented with camera orientation geometry.
        """
        fig = go.Figure() if fig is None else fig
        stl_utils.plotly_show_axes(R=self.rotation_matrix, fig=fig, origin=self.camera_position, name="camera",
                                   **kwargs)
        return fig

    def draw_camera_orientation(self, ax=None):
        """matplotlib.axes.Axes: Plot camera axes relative to the world frame.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axis to draw on; a new orthographic subplot is created when omitted.

        Returns
        -------
        matplotlib.axes.Axes
            Axis showing both camera and world coordinate frames.
        """
        if ax is None:
            ax = plt.subplot(projection="3d", proj_type="ortho")
        # set axes limits
        lim = np.max(np.abs(self.camera_position))

        ax.set_xlim(-1.1 * lim, 1.1 * lim)
        ax.set_ylim(-1.1 * lim, 1.1 * lim)
        ax.set_zlim(-1.1 * lim, 1.1 * lim)
        ax.set_box_aspect((1, 1, 1))
        # set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # draw camera coordinate system in the world coordinate system
        # x, y, z axes in the world coordinate system is the same as axes in the figure
        # draw camera position
        ax.scatter(*self.camera_position, c="k", marker="x", s=100)
        arrow_length = np.linalg.norm(self.camera_position) * 0.2

        # draw camera X-axis
        ax.quiver(*self.camera_position, *(self.camera_x * arrow_length), color="r")
        # draw camera Y-axis
        ax.quiver(*self.camera_position, *(self.camera_y * arrow_length), color="g")
        # draw camera Z-axis
        ax.quiver(*self.camera_position, *(self.camera_z * arrow_length), color="b")

        # draw world coordinate system
        arrow_length = np.mean([ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]]) * 0.8
        ax.quiver(0, 0, 0, arrow_length, 0, 0, color="k")
        ax.quiver(0, 0, 0, 0, arrow_length, 0, color="k")
        ax.quiver(0, 0, 0, 0, 0, arrow_length, color="k")

        return ax

    def print_settings(self):
        """None: Emit a formatted summary of camera, eye, screen, and aperture settings."""
        print("Camera settings:")
        print(f"Camera position: {self._camera_position}")
        print(f"Eye type: {self._eye_type}")
        print(f"Eye position: {[eye.position for eye in self._eyes]}")
        print(f"Eye focal length: {[eye.focal_length for eye in self._eyes]}")
        print(f"Eye shape: {[eye.eye_shape for eye in self._eyes]}")
        print(f"Eye size: {[eye.eye_size for eye in self._eyes]}")
        print(f"Screen shape: {self._screen.screen_shape}")
        print(f"Screen size: {self._screen.screen_size}")
        print(f"pixel shape: {self._screen.pixel_shape}")
        print(f"pixel size: {self._screen.pixel_size}")
        print(f"Aperture position: {[aperture.position for aperture in self._apertures]}")
        print(f"Aperture shape: {[aperture.shape for aperture in self._apertures]}")
        print(f"Aperture size: {[aperture.size for aperture in self._apertures]}")
