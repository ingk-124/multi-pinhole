"""Aperture geometry and STL freeze handling."""
from numbers import Number
from typing import List, Tuple, Union

import numpy as np
from stl import mesh
from typing_extensions import Literal

from .utils import stl_utils

Vector2DLike = Union[np.ndarray, List[Number], Tuple[Number, Number]]
Vector3DLike = Union[np.ndarray, List[Number], Tuple[Number, Number, Number]]

class Aperture:
    """The physical opening that limits light reaching an :class:`Eye`.

    An ``Aperture`` is described either by an analytic shape (circle,
    ellipse, or rectangle) or by an explicit STL mesh. When an analytic
    shape is given, the STL geometry used for visibility/occlusion checks is
    generated on demand via :meth:`set_model`. See ``__init__`` below for the
    full parameter reference.
    """

    def __init__(self,
                 shape: Literal["circle", "ellipse", "rectangle"] = None,
                 size: Union[Number, Vector2DLike] = None,
                 position: Vector3DLike = None,
                 direction: Vector3DLike = None,
                 stl_model: mesh.Mesh = None,
                 **stl_args):
        """Create an aperture object

        Parameters
        ----------
        position : Vector3DLike
            Aperture position in the camera coordinate system (origin is the center of the screen)
        direction : Vector3DLike
            Aperture direction in the camera coordinate system (default: [0, 0, 1])
        shape : Literal["circle", "ellipse", "rectangle"]
            Aperture shape
        size : Union[Number, Vector2DLike]
            Aperture size
            If shape is "circle", size must be float (radius)
            If shape is "ellipse", size must be array of two floats (semi-major axis a and semi-minor axis b)
            If shape is "rectangle", size must be array of two floats (height, width)
        """

        self._stl_model = None
        self._frozen = False
        self._position = np.array(position) if position is not None else np.array([0, 0, 0])
        self._direction = np.array(direction) if direction is not None else np.array([0, 0, 1])

        if isinstance(stl_model, mesh.Mesh):
            self._shape = "stl"
            self._size = None
            self._stl_model = stl_utils.copy_model(stl_model)
            self._stl_model.translate(self._position)
        else:
            if (shape is None) and (size is None):
                pass
            else:
                self._shape, self._size = stl_utils.shape_check(shape, size)
                self.set_model(**stl_args)


    def __eq__(self, other):
        """bool: Compare aperture geometry, ignoring linked STL mesh objects."""
        if isinstance(other, Aperture):
            for k in self.__dict__.keys():
                if k in ("_stl_model", "_frozen"):
                    continue
                elif not np.all(self.__dict__[k] == other.__dict__[k]):
                    return False
            return True
        else:
            return False

    def set_model(self, resolution: int = 20, max_size: Union[Number, Vector2DLike] = None):
        """Set stl model

        Parameters
        ----------
        resolution : int, optional (default is 20)
            Resolution of the stl model
        max_size : Union[Number, Vector2DLike], optional (default is None)
            Maximum size of the stl model
            If max_size is None, the size of the stl model is 1.5 times the size of the aperture

        Returns
        -------
        self : Aperture
        """
        self._ensure_mutable()
        if self._shape == "stl":
            return self

        self._stl_model = stl_utils.generate_aperture_stl(shape=self._shape, size=self._size,
                                                          resolution=resolution, max_size=max_size)
        self._stl_model.translate(self._position)
        return self

    @property
    def frozen(self):
        """bool: Whether this aperture's geometry is immutable."""
        return self._frozen

    def _ensure_mutable(self):
        if self._frozen:
            raise RuntimeError("Aperture geometry is frozen because its Camera is registered in a World")

    def freeze(self):
        """Freeze analytic geometry and the underlying STL data buffer."""
        if not self._frozen:
            self._position.setflags(write=False)
            self._direction.setflags(write=False)
            if isinstance(self._size, np.ndarray):
                self._size.setflags(write=False)
            if self._stl_model is not None:
                self._stl_model.data.setflags(write=False)
            self._frozen = True
        return self

    @property
    def position(self):
        """np.ndarray: Aperture center in camera coordinates as ``(x, y, z)``."""
        return self._position

    @property
    def direction(self):
        """np.ndarray: Unit vector indicating aperture normal in camera space."""
        return self._direction

    @property
    def shape(self):
        """str: Shape keyword such as ``"circle"``, ``"ellipse"``, ``"rectangle"`` or ``"stl"``."""
        return self._shape

    @property
    def size(self):
        """np.ndarray or None: Characteristic dimensions of the aperture opening in millimeters."""
        return self._size

    @property
    def stl_model(self):
        """mesh.Mesh or None: Triangulated aperture surface when generated from STL."""
        return self._stl_model

    def print_info(self):
        """None: Print the aperture's spatial configuration and dimensions."""
        print("position:", self.position)
        print("shape:", self.shape)
        print("size:", self.size)
