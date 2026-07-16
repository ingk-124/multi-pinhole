"""Compatibility facade for the decomposed core optics modules.

The implementations live in ``multi_pinhole.eye``,
``multi_pinhole.aperture``, ``multi_pinhole.screen``, and
``multi_pinhole.camera``. Existing imports from this module remain
supported, including historical pickle references.
"""
from numbers import Number
from typing import List, Tuple, Union

import numpy as np

from .aperture import Aperture
from .camera import Camera
from .eye import Eye
from .rays import Rays
from .screen import (
    Screen,
    _local_etendue_density,
    _rasterize_spots,
    _rectangle_density_average,
    _spot_cell_local_etendue,
    _spot_cell_overlap,
    _unit_circle_primitive,
    _unit_circle_rectangle_overlap,
)
from .utils import stl_utils

VectorLike = Union[np.ndarray, List[Number], Tuple[Number], Number]
Vector2DLike = Union[np.ndarray, List[Number], Tuple[Number, Number]]
Vector3DLike = Union[np.ndarray, List[Number], Tuple[Number, Number, Number]]
MatrixLike = Union[
    np.ndarray,
    List[List[Number]],
    Tuple[List[Number]],
    Tuple[List[Number]],
    Tuple[Tuple[Number]],
]

__all__ = [
    "Rays",
    "Eye",
    "Aperture",
    "Screen",
    "Camera",
    "VectorLike",
    "Vector2DLike",
    "Vector3DLike",
    "MatrixLike",
]
