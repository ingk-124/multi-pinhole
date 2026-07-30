"""Public optics, voxel, projection, and scene-orchestration API."""

from .__about__ import __version__
from .aperture import Aperture
from .camera import Camera
from .eye import Eye
from .projection import EyeProjectionWorkEstimate, ProjectionWorkEstimate
from .rays import Rays
from .screen import Screen
from .voxel import Voxel
from .world import World

__all__ = [
    "Aperture",
    "Camera",
    "Eye",
    "EyeProjectionWorkEstimate",
    "ProjectionWorkEstimate",
    "Rays",
    "Screen",
    "Voxel",
    "World",
    "__version__",
    "cli",
]


def cli() -> None:
    """Entry point for the ``multi-pinhole-sim`` console script.

    This package is intended to be used as a library; the console script
    (registered via ``pyproject.toml``'s ``[project.scripts]``) currently
    only prints a short usage hint and performs no simulation work itself.
    """
    print("This is the multi_pinhole package. "
          "Use it as a library to create multi-pinhole camera simulations.")
