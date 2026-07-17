"""Public optics, voxel, projection, and scene-orchestration API."""

from .core import Rays, Eye, Aperture, Screen, Camera
from .projection import EyeProjectionWorkEstimate, ProjectionWorkEstimate
from .voxel import Voxel
from .world import World


def cli():
    """Entry point for the ``multi-pinhole-sim`` console script.

    This package is intended to be used as a library; the console script
    (registered via ``pyproject.toml``'s ``[project.scripts]``) currently
    only prints a short usage hint and performs no simulation work itself.
    """
    print("This is the multi_pinhole package. "
          "Use it as a library to create multi-pinhole camera simulations.")
