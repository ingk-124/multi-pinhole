"""Multi-pinhole X-ray imaging simulation package.

This top-level package re-exports the public classes and helper functions
needed to build multi-pinhole camera simulations, so most users only need
``from multi_pinhole import ...``:

* Optics: :class:`~multi_pinhole.eye.Eye`, :class:`~multi_pinhole.aperture.Aperture`,
  :class:`~multi_pinhole.screen.Screen`, :class:`~multi_pinhole.camera.Camera`, and the
  :class:`~multi_pinhole.rays.Rays` data container returned by ray tracing.
* Scene modeling: :class:`~multi_pinhole.voxel.Voxel` (the Cartesian voxel grid)
  and :mod:`multi_pinhole.profiles` helpers for evaluating synthetic toroidal
  and poloidal profiles.
* Orchestration: :class:`~multi_pinhole.world.World`, which binds voxels, cameras,
  and optional walls into a simulation-ready scene.

See ``docs/overview.md`` for a walkthrough of the typical workflow.
"""

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
