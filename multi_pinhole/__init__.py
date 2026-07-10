from .core import Rays, Eye, Aperture, Screen, Camera
from .profiles import emission_profile, helical_axis, helical_displacement, hollow, shifted_torus
from .voxel import Voxel
from .world import World


def cli():
    print("This is the multi_pinhole package. "
          "Use it as a library to create multi-pinhole camera simulations.")
