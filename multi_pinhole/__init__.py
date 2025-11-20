from .core import Rays, Eye, Aperture, Screen, Camera
from .voxel import Voxel, shifted_torus, helical_displacement, hollow, emission_profile
from .world import World


def cli():
    print("This is the multi_pinhole package. "
          "Use it as a library to create multi-pinhole camera simulations.")
