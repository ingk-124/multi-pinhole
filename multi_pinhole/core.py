"""Legacy-loading facade for historical optics import and pickle paths.

This module is not a public API for new code.  It intentionally retains only
class globals needed to resolve historical ``multi_pinhole.core.*`` imports
and pickle/dill payloads.  Use top-level ``multi_pinhole`` imports instead.
"""

from .aperture import Aperture
from .camera import Camera
from .eye import Eye
from .rays import Rays
from .screen import Screen

__all__ = [
    "Aperture",
    "Camera",
    "Eye",
    "Rays",
    "Screen",
]
