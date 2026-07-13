"""Optional GUI helpers for building and inspecting multi-pinhole worlds.

The data-model helpers in this package only require the core project
dependencies.  Import :mod:`multi_pinhole.gui.app` to launch the optional
VTK/trame application installed with ``multi-pinhole[gui]``.
"""

from .model import CameraPoseDraft, sample_view_boundary

__all__ = ["CameraPoseDraft", "sample_view_boundary"]
