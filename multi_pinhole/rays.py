"""Ray bundle data structures for multi-pinhole projection calculations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rays:
    """Rays class

    Parameters
    ----------
    Z : np.ndarray
        distance from eye to light source along the main optical axis (n, )
    XY : np.ndarray
        position of the light spot on the screen (n, 2)
    zoom_rate : np.ndarray
        zoom rate of the light spot on the screen (n, )
        It expressed as 1 + f / Z.
    front_and_visible : np.ndarray
        boolean array (n, ) True if the point is in front of the eye and visible
    """

    Z: np.ndarray
    XY: np.ndarray
    zoom_rate: np.ndarray
    front_and_visible: np.ndarray

    @property
    def n(self):
        """int: Total number of sampled rays contained in this instance."""
        return self.Z.size

    @property
    def n_visible(self):
        """int: Count of rays that are both in front of the eye and marked visible."""
        return self.front_and_visible.nonzero()[0].size

    # allow indexing and slicing
    def __getitem__(self, key):
        # for slicing
        return Rays(Z=self.Z[key],
                    XY=self.XY[key],
                    zoom_rate=self.zoom_rate[key],
                    front_and_visible=self.front_and_visible[key])

    def __len__(self):
        return self.n

