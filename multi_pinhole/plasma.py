"""Plasma emissivity model built on top of the Cartesian :class:`Voxel` grid.

``Plasma`` extends :class:`~multi_pinhole.voxel.Voxel` with per-voxel
electron/ion temperature and density fields, and derives simple X-ray
emission spectra from them (relative bremsstrahlung-like scaling and a full
bremsstrahlung formula). It is primarily used to generate synthetic emission
profiles for testing the imaging pipeline (see the ``__main__`` block below).
"""

import numpy as np

from multi_pinhole.voxel import Voxel, torus_coordinates


class Plasma(Voxel):
    """A :class:`~multi_pinhole.voxel.Voxel` grid carrying plasma fields.

    ``Plasma`` accepts the same axis arguments as its parent :class:`Voxel`
    class (``x_axis``, ``y_axis``, ``z_axis``); it does not take torus
    geometry parameters directly. Electron/ion temperature and density are
    stored as flat arrays over the voxel grid points (length ``N_grid``) and
    default to all zeros until assigned via the :attr:`Te`, :attr:`ne`,
    :attr:`Ti`, and :attr:`ni` properties.

    Parameters
    ----------
    x_axis : np.ndarray, optional
        x axis for the grid, forwarded to :class:`Voxel`.
    y_axis : np.ndarray, optional
        y axis for the grid, forwarded to :class:`Voxel`.
    z_axis : np.ndarray, optional
        z axis for the grid, forwarded to :class:`Voxel`.
    """

    def __init__(self, x_axis=None, y_axis=None, z_axis=None):
        """Initialize the voxel grid and zero-fill the plasma field arrays.

        Parameters
        ----------
        x_axis : np.ndarray, optional
            x axis for the grid, forwarded to :class:`Voxel`.
        y_axis : np.ndarray, optional
            y axis for the grid, forwarded to :class:`Voxel`.
        z_axis : np.ndarray, optional
            z axis for the grid, forwarded to :class:`Voxel`.
        """
        super().__init__(x_axis, y_axis, z_axis)
        self._Te, self._ne, self._Ti, self._ni = [np.zeros(self.N_grid)] * 4

    @property
    def Te(self):
        """np.ndarray: Electron temperature evaluated at each grid point."""
        return self._Te

    @Te.setter
    def Te(self, Te):
        """Set the electron temperature field (shape ``(N_grid,)``)."""
        self._Te = Te

    @property
    def ne(self):
        """np.ndarray: Electron density evaluated at each grid point."""
        return self._ne

    @ne.setter
    def ne(self, ne):
        """Set the electron density field (shape ``(N_grid,)``)."""
        self._ne = ne

    @property
    def Ti(self):
        """np.ndarray: Ion temperature evaluated at each grid point."""
        return self._Ti

    @Ti.setter
    def Ti(self, Ti):
        """Set the ion temperature field (shape ``(N_grid,)``)."""
        self._Ti = Ti

    @property
    def ni(self):
        """np.ndarray: Ion density evaluated at each grid point."""
        return self._ni

    @ni.setter
    def ni(self, ni):
        """Set the ion density field (shape ``(N_grid,)``)."""
        self._ni = ni

    def relative_xray_emission(self, photon_energy):
        """Compute the relative x-ray emission spectrum.

        Parameters
        ----------
        photon_energy : np.ndarray
            Photon energy values, expressed in electron volts (eV).

        Returns
        -------
        np.ndarray
            Relative x-ray emission intensity evaluated at ``photon_energy``.
        """
        return self._Te ** 0.5 * np.exp(-photon_energy / self._Te)

    def bremsstrahlung(self, photon_energy, Z):
        """Calculate the bremsstrahlung emission spectrum.

        Parameters
        ----------
        photon_energy : np.ndarray
            Photon energy values, expressed in electron volts (eV).
        Z : float
            Effective charge number of the emitting ion species.

        Returns
        -------
        np.ndarray
            Bremsstrahlung spectrum corresponding to ``photon_energy`` and ``Z``.

        Notes
        -----
        The bremsstrahlung spectrum is calculated as:
            ``spectrum = 1.54e-29 * ne * ni * Z * relative_xray_emission(photon_energy)``
            ``           = 1.54e-29 * ne * ni * Z * Te ** 0.5 * exp(-photon_energy / Te)``
        where ``Te`` is the electron temperature, ``ne`` is the electron density, ``ni`` is
        the ion density, and ``Z`` is the effective ion charge number.
        """
        spectrum = 1.54E-29 * self._ne * self._ni * Z * self.relative_xray_emission(photon_energy)
        return spectrum



if __name__ == '__main__':
    from multi_pinhole.utils import plot

    plasma = Plasma().uniform_axes(ranges=[[-750, 750], [-750, 750], [-250, 250]], shape=[40, 40, 20], show_info=True)
    r, theta, phi = torus_coordinates(major_radius=500, minor_radius=250)(plasma.gravity_center).T
    plasma.Te = 120 * (1 - r ** 2)
    fig = plot.volume_rendering(plasma.Te, plasma.gravity_center,
                                surface_count=15, opacity=1, isomin=-1, isomax=150,
                                opacityscale=[[0, 1], [0.25, 0.8], [0.4, 0.5], [0.5, 0], [0.6, 0.5], [0.75, 0.8],
                                              [1, 1]],
                                caps=dict(x_show=False, y_show=False, z_show=False), colorscale="jet", )

    fig.show()