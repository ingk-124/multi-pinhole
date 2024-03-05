import numpy as np

from multi_pinhole.voxel import Voxel, torus_coordinates


class Plasma(Voxel):
    """
    Plasma object

    Parameters
    ----------
    r : float
        The major radius of the torus.
    a : float
        The minor radius of the torus.
    n : int
        The number of voxels in the radial direction.
    m : int
        The number of voxels in the poloidal direction.
    l : int
        The number of voxels in the toroidal direction.
    """

    def __init__(self, x_axis=None, y_axis=None, z_axis=None):
        super().__init__(x_axis, y_axis, z_axis)
        self._Te, self._ne, self._Ti, self._ni = [np.zeros(self.N_grid)] * 4

    @property
    def Te(self):
        return self._Te

    @Te.setter
    def Te(self, Te):
        self._Te = Te

    @property
    def ne(self):
        return self._ne

    @ne.setter
    def ne(self, ne):
        self._ne = ne

    @property
    def Ti(self):
        return self._Ti

    @Ti.setter
    def Ti(self, Ti):
        self._Ti = Ti

    @property
    def ni(self):
        return self._ni

    @ni.setter
    def ni(self, ni):
        self._ni = ni

    def relative_xray_emission(self, photon_energy):
        """
        relative xray emission

        Parameters
        ----------
        photon_energy : np.ndarray
            photon energy in eV

        Returns
        -------
        relative_xray_emission : np.ndarray
            relative xray emission spectrum
        """
        return self._Te ** 0.5 * np.exp(-photon_energy / self._Te)

    def bremsstrahlung(self, photon_energy, Z):
        """
        Calculate bremsstrahlung spectrum

        Parameters
        ----------
        photon_energy : np.ndarray
            photon energy in eV

        Returns
        -------
        spectrum : np.ndarray
            bremsstrahlung spectrum

        Notes
        -----
        The bremsstrahlung spectrum is calculated as:
            spectrum = 1.42e-32 * Te ** 0.5 * ne * Z ** 2 * photon_energy ** -1.5 * np.exp(-photon_energy / Te)
        where Te is the electron temperature, ne is the electron density, and Z is the atomic number.
        """
        spectrum = 1.54E-29 * self._ne * self._ni * Z * self.relative_xray_emission(photon_energy)
        return spectrum



if __name__ == '__main__':
    from utils import plot

    plasma = Plasma().uniform_axes(ranges=[[-750, 750], [-750, 750], [-250, 250]], shape=[40, 40, 20], show_info=True)
    r, theta, phi = torus_coordinates(major_radius=500, minor_radius=250)(plasma.gravity_center).T
    plasma.Te = 120 * (1 - r ** 2)
    fig = plot.volume_rendering(plasma.Te, plasma.gravity_center,
                                surface_count=15, opacity=1, isomin=-1, isomax=150,
                                opacityscale=[[0, 1], [0.25, 0.8], [0.4, 0.5], [0.5, 0], [0.6, 0.5], [0.75, 0.8],
                                              [1, 1]],
                                caps=dict(x_show=False, y_show=False, z_show=False), colorscale="jet", )

    fig.show()