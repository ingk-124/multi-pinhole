import itertools

import numpy as np
from scipy.special import jn, jn_zeros


def torus_fourier_bessel(k: int, m: int, n: int):
    """Fourier-Bessel functions f_{n, m, k}(r, theta, phi) in the torus coordinate system.

    Fourier-Bessel functions are defined as
    f_{n, m, k}(r, theta, phi) = J_n(k * lmd_n^k * r) * exp(i * m * theta) * exp(-i * n * phi)
                               = J_n(k * lmd_n^k * r) * exp(i * (m * theta - n * phi))
    where r, theta, phi are the normalized radius, poloidal angle, and toroidal angle respectively
    (0 <= r <= 1, 0 <= theta <= 2 * pi, 0 <= phi <= 2 * pi).
    lmd_n^k is the k-th zero of the Bessel function J_n.

    Parameters
    ----------
    k : int
        The radial wave number.
    m : int
        The poloidal wave number.
    n : int
        The toroidal wave number.

    Returns
    -------
    fb_kmn : callable
        The Fourier-Bessel function f_{n, m, k}(r, theta, phi).
    """
    lmd_nk = jn_zeros(m, k)[-1]

    def fb_kmn(r, theta, phi):
        """Fourier-Bessel function f_{n, m, k}(r, theta, phi).

        Parameters
        ----------
        r : np.ndarray
            The normalized radius. (0 <= r <= 1)
        theta : np.ndarray
            The poloidal angle. (0 <= theta <= 2 * pi)
        phi : np.ndarray
            The toroidal angle. (0 <= phi <= 2 * pi)

        Returns
        -------
        fb_kmn : np.ndarray
            The Fourier-Bessel function f_{n, m, k}(r, theta, phi).
        """
        return jn(m, k * lmd_nk * r) * np.exp(1j * (m * theta - n * phi))

    return fb_kmn


def basis_params(k_range: list, m_range: list, n_range: list):
    """
    make a list of basis parameters

    Parameters
    ----------
    k_range : list | int
        range of k or maximum value of k
    m_range : list | int
        range of m or maximum value of m
    n_range : list | int
        range of n or maximum value of n

    Returns
    -------
    basis_params : list
        list of basis parameters
    """
    k_ = range(k_range[0], k_range[1] + 1) if isinstance(k_range, list) else range(1, k_range + 1)
    m_ = range(m_range[0], m_range[1] + 1) if isinstance(m_range, list) else range(0, m_range + 1)
    n_ = range(n_range[0], n_range[1] + 1) if isinstance(n_range, list) else range(0, n_range + 1)
    parameters = itertools.product(k_, m_, n_)
    return list(parameters)


#     TODO: make 3d basis of Fourier-Bessel functions

if __name__ == '__main__':
    params = basis_params(2, 3, 4)

    from plot import volume_rendering, multi_volume_rendering
    import plotly.io as pio
    from multi_pinhole.voxel import Voxel, torus_coordinates

    pio.renderers.default = 'firefox'

    # 3d volume rendering
    # vox = Voxel(x_axis=np.hstack([np.linspace(-750, 0, 5, endpoint=False), np.linspace(0, 750, 31)]),
    #             y_axis=np.linspace(-750, 750, 21),
    #             z_axis=np.linspace(-250, 250, 21))
    vox = Voxel().uniform_axes(ranges=[[-750, 750], [-750, 750], [-250, 250]], shape=[40, 40, 20], show_info=True)
    r, theta, phi = torus_coordinates(major_radius=500, minor_radius=250)(vox.gravity_center).T
    inside = r < 1.0
    grid = vox.gravity_center

    fig = volume_rendering(inside * np.real(torus_fourier_bessel(1, 0, 0)(r, theta, phi)), grid,
                           surface_count=15, opacity=1, isomin=-1, isomax=1,
                           opacityscale=[[0, 1], [0.25, 0.8], [0.4, 0.5], [0.5, 0], [0.6, 0.5], [0.75, 0.8], [1, 1]],
                           caps=dict(x_show=False, y_show=False, z_show=False), colorscale="jet",
                           )
    fig.show()

    fig = multi_volume_rendering([[inside * np.real(torus_fourier_bessel(1, m, 0)(r, theta, phi)) for m in range(5)],
                                  [inside * np.real(torus_fourier_bessel(1, m, 2)(r, theta, phi)) for m in range(5)],
                                  [inside * np.real(torus_fourier_bessel(1, m, 5)(r, theta, phi)) for m in range(5)]],
                                 grid,
                                 surface_count=15, opacity=1, isomin=-1, isomax=1,
                                 opacityscale=[[0, 1], [0.25, 0.8], [0.4, 0.5], [0.5, 0], [0.6, 0.5], [0.75, 0.8],
                                               [1, 1]],
                                 caps=dict(x_show=False, y_show=False, z_show=False), colorscale="jet",
                                 )
    fig.show()

    params = basis_params(2, 3, 4)

