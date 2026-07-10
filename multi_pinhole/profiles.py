"""Synthetic emission-profile helper functions.

These functions are intentionally separate from :mod:`multi_pinhole.voxel` so
the voxel grid container can stay focused on geometry and interpolation.  They
operate on normalized coordinates such as those returned by
``Voxel.normalized_coordinates()``.
"""

import numpy as np


def shifted_torus(r, theta, phi, delta):
    R = r * np.cos(theta) + delta
    Z = r * np.sin(theta)
    r_shifted = np.sqrt(R ** 2 + Z ** 2)
    theta_shifted = np.arctan2(Z, R)
    return r_shifted, theta_shifted, phi


def helical_displacement(r, theta, phi, m_, n_, phi_0, d, r_1, xi_0):
    r_ = r * np.exp(m_ * theta * 1j)
    xi = xi_0 * np.exp(-(r / r_1) ** d) * np.exp(n_ * (phi - phi_0) * 1j)
    r_new_complex = r_ - xi
    r_new = np.abs(r_new_complex)
    theta = np.angle(r_new_complex)
    phi = phi

    return r_new, theta, phi


def hollow(r, A, p, q, h, w):
    f1 = np.maximum(1 - r ** p, 0) ** q
    f2 = np.exp(-(r / w) ** 2)
    return A * (f1 - h * f2)


def helical_axis(r, theta, phi, m_, n_, r_a, phi_0):
    psi = n_ / m_ * phi + phi_0
    dx = r_a * np.cos(psi)
    dy = r_a * np.sin(psi)
    _x = r * np.cos(theta)
    _y = r * np.sin(theta)
    r_new = np.sqrt((_x - dx) ** 2 + (_y - dy) ** 2)
    return r_new


def emission_profile(r, theta, phi, allow_negative=False, flat_inside=False, **params):
    m_ = params.get("m_", 1)
    n_ = params.get("n_", -1)
    delta = params.get("delta", 0)
    phi_0 = params.get("phi_0", 0)
    d = params.get("d", 2)
    r_1 = params.get("r_1", 0.5)
    xi_0 = params.get("xi_0", 0.1)
    A = params.get("A", 1)
    p = params.get("p", 2)
    q = params.get("q", 3)
    h = params.get("h", 0)
    w = params.get("w", 0.5)

    r_shifted, theta_shifted, phi_shifted = shifted_torus(r, theta, phi, delta)
    r_new, theta_new, phi_new = helical_displacement(r_shifted, theta_shifted, phi_shifted,
                                                     m_=m_, n_=n_, phi_0=phi_0, d=d, r_1=r_1, xi_0=xi_0)
    y = hollow(r_new, A=A, p=p, q=q, h=h, w=w)
    if flat_inside:
        r_flat = np.clip(hollow(r_shifted, A=A, p=p, q=q, h=h, w=w),
                         0, hollow(r_1, A=A, p=p, q=q, h=h, w=w))
        y = np.maximum(y, r_flat)
    if not allow_negative:
        y = np.maximum(y, 0)
    return y
