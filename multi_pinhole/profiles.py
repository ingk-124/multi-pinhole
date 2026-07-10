"""Synthetic emission-profile helper functions.

These functions are intentionally separate from :mod:`multi_pinhole.voxel` so
the voxel grid container can stay focused on geometry and interpolation.  They
operate on normalized coordinates such as those returned by
``Voxel.normalized_coordinates()``.
"""

import numpy as np


# TODO: Add a magnetic-surface-based emission helper driven by a Fourier
# spectrum of the flux-surface shape. Keep it as a pure profile model first:
# map (r, theta, phi) to a surface label rho_surf(theta, phi) from configurable
# Fourier coefficients, evaluate the radial emission on that label, then add a
# thin Voxel wrapper through evaluate_torus_profile/evaluate_poloidal_profile.
# Keep plotting, fitting, and diagnostics outside this core profile API.


def _as_array(value):
    return np.asarray(value, dtype=float)


def torus_to_poloidal_cartesian(r, theta, phi):
    """Convert normalized torus coordinates to poloidal Cartesian coordinates.

    Returns ``(x, y, phi)`` with ``x = r*cos(theta)`` and
    ``y = r*sin(theta)``.  ``phi`` is passed through unchanged so callers can
    keep the toroidal coordinate aligned with the returned poloidal plane.
    """
    x = _as_array(r) * np.cos(theta)
    y = _as_array(r) * np.sin(theta)
    return x, y, phi


def helical_phase(theta, phi, m_, n_, phi_0=0):
    """Return the helical phase used by the profile displacement helpers."""
    return m_ * _as_array(theta) - n_ * (_as_array(phi) - phi_0)


def helical_poloidal_coordinates(r, theta, phi, m_, n_, phi_0=0):
    """Convert ``(r, theta, phi)`` to ``(x, y, psi)`` helical coordinates.

    ``psi = m_*theta - n_*(phi - phi_0)`` is the phase of a helical
    perturbation relative to the poloidal angle.
    """
    x, y, _ = torus_to_poloidal_cartesian(r, theta, phi)
    psi = helical_phase(theta, phi, m_=m_, n_=n_, phi_0=phi_0)
    return x, y, psi


def torus_coordinates_from_voxel(voxel, points=None):
    """Return ``(r, theta, phi)`` from a torus-coordinate voxel.

    ``voxel`` must be configured with ``coordinate_type`` equal to ``"torus"``
    or ``"torus_inverse"``.  The optional ``points`` argument is passed through
    to ``voxel.normalized_coordinates(points)``.
    """
    if voxel.coordinate_type not in ("torus", "torus_inverse"):
        raise ValueError(
            "voxel.coordinate_type must be 'torus' or 'torus_inverse' "
            f"for torus profile evaluation, got {voxel.coordinate_type!r}."
        )
    r, theta, phi = voxel.normalized_coordinates(points).T
    return r, theta, phi


def evaluate_torus_profile(voxel, profile_func, *args, points=None, **kwargs):
    """Evaluate a ``profile_func(r, theta, phi, ...)`` directly on a voxel."""
    r, theta, phi = torus_coordinates_from_voxel(voxel, points=points)
    return profile_func(r, theta, phi, *args, **kwargs)


def evaluate_poloidal_profile(voxel, profile_func, *args, points=None, pass_phi=True, **kwargs):
    """Evaluate a ``profile_func(x, y, ...)`` directly on a torus-coordinate voxel.

    When ``pass_phi`` is true, the toroidal angle is passed as a keyword
    argument ``phi`` unless the caller already supplied ``phi`` in ``kwargs``.
    This matches the non-axisymmetric profile helpers in this module while
    still allowing axisymmetric helpers to ignore ``phi`` through ``**kwargs``.
    """
    r, theta, phi = torus_coordinates_from_voxel(voxel, points=points)
    x, y, phi = torus_to_poloidal_cartesian(r, theta, phi)
    if pass_phi and "phi" not in kwargs:
        kwargs = kwargs | {"phi": phi}
    return profile_func(x, y, *args, **kwargs)


def shifted_polar(x, y, cx, cy, normalize_boundary=True):
    """Convert normalized poloidal Cartesian coordinates to shifted polar coordinates.

    ``x`` and ``y`` are normalized to the unshifted circular plasma boundary
    ``x**2 + y**2 = 1``.  The returned ``rho`` is additionally normalized by
    the distance from the shifted origin ``(cx, cy)`` to that original boundary
    along the same angle when ``normalize_boundary`` is true, so points on the
    original boundary remain at ``rho == 1`` after the shift.
    """
    x = _as_array(x)
    y = _as_array(y)
    x_shifted = x - cx
    y_shifted = y - cy
    theta_shifted = np.arctan2(y_shifted, x_shifted)
    rho_raw = np.hypot(x_shifted, y_shifted)
    if not normalize_boundary:
        return rho_raw, theta_shifted

    ex = np.cos(theta_shifted)
    ey = np.sin(theta_shifted)
    c_dot_e = cx * ex + cy * ey
    c2 = cx ** 2 + cy ** 2
    discriminant = np.maximum(c_dot_e ** 2 + 1 - c2, 0)
    rho_boundary = -c_dot_e + np.sqrt(discriminant)
    rho_shifted = rho_raw / rho_boundary
    return rho_shifted, theta_shifted


def rigid_shifted_polar(x, y, delta, xi, phi=0, psi_0=0):
    """Apply a rigid toroidally rotating shift before evaluating shifted polar coordinates."""
    cx = delta + xi * np.cos(phi + psi_0)
    cy = xi * np.sin(phi + psi_0)
    return shifted_polar(x, y, cx, cy)


def gaussian(rho, rho_s, w, d=2, edge=0.02):
    """Evaluate a bounded Gaussian-like envelope on normalized radius ``rho``.

    The exponent uses ``abs((rho - rho_s) / (w / 2))`` so odd exponents remain
    decaying on both sides of ``rho_s``.  The optional edge factor suppresses
    the value at ``rho == 0`` and ``rho == 1`` without relying on a module-level
    machine or experiment constant.
    """
    rho = _as_array(rho)
    x = (rho - rho_s) / (w / 2)
    profile = np.exp(-np.abs(x) ** d)
    if edge is not None and edge > 0:
        profile = profile * (1 - np.exp(-(rho / edge) ** 2)) * (1 - np.exp(-((rho - 1) / edge) ** 2))
    return np.where(rho > 1, 0, profile)


def two_power(rho, alpha, beta):
    """Evaluate ``(1 - rho**alpha)**beta`` inside ``rho <= 1`` and zero outside."""
    rho = _as_array(rho)
    rho_clipped = np.clip(rho, 0, 1)
    profile = (1 - rho_clipped ** alpha) ** beta
    return np.where(rho > 1, 0, profile)


def two_power_derivative(rho, alpha, beta):
    """Derivative of :func:`two_power`, clipped before the power expression."""
    rho = _as_array(rho)
    rho_clipped = np.clip(rho, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        derivative = -alpha * beta * rho_clipped ** (alpha - 1) * (1 - rho_clipped ** alpha) ** (beta - 1)
    return np.where(rho > 1, 0, derivative)


def smooth_maximum(a, b, eps=0.03):
    """Smooth approximation of ``maximum(a, b)`` using ``logaddexp``."""
    return np.logaddexp(_as_array(a) / eps, _as_array(b) / eps) * eps


def smooth_minimum(a, b, eps=0.03):
    """Smooth approximation of ``minimum(a, b)`` using ``logaddexp``."""
    return -smooth_maximum(-_as_array(a), -_as_array(b), eps=eps)


def _distort_theta(theta, gamma, psi_0):
    theta_offset = theta - psi_0
    return theta_offset + gamma * np.sin(theta_offset)


def kinked_rho(x, y, delta, xi_0, rho_s, d, phi=0, psi_0=0):
    """Return the radial coordinate after a decaying rigid-shift kink."""
    rho_shifted, _ = shifted_polar(x, y, delta, 0)
    xi = xi_0 * np.exp(-(rho_shifted / rho_s) ** d)
    return rigid_shifted_polar(x, y, delta, xi, phi=phi, psi_0=psi_0)


def flattening_rho(x, y, delta, xi_0, rho_s, d, w, gamma=0, lam_0=1,
                   phi=0, psi_0=0, psi_1=np.pi):
    """Return the radial coordinate after kink displacement and partial flattening."""
    rho_shifted, _ = shifted_polar(x, y, delta, 0)
    rho_kinked, theta_kinked = kinked_rho(x, y, delta, xi_0, rho_s, d, phi=phi, psi_0=psi_0)
    rho_flat = smooth_maximum(rho_s, rho_shifted)
    theta_distorted = _distort_theta(theta_kinked, gamma=gamma, psi_0=phi + psi_1 + psi_0)
    angular_weight = 0.5 * (1 + np.cos(theta_distorted))
    lam = gaussian(rho_kinked, rho_s, w) * angular_weight * lam_0
    rho_merged = (1 - lam) * rho_kinked + lam * rho_flat
    return rho_merged, theta_distorted


def axisymmetric_profile(x, y, A, delta, alpha, beta, **kwargs):
    """Axisymmetric two-power profile on normalized poloidal coordinates."""
    rho_shifted, _ = shifted_polar(x, y, delta, 0)
    return A * two_power(rho_shifted, alpha, beta)


def kinked_profile(x, y, A, delta, alpha, beta, xi_0, rho_s, d, phi=0, psi_0=0):
    """Two-power profile evaluated on :func:`kinked_rho`."""
    rho_kinked, _ = kinked_rho(x, y, delta, xi_0, rho_s, d, phi=phi, psi_0=psi_0)
    return A * two_power(rho_kinked, alpha, beta)


def flattening_profile(x, y, A, delta, alpha, beta, xi_0, rho_s, d, w,
                       gamma=0, lam_0=1, phi=0, psi_0=0, psi_1=np.pi):
    """Two-power profile evaluated on :func:`flattening_rho`."""
    rho_flattened, _ = flattening_rho(x, y, delta=delta, xi_0=xi_0, rho_s=rho_s, d=d,
                                      w=w, gamma=gamma, lam_0=lam_0,
                                      phi=phi, psi_0=psi_0, psi_1=psi_1)
    return A * two_power(rho_flattened, alpha, beta)
