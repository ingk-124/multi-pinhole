"""Synthetic emission-profile helper functions.

These functions are intentionally separate from :mod:`multi_pinhole.voxel` so
the voxel grid container can stay focused on geometry and interpolation.  They
operate on normalized coordinates such as those returned by
``Voxel.normalized_coordinates()``.
"""

import numpy as np


def _as_array(value):
    return np.asarray(value, dtype=float)


def torus_to_poloidal_cartesian(r, theta, phi):
    """Convert normalized torus coordinates to poloidal Cartesian coordinates.

    Parameters
    ----------
    r, theta : array-like
        Dimensionless minor radius and poloidal angle in radians. Inputs use
        NumPy broadcasting.
    phi : scalar or ndarray
        Toroidal angle in radians. It is returned unchanged, without coercion.

    Returns
    -------
    x, y, phi : tuple
        ``x=r*cos(theta)`` and ``y=r*sin(theta)`` with broadcast shape, plus
        the original ``phi`` object. Non-finite values propagate through NumPy.
    """
    x = _as_array(r) * np.cos(theta)
    y = _as_array(r) * np.sin(theta)
    return x, y, phi


def helical_phase(theta, phi, m_, n_, phi_0=0):
    """Calculate a helical phase ``m*theta - n*(phi - phi_0)``.

    Parameters
    ----------
    theta, phi : array-like
        Poloidal and toroidal angles in radians.
    m_, n_ : scalar or array-like
        Dimensionless mode numbers. Integer values are intended but not enforced.
    phi_0 : scalar or array-like, default=0
        Phase origin in radians.

    Returns
    -------
    ndarray
        Phase in radians with the broadcast input shape. Non-finite values propagate.
    """
    return m_ * _as_array(theta) - n_ * (_as_array(phi) - phi_0)


def helical_poloidal_coordinates(r, theta, phi, m_, n_, phi_0=0):
    """Convert ``(r, theta, phi)`` to ``(x, y, psi)`` helical coordinates.

    Parameters
    ----------
    r : array-like
        Dimensionless minor radius.
    theta, phi : array-like
        Poloidal and toroidal angles in radians.
    m_, n_ : scalar or array-like
        Dimensionless mode numbers; integer values are intended but unchecked.
    phi_0 : scalar or array-like, default=0
        Phase origin in radians.

    Returns
    -------
    x, y, psi : tuple of ndarray
        Broadcast dimensionless poloidal coordinates and helical phase in radians.
    """
    x, y, _ = torus_to_poloidal_cartesian(r, theta, phi)
    psi = helical_phase(theta, phi, m_=m_, n_=n_, phi_0=phi_0)
    return x, y, psi


def torus_coordinates_from_voxel(voxel, points=None):
    """Return ``(r, theta, phi)`` from a torus-coordinate voxel.

    Parameters
    ----------
    voxel : Voxel
        Voxel whose ``coordinate_type`` is ``"torus"`` or ``"torus_inverse"``.
    points : ndarray, optional
        Cartesian points, shape ``(n, 3)``. ``None`` uses voxel centers.

    Returns
    -------
    r, theta, phi : tuple of ndarray
        Arrays with shape ``(n,)``. ``r`` is dimensionless and angles are radians.

    Raises
    ------
    ValueError
        If the voxel is not configured for torus coordinates.

    Notes
    -----
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
    """Evaluate a toroidal-coordinate callable on a voxel.

    Parameters
    ----------
    voxel : Voxel
        Voxel configured as ``"torus"`` or ``"torus_inverse"``.
    profile_func : callable
        Callable with signature ``(r, theta, phi, *args, **kwargs)``.
    points : ndarray, optional
        Cartesian points, shape ``(n, 3)``. ``None`` uses voxel centers.

    Returns
    -------
    ndarray
        Profile result with the callable's broadcast shape.
    """
    r, theta, phi = torus_coordinates_from_voxel(voxel, points=points)
    return profile_func(r, theta, phi, *args, **kwargs)


def evaluate_poloidal_profile(voxel, profile_func, *args, points=None, pass_phi=True, **kwargs):
    """Evaluate a ``profile_func(x, y, ...)`` directly on a torus-coordinate voxel.

    Parameters
    ----------
    voxel : Voxel
        Voxel configured for ``"torus"`` or ``"torus_inverse"`` coordinates.
    profile_func : callable
        Callable with signature ``(x, y, *args, **kwargs)``. With
        ``pass_phi=True`` it must accept keyword ``phi`` or ``**kwargs``.
    *args : tuple
        Additional positional arguments forwarded to ``profile_func``.
    points : ndarray, optional
        Cartesian points, shape ``(n, 3)``. ``None`` uses voxel centers.
    pass_phi : bool, default=True
        Inject the toroidal angle as keyword ``phi`` unless supplied explicitly.
    **kwargs : dict
        Keyword arguments forwarded to ``profile_func``; an explicit ``phi`` wins.

    Returns
    -------
    ndarray
        Callable result with its broadcast shape.

    Notes
    -----
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

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    cx, cy : scalar or array-like
        Dimensionless shifted-origin coordinates. All inputs broadcast.
    normalize_boundary : bool, default=True
        Normalize radius by the ray distance to the original unit circle.

    Returns
    -------
    rho, theta : tuple of ndarray
        Dimensionless radius and shifted angle in radians, with broadcast shape.

    Notes
    -----
    ``x`` and ``y`` are normalized to the unshifted circular plasma boundary
    ``x**2 + y**2 = 1``.  The returned ``rho`` is additionally normalized by
    the distance from the shifted origin ``(cx, cy)`` to that original boundary
    along the same angle when ``normalize_boundary`` is true, so points on the
    original boundary remain at ``rho == 1`` after the shift. The quadratic
    discriminant is clipped to zero. A shifted origin on or outside the unit
    circle can give a zero or nonphysical boundary distance. No validation is
    performed; divisions and non-finite inputs follow NumPy.
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
    """Apply a rigid toroidally rotating shift and return shifted polar coordinates.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    delta, xi : scalar or array-like
        Dimensionless static and rotating displacement amplitudes.
    phi : scalar or array-like, default=0
        Toroidal angle in radians.
    psi_0 : scalar or array-like, default=0
        Phase offset in radians. Inputs broadcast.

    Returns
    -------
    rho, theta : tuple of ndarray
        Boundary-normalized radius and angle in radians with broadcast shape.

    Notes
    -----
    The shift center is ``(delta + xi*cos(phi+psi_0), xi*sin(phi+psi_0))``.
    Singular and non-finite behavior is inherited from :func:`shifted_polar`.
    """
    cx = delta + xi * np.cos(phi + psi_0)
    cy = xi * np.sin(phi + psi_0)
    return shifted_polar(x, y, cx, cy)


def gaussian(rho, rho_s, w, d=2, edge=0.02):
    """Evaluate a bounded Gaussian-like envelope on normalized radius ``rho``.

    Parameters
    ----------
    rho, rho_s, w : array-like
        Dimensionless radius, center, and width; inputs broadcast.
    d : array-like, default=2
        Exponent. Positive values are intended.
    edge : float or None, default=0.02
        Edge scale; ``None`` or a nonpositive value disables suppression.

    Returns
    -------
    ndarray
        Broadcast profile. ``rho > 1`` is zeroed; ``rho < 0`` is not clipped.

    Notes
    -----
    The exponent uses ``abs((rho - rho_s) / (w / 2))`` so odd exponents remain
    decaying on both sides of ``rho_s``.  The optional edge factor suppresses
    the value at ``rho == 0`` and ``rho == 1`` without relying on a module-level
    machine or experiment constant. ``w=0`` and unsuitable exponents can be
    singular; non-finite values otherwise follow NumPy.
    """
    rho = _as_array(rho)
    x = (rho - rho_s) / (w / 2)
    profile = np.exp(-np.abs(x) ** d)
    if edge is not None and edge > 0:
        profile = profile * (1 - np.exp(-(rho / edge) ** 2)) * (1 - np.exp(-((rho - 1) / edge) ** 2))
    return np.where(rho > 1, 0, profile)


def two_power(rho, alpha, beta):
    """Evaluate a clipped two-power radial profile.

    Parameters
    ----------
    rho, alpha, beta : array-like
        Dimensionless radius and exponents; inputs broadcast.

    Returns
    -------
    ndarray
        ``rho`` is clipped to ``[0, 1]`` for the power expression, then
        original ``rho > 1`` is zeroed. Negative radii therefore map to the
        axis value. Exponents are unvalidated and may create singularities.
    """
    rho = _as_array(rho)
    rho_clipped = np.clip(rho, 0, 1)
    profile = (1 - rho_clipped ** alpha) ** beta
    return np.where(rho > 1, 0, profile)


def two_power_derivative(rho, alpha, beta):
    """Evaluate the interior derivative of the clipped two-power profile.

    Parameters
    ----------
    rho, alpha, beta : array-like
        Dimensionless radius and exponents; inputs broadcast.

    Returns
    -------
    ndarray
        Analytic power-expression derivative evaluated after clipping ``rho``
        to ``[0, 1]`` and zeroed for original ``rho > 1``. For ``rho < 0`` it
        equals the axis-side expression, not a derivative of a smooth global
        extension. Unrestricted exponents can yield ``inf`` or ``nan`` at 0 or 1.
    """
    rho = _as_array(rho)
    rho_clipped = np.clip(rho, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        derivative = -alpha * beta * rho_clipped ** (alpha - 1) * (1 - rho_clipped ** alpha) ** (beta - 1)
    return np.where(rho > 1, 0, derivative)


def smooth_maximum(a, b, eps=0.03):
    """Smooth a broadcast maximum using ``logaddexp``.

    Parameters
    ----------
    a, b : array-like
        Values to combine.
    eps : float, default=0.03
        Positive smoothing scale. Zero is singular and is not validated.

    Returns
    -------
    ndarray
        Broadcast smooth maximum; non-finite inputs follow NumPy.
    """
    return np.logaddexp(_as_array(a) / eps, _as_array(b) / eps) * eps


def smooth_minimum(a, b, eps=0.03):
    """Smooth a broadcast minimum using ``-smooth_maximum(-a, -b)``.

    Parameters
    ----------
    a, b : array-like
        Values to combine.
    eps : float, default=0.03
        Positive smoothing scale. Zero is singular and is not validated.

    Returns
    -------
    ndarray
        Broadcast smooth minimum; non-finite inputs follow NumPy.
    """
    return -smooth_maximum(-_as_array(a), -_as_array(b), eps=eps)


def _distort_theta(theta, gamma, psi_0):
    theta_offset = theta - psi_0
    return theta_offset + gamma * np.sin(theta_offset)


def kinked_rho(x, y, delta, xi_0, rho_s, d, phi=0, psi_0=0):
    """Return polar coordinates after a radially decaying rigid-shift kink.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    delta, xi_0, rho_s : scalar or array-like
        Dimensionless static shift, kink amplitude, and decay radius.
    d : scalar or array-like
        Decay exponent; positive values and nonzero ``rho_s`` are intended.
    phi, psi_0 : scalar or array-like, default=0
        Toroidal angle and phase origin in radians. Inputs broadcast.

    Returns
    -------
    rho, theta : tuple of ndarray
        Boundary-normalized radius and angle in radians with broadcast shape.

    Notes
    -----
    ``rho_s=0`` and unsuitable exponents can be singular. Inputs are not
    validated and non-finite values propagate according to NumPy.
    """
    rho_shifted, _ = shifted_polar(x, y, delta, 0)
    xi = xi_0 * np.exp(-(rho_shifted / rho_s) ** d)
    return rigid_shifted_polar(x, y, delta, xi, phi=phi, psi_0=psi_0)


def flattening_rho(x, y, delta, xi_0, rho_s, d, w, gamma=0, lam_0=1,
                   phi=0, psi_0=0, psi_1=np.pi):
    """Return coordinates after kink displacement and smooth partial flattening.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    delta, xi_0, rho_s, w : scalar or array-like
        Dimensionless shift, kink amplitude, target radius, and Gaussian width.
    d : scalar or array-like
        Decay exponent; positive values are intended.
    gamma : scalar or array-like, default=0
        Dimensionless angular-distortion amplitude.
    lam_0 : scalar or array-like, default=1
        Blend amplitude. It is not clipped, so values outside ``[0, 1]`` extrapolate.
    phi, psi_0 : scalar or array-like, default=0
        Toroidal angle and phase origin in radians.
    psi_1 : scalar or array-like, default=pi
        Angular flattening offset in radians. Inputs broadcast.

    Returns
    -------
    rho, theta : tuple of ndarray
        Merged dimensionless radius and distorted angle in radians.

    Notes
    -----
    The maximum uses :func:`smooth_maximum` with its default ``eps=0.03``;
    the blend uses :func:`gaussian` with default edge suppression. ``w=0``,
    ``rho_s=0``, unsuitable exponents, and non-finite inputs follow NumPy.
    """
    rho_shifted, _ = shifted_polar(x, y, delta, 0)
    rho_kinked, theta_kinked = kinked_rho(x, y, delta, xi_0, rho_s, d, phi=phi, psi_0=psi_0)
    rho_flat = smooth_maximum(rho_s, rho_shifted)
    theta_distorted = _distort_theta(theta_kinked, gamma=gamma, psi_0=phi + psi_1 + psi_0)
    angular_weight = 0.5 * (1 + np.cos(theta_distorted))
    lam = gaussian(rho_kinked, rho_s, w) * angular_weight * lam_0
    rho_merged = (1 - lam) * rho_kinked + lam * rho_flat
    return rho_merged, theta_distorted


def axisymmetric_profile(x, y, A, delta, alpha, beta, **kwargs):
    """Evaluate an axisymmetric two-power profile on poloidal coordinates.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    A : scalar or array-like
        Profile amplitude; it may carry application-defined units.
    delta : scalar or array-like
        Dimensionless horizontal shift.
    alpha, beta : scalar or array-like
        Two-power exponents. Positive values are intended but unchecked.
    **kwargs : dict
        Ignored compatibility keywords, including an injected toroidal ``phi``.

    Returns
    -------
    ndarray
        Broadcast profile. Radius behavior and singularities follow :func:`two_power`.
    """
    rho_shifted, _ = shifted_polar(x, y, delta, 0)
    return A * two_power(rho_shifted, alpha, beta)


def kinked_profile(x, y, A, delta, alpha, beta, xi_0, rho_s, d, phi=0, psi_0=0):
    """Evaluate a two-power profile on kink-displaced coordinates.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    A : scalar or array-like
        Profile amplitude; it may carry application-defined units.
    delta, xi_0, rho_s : scalar or array-like
        Dimensionless displacement parameters and decay radius.
    alpha, beta, d : scalar or array-like
        Two-power and decay exponents; positive values are intended.
    phi, psi_0 : scalar or array-like, default=0
        Toroidal angle and phase origin in radians. Inputs broadcast.

    Returns
    -------
    ndarray
        Broadcast profile. Clipping and singularities follow
        :func:`kinked_rho` and :func:`two_power`.
    """
    rho_kinked, _ = kinked_rho(x, y, delta, xi_0, rho_s, d, phi=phi, psi_0=psi_0)
    return A * two_power(rho_kinked, alpha, beta)


def flattening_profile(x, y, A, delta, alpha, beta, xi_0, rho_s, d, w,
                       gamma=0, lam_0=1, phi=0, psi_0=0, psi_1=np.pi):
    """Evaluate a two-power profile on kinked and flattened coordinates.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    A : scalar or array-like
        Profile amplitude; it may carry application-defined units.
    delta, xi_0, rho_s, w : scalar or array-like
        Dimensionless displacement, decay-radius, and width parameters.
    alpha, beta, d : scalar or array-like
        Two-power and decay exponents; positive values are intended.
    gamma : scalar or array-like, default=0
        Dimensionless angular-distortion amplitude.
    lam_0 : scalar or array-like, default=1
        Unclipped blend amplitude.
    phi, psi_0 : scalar or array-like, default=0
        Toroidal angle and phase origin in radians.
    psi_1 : scalar or array-like, default=pi
        Flattening offset in radians. Inputs broadcast.

    Returns
    -------
    ndarray
        Broadcast profile. Range clipping and singular behavior follow
        :func:`flattening_rho` and :func:`two_power`.
    """
    rho_flattened, _ = flattening_rho(x, y, delta=delta, xi_0=xi_0, rho_s=rho_s, d=d,
                                      w=w, gamma=gamma, lam_0=lam_0,
                                      phi=phi, psi_0=psi_0, psi_1=psi_1)
    return A * two_power(rho_flattened, alpha, beta)
