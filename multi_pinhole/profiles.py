"""Generic scalar-profile helper functions.

These functions are intentionally separate from :mod:`multi_pinhole.voxel` so
the voxel grid container can stay focused on geometry and interpolation.  They
operate on normalized coordinates such as those returned by
``Voxel.to_coordinates()``.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_array(value: ArrayLike) -> FloatArray:
    return np.asarray(value, dtype=float)


def helical_center_angle(
        phi: ArrayLike,
        center_angle_xy_ref: ArrayLike,
        *,
        m: ArrayLike,
        n: ArrayLike,
        phi_ref: ArrayLike
) -> FloatArray:
    """Propagate a reference poloidal-center angle along a helical structure.

    Parameters
    ----------
    phi : array-like
        Toroidal angle in radians at each evaluation point.
    center_angle_xy_ref : array-like
        Poloidal-center angle in radians at ``phi_ref``. It is measured
        counter-clockwise from the outward poloidal Cartesian ``+x`` axis
        toward ``+y`` (upward).
    m, n : scalar or array-like
        Signed poloidal and toroidal mode numbers. ``m`` must be nonzero.
    phi_ref : scalar or array-like
        Toroidal reference angle in radians.

    Returns
    -------
    ndarray
        Poloidal Cartesian center angle
        ``center_angle_xy_ref + (n / m) * (phi - phi_ref)`` with the
        broadcast input shape. Angles are not wrapped.

    Raises
    ------
    ValueError
        If any value of ``m`` is zero.

    Notes
    -----
    This function does not infer whether ``phi`` follows the ``torus`` or
    ``torus_inverse`` convention. The caller supplies a signed ``n`` that is
    consistent with the chosen toroidal-angle convention.
    """
    m = _as_array(m)
    if np.any(m == 0):
        raise ValueError("m must be nonzero")
    return (
        _as_array(center_angle_xy_ref)
        + (_as_array(n) / m) * (_as_array(phi) - _as_array(phi_ref))
    )


def shifted_polar(
        x: ArrayLike,
        y: ArrayLike,
        cx: ArrayLike,
        cy: ArrayLike,
        normalize_boundary: bool = True
) -> tuple[FloatArray, FloatArray]:
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


def rigid_shifted_polar(
        x: ArrayLike,
        y: ArrayLike,
        delta: ArrayLike,
        xi: ArrayLike,
        center_angle_xy: ArrayLike = 0
) -> tuple[FloatArray, FloatArray]:
    """Apply a rigid shift and return shifted polar coordinates.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    delta, xi : scalar or array-like
        Dimensionless static and directional displacement amplitudes.
    center_angle_xy : scalar or array-like, default=0
        Displacement angle measured counter-clockwise from the positive
        poloidal ``x`` axis. Inputs broadcast.

    Returns
    -------
    rho, theta : tuple of ndarray
        Boundary-normalized radius and angle in radians with broadcast shape.

    Notes
    -----
    Singular and non-finite behavior is inherited from
    :func:`shifted_polar`.
    """
    cx = delta + xi * np.cos(center_angle_xy)
    cy = xi * np.sin(center_angle_xy)
    return shifted_polar(x, y, cx, cy)


def gaussian(
        rho: ArrayLike,
        rho_s: ArrayLike,
        w: ArrayLike,
        d: ArrayLike = 2,
        edge: float | None = 0.02
) -> FloatArray:
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


def two_power(
        rho: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike
) -> FloatArray:
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


def two_power_derivative(
        rho: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike
) -> FloatArray:
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


def smooth_maximum(
        a: ArrayLike,
        b: ArrayLike,
        eps: float = 0.03
) -> FloatArray:
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


def smooth_minimum(
        a: ArrayLike,
        b: ArrayLike,
        eps: float = 0.03
) -> FloatArray:
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


def _distort_theta(
        theta: ArrayLike,
        gamma: ArrayLike,
        reference_angle: ArrayLike
) -> FloatArray:
    theta_offset = theta - reference_angle
    return theta_offset + gamma * np.sin(theta_offset)


def kinked_rho(
        x: ArrayLike,
        y: ArrayLike,
        delta: ArrayLike,
        xi_0: ArrayLike,
        rho_s: ArrayLike,
        d: ArrayLike,
        center_angle_xy: ArrayLike = 0
) -> tuple[FloatArray, FloatArray]:
    """Return polar coordinates after a radially decaying rigid-shift kink.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    delta, xi_0, rho_s : scalar or array-like
        Dimensionless static shift, kink amplitude, and decay radius.
    d : scalar or array-like
        Decay exponent; positive values and nonzero ``rho_s`` are intended.
    center_angle_xy : scalar or array-like, default=0
        Kink displacement angle measured counter-clockwise from the positive
        poloidal ``x`` axis.

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
    return rigid_shifted_polar(
        x, y, delta, xi,
        center_angle_xy=center_angle_xy,
    )


def flattening_rho(
        x: ArrayLike,
        y: ArrayLike,
        delta: ArrayLike,
        xi_0: ArrayLike,
        rho_s: ArrayLike,
        d: ArrayLike,
        w: ArrayLike,
        gamma: ArrayLike = 0,
        lam_0: ArrayLike = 1,
        center_angle_xy: ArrayLike = 0,
        flattening_angle_offset: ArrayLike = np.pi,
) -> tuple[FloatArray, FloatArray]:
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
    center_angle_xy : scalar or array-like, default=0
        Kink displacement angle measured counter-clockwise from the positive
        poloidal ``x`` axis.
    flattening_angle_offset : scalar or array-like, default=pi
        Angular offset of the flattening region relative to the kink
        displacement direction. Inputs broadcast.

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
    rho_kinked, theta_kinked = kinked_rho(
        x, y, delta, xi_0, rho_s, d,
        center_angle_xy=center_angle_xy,
    )
    rho_flat = smooth_maximum(rho_s, rho_shifted)
    theta_distorted = _distort_theta(
        theta_kinked,
        gamma=gamma,
        reference_angle=center_angle_xy + flattening_angle_offset,
    )
    angular_weight = 0.5 * (1 + np.cos(theta_distorted))
    lam = gaussian(rho_kinked, rho_s, w) * angular_weight * lam_0
    rho_merged = (1 - lam) * rho_kinked + lam * rho_flat
    return rho_merged, theta_distorted


def _profile_from_rho(
        x: ArrayLike,
        y: ArrayLike,
        rho: ArrayLike,
        A: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        edge_value: ArrayLike = 0
) -> FloatArray:
    """Scale a two-power shape within the unit poloidal disk."""
    shape = two_power(rho, alpha, beta)
    return np.where(
        np.asarray(x) ** 2 + np.asarray(y) ** 2 <= 1,
        edge_value + (A - edge_value) * shape,
        0,
    )


def axisymmetric_profile(
        x: ArrayLike,
        y: ArrayLike,
        A: ArrayLike,
        delta: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        edge_value: ArrayLike = 0,
        **kwargs: object
) -> FloatArray:
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
    edge_value : scalar or array-like, default=0
        Profile value where the effective ``rho=1``. The central value
        remains ``A`` and the profile is zero where ``x**2 + y**2 > 1``.
    **kwargs : dict
        Ignored compatibility keywords, including an injected toroidal ``phi``.

    Returns
    -------
    ndarray
        Broadcast profile. Radius behavior and singularities follow :func:`two_power`.
    """
    rho_shifted, _ = shifted_polar(x, y, delta, 0)
    return _profile_from_rho(x, y, rho_shifted, A, alpha, beta, edge_value)


def kinked_profile(
        x: ArrayLike,
        y: ArrayLike,
        A: ArrayLike,
        delta: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        xi_0: ArrayLike,
        rho_s: ArrayLike,
        d: ArrayLike,
        center_angle_xy: ArrayLike = 0,
        edge_value: ArrayLike = 0,
) -> FloatArray:
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
    center_angle_xy : scalar or array-like, default=0
        Kink displacement angle measured counter-clockwise from the positive
        poloidal ``x`` axis.
    edge_value : scalar or array-like, default=0
        Profile value where the effective ``rho=1``. The central value
        remains ``A`` and the profile is zero where ``x**2 + y**2 > 1``.

    Returns
    -------
    ndarray
        Broadcast profile. Clipping and singularities follow
        :func:`kinked_rho` and :func:`two_power`.
    """
    rho_kinked, _ = kinked_rho(
        x, y, delta, xi_0, rho_s, d,
        center_angle_xy=center_angle_xy,
    )
    return _profile_from_rho(x, y, rho_kinked, A, alpha, beta, edge_value)


def flattening_profile(
        x: ArrayLike,
        y: ArrayLike,
        A: ArrayLike,
        delta: ArrayLike,
        alpha: ArrayLike,
        beta: ArrayLike,
        xi_0: ArrayLike,
        rho_s: ArrayLike,
        d: ArrayLike,
        w: ArrayLike,
        gamma: ArrayLike = 0,
        lam_0: ArrayLike = 1,
        center_angle_xy: ArrayLike = 0,
        flattening_angle_offset: ArrayLike = np.pi,
        edge_value: ArrayLike = 0,
) -> FloatArray:
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
    center_angle_xy : scalar or array-like, default=0
        Kink displacement angle measured counter-clockwise from the positive
        poloidal ``x`` axis.
    flattening_angle_offset : scalar or array-like, default=pi
        Angular offset of the flattening region relative to the kink
        displacement direction. Inputs broadcast.
    edge_value : scalar or array-like, default=0
        Profile value where the effective ``rho=1``. The central value
        remains ``A`` and the profile is zero where ``x**2 + y**2 > 1``.

    Returns
    -------
    ndarray
        Broadcast profile. Range clipping and singular behavior follow
        :func:`flattening_rho` and :func:`two_power`.
    """
    rho_flattened, _ = flattening_rho(x, y, delta=delta, xi_0=xi_0, rho_s=rho_s, d=d,
                                      w=w, gamma=gamma, lam_0=lam_0,
                                      center_angle_xy=center_angle_xy,
                                      flattening_angle_offset=flattening_angle_offset)
    return _profile_from_rho(x, y, rho_flattened, A, alpha, beta, edge_value)
