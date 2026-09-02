"""Generic scalar-profile helper functions.

These functions are intentionally separate from :mod:`multi_pinhole.voxel` so
the voxel grid container can stay focused on geometry and interpolation.  They
operate on normalized coordinates such as those returned by
``Voxel.to_coordinates()``.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import lambertw

FloatArray = NDArray[np.float64]


def _as_array(value: ArrayLike) -> FloatArray:
    return np.asarray(value, dtype=float)


def _finite_scalar(value: object, name: str) -> float:
    """Return a finite real scalar or raise a parameter-specific error."""
    array = np.asarray(value)
    if (
        array.ndim != 0
        or np.iscomplexobj(array)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        scalar = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar


def _positive_scalar(value: object, name: str) -> float:
    scalar = _finite_scalar(value, name)
    if scalar <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return scalar


def _nonnegative_scalar(value: object, name: str) -> float:
    scalar = _finite_scalar(value, name)
    if scalar < 0:
        raise ValueError(f"{name} must be nonnegative")
    return scalar


def _unit_interval_scalar(value: object, name: str) -> float:
    scalar = _finite_scalar(value, name)
    if not 0 <= scalar <= 1:
        raise ValueError(f"{name} must lie in the closed interval [0, 1]")
    return scalar


def _bool_scalar(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a bool")
    return bool(value)


def _finite_array(value: ArrayLike, name: str) -> FloatArray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must contain only finite real values")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite real values") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite real values")
    return array


def _shifted_polar_impl(
    x: ArrayLike,
    y: ArrayLike,
    cx: ArrayLike,
    cy: ArrayLike,
    normalize_boundary: bool,
) -> tuple[FloatArray, FloatArray]:
    """Apply a scalar or elementwise origin shift after public validation."""
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
    c2 = cx**2 + cy**2
    discriminant = np.maximum(c_dot_e**2 + 1 - c2, 0)
    rho_boundary = -c_dot_e + np.sqrt(discriminant)
    return rho_raw / rho_boundary, theta_shifted


def helical_center_angle(
    phi: ArrayLike,
    center_angle_xy_ref: ArrayLike,
    *,
    m: float,
    n: float,
    phi_ref: float,
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
    m, n : float
        Finite signed poloidal and toroidal mode numbers. ``m`` must be
        nonzero.
    phi_ref : float
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
        If ``m``, ``n``, or ``phi_ref`` is not a finite scalar, if
        ``center_angle_xy_ref`` contains a non-finite value, or if ``m`` is
        zero.

    Notes
    -----
    This function does not infer whether ``phi`` follows the ``torus`` or
    ``torus_inverse`` convention. The caller supplies a signed ``n`` that is
    consistent with the chosen toroidal-angle convention.
    """
    center_angle_xy_ref = _finite_array(center_angle_xy_ref, "center_angle_xy_ref")
    m = _finite_scalar(m, "m")
    n = _finite_scalar(n, "n")
    phi_ref = _finite_scalar(phi_ref, "phi_ref")
    if m == 0:
        raise ValueError("m must be nonzero")
    return center_angle_xy_ref + (n / m) * (_as_array(phi) - phi_ref)


def shifted_polar(
    x: ArrayLike,
    y: ArrayLike,
    cx: float,
    cy: float,
    *,
    normalize_boundary: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """Convert normalized poloidal Cartesian coordinates to shifted polar coordinates.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    cx, cy : float
        Finite dimensionless shifted-origin coordinates.
    normalize_boundary : bool, default=True
        Normalize radius by the ray distance to the original unit circle.

    Returns
    -------
    rho, theta : tuple of ndarray
        Dimensionless radius and shifted angle in radians, with broadcast shape.

    Raises
    ------
    ValueError
        If ``cx`` or ``cy`` is not a finite scalar, if
        ``normalize_boundary`` is not boolean, or if boundary normalization
        is requested with the shifted origin on or outside the unit circle.

    Notes
    -----
    ``x`` and ``y`` are normalized to the unshifted circular plasma boundary
    ``x**2 + y**2 = 1``.  The returned ``rho`` is additionally normalized by
    the distance from the shifted origin ``(cx, cy)`` to that original boundary
    along the same angle when ``normalize_boundary`` is true, so points on the
    original boundary remain at ``rho == 1`` after the shift. The quadratic
    discriminant is clipped to zero. A shifted origin on or outside the unit
    circle is rejected when boundary normalization is enabled. Non-finite
    values in the evaluation coordinates follow NumPy.
    """
    cx = _finite_scalar(cx, "cx")
    cy = _finite_scalar(cy, "cy")
    normalize_boundary = _bool_scalar(normalize_boundary, "normalize_boundary")
    if normalize_boundary and cx**2 + cy**2 >= 1:
        raise ValueError("the shifted origin must lie inside the unit boundary")
    return _shifted_polar_impl(x, y, cx, cy, normalize_boundary)


def _kink_displaced_polar(
    x: ArrayLike,
    y: ArrayLike,
    delta: float,
    xi: ArrayLike,
    *,
    center_angle_xy: ArrayLike = 0,
    normalize_kink: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Apply a local kink displacement and return polar coordinates.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    delta : float
        Finite dimensionless horizontal static shift. Its magnitude must be
        less than one.
    xi : array-like
        Finite dimensionless directional displacement amplitude. It may vary
        over the evaluation points.
    center_angle_xy : array-like, default=0
        Displacement angle measured counter-clockwise from the positive
        poloidal ``x`` axis. Inputs broadcast.
    normalize_kink : bool, default=False
        Renormalize the kink-displaced coordinates to the unit boundary.
        The static ``delta`` shift is always boundary-normalized first.

    Returns
    -------
    rho, theta : tuple of ndarray
        Radius and angle in radians with broadcast shape. The static shift is
        boundary-normalized; the kinked boundary is normalized only when
        ``normalize_kink`` is true.

    Raises
    ------
    ValueError
        If ``delta`` is not a finite scalar or has magnitude at least one, if
        ``xi`` or ``center_angle_xy`` contains a non-finite value, if
        ``normalize_kink`` is not boolean, or if a normalized kink origin is
        on or outside the unit circle.

    Notes
    -----
    The static horizontal shift is always normalized to keep the original
    unit-circle wall at ``rho=1``. With the default ``normalize_kink=False``,
    the subsequent kink displacement is not renormalized and may therefore
    remain nonzero at the wall.
    """
    delta = _finite_scalar(delta, "delta")
    if abs(delta) >= 1:
        raise ValueError("abs(delta) must be less than one")
    xi = _finite_array(xi, "xi")
    center_angle_xy = _finite_array(center_angle_xy, "center_angle_xy")
    normalize_kink = _bool_scalar(normalize_kink, "normalize_kink")
    cx = delta + xi * np.cos(center_angle_xy)
    cy = xi * np.sin(center_angle_xy)
    if normalize_kink:
        if np.any(cx**2 + cy**2 >= 1):
            raise ValueError("the kinked origin must lie inside the unit boundary")
        return _shifted_polar_impl(x, y, cx, cy, normalize_boundary=True)

    x_kinked = _as_array(x) - cx
    y_kinked = _as_array(y) - cy
    theta_kinked = np.arctan2(y_kinked, x_kinked)
    rho_raw = np.hypot(x_kinked, y_kinked)

    # Keep the static Shafranov-like shift normalized to the original wall,
    # while leaving the kink displacement itself unnormalized.
    c_dot_e = delta * np.cos(theta_kinked)
    discriminant = np.maximum(c_dot_e**2 + 1 - delta**2, 0)
    boundary_distance = -c_dot_e + np.sqrt(discriminant)
    return rho_raw / boundary_distance, theta_kinked


def gaussian(
    rho: ArrayLike,
    rho_s: float,
    w: float,
    *,
    d: float = 2,
    edge: float | None = None,
) -> FloatArray:
    """Evaluate a bounded Gaussian-like envelope on normalized radius ``rho``.

    Parameters
    ----------
    rho : array-like
        Dimensionless evaluation radius.
    rho_s : float
        Finite dimensionless center radius.
    w : float
        Finite positive full e-folding width. The profile reaches ``exp(-1)``
        at ``abs(rho - rho_s) == w / 2``.
    d : float, default=2
        Finite positive exponent.
    edge : float or None, default=None
        Positive taper width at ``rho=0`` and ``rho=1``. ``None`` disables
        edge suppression.

    Returns
    -------
    ndarray
        Broadcast profile. ``rho > 1`` is zeroed; ``rho < 0`` is not clipped.

    Raises
    ------
    ValueError
        If ``rho_s``, ``w``, or ``d`` is not a finite scalar, if ``w`` or
        ``d`` is not positive, or if ``edge`` is supplied but is not a finite
        positive scalar.

    Notes
    -----
    The exponent uses ``abs(2 * (rho - rho_s) / w)`` so odd exponents remain
    decaying on both sides of ``rho_s``. For ``d=2``, ``w`` is not a standard
    deviation; the equivalent Gaussian standard deviation is
    ``w / (2 * sqrt(2))``. The optional edge factor suppresses the value at
    ``rho == 0`` and ``rho == 1`` without relying on a module-level machine or
    experiment constant. Non-finite values in ``rho`` follow NumPy.
    """
    rho_s = _finite_scalar(rho_s, "rho_s")
    w = _positive_scalar(w, "w")
    d = _positive_scalar(d, "d")
    if edge is not None:
        edge = _positive_scalar(edge, "edge")
    rho = _as_array(rho)
    x = (rho - rho_s) / (w / 2)
    profile = np.exp(-(np.abs(x) ** d))
    if edge is not None:
        profile = (
            profile
            * (1 - np.exp(-((rho / edge) ** 2)))
            * (1 - np.exp(-(((rho - 1) / edge) ** 2)))
        )
    return np.where(rho > 1, 0, profile)


def two_power(rho: ArrayLike, alpha: float, beta: float) -> FloatArray:
    """Evaluate a clipped two-power radial profile.

    Parameters
    ----------
    rho : array-like
        Dimensionless evaluation radius.
    alpha, beta : float
        Finite positive exponents.

    Returns
    -------
    ndarray
        ``rho`` is clipped to ``[0, 1]`` for the power expression, then
        original ``rho > 1`` is zeroed. Negative radii therefore map to the
        axis value.

    Raises
    ------
    ValueError
        If ``alpha`` or ``beta`` is not a finite positive scalar.
    """
    alpha = _positive_scalar(alpha, "alpha")
    beta = _positive_scalar(beta, "beta")
    rho = _as_array(rho)
    rho_clipped = np.clip(rho, 0, 1)
    profile = (1 - rho_clipped**alpha) ** beta
    return np.where(rho > 1, 0, profile)


def two_power_derivative(rho: ArrayLike, alpha: float, beta: float) -> FloatArray:
    """Evaluate the interior derivative of the clipped two-power profile.

    Parameters
    ----------
    rho : array-like
        Dimensionless evaluation radius.
    alpha, beta : float
        Finite positive exponents.

    Returns
    -------
    ndarray
        Analytic power-expression derivative evaluated after clipping ``rho``
        to ``[0, 1]`` and zeroed for original ``rho > 1``. For ``rho < 0`` it
        equals the axis-side expression, not a derivative of a smooth global
        extension. Valid exponents can still yield an infinite endpoint
        derivative when either exponent is below one.

    Raises
    ------
    ValueError
        If ``alpha`` or ``beta`` is not a finite positive scalar.
    """
    alpha = _positive_scalar(alpha, "alpha")
    beta = _positive_scalar(beta, "beta")
    rho = _as_array(rho)
    rho_clipped = np.clip(rho, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        derivative = (
            -alpha
            * beta
            * rho_clipped ** (alpha - 1)
            * (1 - rho_clipped**alpha) ** (beta - 1)
        )
    return np.where(rho > 1, 0, derivative)


def smooth_maximum(a: ArrayLike, b: ArrayLike, *, eps: float = 0) -> FloatArray:
    """Smooth a broadcast maximum using ``logaddexp``.

    Parameters
    ----------
    a, b : array-like
        Values to combine.
    eps : float, default=0
        Nonnegative smoothing scale. Zero selects the exact maximum.

    Returns
    -------
    ndarray
        Broadcast smooth maximum; non-finite inputs follow NumPy.

    Raises
    ------
    ValueError
        If ``eps`` is not a finite nonnegative scalar.
    """
    eps = _nonnegative_scalar(eps, "eps")
    if eps == 0:
        return np.maximum(_as_array(a), _as_array(b))
    return np.logaddexp(_as_array(a) / eps, _as_array(b) / eps) * eps


def smooth_minimum(a: ArrayLike, b: ArrayLike, *, eps: float = 0) -> FloatArray:
    """Smooth a broadcast minimum using ``-smooth_maximum(-a, -b)``.

    Parameters
    ----------
    a, b : array-like
        Values to combine.
    eps : float, default=0
        Nonnegative smoothing scale. Zero selects the exact minimum.

    Returns
    -------
    ndarray
        Broadcast smooth minimum; non-finite inputs follow NumPy.

    Raises
    ------
    ValueError
        If ``eps`` is not a finite nonnegative scalar.
    """
    return -smooth_maximum(-_as_array(a), -_as_array(b), eps=eps)


def _distort_theta(
    theta: ArrayLike, gamma: float, reference_angle: ArrayLike
) -> FloatArray:
    gamma = _finite_scalar(gamma, "gamma")
    theta_offset = theta - reference_angle
    return theta_offset + gamma * np.sin(theta_offset)


def _fully_flattened_rho(
    rho_0: ArrayLike,
    rho_kinked: ArrayLike,
    rho_flat: float,
    smooth_eps: float,
) -> FloatArray:
    """Clip a kinked radius by the full-angle island limit."""
    rho_flat = _positive_scalar(rho_flat, "rho_flat")
    rho_limit = smooth_maximum(rho_flat, rho_0, eps=smooth_eps)
    return smooth_minimum(rho_kinked, rho_limit, eps=smooth_eps)


def _kinked_coordinates(
    x: ArrayLike,
    y: ArrayLike,
    delta: float,
    xi_0: float,
    rho_s: float,
    d: float,
    center_angle_xy: ArrayLike,
    normalize_kink: bool,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the pre-kink radius and kinked polar coordinates."""
    delta = _finite_scalar(delta, "delta")
    if abs(delta) >= 1:
        raise ValueError("abs(delta) must be less than one")
    xi_0 = _finite_scalar(xi_0, "xi_0")
    rho_s = _positive_scalar(rho_s, "rho_s")
    d = _positive_scalar(d, "d")
    center_angle_xy = _finite_array(center_angle_xy, "center_angle_xy")
    normalize_kink = _bool_scalar(normalize_kink, "normalize_kink")
    rho_0, _ = shifted_polar(x, y, delta, 0)
    xi = xi_0 * np.exp(-((rho_0 / rho_s) ** d))
    rho_kinked, theta_kinked = _kink_displaced_polar(
        x,
        y,
        delta,
        xi,
        center_angle_xy=center_angle_xy,
        normalize_kink=normalize_kink,
    )
    return rho_0, rho_kinked, theta_kinked


def kinked_rho(
    x: ArrayLike,
    y: ArrayLike,
    delta: float,
    xi_0: float,
    rho_s: float,
    d: float,
    *,
    center_angle_xy: ArrayLike = 0,
    normalize_kink: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Return polar coordinates after a radially decaying rigid-shift kink.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    delta : float
        Finite dimensionless horizontal static shift with magnitude less than
        one.
    xi_0 : float
        Finite dimensionless Cartesian kink-displacement amplitude.
    rho_s : float
        Finite positive dimensionless decay radius.
    d : float
        Finite positive decay exponent.
    center_angle_xy : array-like, default=0
        Kink displacement angle measured counter-clockwise from the positive
        poloidal ``x`` axis. It may vary over the evaluation points.
    normalize_kink : bool, default=False
        Renormalize the kink-displaced coordinates to the unit boundary.
        The static ``delta`` shift is always boundary-normalized.

    Returns
    -------
    rho, theta : tuple of ndarray
        Radius and angle in radians with broadcast shape. By default the
        kink displacement may remain nonzero at the wall.

    Raises
    ------
    ValueError
        If a shape parameter is not a finite scalar, if ``abs(delta) >= 1``,
        if ``rho_s`` or ``d`` is not positive, if ``center_angle_xy`` contains
        a non-finite value, if ``normalize_kink`` is not boolean, or if a
        normalized kink origin reaches or crosses the unit boundary.

    Notes
    -----
    The horizontal static shift is always boundary-normalized. The kink is
    applied afterward and, by default, is not renormalized to the wall. Its
    displacement envelope is evaluated on the pre-kink radius ``rho_0``.
    """
    _, rho_kinked, theta_kinked = _kinked_coordinates(
        x,
        y,
        delta,
        xi_0,
        rho_s,
        d,
        center_angle_xy,
        normalize_kink,
    )
    return rho_kinked, theta_kinked


def _crescent_folds(
    kink_amplitude: ArrayLike,
    rho_s: float,
    d: float,
) -> tuple[np.ndarray, FloatArray, FloatArray]:
    """Return fold masks, locations, and effective radii elementwise.

    The radial map along the kink direction is
    ``rho + kink_amplitude * exp(-(rho / rho_s)**d)``. The ``-1`` branch of Lambert W
    gives its outer stationary point. Entries outside the real branch are
    marked false in the returned mask; their radius values are placeholders.
    """
    kink_amplitude = _finite_array(kink_amplitude, "kink_amplitude")
    if np.any(kink_amplitude <= 0):
        raise ValueError("kink_amplitude must be greater than zero")
    exponent_ratio = (d - 1) / d
    scale_ratio = rho_s / (kink_amplitude * d)
    lambert_argument = -(scale_ratio ** (1 / exponent_ratio)) / exponent_ratio
    has_fold = lambert_argument >= -1 / np.e
    safe_argument = np.where(has_fold, lambert_argument, -1 / np.e)
    normalized_power_at_fold = -exponent_ratio * lambertw(safe_argument, k=-1).real
    base_radius_at_fold = rho_s * normalized_power_at_fold ** (1 / d)
    effective_radius_at_fold = base_radius_at_fold + kink_amplitude * np.exp(
        -normalized_power_at_fold
    )
    return has_fold, base_radius_at_fold, effective_radius_at_fold


def _crescent_fold(
    kink_amplitude: float,
    rho_s: float,
    d: float,
) -> tuple[float, float] | None:
    """Return one fold location and effective radius, or ``None``."""
    has_fold, base_radius, effective_radius = _crescent_folds(kink_amplitude, rho_s, d)
    if not bool(has_fold):
        return None
    return float(base_radius), float(effective_radius)


def crescent_rho(
    x: ArrayLike,
    y: ArrayLike,
    delta: float,
    xi_0: float,
    rho_s: float,
    d: float,
    *,
    center_angle_xy: ArrayLike = 0,
    smooth_eps: float = 0,
) -> tuple[FloatArray, FloatArray]:
    """Flatten the non-monotonic part of a kinked radial coordinate.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian evaluation coordinates.
    delta : float
        Finite dimensionless horizontal static shift with magnitude less than
        one.
    xi_0 : float
        Finite positive dimensionless Cartesian kink-displacement amplitude.
    rho_s : float
        Finite positive dimensionless kink decay radius.
    d : float
        Finite decay exponent greater than one.
    center_angle_xy : array-like, default=0
        Kink displacement angle measured counter-clockwise from the positive
        poloidal ``x`` axis. It may vary over the evaluation points.
    smooth_eps : float, default=0
        Finite nonnegative smoothing scale. Zero uses hard minimum and maximum
        operations.

    Returns
    -------
    rho, theta : tuple of ndarray
        Crescent-flattened dimensionless radius and kinked angle in radians,
        with broadcast shape.

    Raises
    ------
    ValueError
        If a scalar parameter is non-finite or has the wrong shape, if
        ``abs(delta) >= 1``, if ``xi_0`` or ``rho_s`` is not positive, if
        ``d <= 1``, if ``smooth_eps`` is negative, or if
        ``center_angle_xy`` contains a non-finite value.

    Notes
    -----
    The horizontal static shift is boundary-normalized, but the kink
    displacement is not. ``xi_0`` is measured in the original normalized
    Cartesian frame and is converted to the statically normalized radial
    coordinate using the wall distance opposite the kink.

    The outer stationary point of the radial kink map is found analytically
    with the ``-1`` branch of Lambert W. The transformed radius at that fold
    sets the clipping level. If the radial map is monotonic and has no fold,
    the unmodified kinked coordinates are returned.

    Kink-boundary renormalization is intentionally unsupported because it
    invalidates the Lambert W fold equation. A positive ``smooth_eps`` applies
    the same smoothing scale to both clipping operations.
    """
    xi_0 = _positive_scalar(xi_0, "xi_0")
    rho_s = _positive_scalar(rho_s, "rho_s")
    d = _positive_scalar(d, "d")
    if d <= 1:
        raise ValueError("d must be greater than one for a crescent profile")
    smooth_eps = _nonnegative_scalar(smooth_eps, "smooth_eps")
    rho_0, rho_kinked, theta_kinked = _kinked_coordinates(
        x,
        y,
        delta,
        xi_0,
        rho_s,
        d,
        center_angle_xy,
        False,
    )
    center_angle_xy = _finite_array(center_angle_xy, "center_angle_xy")
    opposite_boundary_radius = delta * np.cos(center_angle_xy) + np.sqrt(
        1 - delta**2 * np.sin(center_angle_xy) ** 2
    )
    effective_kink_amplitude = xi_0 / opposite_boundary_radius
    has_fold, _, effective_radius_at_fold = _crescent_folds(
        effective_kink_amplitude, rho_s, d
    )
    if not np.any(has_fold):
        return rho_kinked, theta_kinked

    rho_flat = smooth_maximum(effective_radius_at_fold, rho_0, eps=smooth_eps)
    rho_flattened = smooth_minimum(rho_kinked, rho_flat, eps=smooth_eps)
    return np.where(has_fold, rho_flattened, rho_kinked), theta_kinked


def _profile_from_rho(
    x: ArrayLike,
    y: ArrayLike,
    rho: ArrayLike,
    A: float,
    alpha: float,
    beta: float,
    edge_value: float = 0,
) -> FloatArray:
    """Scale a two-power shape within the unit poloidal disk."""
    A = _finite_scalar(A, "A")
    edge_value = _finite_scalar(edge_value, "edge_value")
    shape = two_power(rho, alpha, beta)
    return np.where(
        np.asarray(x) ** 2 + np.asarray(y) ** 2 <= 1,
        edge_value + (A - edge_value) * shape,
        0,
    )


def axisymmetric_profile(
    x: ArrayLike,
    y: ArrayLike,
    A: float,
    delta: float,
    alpha: float,
    beta: float,
    *,
    edge_value: float = 0,
    **kwargs: object,
) -> FloatArray:
    """Evaluate an axisymmetric two-power profile on poloidal coordinates.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    A : float
        Profile amplitude; it may carry application-defined units.
    delta : float
        Finite dimensionless horizontal shift with magnitude less than one.
    alpha, beta : float
        Finite positive two-power exponents.
    edge_value : float, default=0
        Profile value where the effective ``rho=1``. The central value
        remains ``A`` and the profile is zero where ``x**2 + y**2 > 1``.
    **kwargs : dict
        Ignored compatibility keywords, including an injected toroidal ``phi``.

    Returns
    -------
    ndarray
        Broadcast profile. Radius behavior follows :func:`two_power`.

    Raises
    ------
    ValueError
        If a profile parameter is not a finite scalar, if
        ``abs(delta) >= 1``, or if ``alpha`` or ``beta`` is not positive.
    """
    rho_0, _ = shifted_polar(x, y, delta, 0)
    return _profile_from_rho(x, y, rho_0, A, alpha, beta, edge_value)


def kinked_profile(
    x: ArrayLike,
    y: ArrayLike,
    A: float,
    delta: float,
    alpha: float,
    beta: float,
    xi_0: float,
    rho_s: float,
    d: float,
    *,
    center_angle_xy: ArrayLike = 0,
    normalize_kink: bool = False,
    edge_value: float = 0,
) -> FloatArray:
    """Evaluate a two-power profile on kink-displaced coordinates.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    A : float
        Profile amplitude; it may carry application-defined units.
    delta : float
        Finite dimensionless horizontal static shift with magnitude less than
        one.
    xi_0 : float
        Finite dimensionless Cartesian kink-displacement amplitude.
    rho_s : float
        Finite positive dimensionless decay radius.
    alpha, beta, d : float
        Finite positive two-power and decay exponents.
    center_angle_xy : array-like, default=0
        Kink displacement angle measured counter-clockwise from the positive
        poloidal ``x`` axis.
    normalize_kink : bool, default=False
        Renormalize the kink-displaced coordinates to the unit boundary.
    edge_value : float, default=0
        Profile value where the effective ``rho=1``. The central value
        remains ``A`` and the profile is zero where ``x**2 + y**2 > 1``.

    Returns
    -------
    ndarray
        Broadcast profile. Clipping follows
        :func:`kinked_rho` and :func:`two_power`.

    Raises
    ------
    ValueError
        If a scalar parameter is non-finite or has the wrong shape, if
        ``abs(delta) >= 1``, if ``rho_s``, ``alpha``, ``beta``, or ``d`` is
        not positive, if ``center_angle_xy`` contains a non-finite value, if
        ``normalize_kink`` is not boolean, or if a normalized kink origin
        reaches or crosses the unit boundary.
    """
    rho_kinked, _ = kinked_rho(
        x,
        y,
        delta,
        xi_0,
        rho_s,
        d,
        center_angle_xy=center_angle_xy,
        normalize_kink=normalize_kink,
    )
    return _profile_from_rho(x, y, rho_kinked, A, alpha, beta, edge_value)


def flattening_profile(
    x: ArrayLike,
    y: ArrayLike,
    A: float,
    delta: float,
    alpha: float,
    beta: float,
    xi_0: float,
    rho_s: float,
    d: float,
    *,
    localized: bool = False,
    lam_0: float = 1,
    rho_flat: float | None = None,
    w: float | None = None,
    gamma: float = 0,
    blend_edge: float | None = None,
    center_angle_xy: ArrayLike = 0,
    flattening_angle_offset: float = np.pi,
    smooth_eps: float = 0,
    normalize_kink: bool = False,
    edge_value: float = 0,
) -> FloatArray:
    """Model full-angle or localized island density flattening.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    A : float
        Profile amplitude; it may carry application-defined units.
    delta : float
        Finite dimensionless horizontal static shift with magnitude less than
        one.
    xi_0 : float
        Finite dimensionless Cartesian kink-displacement amplitude.
    rho_s : float
        Finite positive dimensionless kink-decay radius.
    alpha, beta, d : float
        Finite positive two-power and decay exponents.
    localized : bool, default=False
        If false, apply ``lam_0`` at every poloidal angle. If true, localize
        it with Gaussian radial and cosine angular weights.
    lam_0 : float, default=1
        Density-flattening fraction in the closed interval ``[0, 1]``.
    rho_flat : float or None, default=None
        Positive island-flattening radius. ``None`` uses ``rho_s``.
    w : float or None, default=None
        Positive Gaussian full e-folding width required when
        ``localized=True``. It is ignored for full-angle flattening.
    gamma : float, default=0
        Finite angular-weight distortion used only for localized flattening;
        ignored otherwise.
    blend_edge : float or None, default=None
        Optional positive taper width for the localized Gaussian weight;
        ignored when ``localized=False``.
    center_angle_xy : array-like, default=0
        Kink displacement angle measured counter-clockwise from the positive
        poloidal ``x`` axis.
    flattening_angle_offset : float, default=pi
        Angular offset of the localized island region relative to the kink;
        ignored when ``localized=False``.
    smooth_eps : float, default=0
        Nonnegative smoothing scale used for both radial clipping operations.
    normalize_kink : bool, default=False
        Renormalize the kink-displaced coordinates to the unit boundary.
    edge_value : float, default=0
        Profile value where the effective ``rho=1``. The central value
        remains ``A`` and the profile is zero where ``x**2 + y**2 > 1``.

    Returns
    -------
    ndarray
        Density blended between the kinked and island-flattened profiles.

    Raises
    ------
    ValueError
        If a scalar parameter is invalid, if ``localized`` or
        ``normalize_kink`` is not boolean, if ``lam_0`` lies outside
        ``[0, 1]``, if localized-only controls are inconsistent with the
        selected mode, or if the underlying coordinate transforms fail.

    Notes
    -----
    The island target is built by clipping ``rho_kink`` from above by
    ``max(rho_flat, rho_0)``. The returned density, rather than the radius,
    is linearly blended between the kinked and island target profiles.

    ``edge_value`` is the value at effective ``rho=1``. When
    ``normalize_kink=False``, it is not guaranteed to be the value everywhere
    on the physical unit-circle wall. Values outside that wall are zero.
    """
    localized = _bool_scalar(localized, "localized")
    lam_0 = _unit_interval_scalar(lam_0, "lam_0")
    rho_s = _positive_scalar(rho_s, "rho_s")
    if rho_flat is None:
        rho_flat = rho_s
    else:
        rho_flat = _positive_scalar(rho_flat, "rho_flat")
    if localized:
        if w is None:
            raise ValueError("w is required when localized=True")
        w = _positive_scalar(w, "w")
        gamma = _finite_scalar(gamma, "gamma")
        flattening_angle_offset = _finite_scalar(
            flattening_angle_offset, "flattening_angle_offset"
        )
        if blend_edge is not None:
            blend_edge = _positive_scalar(blend_edge, "blend_edge")

    rho_0, rho_kinked, theta_kinked = _kinked_coordinates(
        x,
        y,
        delta,
        xi_0,
        rho_s,
        d,
        center_angle_xy,
        normalize_kink,
    )
    smooth_eps = _nonnegative_scalar(smooth_eps, "smooth_eps")
    rho_island = _fully_flattened_rho(rho_0, rho_kinked, rho_flat, smooth_eps)
    n_kinked = _profile_from_rho(x, y, rho_kinked, A, alpha, beta, edge_value)
    n_island = _profile_from_rho(x, y, rho_island, A, alpha, beta, edge_value)

    if localized:
        center_angle_xy = _finite_array(center_angle_xy, "center_angle_xy")
        theta_distorted = _distort_theta(
            theta_kinked,
            gamma=gamma,
            reference_angle=center_angle_xy + flattening_angle_offset,
        )
        angular_weight = 0.5 * (1 + np.cos(theta_distorted))
        lam = (
            lam_0 * gaussian(rho_kinked, rho_flat, w, edge=blend_edge) * angular_weight
        )
    else:
        lam = lam_0
    return (1 - lam) * n_kinked + lam * n_island


def crescent_profile(
    x: ArrayLike,
    y: ArrayLike,
    A: float,
    delta: float,
    alpha: float,
    beta: float,
    xi_0: float,
    rho_s: float,
    d: float,
    *,
    center_angle_xy: ArrayLike = 0,
    smooth_eps: float = 0,
    edge_value: float = 0,
) -> FloatArray:
    """Evaluate a two-power profile on kinked and crescent-shaped coordinates.

    Parameters
    ----------
    x, y : array-like
        Dimensionless poloidal Cartesian coordinates.
    A : float
        Profile amplitude; it may carry application-defined units.
    delta : float
        Finite dimensionless horizontal static shift with magnitude less than
        one.
    xi_0 : float
        Finite positive dimensionless Cartesian kink-displacement amplitude.
    rho_s : float
        Finite positive dimensionless decay radius.
    alpha, beta : float
        Finite positive two-power exponents.
    d : float
        Finite decay exponent greater than one.
    center_angle_xy : array-like, default=0
        Kink displacement angle measured counter-clockwise from the positive
        poloidal ``x`` axis.
    smooth_eps : float, default=0
        Nonnegative smoothing scale for the maximum-minimum clip. Zero uses
        exact minimum and maximum operations.
    edge_value : float, default=0
        Profile value where the effective ``rho=1``. The central value
        remains ``A`` and the profile is zero where ``x**2 + y**2 > 1``.

    Returns
    -------
    ndarray
        Broadcast profile. Range clipping follows
        :func:`crescent_rho` and :func:`two_power`.

    Raises
    ------
    ValueError
        If a scalar parameter is non-finite or has the wrong shape, if
        ``abs(delta) >= 1``, if ``xi_0``, ``rho_s``, ``alpha``, or ``beta``
        is not positive, if ``d <= 1``, if ``smooth_eps`` is negative, or if
        ``center_angle_xy`` contains a non-finite value.

    Notes
    -----
    The horizontal static shift is boundary-normalized, but the kink is not.
    If the kinked radial map has no fold, this is identical to
    :func:`kinked_profile` with ``normalize_kink=False`` for the same shape
    parameters.
    """
    rho_flattened, _ = crescent_rho(
        x,
        y,
        delta=delta,
        xi_0=xi_0,
        rho_s=rho_s,
        d=d,
        center_angle_xy=center_angle_xy,
        smooth_eps=smooth_eps,
    )
    return _profile_from_rho(x, y, rho_flattened, A, alpha, beta, edge_value)
