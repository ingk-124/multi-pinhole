"""Physical and normalized transforms for Cartesian voxel-grid points.

The voxel grid itself is always Cartesian. These helpers convert Cartesian
points into normalized coordinates for profile evaluation, for example:

    r, theta, phi = voxel.normalized_coordinates().T
"""

import numpy as np


COORDINATE_TYPES = (
    "cartesian",
    "torus",
    "torus_inverse",
    "cylindrical",
    "spherical",
)
COORDINATE_PARAMETER_KEYS = {
    "cartesian": [["width", "depth", "height"], ["X", "Y", "Z"]],
    "torus": [["major_radius", "minor_radius"], ["R_0", "a"]],
    "torus_inverse": [["major_radius", "minor_radius"], ["R_0", "a"]],
    "cylindrical": [["radius", "height"], ["a", "h"]],
    "spherical": [["radius"], ["a"]],
}

_COORDINATE_COMPONENT_KEYS = {
    "cartesian": ("x", "y", "z"),
    "torus": ("rho", "theta", "phi"),
    "torus_inverse": ("rho", "theta", "phi"),
    "cylindrical": ("R", "phi", "Z"),
    "spherical": ("r", "theta", "phi"),
}


def _missing_component_error(coordinate_type, components):
    """Return an informative error for an incomplete component set."""
    required = _COORDINATE_COMPONENT_KEYS[coordinate_type]
    missing = [name for name in required if name not in components]
    return ValueError(
        f"{coordinate_type} coordinates require components "
        f"{', '.join(required)}; missing {', '.join(missing)}"
    )


def cartesian_coordinates(width: float, depth: float, height: float):
    """Build a normalized Cartesian-coordinate transform.

    Parameters
    ----------
    width, depth, height : float
        Full extents along ``x``, ``y``, and ``z``. Nonzero, positive values
        are intended. Use one consistent length unit (mm in the examples).

    Returns
    -------
    callable
        Function mapping Cartesian ``points``, shape ``(n, 3)``, to
        dimensionless normalized coordinates with shape ``(n, 3)``.

    Notes
    -----
    Extent signs are ignored. Zero extents and non-finite values follow NumPy
    division rules; this factory performs no validation.
    """
    scale = np.abs([width / 2, depth / 2, height / 2])

    def normalized_coordinates(points: np.ndarray):
        return points / scale[None, :]

    return normalized_coordinates


def torus_coordinates(major_radius: float, minor_radius: float):
    """Return the standard right-handed normalized torus coordinate transform.

    Parameters
    ----------
    major_radius : float
        Major radius ``R0`` in the same length unit as input points. Positive
        values are physically intended but are not validated.
    minor_radius : float
        Minor reference radius ``a``. It must be nonzero for a finite
        normalized radius; positive values are physically intended.

    Returns
    -------
    callable
        Function mapping Cartesian ``(x, y, z)`` points, shape ``(n, 3)``, to
        ``(r, theta, phi)``, shape ``(n, 3)``. ``r`` is dimensionless and both
        angles are in radians.

    Notes
    -----
    The input points are Cartesian ``(x, y, z)``. Returned coordinates are
    ``(r, theta, phi)`` where:

    * ``r = sqrt((R - R0)^2 + z^2) / a``
    * ``theta = atan2(z, R - R0)``
    * ``phi = atan2(-y, x)``

    This convention places ``theta=0`` on the outboard midplane and makes
    ``phi`` increase clockwise when viewed from ``+z``. The ordered basis
    ``(r, theta, phi)`` is right-handed.
    ``theta`` lies in ``[-pi, pi]`` and is undefined on the toroidal centerline
    ``R=R0, z=0``. ``phi`` lies in ``[-pi, pi]`` and is undefined on the
    ``z`` axis; signed-zero behavior follows :func:`numpy.arctan2`. A zero
    minor radius yields NumPy division results, and non-finite values propagate.
    """
    R_0 = major_radius
    a = minor_radius

    def normalized_coordinates(points: np.ndarray):
        R = np.linalg.norm(points[:, :2], axis=1)
        r = np.linalg.norm([R - R_0, points[:, 2]], axis=0) / a
        theta = np.arctan2(points[:, 2], R - R_0)
        phi = np.arctan2(-points[:, 1], points[:, 0])
        return np.stack([r, theta, phi], axis=1)

    return normalized_coordinates


def torus_inverse_coordinates(major_radius: float, minor_radius: float):
    """Return the inverse-angle right-handed normalized torus coordinate transform.

    Parameters
    ----------
    major_radius : float
        Major radius ``R0`` in the same length unit as input points. Positive
        values are physically intended but are not validated.
    minor_radius : float
        Minor reference radius ``a``. It must be nonzero for a finite
        normalized radius; positive values are physically intended.

    Returns
    -------
    callable
        Function mapping Cartesian ``(x, y, z)`` points, shape ``(n, 3)``, to
        ``(r, theta, phi)``, shape ``(n, 3)``. ``r`` is dimensionless and both
        angles are in radians.

    Notes
    -----
    Compared with :func:`torus_coordinates`, both angular directions are
    reversed:

    * ``theta = atan2(z, R0 - R)``
    * ``phi = atan2(y, x)``

    This places ``theta=0`` on the inboard midplane and makes ``phi`` increase
    counter-clockwise when viewed from ``+z``. The ordered basis
    ``(r, theta, phi)`` remains right-handed.
    Both angles lie in ``[-pi, pi]``. The poloidal angle is undefined on the
    toroidal centerline and azimuth is undefined on the ``z`` axis. No shape,
    positivity, or finiteness validation is performed; NumPy behavior applies.
    """
    R_0 = major_radius
    a = minor_radius

    def normalized_coordinates(points: np.ndarray):
        R = np.linalg.norm(points[:, :2], axis=1)
        r = np.linalg.norm([R - R_0, points[:, 2]], axis=0) / a
        theta = np.arctan2(points[:, 2], R_0 - R)
        phi = np.arctan2(points[:, 1], points[:, 0])
        return np.stack([r, theta, phi], axis=1)

    return normalized_coordinates


def cylindrical_coordinates(radius: float, height: float):
    """Build a normalized cylindrical-coordinate transform.

    Parameters
    ----------
    radius : float
        Nonzero radial scale in the same length unit as ``points``.
    height : float
        Nonzero full axial extent in the same length unit as ``points``.

    Returns
    -------
    callable
        Function mapping Cartesian ``points``, shape ``(n, 3)``, to
        ``(r, theta, z)``, shape ``(n, 3)``. ``r`` and ``z`` are dimensionless;
        ``theta=atan2(y, x)`` is in radians, increasing counter-clockwise from
        ``+x`` when viewed from ``+z``.

    Notes
    -----
    The angle is undefined on the axis (NumPy returns zero at the origin).
    Zero scales and non-finite inputs are not validated.
    """
    a = radius
    h = height

    def normalized_coordinates(points: np.ndarray):
        r = np.linalg.norm(points[:, :2], axis=1) / a
        theta = np.arctan2(points[:, 1], points[:, 0])
        z = points[:, 2] / (h / 2)
        return np.stack([r, theta, z], axis=1)

    return normalized_coordinates


def spherical_coordinates(radius: float):
    """Build a normalized spherical-coordinate transform.

    Parameters
    ----------
    radius : float
        Nonzero radial scale in the same length unit as ``points``.

    Returns
    -------
    callable
        Function mapping Cartesian ``points``, shape ``(n, 3)``, to
        ``(r, theta, phi)``, shape ``(n, 3)``. ``r`` is dimensionless;
        ``theta`` is polar from ``+z`` and ``phi`` is counter-clockwise from
        ``+x``, in radians.

    Notes
    -----
    The reference radius scales only ``r``; it does not affect either angle.
    At the origin ``theta`` is ``nan``. Azimuth is mathematically undefined on
    the ``z`` axis, where the result follows :func:`numpy.arctan2`. Non-finite
    inputs propagate according to NumPy. This factory performs no validation.
    """
    a = radius

    def normalized_coordinates(points: np.ndarray):
        distance = np.linalg.norm(points, axis=1)
        r = distance / a
        cos_theta = np.divide(
            points[:, 2],
            distance,
            out=np.full_like(distance, np.nan),
            where=distance != 0,
        )
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        phi = np.arctan2(points[:, 1], points[:, 0])
        return np.stack([r, theta, phi], axis=1)

    return normalized_coordinates


def coordinate_transform(coordinate_type: str, coordinate_parameters: dict):
    """Build a normalized-coordinate transform selected by name.

    Parameters
    ----------
    coordinate_type : {"cartesian", "torus", "torus_inverse", "cylindrical", "spherical"}
        Coordinate convention to construct.
    coordinate_parameters : dict
        Keyword arguments required by the selected factory: ``width``,
        ``depth``, and ``height``; ``major_radius`` and ``minor_radius``;
        ``radius`` and ``height``; or ``radius``, respectively. Voxel accepts
        alternative key sets listed in :data:`COORDINATE_PARAMETER_KEYS`, but
        this low-level dispatcher passes keys directly to the factory.

    Returns
    -------
    callable
        Function accepting Cartesian ``points`` with shape ``(n, 3)`` and
        returning coordinates with shape ``(n, 3)``.

    Raises
    ------
    ValueError
        If ``coordinate_type`` is unsupported.
    TypeError
        If parameter keys do not match the selected factory.
    """
    if coordinate_type == "cartesian":
        return cartesian_coordinates(**coordinate_parameters)
    if coordinate_type == "torus":
        return torus_coordinates(**coordinate_parameters)
    if coordinate_type == "torus_inverse":
        return torus_inverse_coordinates(**coordinate_parameters)
    if coordinate_type == "cylindrical":
        return cylindrical_coordinates(**coordinate_parameters)
    if coordinate_type == "spherical":
        return spherical_coordinates(**coordinate_parameters)
    raise ValueError(f"Unsupported coordinate_type: {coordinate_type}")


def _parameter(parameters, name, alias=None, *, required=False):
    value = parameters.get(name)
    if value is None and alias is not None:
        value = parameters.get(alias)
    if required and value is None:
        raise ValueError(f"{name} is required for this coordinate conversion")
    return value


def _positive_scale(value, name):
    if value is None:
        raise ValueError(f"{name} is required when normalized=True")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def convert_from_cartesian(points, coordinate_type: str, *, normalized=False,
                           **coordinate_parameters):
    """Convert Cartesian points to a selected coordinate convention.

    Unlike the legacy transform factories, this function can return physical
    coordinates. Input leading dimensions are preserved.

    Parameters
    ----------
    points : array-like
        Cartesian coordinates with shape ``(..., 3)``.
    coordinate_type : {"cartesian", "torus", "torus_inverse", "cylindrical", "spherical"}
        Convention of the returned coordinates.
    normalized : bool, default=False
        Whether to divide physical radial/axial components by their supplied
        reference scales. Angles are always radians and are never scaled.
    **coordinate_parameters
        Geometry and scale parameters. Normalized output requires every scale
        for the selected convention. Toroidal output always requires
        ``major_radius`` because it defines the coordinate geometry.

    Returns
    -------
    np.ndarray
        Converted coordinates with the same leading shape as ``points`` and
        final axis length three.

    Raises
    ------
    ValueError
        If shapes, coordinate names, required parameters, or scales are invalid.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")

    x, y, z = np.moveaxis(points, -1, 0)
    if coordinate_type == "cartesian":
        if not normalized:
            return points.copy()
        width = _positive_scale(
            _parameter(coordinate_parameters, "width", "X"), "width",
        )
        depth = _positive_scale(
            _parameter(coordinate_parameters, "depth", "Y"), "depth",
        )
        height = _positive_scale(
            _parameter(coordinate_parameters, "height", "Z"), "height",
        )
        return points / np.array([width / 2, depth / 2, height / 2])

    R = np.hypot(x, y)
    if coordinate_type == "cylindrical":
        radial = R
        axial = z
        if normalized:
            radius = _positive_scale(
                _parameter(coordinate_parameters, "radius", "a"), "radius",
            )
            height = _positive_scale(
                _parameter(coordinate_parameters, "height", "h"), "height",
            )
            radial = radial / radius
            axial = axial / (height / 2)
        phi = np.arctan2(y, x)
        return np.stack([radial, phi, axial], axis=-1)

    if coordinate_type in ("torus", "torus_inverse"):
        major_radius = _parameter(
            coordinate_parameters, "major_radius", "R_0", required=True,
        )
        rho = np.hypot(R - major_radius, z)
        if normalized:
            minor_radius = _positive_scale(
                _parameter(coordinate_parameters, "minor_radius", "a"),
                "minor_radius",
            )
            rho = rho / minor_radius
        if coordinate_type == "torus":
            theta = np.arctan2(z, R - major_radius)
            phi = np.arctan2(-y, x)
        else:
            theta = np.arctan2(z, major_radius - R)
            phi = np.arctan2(y, x)
        return np.stack([rho, theta, phi], axis=-1)

    if coordinate_type == "spherical":
        distance = np.linalg.norm(points, axis=-1)
        radial = distance
        if normalized:
            radius = _positive_scale(
                _parameter(coordinate_parameters, "radius", "a"), "radius",
            )
            radial = radial / radius
        cos_theta = np.divide(
            z, distance, out=np.full_like(distance, np.nan),
            where=distance != 0,
        )
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        phi = np.arctan2(y, x)
        return np.stack([radial, theta, phi], axis=-1)

    raise ValueError(f"Unsupported coordinate_type: {coordinate_type}")


def convert_to_cartesian(coordinate_type: str, *, normalized=False,
                         **components):
    """Convert broadcastable keyword coordinate components to Cartesian.

    Parameters
    ----------
    coordinate_type : {"cartesian", "torus", "torus_inverse", "cylindrical", "spherical"}
        Convention of the keyword coordinate components.
    normalized : bool, default=False
        Whether radial/axial inputs are dimensionless normalized components.
        The corresponding physical scales are then required.
    **components
        Broadcastable keyword components. Cylindrical inputs are ``R``,
        ``phi``, and ``Z``; spherical inputs are ``r``, ``theta``, and
        ``phi``; toroidal inputs are ``rho``, ``theta``, and ``phi``.

    Returns
    -------
    np.ndarray
        Cartesian coordinates with shape ``broadcast_shape + (3,)``.

    Raises
    ------
    ValueError
        If a component, geometry parameter, or normalized scale is missing or
        invalid, or if ``coordinate_type`` is unsupported.
    """
    if coordinate_type == "cartesian":
        try:
            x, y, z = np.broadcast_arrays(
                components["x"], components["y"], components["z"],
            )
        except KeyError:
            raise _missing_component_error("cartesian", components) from None
        x, y, z = (np.asarray(value, dtype=float) for value in (x, y, z))
        if normalized:
            width = _positive_scale(
                _parameter(components, "width", "X"), "width",
            )
            depth = _positive_scale(
                _parameter(components, "depth", "Y"), "depth",
            )
            height = _positive_scale(
                _parameter(components, "height", "Z"), "height",
            )
            x, y, z = x * width / 2, y * depth / 2, z * height / 2
        return np.stack([x, y, z], axis=-1)

    if coordinate_type == "cylindrical":
        try:
            R, phi, Z = np.broadcast_arrays(
                components["R"], components["phi"], components["Z"],
            )
        except KeyError:
            raise _missing_component_error("cylindrical", components) from None
        R, phi, Z = (np.asarray(value, dtype=float) for value in (R, phi, Z))
        if normalized:
            radius = _positive_scale(
                _parameter(components, "radius", "a"), "radius",
            )
            height = _positive_scale(
                _parameter(components, "height", "h"), "height",
            )
            R, Z = R * radius, Z * height / 2
        return np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=-1)

    if coordinate_type in ("torus", "torus_inverse"):
        try:
            rho, theta, phi = np.broadcast_arrays(
                components["rho"], components["theta"], components["phi"],
            )
        except KeyError:
            raise _missing_component_error(coordinate_type, components) from None
        rho, theta, phi = (
            np.asarray(value, dtype=float) for value in (rho, theta, phi)
        )
        major_radius = float(_parameter(
            components, "major_radius", "R_0", required=True,
        ))
        if normalized:
            minor_radius = _positive_scale(
                _parameter(components, "minor_radius", "a"), "minor_radius",
            )
            rho = rho * minor_radius
        if coordinate_type == "torus":
            R = major_radius + rho * np.cos(theta)
            x, y = R * np.cos(phi), -R * np.sin(phi)
        else:
            R = major_radius - rho * np.cos(theta)
            x, y = R * np.cos(phi), R * np.sin(phi)
        z = rho * np.sin(theta)
        return np.stack([x, y, z], axis=-1)

    if coordinate_type == "spherical":
        try:
            r, theta, phi = np.broadcast_arrays(
                components["r"], components["theta"], components["phi"],
            )
        except KeyError:
            raise _missing_component_error("spherical", components) from None
        r, theta, phi = (
            np.asarray(value, dtype=float) for value in (r, theta, phi)
        )
        if normalized:
            radius = _positive_scale(
                _parameter(components, "radius", "a"), "radius",
            )
            r = r * radius
        sin_theta = np.sin(theta)
        return np.stack([
            r * sin_theta * np.cos(phi),
            r * sin_theta * np.sin(phi),
            r * np.cos(theta),
        ], axis=-1)

    raise ValueError(
        f"Unsupported coordinate_type: {coordinate_type!r}; "
        f"choose one of {', '.join(COORDINATE_TYPES)}"
    )
