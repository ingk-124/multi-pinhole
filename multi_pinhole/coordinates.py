"""Coordinate transforms used to evaluate fields on Cartesian voxel grids.

The voxel grid itself is always Cartesian. These helpers convert Cartesian
points into normalized coordinates for profile evaluation, for example:

    r, theta, phi = voxel.normalized_coordinates().T
"""

import numpy as np


COORDINATE_TYPES = ["cartesian", "torus", "torus_inverse", "cylindrical", "spherical"]
COORDINATE_PARAMETER_KEYS = {
    "cartesian": [["width", "depth", "height"], ["X", "Y", "Z"]],
    "torus": [["major_radius", "minor_radius"], ["R_0", "a"]],
    "torus_inverse": [["major_radius", "minor_radius"], ["R_0", "a"]],
    "cylindrical": [["radius", "height"], ["a", "h"]],
    "spherical": [["radius"], ["a"]],
}


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
