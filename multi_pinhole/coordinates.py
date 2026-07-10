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
    """Return a transform from Cartesian points to normalized Cartesian coordinates."""
    scale = np.abs([width / 2, depth / 2, height / 2])

    def normalized_coordinates(points: np.ndarray):
        return points / scale[None, :]

    return normalized_coordinates


def torus_coordinates(major_radius: float, minor_radius: float):
    """Return the standard right-handed normalized torus coordinate transform.

    The input points are Cartesian ``(x, y, z)``. Returned coordinates are
    ``(r, theta, phi)`` where:

    * ``r = sqrt((R - R0)^2 + z^2) / a``
    * ``theta = atan2(z, R - R0)``
    * ``phi = atan2(-y, x)``

    This convention places ``theta=0`` on the outboard midplane and makes
    ``phi`` increase clockwise when viewed from ``+z``. The ordered basis
    ``(r, theta, phi)`` is right-handed.
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

    Compared with :func:`torus_coordinates`, both angular directions are
    reversed:

    * ``theta = atan2(z, R0 - R)``
    * ``phi = atan2(y, x)``

    This places ``theta=0`` on the inboard midplane and makes ``phi`` increase
    counter-clockwise when viewed from ``+z``. The ordered basis
    ``(r, theta, phi)`` remains right-handed.
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
    """Return a transform from Cartesian points to normalized cylindrical coordinates."""
    a = radius
    h = height

    def normalized_coordinates(points: np.ndarray):
        r = np.linalg.norm(points[:, :2], axis=1) / a
        theta = np.arctan2(points[:, 1], points[:, 0])
        z = points[:, 2] / (h / 2)
        return np.stack([r, theta, z], axis=1)

    return normalized_coordinates


def spherical_coordinates(radius: float):
    """Return a transform from Cartesian points to normalized spherical coordinates."""
    a = radius

    def normalized_coordinates(points: np.ndarray):
        r = np.linalg.norm(points, axis=1) / a
        theta = np.arccos(points[:, 2] / r)
        phi = np.arctan2(points[:, 1], points[:, 0])
        return np.stack([r, theta, phi], axis=1)

    return normalized_coordinates


def coordinate_transform(coordinate_type: str, coordinate_parameters: dict):
    """Build the normalized-coordinate transform for ``coordinate_type``."""
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
