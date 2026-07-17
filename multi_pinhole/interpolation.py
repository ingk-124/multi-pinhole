"""User-facing interpolation of values sampled at Voxel gravity centers."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class VoxelCenterInterpolator:
    """Interpolate scalar or vector values sampled on a Voxel center grid.

    Instances are created by :meth:`multi_pinhole.voxel.Voxel.center_interpolator`.
    Query points may be Cartesian arrays or broadcastable keyword components
    in any coordinate convention supported by the owning Voxel.
    """

    def __init__(self, voxel, values, **interpolator_kwargs):
        values = np.asarray(values)
        grid_shape = tuple(voxel.voxel_shape)
        if values.shape[:3] == grid_shape:
            grid_values = values
        elif values.ndim >= 1 and values.shape[0] == voxel.N_voxel:
            grid_values = values.reshape(grid_shape + values.shape[1:])
        else:
            raise ValueError(
                "values must have shape voxel.voxel_shape + value_shape or "
                "(voxel.N_voxel,) + value_shape"
            )
        self._voxel = voxel
        self._interpolator = RegularGridInterpolator(
            (voxel.cx_axis, voxel.cy_axis, voxel.cz_axis),
            grid_values,
            **interpolator_kwargs,
        )

    def __call__(self, points=None, *, coordinate_type=None,
                 normalized=False, **components):
        """Evaluate the center-grid field at Cartesian or named coordinates.

        Parameters
        ----------
        points : array-like, optional
            Cartesian query points with shape ``(..., 3)``. Required when
            ``coordinate_type`` is ``None`` and forbidden otherwise.
        coordinate_type : str, optional
            Coordinate convention for keyword ``components``. When supplied,
            components are converted to Cartesian before interpolation.
        normalized : bool, default=False
            Whether keyword radial/axial components are normalized.
        **components
            Broadcastable keyword coordinate components and required scales.

        Returns
        -------
        np.ndarray
            Interpolated values with the query leading shape followed by the
            field's optional value shape.

        Raises
        ------
        ValueError
            If Cartesian points and named components are missing, mixed, or
            inconsistent with ``normalized``.
        """
        if coordinate_type is None:
            if points is None:
                raise ValueError("points are required when coordinate_type is None")
            if components:
                raise ValueError(
                    "coordinate components require an explicit coordinate_type"
                )
            if normalized:
                raise ValueError(
                    "normalized=True requires an explicit coordinate_type"
                )
            cartesian = np.asarray(points, dtype=float)
            if cartesian.ndim == 0 or cartesian.shape[-1] != 3:
                raise ValueError("points must have shape (..., 3)")
        else:
            if points is not None:
                raise ValueError(
                    "points and coordinate_type/components are mutually exclusive"
                )
            cartesian = self._voxel.from_coordinates(
                coordinate_type,
                normalized=normalized,
                **components,
            )
        single_point = cartesian.ndim == 1
        result = self._interpolator(cartesian)
        return result[0] if single_point else result
