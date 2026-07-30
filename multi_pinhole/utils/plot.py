"""Plotting helpers for scalar fields defined on voxel centers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh

    from ..voxel import Voxel


def _voxel_values(voxel: Voxel, emission: Any) -> np.ndarray:
    """Validate and flatten one voxel-centered scalar field."""
    values = np.asarray(emission)
    if values.shape == voxel.shape:
        return values.ravel()
    if values.shape == (voxel.N_voxel,):
        return values
    raise ValueError(
        "emission must have shape "
        f"{voxel.shape} or ({voxel.N_voxel},), got {values.shape}"
    )


def _finite_volume_data(
    values: np.ndarray,
    points: np.ndarray,
    mask: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select finite values and optionally apply a voxel mask."""
    selected = np.isfinite(values)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool).ravel()
        if mask_array.shape != values.shape:
            raise ValueError(
                f"mask must have shape {values.shape}, got {mask_array.shape}"
            )
        selected &= mask_array
    if not np.any(selected):
        raise ValueError("emission contains no finite, selected voxels")
    return values[selected], points[selected]


def get_row_col(
    val_list: list[list[np.ndarray]] | list[np.ndarray] | np.ndarray,
) -> tuple[int, int, np.ndarray]:
    """Normalize a rectangular collection of scalar fields.

    Parameters
    ----------
    val_list : list or np.ndarray
        A flat or rectangular nested collection of equal-shaped fields.

    Returns
    -------
    rows, cols : int
        Subplot dimensions.
    values : np.ndarray
        Object or numeric array whose first two axes index subplots.
    """
    if not isinstance(val_list, (list, np.ndarray)):
        raise TypeError("val_list must be a list or numpy.ndarray")
    try:
        values = np.asarray(val_list)
    except ValueError as exc:
        raise ValueError("The columns of val_list must have equal length.") from exc
    values = np.array(values, ndmin=3)
    return values.shape[0], values.shape[1], values


def volume_rendering(
    f_val: np.ndarray,
    grid: np.ndarray,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
    isomin: float | None = None,
    isomax: float | None = None,
    opacity: float = 0.8,
    surface_count: int = 7,
    **volumekw: Any,
) -> go.Figure:
    """Add one Plotly volume trace from explicit point coordinates.

    This low-level compatibility API accepts a coordinate array. New code
    should normally use :func:`plot_voxel_volume`, which validates a
    :class:`~multi_pinhole.voxel.Voxel` and its emission field together.

    Parameters
    ----------
    f_val : array-like, shape (N,)
        Scalar values at ``grid`` points. Non-finite samples are omitted.
    grid : array-like, shape (N, 3)
        Cartesian coordinates in the user's length unit.
    fig : plotly.graph_objects.Figure, optional
        Existing figure.
    row, col : int, optional
        One-based subplot indices. Both are required with a subplot figure.
    isomin, isomax : float, optional
        Rendered scalar range. Finite data extrema are used by default.
    opacity : float, default 0.8
        Isosurface opacity.
    surface_count : int, default 7
        Number of isosurfaces.
    **volumekw
        Additional arguments for :class:`plotly.graph_objects.Volume`.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing the volume trace.
    """
    values = np.asarray(f_val).ravel()
    points = np.asarray(grid)
    if points.shape != (values.size, 3):
        raise ValueError(f"grid must have shape ({values.size}, 3), got {points.shape}")
    values, points = _finite_volume_data(values, points, mask=None)
    if fig is None:
        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "volume"}]])
        row, col = 1, 1
    elif (row is None) != (col is None):
        raise ValueError("row and col must be supplied together")

    isomin = float(values.min()) if isomin is None else isomin
    isomax = float(values.max()) if isomax is None else isomax
    fig.add_trace(
        go.Volume(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            value=values,
            isomin=isomin,
            isomax=isomax,
            opacity=opacity,
            surface_count=surface_count,
            **volumekw,
        ),
        row=row,
        col=col,
    )
    return fig


def plot_voxel_volume(
    voxel: Voxel,
    emission: Any,
    *,
    mask: Any | None = None,
    fig: go.Figure | None = None,
    row: int | None = None,
    col: int | None = None,
    length_unit: str = "",
    value_label: str = "Emission",
    **volume_kwargs: Any,
) -> go.Figure:
    """Render a voxel-centered emission profile as a Plotly volume.

    Parameters
    ----------
    voxel : Voxel
        Cartesian voxel grid defining the sample positions.
    emission : array-like, shape (N_voxel,) or voxel.shape
        Scalar emission at voxel gravity centers.
    mask : array-like of bool, shape (N_voxel,) or voxel.shape, optional
        Voxels to include. Non-finite emission is always omitted.
    fig : plotly.graph_objects.Figure, optional
        Existing figure.
    row, col : int, optional
        One-based subplot indices.
    length_unit : str, optional
        Coordinate unit appended to axis labels, for example ``"mm"``.
    value_label : str, default "Emission"
        Colorbar title.
    **volume_kwargs
        Arguments forwarded to :func:`volume_rendering`.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing the volume trace.
    """
    values = _voxel_values(voxel, emission)
    values, points = _finite_volume_data(values, voxel.gravity_center, mask)
    volume_kwargs.setdefault("colorbar_title", value_label)
    volume_kwargs.setdefault("opacity", 0.2)
    volume_kwargs.setdefault("surface_count", 15)
    fig = volume_rendering(values, points, fig=fig, row=row, col=col, **volume_kwargs)
    suffix = f" [{length_unit}]" if length_unit else ""
    scene_update = {
        "xaxis_title": f"x{suffix}",
        "yaxis_title": f"y{suffix}",
        "zaxis_title": f"z{suffix}",
        "aspectmode": "data",
    }
    fig.update_scenes(row=row, col=col, **scene_update)
    return fig


def plot_voxel_slice(
    voxel: Voxel,
    emission: Any,
    *,
    axis: str = "z",
    index: int | None = None,
    coordinate: float | None = None,
    ax: Axes | None = None,
    length_unit: str = "",
    colorbar: bool = True,
    colorbar_label: str = "Emission",
    **pcolormesh_kwargs: Any,
) -> tuple[Axes, QuadMesh]:
    """Plot one axis-aligned slice of a voxel-centered emission profile.

    Exactly one of ``index`` and ``coordinate`` may be supplied. A coordinate
    selects the nearest voxel-center plane; when both are omitted, the middle
    plane is used.

    Parameters
    ----------
    voxel : Voxel
        Cartesian voxel grid.
    emission : array-like, shape (N_voxel,) or voxel.shape
        Scalar values at voxel gravity centers.
    axis : {"x", "y", "z"}, default "z"
        Axis normal to the slice.
    index : int, optional
        Voxel-center plane index.
    coordinate : float, optional
        Physical coordinate whose nearest center plane is selected.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.
    length_unit : str, optional
        Coordinate unit appended to labels.
    colorbar : bool, default True
        Add a colorbar to the axes.
    colorbar_label : str, default "Emission"
        Colorbar label.
    **pcolormesh_kwargs
        Additional arguments for :meth:`matplotlib.axes.Axes.pcolormesh`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the slice.
    mesh : matplotlib.collections.QuadMesh
        Created pcolormesh artist.

    Raises
    ------
    ValueError
        If the axis, field shape, or selection arguments are invalid.
    IndexError
        If ``index`` is outside the chosen voxel axis.
    """
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    if index is not None and coordinate is not None:
        raise ValueError("index and coordinate are mutually exclusive")

    values = _voxel_values(voxel, emission).reshape(voxel.shape)
    axis_number = {"x": 0, "y": 1, "z": 2}[axis]
    centers = (voxel.cx_axis, voxel.cy_axis, voxel.cz_axis)
    if coordinate is not None:
        index = int(np.argmin(np.abs(centers[axis_number] - coordinate)))
    if index is None:
        index = voxel.shape[axis_number] // 2
    if not 0 <= index < voxel.shape[axis_number]:
        raise IndexError(f"{axis}-slice index {index} is out of range")

    plane = np.take(values, index, axis=axis_number)
    edge_axes = (voxel.x_axis, voxel.y_axis, voxel.z_axis)
    plotted_axes = [number for number in range(3) if number != axis_number]
    horizontal, vertical = plotted_axes
    if ax is None:
        _, ax = plt.subplots()
    mesh = ax.pcolormesh(
        edge_axes[horizontal],
        edge_axes[vertical],
        plane.T,
        shading="flat",
        **pcolormesh_kwargs,
    )
    suffix = f" [{length_unit}]" if length_unit else ""
    names = "xyz"
    ax.set_xlabel(f"{names[horizontal]}{suffix}")
    ax.set_ylabel(f"{names[vertical]}{suffix}")
    ax.set_aspect("equal")
    ax.set_title(f"{axis} = {centers[axis_number][index]:g}{suffix}")
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=colorbar_label)
    return ax, mesh


def multi_volume_rendering(
    val_list: list[np.ndarray] | list[list[np.ndarray]],
    grid: np.ndarray,
    fig: go.Figure | None = None,
    isomin: float | None = None,
    isomax: float | None = None,
    opacity: float = 0.8,
    surface_count: int = 7,
    **volumekw: Any,
) -> go.Figure:
    """Render a rectangular collection of fields on one shared point grid.

    Parameters
    ----------
    val_list : list
        Flat or rectangular nested collection of fields, each with shape
        ``(N,)``.
    grid : array-like, shape (N, 3)
        Shared Cartesian sample coordinates.
    fig : plotly.graph_objects.Figure, optional
        Existing subplot figure.
    isomin, isomax : float, optional
        Shared scalar range. Finite extrema across all fields are used by
        default.
    opacity : float, default 0.8
        Isosurface opacity.
    surface_count : int, default 7
        Number of isosurfaces per field.
    **volumekw
        Additional arguments for :func:`volume_rendering`.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with one volume trace per field.
    """
    rows, cols, values = get_row_col(val_list)
    if fig is None:
        fig = make_subplots(
            rows=rows, cols=cols, specs=[[{"type": "volume"}] * cols] * rows
        )
    finite_fields = [
        np.asarray(values[i, j]).ravel() for i in range(rows) for j in range(cols)
    ]
    finite_values = np.concatenate(
        [field[np.isfinite(field)] for field in finite_fields]
    )
    if finite_values.size == 0:
        raise ValueError("fields contain no finite values")
    shared_min = float(finite_values.min()) if isomin is None else isomin
    shared_max = float(finite_values.max()) if isomax is None else isomax
    for i in range(rows):
        for j in range(cols):
            volume_rendering(
                values[i, j],
                grid,
                fig=fig,
                row=i + 1,
                col=j + 1,
                isomin=shared_min,
                isomax=shared_max,
                opacity=opacity,
                surface_count=surface_count,
                **volumekw,
            )
    return fig
