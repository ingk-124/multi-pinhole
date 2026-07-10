"""Plotly-based 3D volume-rendering helpers for voxel field data.

:func:`volume_rendering` adds a single ``plotly.graph_objects.Volume`` trace
for one field; :func:`multi_volume_rendering` arranges several fields (given
as a nested list) into a grid of volume-rendering subplots for side-by-side
comparison.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_row_col(val_list: list[list[np.ndarray]] | list[np.ndarray] | np.ndarray) -> tuple[int, int, np.ndarray]:
    """Get the number of rows and columns of a 2d list.

    Parameters
    ----------
    val_list : list[list[np.ndarray]] | list[np.ndarray] | np.ndarray
        The list of values (2d list).

    Returns
    -------
    rows : int
        The number of rows of the 2d list.
    cols : int
        The number of columns of the 2d list.

    Raises
    ------
    ValueError
        If the columns of the 2d list are not the same.
    """
    if isinstance(val_list, list):
        try:
            val_list = np.asarray(val_list)
        except ValueError:
            raise ValueError('The columns of the 2d list are not the same.')
    elif isinstance(val_list, np.ndarray):
        pass
    else:
        raise TypeError('The type of val_list is not supported.')

    val_list = np.array(val_list, ndmin=3)
    rows, cols = val_list.shape[:2]
    return rows, cols, val_list


def volume_rendering(f_val, grid, fig=None, row=None, col=None,
                     isomin=None, isomax=None, opacity=0.8, surface_count=7, **volumekw):
    """Add a Plotly 3D volume-rendering trace for a scalar field on a grid.

    Parameters
    ----------
    f_val : np.ndarray
        Scalar field values, one per grid point (any shape; flattened via
        ``.ravel()``).
    grid : np.ndarray
        Grid point coordinates with shape ``(N, 3)``, e.g.
        ``voxel.gravity_center``. Unpacked into ``X, Y, Z = grid.T``.
    fig : plotly.graph_objects.Figure, optional
        Figure to add the trace to. When ``None`` (default), a new
        single-subplot figure with a ``"volume"`` scene is created and
        ``row``/``col`` are set to ``1``.
    row : int, optional
        Subplot row to add the trace to (1-indexed). Required together with
        ``col`` when ``fig`` is provided.
    col : int, optional
        Subplot column to add the trace to (1-indexed). Required together
        with ``row`` when ``fig`` is provided.
    isomin : float, optional
        Minimum isosurface value. Defaults to ``f_val.min()``.
    isomax : float, optional
        Maximum isosurface value. Defaults to ``f_val.max()``.
    opacity : float, optional
        Opacity of each isosurface. Defaults to ``0.8``.
    surface_count : int, optional
        Number of isosurfaces to render. Defaults to ``7``.
    **volumekw
        Additional keyword arguments forwarded to
        ``plotly.graph_objects.Volume``.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure with the new volume trace added (either ``fig`` or a
        newly created figure).
    """
    if fig is None:
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'volume'}]])
        row = 1
        col = 1

    X, Y, Z = grid.T

    if isomax is None:
        isomax = f_val.max()
    if isomin is None:
        isomin = f_val.min()
    fig.add_trace(go.Volume(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=f_val.ravel() + 0.,
                            isomin=isomin, isomax=isomax, opacity=opacity,
                            surface_count=surface_count, **volumekw),
                  row=row, col=col)
    return fig


def multi_volume_rendering(val_list: list[np.ndarray] | list[list[np.ndarray]], grid: np.ndarray,
                           fig=None, isomin=10, isomax=None, opacity=0.8, surface_count=7, **volumekw):
    """Render a grid of Plotly 3D volume subplots, one per field in ``val_list``.

    Parameters
    ----------
    val_list : list[np.ndarray] or list[list[np.ndarray]]
        Scalar fields to render, one per subplot. A flat list produces a
        single row; a nested list is treated as ``(row, col)`` fields (see
        :func:`get_row_col`), all sharing the same ``grid``.
    grid : np.ndarray
        Grid point coordinates with shape ``(N, 3)``, shared by every field.
    fig : plotly.graph_objects.Figure, optional
        Figure to add traces to. When ``None`` (default), a new figure with
        one ``"volume"`` scene per subplot is created.
    isomin : float, optional
        Minimum isosurface value used for every subplot. Defaults to ``10``.
    isomax : float, optional
        Maximum isosurface value. When ``None`` (default) it is computed
        from the first rendered field's maximum and then reused, unchanged,
        for all subsequent subplots (see Notes).
    opacity : float, optional
        Opacity of each isosurface. Defaults to ``0.8``.
    surface_count : int, optional
        Number of isosurfaces per subplot. Defaults to ``7``.
    **volumekw
        Additional keyword arguments forwarded to :func:`volume_rendering`
        for every subplot.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing one volume-rendering subplot per field.

    Notes
    -----
    When ``isomax`` is left as ``None``, it is only computed once (from the
    first ``(row, col)`` field processed) and then reused for every other
    subplot, rather than being recomputed per field. Pass an explicit
    ``isomax`` if each subplot needs its own scale.
    """
    rows, cols, val_list = get_row_col(val_list)
    if fig is None:
        fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'volume'}] * cols] * rows)

    for i in range(rows):
        for j in range(cols):
            f_val = val_list[i, j].ravel()
            isomax = f_val.max() if isomax is None else isomax
            volume_rendering(f_val, grid, fig=fig, row=i + 1, col=j + 1,
                             isomin=isomin, isomax=isomax, opacity=opacity,
                             surface_count=surface_count, **volumekw)
    return fig


if __name__ == '__main__':
    a = np.arange(5)
    b = np.arange(5) + 10
    c = np.arange(5) + 20

    print(get_row_col([a, b, c]))
    print(get_row_col([a, b, c, a]))
    print(get_row_col([[a, b],
                       [c, a]]))
