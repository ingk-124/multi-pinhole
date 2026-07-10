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
