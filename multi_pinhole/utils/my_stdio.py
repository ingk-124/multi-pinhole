"""Thin, verbosity-aware wrappers around ``print`` and ``tqdm`` primitives.

These helpers let long-running geometry/projection routines expose optional
progress reporting without hard-coding a dependency on ``tqdm`` at every call
site: pass a boolean ``show``/``disable`` flag (typically derived from a
``verbose`` argument) to toggle output on or off.
"""

import sys

from tqdm import trange, tqdm
from tqdm.contrib import tzip

NCOLS = 80
def my_print(s, show=True):
    """Print ``s`` when ``show`` is true; otherwise do nothing.

    Parameters
    ----------
    s : Any
        Value passed to the built-in ``print``.
    show : bool, optional
        Gate controlling whether ``s`` is printed. Defaults to ``True``.
    """
    if show:
        print(s)


def my_range(*args, **kwargs):
    """Return a ``tqdm.trange`` progress-bar iterator with fixed formatting.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :func:`tqdm.trange` (typically the
        ``range``-style ``start``/``stop``/``step`` values).
    **kwargs
        Keyword arguments forwarded to :func:`tqdm.trange`, e.g. ``disable``
        to suppress the progress bar or ``desc`` to label it.

    Returns
    -------
    tqdm.std.tqdm
        Progress-bar-wrapped range iterator with ``ncols=80`` and
        ``position=0``.
    """
    return trange(*args, **kwargs,
                  ncols=NCOLS,
                  # file=sys.stdout,
                  position=0)


def my_tqdm(iterable, *args, **kwargs):
    """Wrap ``iterable`` with a ``tqdm`` progress bar using fixed formatting.

    Parameters
    ----------
    iterable : Iterable
        Iterable to wrap.
    *args
        Additional positional arguments forwarded to :func:`tqdm.tqdm`.
    **kwargs
        Keyword arguments forwarded to :func:`tqdm.tqdm`, e.g. ``disable`` to
        suppress the progress bar or ``desc`` to label it.

    Returns
    -------
    tqdm.std.tqdm
        Progress-bar-wrapped iterator with ``ncols=80`` and ``position=0``.
    """
    return tqdm(iterable, *args, **kwargs,
                ncols=NCOLS,
                # file=sys.stdout,
                position=0)


def my_zip(*iterables, **kwargs):
    """Zip ``iterables`` together while displaying a ``tqdm`` progress bar.

    Parameters
    ----------
    *iterables
        Iterables to zip, forwarded to :func:`tqdm.contrib.tzip`.
    **kwargs
        Keyword arguments forwarded to :func:`tqdm.contrib.tzip`, e.g.
        ``disable`` to suppress the progress bar or ``desc`` to label it.

    Returns
    -------
    Iterator
        Progress-bar-wrapped iterator yielding tuples like the built-in
        ``zip``.
    """
    return tzip(*iterables, **kwargs,
                ncols=NCOLS,
                # file=sys.stdout,
                position=0)
