"""Internal utility helpers for the multi-pinhole simulation package.

Import as ``multi_pinhole.utils`` (previously a top-level ``utils`` package
before the layout reorganization). Submodules:

* :mod:`multi_pinhole.utils.my_stdio` -- thin ``tqdm``-aware progress/print wrappers.
* :mod:`multi_pinhole.utils.stl_utils` -- STL mesh construction, transforms, and
  ray/visibility geometry primitives.

This package's own namespace also exposes :func:`type_check_and_list`, a
small normalization helper used throughout :mod:`multi_pinhole.world` and
:mod:`multi_pinhole.core` to accept either a single object or a list.
"""

# utils package
from . import my_stdio
from . import stl_utils

__all__ = ["my_stdio", "stl_utils", "type_check_and_list"]


def type_check_and_list(obj, type_, default=None):
    """
    Check the type of the input object and convert it to a list if it is not a list.

    Parameters
    ----------
    obj : Any
        The input object.
    type_ : type
        The type of the input object.
    default : Any
        The default value of the input object.

    Returns
    -------
    obj : list
        The input object as a list.
    """

    if obj is None:
        obj = default if default is not None else []
    elif isinstance(obj, type_):
        obj = [obj]
    elif isinstance(obj, list):
        for i, o in enumerate(obj):
            if not isinstance(o, type_):
                raise TypeError(f'The type of obj[{i}] is not {type_}.')
    else:
        raise TypeError(f'The type of obj is not {type_} or list.')

    return obj
