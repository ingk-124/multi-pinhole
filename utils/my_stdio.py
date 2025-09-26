from tqdm import trange, tqdm
from tqdm.contrib import tzip


def my_print(s, show=True):
    if show:
        print(s)


def my_range(*args, verbose=True, **kwargs):
    if verbose:
        return trange(*args, **kwargs)
    else:
        return range(*args)


def my_tqdm(iterable, verbose=True, *args, **kwargs):
    if verbose:
        return tqdm(iterable, *args, **kwargs)
    else:
        return iterable


def my_zip(*iterables, verbose=True, **kwargs):
    if verbose:
        return tzip(*iterables, **kwargs)
    else:
        return zip(*iterables)
