import sys

from tqdm import trange, tqdm
from tqdm.contrib import tzip

NCOLS = 80
def my_print(s, show=True):
    if show:
        print(s)


def my_range(*args, **kwargs):
    return trange(*args, **kwargs,
                  ncols=NCOLS,
                  # file=sys.stdout,
                  position=0)


def my_tqdm(iterable, *args, **kwargs):
    return tqdm(iterable, *args, **kwargs,
                ncols=NCOLS,
                # file=sys.stdout,
                position=0)


def my_zip(*iterables, **kwargs):
    return tzip(*iterables, **kwargs,
                ncols=NCOLS,
                # file=sys.stdout,
                position=0)
