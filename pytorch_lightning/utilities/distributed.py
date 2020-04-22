from functools import wraps
from typing import Callable
import warnings


class rank_zero_only(object):
    """ This is a decorator (in form of a class instead of function). """

    # this is a class attribute, it is constant across all instances of "rank_zero_only"
    # it's like the old rank global, but now it is limited to the scope of this class/decorator
    _rank = 0

    def __init__(self, fn):
        # this is the function we try to decorate.
        self._function = fn

    def __call__(self, *args, **kwargs):
        if self._rank == 0:
            return self._function(*args, **kwargs)

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, rank):
        self._rank = rank


def _warn(*args, **kwargs):
    warnings.warn(*args, **kwargs)


rank_zero_warn = rank_zero_only(_warn)
