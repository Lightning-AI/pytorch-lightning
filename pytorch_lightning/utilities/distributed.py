from functools import wraps
from typing import Callable
import warnings

_proc_rank = 0


def rank_zero_only(fn: Callable):
    """Decorate a method to run it only on the process with rank 0.

    Args:
        fn: Function to decorate
    """
    global _proc_rank
    @wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        if _proc_rank == 0:
            fn(self, *args, **kwargs)

    return wrapped_fn


def set_proc_rank(value: int) -> None:
    """Set the (sub)process rank."""
    global _proc_rank
    _proc_rank = value


def get_proc_rank() -> int:
    """Set the (sub)process rank."""
    global _proc_rank
    return _proc_rank


def rank_zero_warn(*args, **kwargs) -> None:
    """Warning only if (sub)process has rank 0."""
    global _proc_rank
    if _proc_rank == 0:
        warnings.warn(*args, **kwargs)
