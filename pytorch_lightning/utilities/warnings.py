"""Custom Lightning warnings"""

import warnings

_proc_rank = 0


def set_proc_rank(value: int) -> None:
    """Set the (sub)process rank."""
    global _proc_rank
    _proc_rank = value


def rank_zero_warn(*args, **kwargs) -> None:
    """Warning only if (sub)process has rank 0."""
    global _proc_rank
    if _proc_rank == 0:
        warnings.warn(*args, **kwargs)
