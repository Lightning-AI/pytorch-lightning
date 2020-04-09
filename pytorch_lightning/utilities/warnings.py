"""Custom Lightning warnings"""

import warnings

_proc_rank = 0


def set_proc_rank(value: int) -> None:
    global _proc_rank
    _proc_rank = value


def rank_zero_warn(*args, **kwargs):
    global _proc_rank
    if _proc_rank:
        rank_zero_warn(*args, **kwargs)
