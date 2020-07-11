from functools import wraps
import warnings
from pytorch_lightning import _logger as log
import os


def rank_zero_only(fn):

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


try:
    # add the attribute to the function but don't overwrite in case Trainer has already set it
    getattr(rank_zero_only, 'rank')
except AttributeError:
    rank_zero_only.rank = os.environ.get('LOCAL_RANK', 0)


def _warn(*args, **kwargs):
    warnings.warn(*args, **kwargs)


def _info(*args, **kwargs):
    log.info(*args, **kwargs)


def _debug(*args, **kwargs):
    log.info(*args, **kwargs)


rank_zero_debug = rank_zero_only(_debug)
rank_zero_info = rank_zero_only(_info)
rank_zero_warn = rank_zero_only(_warn)
