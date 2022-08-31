import logging
import os
import warnings
from functools import partial, wraps
from platform import python_version
from typing import Any, Callable, Optional, Union

import lightning_lite as lite

log = logging.getLogger(__name__)


def _get_rank(strategy: Optional["lite.strategies.Strategy"] = None) -> Optional[int]:
    if strategy is not None:
        return strategy.global_rank
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


def rank_zero_only(fn: Callable) -> Callable:
    """Function that can be used as a decorator to enable a function/method being called only on global rank 0."""

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, "rank", _get_rank() or 0)


def _info(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
    if python_version() >= "3.8.0":
        kwargs["stacklevel"] = stacklevel
    log.info(*args, **kwargs)


def _debug(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
    if python_version() >= "3.8.0":
        kwargs["stacklevel"] = stacklevel
    log.debug(*args, **kwargs)


@rank_zero_only
def rank_zero_debug(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
    """Function used to log debug-level messages only on global rank 0."""
    _debug(*args, stacklevel=stacklevel, **kwargs)


@rank_zero_only
def rank_zero_info(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
    """Function used to log info-level messages only on global rank 0."""
    _info(*args, stacklevel=stacklevel, **kwargs)


def _warn(message: Union[str, Warning], stacklevel: int = 2, **kwargs: Any) -> None:
    if type(stacklevel) is type and issubclass(stacklevel, Warning):
        rank_zero_deprecation(
            "Support for passing the warning category positionally is deprecated in v1.6 and will be removed in v1.8"
            f" Please, use `category={stacklevel.__name__}`."
        )
        kwargs["category"] = stacklevel
        stacklevel = kwargs.pop("stacklevel", 2)
    warnings.warn(message, stacklevel=stacklevel, **kwargs)


@rank_zero_only
def rank_zero_warn(message: Union[str, Warning], stacklevel: int = 4, **kwargs: Any) -> None:
    """Function used to log warn-level messages only on global rank 0."""
    _warn(message, stacklevel=stacklevel, **kwargs)


class LightningDeprecationWarning(DeprecationWarning):
    """Deprecation warnings raised by Lightning."""


rank_zero_deprecation = partial(rank_zero_warn, category=LightningDeprecationWarning)


def _rank_prefixed_message(message: str, rank: Optional[int]) -> str:
    if rank is not None:
        # specify the rank of the process being logged
        return f"[rank: {rank}] {message}"
    return message
