# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Utilities that can be used for calling functions on a particular rank."""
import logging
import warnings
from functools import wraps
from platform import python_version
from typing import Any, Callable, Optional, TypeVar, Union

from typing_extensions import ParamSpec

log = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


def rank_zero_only(fn: Callable[P, T]) -> Callable[P, Optional[T]]:
    """Wrap a function to call internal function only in rank zero.

    Function that can be used as a decorator to enable a function/method being called only on global rank 0.
    """

    @wraps(fn)
    def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
        rank = getattr(rank_zero_only, "rank", None)
        if rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


def _debug(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
    if python_version() >= "3.8.0":
        kwargs["stacklevel"] = stacklevel
    log.debug(*args, **kwargs)


@rank_zero_only
def rank_zero_debug(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
    """Emit debug-level messages only on global rank 0."""
    _debug(*args, stacklevel=stacklevel, **kwargs)


def _info(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
    if python_version() >= "3.8.0":
        kwargs["stacklevel"] = stacklevel
    log.info(*args, **kwargs)


@rank_zero_only
def rank_zero_info(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
    """Emit info-level messages only on global rank 0."""
    _info(*args, stacklevel=stacklevel, **kwargs)


def _warn(message: Union[str, Warning], stacklevel: int = 2, **kwargs: Any) -> None:
    warnings.warn(message, stacklevel=stacklevel, **kwargs)


@rank_zero_only
def rank_zero_warn(message: Union[str, Warning], stacklevel: int = 4, **kwargs: Any) -> None:
    """Emit warn-level messages only on global rank 0."""
    _warn(message, stacklevel=stacklevel, **kwargs)


rank_zero_deprecation_category = DeprecationWarning


def rank_zero_deprecation(message: Union[str, Warning], stacklevel: int = 5, **kwargs: Any) -> None:
    """Emit a deprecation warning only on global rank 0."""
    category = kwargs.pop("category", rank_zero_deprecation_category)
    rank_zero_warn(message, stacklevel=stacklevel, category=category, **kwargs)


def rank_prefixed_message(message: str, rank: Optional[int]) -> str:
    """Add a prefix with the rank to a message."""
    if rank is not None:
        # specify the rank of the process being logged
        return f"[rank: {rank}] {message}"
    return message


class WarningCache(set):
    """Cache for warnings."""

    def warn(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        """Trigger warning message."""
        if message not in self:
            self.add(message)
            rank_zero_warn(message, stacklevel=stacklevel, **kwargs)

    def deprecation(self, message: str, stacklevel: int = 6, **kwargs: Any) -> None:
        """Trigger deprecation message."""
        if message not in self:
            self.add(message)
            rank_zero_deprecation(message, stacklevel=stacklevel, **kwargs)

    def info(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        """Trigger info message."""
        if message not in self:
            self.add(message)
            rank_zero_info(message, stacklevel=stacklevel, **kwargs)
