# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Utilities that can be used for calling functions on a particular rank."""

import logging
import warnings
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

from typing_extensions import ParamSpec, overload

log = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


@overload
def rank_zero_only(fn: Callable[P, T]) -> Callable[P, Optional[T]]: ...


@overload
def rank_zero_only(fn: Callable[P, T], default: T) -> Callable[P, T]: ...


def rank_zero_only(fn: Callable[P, T], default: Optional[T] = None) -> Callable[P, Optional[T]]:
    """Decorator to run the wrapped function only on global rank 0.

    Set ``rank_zero_only.rank`` before use. On non-zero ranks, the function is skipped and the provided
    ``default`` is returned (or ``None`` if not given).

    """

    @wraps(fn)
    def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
        rank = getattr(rank_zero_only, "rank", None)
        if rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if rank == 0:
            return fn(*args, **kwargs)
        return default

    return wrapped_fn


def _debug(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
    kwargs["stacklevel"] = stacklevel
    log.debug(*args, **kwargs)


@rank_zero_only
def rank_zero_debug(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
    """Emit debug-level messages only on global rank 0."""
    _debug(*args, stacklevel=stacklevel, **kwargs)


def _info(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
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
    """Add a ``[rank: X]`` prefix to the message if ``rank`` is provided; otherwise return the message unchanged."""
    if rank is not None:
        # specify the rank of the process being logged
        return f"[rank: {rank}] {message}"
    return message


class WarningCache(set):
    """A simple de-duplication cache for messages to avoid emitting the same warning/info multiple times."""

    def warn(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        """Emit a warning once on global rank 0; subsequent identical messages are suppressed."""
        if message not in self:
            self.add(message)
            rank_zero_warn(message, stacklevel=stacklevel, **kwargs)

    def deprecation(self, message: str, stacklevel: int = 6, **kwargs: Any) -> None:
        """Emit a deprecation warning once on global rank 0; subsequent identical messages are suppressed."""
        if message not in self:
            self.add(message)
            rank_zero_deprecation(message, stacklevel=stacklevel, **kwargs)

    def info(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        """Emit an info-level log once on global rank 0; subsequent identical messages are suppressed."""
        if message not in self:
            self.add(message)
            rank_zero_info(message, stacklevel=stacklevel, **kwargs)
