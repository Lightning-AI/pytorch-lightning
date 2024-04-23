# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities that can be used for calling functions on a particular rank."""

import logging
import os
from functools import wraps
from typing import Callable, Optional, TypeVar, overload

import lightning_utilities.core.rank_zero as rank_zero_module

# note: we want to keep these indirections so the `rank_zero_only.rank` is set on import
from lightning_utilities.core.rank_zero import (  # noqa: F401
    WarningCache,
    rank_prefixed_message,
    rank_zero_debug,
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_warn,
)
from typing_extensions import ParamSpec

from lightning.fabric.utilities.imports import _UTILITIES_GREATER_EQUAL_0_10

rank_zero_module.log = logging.getLogger(__name__)


def _get_rank() -> Optional[int]:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


if not _UTILITIES_GREATER_EQUAL_0_10:
    T = TypeVar("T")
    P = ParamSpec("P")

    @overload
    def rank_zero_only(fn: Callable[P, T]) -> Callable[P, Optional[T]]:
        """Rank zero only."""

    @overload
    def rank_zero_only(fn: Callable[P, T], default: T) -> Callable[P, T]:
        """Rank zero only."""

    def rank_zero_only(fn: Callable[P, T], default: Optional[T] = None) -> Callable[P, Optional[T]]:
        @wraps(fn)
        def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
            rank = getattr(rank_zero_only, "rank", None)
            if rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            if rank == 0:
                return fn(*args, **kwargs)
            return default

        return wrapped_fn

    rank_zero_module.rank_zero_only.rank = getattr(rank_zero_module.rank_zero_only, "rank", _get_rank() or 0)
else:
    rank_zero_only = rank_zero_module.rank_zero_only

# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, "rank", _get_rank() or 0)


class LightningDeprecationWarning(DeprecationWarning):
    """Deprecation warnings raised by Lightning."""


rank_zero_module.rank_zero_deprecation_category = LightningDeprecationWarning
