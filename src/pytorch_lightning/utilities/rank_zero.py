# Copyright The PyTorch Lightning team.
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
from typing import Any, Optional, Union

import pytorch_lightning as pl
from lightning_lite.utilities.rank_zero import LightningDeprecationWarning  # noqa: F401
from lightning_lite.utilities.rank_zero import rank_zero_debug as new_rank_zero_debug
from lightning_lite.utilities.rank_zero import rank_zero_deprecation as new_rank_zero_deprecation
from lightning_lite.utilities.rank_zero import rank_zero_info as new_rank_zero_info
from lightning_lite.utilities.rank_zero import rank_zero_only as new_rank_zero_only
from lightning_lite.utilities.rank_zero import rank_zero_warn as new_rank_zero_warn

log = logging.getLogger(__name__)


def _get_rank(trainer: Optional["pl.Trainer"] = None) -> Optional[int]:
    # TODO(lite): Refactor usages in PL to lightning_lite.utilities._get_rank
    if trainer is not None:
        return trainer.global_rank
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


# add the attribute to the function but don't overwrite in case Trainer has already set it
new_rank_zero_only.rank = getattr(new_rank_zero_only, "rank", _get_rank() or 0)


@new_rank_zero_only
def rank_zero_debug(*args: Any, stacklevel: int = 5, **kwargs: Any) -> None:
    """Function used to log debug-level messages only on global rank 0."""
    new_rank_zero_deprecation(
        "`pytorch_lightning.utilities.rank_zero.rank_zero_debug` has been deprecated in v1.8.0 and will be"
        " removed in v1.10.0. Please use `lightning_lite.utilities.rank_zero.rank_zero_debug` instead."
    )
    new_rank_zero_debug(*args, stacklevel=stacklevel, **kwargs)


@new_rank_zero_only
def rank_zero_info(*args: Any, stacklevel: int = 5, **kwargs: Any) -> None:
    """Function used to log info-level messages only on global rank 0."""
    new_rank_zero_deprecation(
        "`pytorch_lightning.utilities.rank_zero.rank_zero_info` has been deprecated in v1.8.0 and will be"
        " removed in v1.10.0. Please use `lightning_lite.utilities.rank_zero.rank_zero_info` instead."
    )
    new_rank_zero_info(*args, stacklevel=stacklevel, **kwargs)


@new_rank_zero_only
def rank_zero_warn(message: Union[str, Warning], stacklevel: int = 5, **kwargs: Any) -> None:
    """Function used to log warn-level messages only on global rank 0."""
    new_rank_zero_deprecation(
        "`pytorch_lightning.utilities.rank_zero.rank_zero_warn` has been deprecated in v1.8.0 and will be"
        " removed in v1.10.0. Please use `lightning_lite.utilities.rank_zero.rank_zero_warn` instead."
    )
    new_rank_zero_warn(message, stacklevel=stacklevel, **kwargs)
