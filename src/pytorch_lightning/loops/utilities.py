# Copyright The Lightning team.
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
from contextlib import contextmanager
from typing import Generator, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.loops import _Loop
from pytorch_lightning.loops.progress import BaseProgress
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.rank_zero import rank_zero_warn


def check_finite_loss(loss: Optional[Tensor]) -> None:
    """Checks for finite loss value.

    Args:
        loss: the loss value to check to be finite
    """
    if loss is not None and not torch.isfinite(loss).all():
        raise ValueError(f"The loss returned in `training_step` is {loss}.")


def _parse_loop_limits(
    min_steps: Optional[int],
    max_steps: int,
    min_epochs: Optional[int],
    max_epochs: Optional[int],
    trainer: "pl.Trainer",
) -> Tuple[int, int]:
    """This utility computes the default values for the minimum and maximum number of steps and epochs given the
    values the user has selected.

    Args:
        min_steps: Minimum number of steps.
        max_steps: Maximum number of steps.
        min_epochs: Minimum number of epochs.
        max_epochs: Maximum number of epochs.
        trainer: Trainer instance.

    Returns:
        The parsed limits, with default values being set for the ones that the user did not specify.
    """
    if max_epochs is None:
        if max_steps == -1 and not any(isinstance(cb, Timer) for cb in trainer.callbacks):
            rank_zero_warn(
                "`max_epochs` was not set. Setting it to 1000 epochs. To train without an epoch limit,"
                " set `max_epochs=-1`.",
                category=PossibleUserWarning,
            )
            max_epochs = 1000
        else:
            max_epochs = -1

    if min_epochs is None and min_steps is not None:
        # setting this allows FitLoop.done to re-evaluate should_stop when it gets triggered `on_fit_start`
        min_epochs = 1

    if min_epochs is None:
        # the default value is 0 so no training will be done when should_stop is triggered `on_fit_start`
        min_epochs = 0

    return min_epochs, max_epochs


@contextmanager
def _block_parallel_sync_behavior(strategy: Strategy, block: bool = True) -> Generator[None, None, None]:
    """Blocks synchronization in :class:`~pytorch_lightning.strategies.parallel.ParallelStrategy`. This is useful
    for example when accumulating gradients to reduce communication when it is not needed.

    Args:
        strategy: the strategy instance to use.
        block: whether the context manager is enabled or not

    Returns:
        context manager with sync behaviour off
    """
    if isinstance(strategy, ParallelStrategy) and block:
        with strategy.block_backward_sync():
            yield None
    else:
        yield None


def _is_max_limit_reached(current: int, maximum: int = -1) -> bool:
    """Check if the limit has been reached (if enabled).

    Args:
        current: the current value
        maximum: the maximum value (or -1 to disable limit)

    Returns:
        bool: whether the limit has been reached
    """
    return maximum != -1 and current >= maximum


def _reset_progress(loop: _Loop) -> None:
    for v in vars(loop).values():
        if isinstance(v, BaseProgress):
            v.reset()
        elif isinstance(v, _Loop):
            _reset_progress(v)


def _set_sampler_epoch(dataloader: Union[DataLoader, CombinedLoader], epoch: int) -> None:
    """Calls the ``set_epoch`` method on either the sampler or the batch sampler of the given dataloader.

    Every PyTorch dataloader has either a sampler or a batch sampler, and if it is wrapped by a
    :class:`~torch.utils.data.distributed.DistributedSampler`, ``set_epoch`` must be called at the beginning
    of every epoch to ensure shuffling applies a new ordering. This has no effect if shuffling is off.
    """

    for sampler_name in ("sampler", "batch_sampler"):
        sampler = getattr(dataloader, sampler_name, None)
        if sampler is not None and callable(getattr(sampler, "set_epoch", None)):
            sampler.set_epoch(epoch)
