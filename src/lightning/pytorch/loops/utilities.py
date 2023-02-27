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
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

import lightning.pytorch as pl
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.accelerators import TPUAccelerator
from lightning.pytorch.callbacks.timer import Timer
from lightning.pytorch.loops import _Loop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher, _PrefetchDataFetcher
from lightning.pytorch.loops.progress import BaseProgress
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.pytorch.strategies.strategy import Strategy
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from lightning.pytorch.utilities.signature_utils import is_param_in_hook_signature


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
    """Blocks synchronization in :class:`~lightning.pytorch.strategies.parallel.ParallelStrategy`. This is useful
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


def _select_data_fetcher(trainer: "pl.Trainer") -> _DataFetcher:
    lightning_module = trainer.lightning_module
    if trainer.testing:
        step_fx_name = "test_step"
    elif trainer.training:
        step_fx_name = "training_step"
    elif trainer.validating or trainer.sanity_checking:
        step_fx_name = "validation_step"
    elif trainer.predicting:
        step_fx_name = "predict_step"
    else:
        raise RuntimeError(f"DataFetcher is unsupported for {trainer.state.stage}")
    step_fx = getattr(lightning_module, step_fx_name)
    if is_param_in_hook_signature(step_fx, "dataloader_iter", explicit=True):
        rank_zero_warn(
            f"Found `dataloader_iter` argument in the `{step_fx_name}`. Note that the support for "
            "this signature is experimental and the behavior is subject to change."
        )
        return _DataLoaderIterDataFetcher()
    return _PrefetchDataFetcher()


def _no_grad_context(loop_run: Callable) -> Callable:
    def _decorator(self: _Loop, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(self, _Loop):
            raise TypeError(f"`{type(self).__name__}` needs to be a Loop.")
        if not hasattr(self, "inference_mode"):
            raise TypeError(f"`{type(self).__name__}.inference_mode` needs to be defined")
        context_manager = (
            torch.inference_mode
            if (
                self.inference_mode
                # inference mode is not supported with gloo backend (#9431) and TPU accelerators.
                and not (dist.is_available() and dist.is_initialized() and dist.get_backend() == "gloo")
                and not isinstance(self.trainer.accelerator, TPUAccelerator)
            )
            else torch.no_grad
        )
        with context_manager():
            return loop_run(self, *args, **kwargs)

    return _decorator
