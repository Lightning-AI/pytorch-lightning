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
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins import ParallelPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import STEP_OUTPUT


def check_finite_loss(loss: Optional[torch.Tensor]) -> None:
    """Checks for finite loss value.

    Args:
        loss: the loss value to check to be finite
    """
    if loss is not None and not torch.isfinite(loss).all():
        raise ValueError(f"The loss returned in `training_step` is {loss}.")


def _extract_hiddens(training_step_output: STEP_OUTPUT, truncated_bptt_steps: int) -> Optional[Any]:
    """Get the hidden state if present from the training step output.

    Raises:
        MisconfigurationException: If :attr:`~pytorch_lightning.core.Lightning.LightningModule.truncated_bptt_steps` is
            not enabled and hiddens are returned or vice versa.
    """
    is_dict = isinstance(training_step_output, dict)
    if not truncated_bptt_steps:
        if is_dict and "hiddens" in training_step_output:
            raise MisconfigurationException(
                'You returned "hiddens" in your `training_step` but `truncated_bptt_steps` is disabled'
            )
        return
    elif not is_dict or "hiddens" not in training_step_output:
        raise MisconfigurationException(
            'You enabled `truncated_bptt_steps` but did not return "hiddens" in your `training_step`'
        )
    # detach hiddens to avoid `RuntimeError: Trying to backward through the graph a second time`
    hiddens = recursive_detach(training_step_output["hiddens"])
    return hiddens


def _build_training_step_kwargs(
    lightning_module: "pl.LightningModule",
    optimizers: Sequence[Optimizer],
    batch: Any,
    batch_idx: int,
    opt_idx: Optional[int],
    hiddens: Optional[Any],
) -> Dict[str, Any]:
    """Builds the keyword arguments for training_step.

    Args:
        lightning_module: the LightningModule with a `training_step` hook implementation
        optimizers: the list of optimizers from the Trainer
        batch: the batch to train on
        batch_idx: the index of the current batch
        opt_idx: the index of the current optimizer
        hiddens: the hidden state of the previous RNN iteration

    Returns:
        the keyword arguments for the training step
    """
    # enable not needing to add opt_idx to training_step
    step_kwargs = OrderedDict([("batch", batch)])

    training_step_fx = getattr(lightning_module, "training_step")

    if is_param_in_hook_signature(training_step_fx, "batch_idx", min_args=2):
        step_kwargs["batch_idx"] = batch_idx

    if len(optimizers) > 1:
        has_opt_idx_in_train_step = is_param_in_hook_signature(training_step_fx, "optimizer_idx")
        if has_opt_idx_in_train_step:
            if not lightning_module.automatic_optimization:
                raise ValueError(
                    "Your `LightningModule.training_step` signature contains an `optimizer_idx` argument but"
                    " in manual optimization optimizers must be handled by the user. Remove the optimizer_idx"
                    " argument or set `self.automatic_optimization = True`."
                )
            step_kwargs["optimizer_idx"] = opt_idx
        elif not has_opt_idx_in_train_step and lightning_module.automatic_optimization:
            raise ValueError(
                f"Your LightningModule defines {len(optimizers)} optimizers but"
                " `training_step` is missing the `optimizer_idx` argument."
            )

    # pass hiddens if using tbptt
    if lightning_module.truncated_bptt_steps > 0:
        step_kwargs["hiddens"] = hiddens

    return step_kwargs


def _update_dataloader_iter(data_fetcher: AbstractDataFetcher, batch_idx: int) -> Iterator:
    """Attach the dataloader."""
    if not isinstance(data_fetcher, DataLoaderIterDataFetcher):
        # restore iteration
        dataloader_iter = enumerate(data_fetcher, batch_idx)
    else:
        dataloader_iter = iter(data_fetcher)
    return dataloader_iter


@contextmanager
def _block_parallel_sync_behavior(trainer: "pl.Trainer", block: bool = True) -> Generator[None, None, None]:
    """Blocks synchronization in :class:`~pytorch_lightning.plugins.training_type.parallel.ParallelPlugin`. This is
    useful for example when when accumulating gradients to reduce communication when it is not needed.

    Args:
        trainer: the trainer instance with a reference to a training type plugin
        block: whether the context manager is enabled or not

    Returns:
        context manager with sync behaviour off
    """
    if isinstance(trainer.training_type_plugin, ParallelPlugin) and block:
        with trainer.training_type_plugin.block_backward_sync():
            yield None
    else:
        yield None


@lru_cache(1)
def _cumulative_optimizer_frequencies(frequencies: Tuple[int]):
    return np.cumsum(frequencies)


def _get_active_optimizers(
    optimizers: List[Optimizer], frequencies: List[int], batch_idx: Optional[int] = None
) -> List[Tuple[int, Optimizer]]:
    """Returns the currently active optimizers. When multiple optimizers are used with different frequencies, only
    one of the optimizers is active at a time.

    Returns:
        A list of tuples (opt_idx, optimizer) of currently active optimizers.
    """
    if not frequencies:
        # call training_step once per optimizer
        return list(enumerate(optimizers))

    freq_cumsum = _cumulative_optimizer_frequencies(tuple(frequencies))
    optimizers_loop_length = freq_cumsum[-1]
    current_place_in_loop = batch_idx % optimizers_loop_length

    # find optimizer index by looking for the first {item > current_place} in the cumsum list
    opt_idx = np.searchsorted(freq_cumsum, current_place_in_loop, side="right")
    return [(opt_idx, optimizers[opt_idx])]


def _is_max_limit_reached(current: int, maximum: int = -1) -> bool:
    """Check if the limit has been reached (if enabled).

    Args:
        current: the current value
        maximum: the maximum value (or -1 to disable limit)

    Returns:
        bool: whether the limit has been reached
    """
    return maximum != -1 and current >= maximum
