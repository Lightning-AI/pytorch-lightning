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
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional, Tuple, Generator

import torch

import pytorch_lightning as pl
from pytorch_lightning.plugins import ParallelPlugin
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher
from pytorch_lightning.utilities.finite_checks import detect_nan_parameters
from pytorch_lightning.utilities.types import STEP_OUTPUT


def check_finite_loss(model: "pl.LightningModule", loss: torch.Tensor) -> None:
    """Checks for finite parameters and loss values.

    Args:
        model: a reference to the ``LightningModule``
        loss: the loss value to check to be finite
    """
    if not torch.isfinite(loss).all():
        raise ValueError(f"The loss returned in `training_step` is {loss}.")
    detect_nan_parameters(model)


def _check_training_step_output(model: "pl.LightningModule", training_step_output: STEP_OUTPUT) -> None:
    """Sanity checks that training produced a valid output and optimizer step has already been called in manual
    optimization.

    Args:
        model: a reference to the trainer
        training_step_output: the output of the training step (before wrapping in an AttributeDict)
    """
    if isinstance(training_step_output, torch.Tensor) and not model.automatic_optimization:
        if training_step_output.grad_fn is None:
            # TODO: Find why - RuntimeError: Expected to mark a variable ready only once ...
            raise MisconfigurationException("In manual optimization, `training_step` should not return a Tensor")
    elif model.automatic_optimization:
        if not any(
            (
                isinstance(training_step_output, torch.Tensor),
                (isinstance(training_step_output, Mapping) and "loss" in training_step_output),
                training_step_output is None,
            )
        ):
            raise MisconfigurationException(
                "In automatic optimization, `training_step` must either return a Tensor, "
                "a dict with key 'loss' or None (where the step will be skipped)."
            )


def _process_training_step_output(
    trainer: "pl.Trainer", training_step_output: STEP_OUTPUT
) -> Tuple[Optional[ResultCollection], Optional[Any]]:
    """Adds the :param:`training_step_output` to the trainer's results

    Args:
        trainer: a reference to the trainer
        training_step_output: the output of the training step (before wrapping into an AttributeDict)

    Returns:
        the updated results (None if the training_step's output was None) and hiddens exract from the results
    """
    if training_step_output is None:
        return None, None

    results = trainer._results

    loss = None
    hiddens = None

    # handle dict return
    if isinstance(training_step_output, dict):
        # this should not modify the `training_step_output`, as the user could be using it after `training_step_end`
        loss = training_step_output.get("loss")
        hiddens = training_step_output.get("hiddens")
        # detach hiddens to avoid `RuntimeError: Trying to backward through the graph a second time`
        hiddens = apply_to_collection(hiddens, torch.Tensor, lambda t: t.detach())
        # use the setter instead of `dict.update` because it calls `detach` on the tensor items
        results.extra = {k: v for k, v in training_step_output.items() if k not in ("loss", "hiddens")}

    # handle scalar return
    elif isinstance(training_step_output, torch.Tensor):
        loss = training_step_output

    # map to results under the hood
    results.minimize = loss

    if trainer.move_metrics_to_cpu:
        results.cpu()
    return results, hiddens


def _prepare_dataloader_iter(data_fetcher: AbstractDataFetcher, batch_idx: int) -> Iterator:
    """Attach the dataloader"""
    if not isinstance(data_fetcher, DataLoaderIterDataFetcher):
        # restore iteration
        dataloader_iter = enumerate(data_fetcher, batch_idx)
    else:
        dataloader_iter = iter(data_fetcher)
    return dataloader_iter


@contextmanager
def _block_parallel_sync_behavior(trainer: "pl.Trainer", block: bool = True) -> Generator[None, None, None]:
    """Blocks synchronization in :class:`pytorch_lightning.plugins.training_type.parallel.ParallelPlugin`.
    This is useful for example when when accumulating gradients to reduce communication when it is not needed.

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
