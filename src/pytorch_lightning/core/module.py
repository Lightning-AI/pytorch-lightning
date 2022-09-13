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
"""The LightningModule - an nn.Module with many additional features."""

import collections.abc
import inspect
import logging
import numbers
import os
import tempfile
import warnings
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, overload, Sequence, Tuple, Union

import torch
from torch import ScriptModule, Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric
from typing_extensions import Literal

import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.core.hooks import CheckpointHooks, DataHooks, ModelHooks
from pytorch_lightning.core.mixins import DeviceDtypeModuleMixin, HyperparametersMixin
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.core.saving import ModelIO
from pytorch_lightning.loggers import Logger, LoggerCollection
from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import _FxValidator
from pytorch_lightning.utilities import _IS_WINDOWS, _TORCH_GREATER_EQUAL_1_10, GradClipAlgorithmType
from pytorch_lightning.utilities.apply_func import apply_to_collection, convert_to_tensors
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.distributed import distributed_available, sync_ddp
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_11, _TORCH_GREATER_EQUAL_1_13
from pytorch_lightning.utilities.parsing import collect_init_args
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import _METRIC_COLLECTION, EPOCH_OUTPUT, LRSchedulerTypeUnion, STEP_OUTPUT
from pytorch_lightning.utilities.warnings import WarningCache

warning_cache = WarningCache()
log = logging.getLogger(__name__)


class LightningModule(
    DeviceDtypeModuleMixin,
    HyperparametersMixin,
    ModelIO,
    ModelHooks,
    DataHooks,
    CheckpointHooks,
    Module,
):
    # Below is for property support of JIT
    # since none of these are important when using JIT, we are going to ignore them.
    __jit_unused_properties__ = (
        [
            "example_input_array",
            "on_gpu",
            "current_epoch",
            "global_step",
            "global_rank",
            "local_rank",
            "logger",
            "loggers",
            "automatic_optimization",
            "truncated_bptt_steps",
            "use_amp",
            "trainer",
            "_running_torchscript",
        ]
        + DeviceDtypeModuleMixin.__jit_unused_properties__
        + HyperparametersMixin.__jit_unused_properties__
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # see (https://github.com/pytorch/pytorch/blob/3e6bb5233f9ca2c5aa55d9cda22a7ee85439aa6e/
        # torch/nn/modules/module.py#L227)
        torch._C._log_api_usage_once(f"lightning.module.{self.__class__.__name__}")

        # pointer to the trainer object
        self._trainer: Optional["pl.Trainer"] = None

        self._use_amp: bool = False

        # the precision used
        self.precision: int = 32

        # optionally can be set by user
        self._example_input_array = None
        self._current_fx_name: Optional[str] = None
        self._automatic_optimization: bool = True
        self._truncated_bptt_steps: int = 0
        self._param_requires_grad_state = {}
        self._metric_attributes: Optional[Dict[int, str]] = None
        self._should_prevent_trainer_and_dataloaders_deepcopy: bool = False
        self._running_torchscript_internal = False  # workaround for https://github.com/pytorch/pytorch/issues/67146

        self._register_sharded_tensor_state_dict_hooks_if_available()

    @overload
    def optimizers(self, use_pl_optimizer: Literal[True] = True) -> Union[LightningOptimizer, List[LightningOptimizer]]:
        ...

    @overload
    def optimizers(self, use_pl_optimizer: Literal[False]) -> Union[Optimizer, List[Optimizer]]:
        ...

    @overload
    def optimizers(
        self, use_pl_optimizer: bool
    ) -> Union[Optimizer, LightningOptimizer, List[Optimizer], List[LightningOptimizer]]:
        ...

    def optimizers(
        self, use_pl_optimizer: bool = True
    ) -> Union[Optimizer, LightningOptimizer, List[Optimizer], List[LightningOptimizer]]:
        """Returns the optimizer(s) that are being used during training. Useful for manual optimization.

        Args:
            use_pl_optimizer: If ``True``, will wrap the optimizer(s) in a
                :class:`~pytorch_lightning.core.optimizer.LightningOptimizer` for automatic handling of precision and
                profiling.

        Returns:
            A single optimizer, or a list of optimizers in case multiple ones are present.
        """
        if use_pl_optimizer:
            opts = list(self.trainer.strategy._lightning_optimizers.values())
        else:
            opts = self.trainer.optimizers

        # single optimizer
        if isinstance(opts, list) and len(opts) == 1 and isinstance(opts[0], (Optimizer, LightningOptimizer)):
            return opts[0]
        # multiple opts
        return opts

    def lr_schedulers(self) -> Optional[Union[LRSchedulerTypeUnion, List[LRSchedulerTypeUnion]]]:
        """Returns the learning rate scheduler(s) that are being used during training. Useful for manual
        optimization.

        Returns:
            A single scheduler, or a list of schedulers in case multiple ones are present, or ``None`` if no
            schedulers were returned in :meth:`configure_optimizers`.
        """
        if not self.trainer.lr_scheduler_configs:
            return None

        # ignore other keys "interval", "frequency", etc.
        lr_schedulers = [config.scheduler for config in self.trainer.lr_scheduler_configs]

        # single scheduler
        if len(lr_schedulers) == 1:
            return lr_schedulers[0]

        # multiple schedulers
        return lr_schedulers

    @property
    def trainer(self) -> "pl.Trainer":
        if not self._running_torchscript and self._trainer is None:
            raise RuntimeError(f"{self.__class__.__qualname__} is not attached to a `Trainer`.")
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: Optional["pl.Trainer"]) -> None:
        for v in self.children():
            if isinstance(v, LightningModule):
                v.trainer = trainer
        if trainer is not None and not isinstance(trainer, weakref.ProxyTypes):
            trainer = weakref.proxy(trainer)
        self._trainer = trainer

    @property
    def example_input_array(self) -> Any:
        """The example input array is a specification of what the module can consume in the :meth:`forward` method.
        The return type is interpreted as follows:

        -   Single tensor: It is assumed the model takes a single argument, i.e.,
            ``model.forward(model.example_input_array)``
        -   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,
            ``model.forward(*model.example_input_array)``
        -   Dict: The input array represents named keyword arguments, i.e.,
            ``model.forward(**model.example_input_array)``
        """
        return self._example_input_array

    @example_input_array.setter
    def example_input_array(self, example: Any) -> None:
        self._example_input_array = example

    @property
    def current_epoch(self) -> int:
        """The current epoch in the ``Trainer``, or 0 if not attached."""
        return self.trainer.current_epoch if self._trainer else 0

    @property
    def global_step(self) -> int:
        """Total training batches seen across all epochs.

        If no Trainer is attached, this propery is 0.
        """
        return self.trainer.global_step if self._trainer else 0

    @property
    def global_rank(self) -> int:
        """The index of the current process across all nodes and devices."""
        return self.trainer.global_rank if self._trainer else 0

    @property
    def local_rank(self) -> int:
        """The index of the current process within a single node."""
        return self.trainer.local_rank if self._trainer else 0

    @property
    def on_gpu(self):
        """Returns ``True`` if this model is currently located on a GPU.

        Useful to set flags around the LightningModule for different CPU vs GPU behavior.
        """
        return self.device.type == "cuda"

    @property
    def automatic_optimization(self) -> bool:
        """If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``."""
        return self._automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, automatic_optimization: bool) -> None:
        self._automatic_optimization = automatic_optimization

    @property
    def truncated_bptt_steps(self) -> int:
        """Enables `Truncated Backpropagation Through Time` in the Trainer when set to a positive integer.

        It represents
        the number of times :meth:`training_step` gets called before backpropagation. If this is > 0, the
        :meth:`training_step` receives an additional argument ``hiddens`` and is expected to return a hidden state.
        """
        return self._truncated_bptt_steps

    @truncated_bptt_steps.setter
    def truncated_bptt_steps(self, truncated_bptt_steps: int) -> None:
        self._truncated_bptt_steps = truncated_bptt_steps

    @property
    def logger(self) -> Optional[Logger]:
        """Reference to the logger object in the Trainer."""
        # this should match the implementation of `trainer.logger`
        # we don't reuse it so we can properly set the deprecation stacklevel
        if self._trainer is None:
            return
        loggers = self.trainer.loggers
        if len(loggers) == 0:
            return None
        if len(loggers) == 1:
            return loggers[0]
        else:
            if not self._running_torchscript:
                rank_zero_deprecation(
                    "Using `lightning_module.logger` when multiple loggers are configured."
                    " This behavior will change in v1.8 when `LoggerCollection` is removed, and"
                    " `lightning_module.logger` will return the first logger available.",
                    stacklevel=5,
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return LoggerCollection(loggers)

    @property
    def loggers(self) -> List[Logger]:
        """Reference to the list of loggers in the Trainer."""
        return self.trainer.loggers if self._trainer else []

    @property
    def _running_torchscript(self) -> bool:
        return self._running_torchscript_internal

    @_running_torchscript.setter
    def _running_torchscript(self, value: bool) -> None:
        for v in self.children():
            if isinstance(v, LightningModule):
                v._running_torchscript_internal = value
        self._running_torchscript_internal = value

    def _call_batch_hook(self, hook_name: str, *args: Any) -> Any:
        if self._trainer:
            datahook_selector = self._trainer._data_connector._datahook_selector
            assert datahook_selector is not None
            obj = datahook_selector.get_instance(hook_name)
            if isinstance(obj, self.__class__):
                trainer_method = self._trainer._call_lightning_module_hook
            else:
                trainer_method = self._trainer._call_lightning_datamodule_hook

            return trainer_method(hook_name, *args)
        else:
            hook = getattr(self, hook_name)
            return hook(*args)

    def _on_before_batch_transfer(self, batch: Any, dataloader_idx: int = 0) -> Any:
        return self._call_batch_hook("on_before_batch_transfer", batch, dataloader_idx)

    def _apply_batch_transfer_handler(
        self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0
    ) -> Any:
        device = device or self.device

        def call_hook(hook_name, *args):
            if self._trainer:
                datahook_selector = self._trainer._data_connector._datahook_selector
                obj = datahook_selector.get_instance(hook_name)
                trainer_method = (
                    self._trainer._call_lightning_module_hook
                    if isinstance(obj, self.__class__)
                    else self._trainer._call_lightning_datamodule_hook
                )
                return trainer_method(hook_name, *args)
            else:
                hook = getattr(self, hook_name)
                return hook(*args)

        batch = call_hook("on_before_batch_transfer", batch, dataloader_idx)
        batch = call_hook("transfer_batch_to_device", batch, device, dataloader_idx)
        batch = call_hook("on_after_batch_transfer", batch, dataloader_idx)
        return batch

    def print(self, *args, **kwargs) -> None:
        r"""
        Prints only from process 0. Use this in any distributed mode to log only once.

        Args:
            *args: The thing to print. The same as for Python's built-in print function.
            **kwargs: The same as for Python's built-in print function.

        Example::

            def forward(self, x):
                self.print(x, 'in forward')

        """
        if self.trainer.is_global_zero:
            progress_bar = self.trainer.progress_bar_callback
            if progress_bar is not None and progress_bar.is_enabled:
                progress_bar.print(*args, **kwargs)
            else:
                print(*args, **kwargs)

    def log(
        self,
        name: str,
        value: _METRIC_COLLECTION,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Union[str, Callable] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,
    ) -> None:
        """Log a key, value pair.

        Example::

            self.log('train_loss', loss)

        The default behavior per hook is documented here: :ref:`extensions/logging:Automatic Logging`.

        Args:
            name: key to log.
            value: value to log. Can be a ``float``, ``Tensor``, ``Metric``, or a dictionary of the former.
            prog_bar: if ``True`` logs to the progress bar.
            logger: if ``True`` logs to the logger.
            on_step: if ``True`` logs at this step. The default value is determined by the hook.
                See :ref:`extensions/logging:Automatic Logging` for details.
            on_epoch: if ``True`` logs epoch accumulated metrics. The default value is determined by the hook.
                See :ref:`extensions/logging:Automatic Logging` for details.
            reduce_fx: reduction function over step values for end of epoch. :meth:`torch.mean` by default.
            enable_graph: if ``True``, will not auto detach the graph.
            sync_dist: if ``True``, reduces the metric across devices. Use with care as this may lead to a significant
                communication overhead.
            sync_dist_group: the DDP group to sync across.
            add_dataloader_idx: if ``True``, appends the index of the current dataloader to
                the name (when using multiple dataloaders). If False, user needs to give unique names for
                each dataloader to not mix the values.
            batch_size: Current batch_size. This will be directly inferred from the loaded batch,
                but for some data structures you might need to explicitly provide it.
            metric_attribute: To restore the metric state, Lightning requires the reference of the
                :class:`torchmetrics.Metric` in your model. This is found automatically if it is a model attribute.
            rank_zero_only: Whether the value will be logged only on rank 0. This will prevent synchronization which
                would produce a deadlock as not all processes would perform this log call.
        """
        # check for invalid values
        apply_to_collection(value, dict, self.__check_not_nested, name)
        apply_to_collection(
            value, object, self.__check_allowed, name, value, wrong_dtype=(numbers.Number, Metric, Tensor, dict)
        )

        if self._trainer is None:
            # not an error to support testing the `*_step` methods without a `Trainer` reference
            rank_zero_warn(
                "You are trying to `self.log()` but the `self.trainer` reference is not registered on the model yet."
                " This is most likely because the model hasn't been passed to the `Trainer`"
            )
            return
        results = self.trainer._results
        if results is None:
            raise MisconfigurationException(
                "You are trying to `self.log()` but the loop's result collection is not registered"
                " yet. This is most likely because you are trying to log in a `predict` hook,"
                " but it doesn't support logging"
            )
        if self._current_fx_name is None:
            raise MisconfigurationException(
                "You are trying to `self.log()` but it is not managed by the `Trainer` control flow"
            )

        on_step, on_epoch = _FxValidator.check_logging_and_get_default_levels(
            self._current_fx_name, on_step=on_step, on_epoch=on_epoch
        )

        # make sure user doesn't introduce logic for multi-dataloaders
        if "/dataloader_idx_" in name:
            raise MisconfigurationException(
                f"You called `self.log` with the key `{name}`"
                " but it should not contain information about `dataloader_idx`"
            )

        value = apply_to_collection(value, (torch.Tensor, numbers.Number), self.__to_tensor, name)

        if self.trainer._logger_connector.should_reset_tensors(self._current_fx_name):
            # if we started a new epoch (running its first batch) the hook name has changed
            # reset any tensors for the new hook name
            results.reset(metrics=False, fx=self._current_fx_name)

        if metric_attribute is None and isinstance(value, Metric):
            if self._metric_attributes is None:
                # compute once
                self._metric_attributes = {
                    id(module): name for name, module in self.named_modules() if isinstance(module, Metric)
                }
                if not self._metric_attributes:
                    raise MisconfigurationException(
                        "Could not find the `LightningModule` attribute for the `torchmetrics.Metric` logged."
                        " You can fix this by setting an attribute for the metric in your `LightningModule`."
                    )
            # try to find the passed metric in the LightningModule
            metric_attribute = self._metric_attributes.get(id(value), None)
            if metric_attribute is None:
                raise MisconfigurationException(
                    "Could not find the `LightningModule` attribute for the `torchmetrics.Metric` logged."
                    f" You can fix this by calling `self.log({name}, ..., metric_attribute=name)` where `name` is one"
                    f" of {list(self._metric_attributes.values())}"
                )

        if (
            self.trainer.training
            and is_param_in_hook_signature(self.training_step, "dataloader_iter", explicit=True)
            and batch_size is None
        ):
            raise MisconfigurationException(
                "With `def training_step(self, dataloader_iter)`, `self.log(..., batch_size=...)` should be provided."
            )

        results.log(
            self._current_fx_name,
            name,
            value,
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            enable_graph=enable_graph,
            add_dataloader_idx=add_dataloader_idx,
            batch_size=batch_size,
            sync_dist=sync_dist and distributed_available(),
            sync_dist_fn=self.trainer.strategy.reduce or sync_ddp,
            sync_dist_group=sync_dist_group,
            metric_attribute=metric_attribute,
            rank_zero_only=rank_zero_only,
        )

        self.trainer._logger_connector._current_fx = self._current_fx_name

    def log_dict(
        self,
        dictionary: Mapping[str, _METRIC_COLLECTION],
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Union[str, Callable] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        rank_zero_only: bool = False,
    ) -> None:
        """Log a dictionary of values at once.

        Example::

            values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
            self.log_dict(values)

        Args:
            dictionary: key value pairs.
                The values can be a ``float``, ``Tensor``, ``Metric``, or a dictionary of the former.
            prog_bar: if ``True`` logs to the progress base.
            logger: if ``True`` logs to the logger.
            on_step: if ``True`` logs at this step.
                ``None`` auto-logs for training_step but not validation/test_step.
                The default value is determined by the hook.
                See :ref:`extensions/logging:Automatic Logging` for details.
            on_epoch: if ``True`` logs epoch accumulated metrics.
                ``None`` auto-logs for val/test step but not ``training_step``.
                The default value is determined by the hook.
                See :ref:`extensions/logging:Automatic Logging` for details.
            reduce_fx: reduction function over step values for end of epoch. :meth:`torch.mean` by default.
            enable_graph: if ``True``, will not auto-detach the graph
            sync_dist: if ``True``, reduces the metric across GPUs/TPUs. Use with care as this may lead to a significant
                communication overhead.
            sync_dist_group: the ddp group to sync across.
            add_dataloader_idx: if ``True``, appends the index of the current dataloader to
                the name (when using multiple). If ``False``, user needs to give unique names for
                each dataloader to not mix values.
            batch_size: Current batch size. This will be directly inferred from the loaded batch,
                but some data structures might need to explicitly provide it.
            rank_zero_only: Whether the value will be logged only on rank 0. This will prevent synchronization which
                would produce a deadlock as not all processes would perform this log call.
        """
        for k, v in dictionary.items():
            self.log(
                name=k,
                value=v,
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                reduce_fx=reduce_fx,
                enable_graph=enable_graph,
                sync_dist=sync_dist,
                sync_dist_group=sync_dist_group,
                add_dataloader_idx=add_dataloader_idx,
                batch_size=batch_size,
                rank_zero_only=rank_zero_only,
            )

    @staticmethod
    def __check_not_nested(value: dict, name: str) -> None:
        # self-imposed restriction. for simplicity
        if any(isinstance(v, dict) for v in value.values()):
            raise ValueError(f"`self.log({name}, {value})` was called, but nested dictionaries cannot be logged")

    @staticmethod
    def __check_allowed(v: Any, name: str, value: Any) -> None:
        raise ValueError(f"`self.log({name}, {value})` was called, but `{type(v).__name__}` values cannot be logged")

    def __to_tensor(self, value: Union[torch.Tensor, numbers.Number], name: str) -> Tensor:
        value = (
            value.clone().detach().to(self.device)
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, device=self.device)
        )
        if not torch.numel(value) == 1:
            raise ValueError(
                f"`self.log({name}, {value})` was called, but the tensor must have a single element."
                f" You can try doing `self.log({name}, {value}.mean())`"
            )
        value = value.squeeze()
        return value

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        """Override this method to change the default behaviour of ``log_grad_norm``.

        If clipping gradients, the gradients will not have been clipped yet.

        Args:
            grad_norm_dict: Dictionary containing current grad norm metrics

        Example::

            # DEFAULT
            def log_grad_norm(self, grad_norm_dict):
                self.log_dict(grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        """
        self.log_dict(grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)

    def all_gather(self, data: Union[Tensor, Dict, List, Tuple], group: Optional[Any] = None, sync_grads: bool = False):
        r"""
        Allows users to call ``self.all_gather()`` from the LightningModule, thus making the ``all_gather`` operation
        accelerator agnostic. ``all_gather`` is a function provided by accelerators to gather a tensor from several
        distributed processes.

        Args:
            data: int, float, tensor of shape (batch, ...), or a (possibly nested) collection thereof.
            group: the process group to gather results from. Defaults to all processes (world)
            sync_grads: flag that allows users to synchronize gradients for the all_gather operation

        Return:
            A tensor of shape (world_size, batch, ...), or if the input was a collection
            the output will also be a collection with tensors of this shape.
        """
        group = group if group is not None else torch.distributed.group.WORLD
        all_gather = self.trainer.strategy.all_gather
        data = convert_to_tensors(data, device=self.device)
        return apply_to_collection(data, Tensor, all_gather, group=group, sync_grads=sync_grads)

    def forward(self, *args, **kwargs) -> Any:
        r"""
        Same as :meth:`torch.nn.Module.forward()`.

        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.

        Return:
            Your model's output
        """
        return super().forward(*args, **kwargs)

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        r"""
        Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.

        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (``int``): Integer displaying index of this batch
            optimizer_idx (``int``): When using multiple optimizers, this argument will also be present.
            hiddens (``Any``): Passed in if
                :paramref:`~pytorch_lightning.core.module.LightningModule.truncated_bptt_steps` > 0.

        Return:
            Any of.

            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``
            - ``None`` - Training will skip to the next batch. This is only for automatic optimization.
                This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.

        Example::

            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss

        If you define multiple optimizers, this step will be called with an additional
        ``optimizer_idx`` parameter.

        .. code-block:: python

            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx, optimizer_idx):
                if optimizer_idx == 0:
                    # do training_step with encoder
                    ...
                if optimizer_idx == 1:
                    # do training_step with decoder
                    ...


        If you add truncated back propagation through time you will also get an additional
        argument with the hidden states of the previous step.

        .. code-block:: python

            # Truncated back-propagation through time
            def training_step(self, batch, batch_idx, hiddens):
                # hiddens are the hidden states from the previous truncated backprop step
                out, hiddens = self.lstm(data, hiddens)
                loss = ...
                return {"loss": loss, "hiddens": hiddens}

        Note:
            The loss value shown in the progress bar is smoothed (averaged) over the last values,
            so it differs from the actual loss returned in train/validation step.
        """
        rank_zero_warn("`training_step` must be implemented to be used with the Lightning Trainer")

    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        """Use this when training with dp because :meth:`training_step` will operate on only part of the batch.
        However, this is still optional and only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be called
            so that you don't have to change your code

        .. code-block:: python

            # pseudocode
            sub_batches = split_batches_for_dp(batch)
            step_output = [training_step(sub_batch) for sub_batch in sub_batches]
            training_step_end(step_output)

        Args:
            step_output: What you return in `training_step` for each batch part.

        Return:
            Anything

        When using the DP strategy, only a portion of the batch is inside the training_step:

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self(x)

                # softmax uses only a portion of the batch in the denominator
                loss = self.softmax(out)
                loss = nce_loss(loss)
                return loss

        If you wish to do something with all the parts of the batch, then use this method to do it:

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self.encoder(x)
                return {"pred": out}


            def training_step_end(self, training_step_outputs):
                gpu_0_pred = training_step_outputs[0]["pred"]
                gpu_1_pred = training_step_outputs[1]["pred"]
                gpu_n_pred = training_step_outputs[n]["pred"]

                # this softmax now uses the full batch
                loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
                return loss

        See Also:
            See the :ref:`Multi GPU Training <gpu_intermediate>` guide for more details.
        """

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """Called at the end of the training epoch with the outputs of all training steps. Use this in case you
        need to do something with all the outputs returned by :meth:`training_step`.

        .. code-block:: python

            # the pseudocode for these calls
            train_outs = []
            for train_batch in train_data:
                out = training_step(train_batch)
                train_outs.append(out)
            training_epoch_end(train_outs)

        Args:
            outputs: List of outputs you defined in :meth:`training_step`. If there are multiple optimizers or when
                using ``truncated_bptt_steps > 0``, the lists have the dimensions
                (n_batches, tbptt_steps, n_optimizers). Dimensions of length 1 are squeezed.

        Return:
            None

        Note:
            If this method is not overridden, this won't be called.

        .. code-block:: python

            def training_epoch_end(self, training_step_outputs):
                # do something with all training_step outputs
                for out in training_step_outputs:
                    ...
        """

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        r"""
        Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.

        .. code-block:: python

            # the pseudocode for these calls
            val_outs = []
            for val_batch in val_data:
                out = validation_step(val_batch)
                val_outs.append(out)
            validation_epoch_end(val_outs)

        Args:
            batch: The output of your :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple val dataloaders used)

        Return:
            - Any object or value
            - ``None`` - Validation will skip to the next batch

        .. code-block:: python

            # pseudocode of order
            val_outs = []
            for val_batch in val_data:
                out = validation_step(val_batch)
                if defined("validation_step_end"):
                    out = validation_step_end(out)
                val_outs.append(out)
            val_outs = validation_epoch_end(val_outs)


        .. code-block:: python

            # if you have one val dataloader:
            def validation_step(self, batch, batch_idx):
                ...


            # if you have multiple val dataloaders:
            def validation_step(self, batch, batch_idx, dataloader_idx=0):
                ...

        Examples::

            # CASE 1: A single validation dataset
            def validation_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self(x)
                loss = self.loss(out, y)

                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # log the outputs!
                self.log_dict({'val_loss': loss, 'val_acc': val_acc})

        If you pass in multiple val dataloaders, :meth:`validation_step` will have an additional argument. We recommend
        setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.

        .. code-block:: python

            # CASE 2: multiple validation dataloaders
            def validation_step(self, batch, batch_idx, dataloader_idx=0):
                # dataloader_idx tells you which dataset this is.
                ...

        Note:
            If you don't need to validate you don't need to implement this method.

        Note:
            When the :meth:`validation_step` is called, the model has been put in eval mode
            and PyTorch gradients have been disabled. At the end of validation,
            the model goes back to training mode and gradients are enabled.
        """

    def validation_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        """Use this when validating with dp because :meth:`validation_step` will operate on only part of the batch.
        However, this is still optional and only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be called
            so that you don't have to change your code.

        .. code-block:: python

            # pseudocode
            sub_batches = split_batches_for_dp(batch)
            step_output = [validation_step(sub_batch) for sub_batch in sub_batches]
            validation_step_end(step_output)

        Args:
            step_output: What you return in :meth:`validation_step` for each batch part.

        Return:
            None or anything

        .. code-block:: python

            # WITHOUT validation_step_end
            # if used in DP, this batch is 1/num_gpus large
            def validation_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self.encoder(x)
                loss = self.softmax(out)
                loss = nce_loss(loss)
                self.log("val_loss", loss)


            # --------------
            # with validation_step_end to do softmax over the full batch
            def validation_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self(x)
                return out


            def validation_step_end(self, val_step_outputs):
                for out in val_step_outputs:
                    ...

        See Also:
            See the :ref:`Multi GPU Training <gpu_intermediate>` guide for more details.
        """

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        """Called at the end of the validation epoch with the outputs of all validation steps.

        .. code-block:: python

            # the pseudocode for these calls
            val_outs = []
            for val_batch in val_data:
                out = validation_step(val_batch)
                val_outs.append(out)
            validation_epoch_end(val_outs)

        Args:
            outputs: List of outputs you defined in :meth:`validation_step`, or if there
                are multiple dataloaders, a list containing a list of outputs for each dataloader.

        Return:
            None

        Note:
            If you didn't define a :meth:`validation_step`, this won't be called.

        Examples:
            With a single dataloader:

            .. code-block:: python

                def validation_epoch_end(self, val_step_outputs):
                    for out in val_step_outputs:
                        ...

            With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
            one entry per dataloader, while the inner list contains the individual outputs of
            each validation step for that dataloader.

            .. code-block:: python

                def validation_epoch_end(self, outputs):
                    for dataloader_output_result in outputs:
                        dataloader_outs = dataloader_output_result.dataloader_i_outputs

                    self.log("final_metric", final_value)
        """

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        r"""
        Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest
        such as accuracy.

        .. code-block:: python

            # the pseudocode for these calls
            test_outs = []
            for test_batch in test_data:
                out = test_step(test_batch)
                test_outs.append(out)
            test_epoch_end(test_outs)

        Args:
            batch: The output of your :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_id: The index of the dataloader that produced this batch.
                (only if multiple test dataloaders used).

        Return:
           Any of.

            - Any object or value
            - ``None`` - Testing will skip to the next batch

        .. code-block:: python

            # if you have one test dataloader:
            def test_step(self, batch, batch_idx):
                ...


            # if you have multiple test dataloaders:
            def test_step(self, batch, batch_idx, dataloader_idx=0):
                ...

        Examples::

            # CASE 1: A single test dataset
            def test_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self(x)
                loss = self.loss(out, y)

                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # log the outputs!
                self.log_dict({'test_loss': loss, 'test_acc': test_acc})

        If you pass in multiple test dataloaders, :meth:`test_step` will have an additional argument. We recommend
        setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.

        .. code-block:: python

            # CASE 2: multiple test dataloaders
            def test_step(self, batch, batch_idx, dataloader_idx=0):
                # dataloader_idx tells you which dataset this is.
                ...

        Note:
            If you don't need to test you don't need to implement this method.

        Note:
            When the :meth:`test_step` is called, the model has been put in eval mode and
            PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
            to training mode and gradients are enabled.
        """

    def test_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        """Use this when testing with DP because :meth:`test_step` will operate on only part of the batch. However,
        this is still optional and only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be called
            so that you don't have to change your code.

        .. code-block:: python

            # pseudocode
            sub_batches = split_batches_for_dp(batch)
            step_output = [test_step(sub_batch) for sub_batch in sub_batches]
            test_step_end(step_output)

        Args:
            step_output: What you return in :meth:`test_step` for each batch part.

        Return:
            None or anything

        .. code-block:: python

            # WITHOUT test_step_end
            # if used in DP, this batch is 1/num_gpus large
            def test_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self(x)
                loss = self.softmax(out)
                self.log("test_loss", loss)


            # --------------
            # with test_step_end to do softmax over the full batch
            def test_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self.encoder(x)
                return out


            def test_step_end(self, output_results):
                # this out is now the full size of the batch
                all_test_step_outs = output_results.out
                loss = nce_loss(all_test_step_outs)
                self.log("test_loss", loss)

        See Also:
            See the :ref:`Multi GPU Training <gpu_intermediate>` guide for more details.
        """

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        """Called at the end of a test epoch with the output of all test steps.

        .. code-block:: python

            # the pseudocode for these calls
            test_outs = []
            for test_batch in test_data:
                out = test_step(test_batch)
                test_outs.append(out)
            test_epoch_end(test_outs)

        Args:
            outputs: List of outputs you defined in :meth:`test_step_end`, or if there
                are multiple dataloaders, a list containing a list of outputs for each dataloader

        Return:
            None

        Note:
            If you didn't define a :meth:`test_step`, this won't be called.

        Examples:
            With a single dataloader:

            .. code-block:: python

                def test_epoch_end(self, outputs):
                    # do something with the outputs of all test batches
                    all_test_preds = test_step_outputs.predictions

                    some_result = calc_all_results(all_test_preds)
                    self.log(some_result)

            With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
            one entry per dataloader, while the inner list contains the individual outputs of
            each test step for that dataloader.

            .. code-block:: python

                def test_epoch_end(self, outputs):
                    final_value = 0
                    for dataloader_outputs in outputs:
                        for test_step_out in dataloader_outputs:
                            # do something
                            final_value += test_step_out

                    self.log("final_metric", final_value)
        """

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Step function called during :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`. By default, it
        calls :meth:`~pytorch_lightning.core.module.LightningModule.forward`. Override to add any processing logic.

        The :meth:`~pytorch_lightning.core.module.LightningModule.predict_step` is used
        to scale inference on multi-devices.

        To prevent an OOM error, it is possible to use :class:`~pytorch_lightning.callbacks.BasePredictionWriter`
        callback to write the predictions to disk or database after each batch or on epoch end.

        The :class:`~pytorch_lightning.callbacks.BasePredictionWriter` should be used while using a spawn
        based accelerator. This happens for ``Trainer(strategy="ddp_spawn")``
        or training on 8 TPU cores with ``Trainer(accelerator="tpu", devices=8)`` as predictions won't be returned.

        Example ::

            class MyModel(LightningModule):

                def predict_step(self, batch, batch_idx, dataloader_idx=0):
                    return self(batch)

            dm = ...
            model = MyModel()
            trainer = Trainer(accelerator="gpu", devices=2)
            predictions = trainer.predict(model, dm)


        Args:
            batch: Current batch.
            batch_idx: Index of current batch.
            dataloader_idx: Index of the current dataloader.

        Return:
            Predicted output
        """
        return self(batch)

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        """Configure model-specific callbacks. When the model gets attached, e.g., when ``.fit()`` or ``.test()``
        gets called, the list or a callback returned here will be merged with the list of callbacks passed to the
        Trainer's ``callbacks`` argument. If a callback returned here has the same type as one or several callbacks
        already present in the Trainer's callbacks list, it will take priority and replace them. In addition,
        Lightning will make sure :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` callbacks
        run last.

        Return:
            A callback or a list of callbacks which will extend the list of callbacks in the Trainer.

        Example::

            def configure_callbacks(self):
                early_stop = EarlyStopping(monitor="val_acc", mode="max")
                checkpoint = ModelCheckpoint(monitor="val_loss")
                return [early_stop, checkpoint]

        Note:
            Certain callback methods like :meth:`~pytorch_lightning.callbacks.base.Callback.on_init_start`
            will never be invoked on the new callbacks returned here.
        """
        return []

    def configure_optimizers(self):
        r"""
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - **Single optimizer**.
            - **List or Tuple** of optimizers.
            - **Two lists** - The first list has multiple optimizers, and the second has multiple LR schedulers
              (or multiple ``lr_scheduler_config``).
            - **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``
              key whose value is a single LR scheduler or ``lr_scheduler_config``.
            - **Tuple of dictionaries** as described above, with an optional ``"frequency"`` key.
            - **None** - Fit will run without any optimizer.

        The ``lr_scheduler_config`` is a dictionary which contains the scheduler and its associated configuration.
        The default configuration is shown below.

        .. code-block:: python

            lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val_loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }

        When there are schedulers in which the ``.step()`` method is conditioned on a value, such as the
        :class:`torch.optim.lr_scheduler.ReduceLROnPlateau` scheduler, Lightning requires that the
        ``lr_scheduler_config`` contains the keyword ``"monitor"`` set to the metric name that the scheduler
        should be conditioned on.

        .. testcode::

            # The ReduceLROnPlateau scheduler requires a monitor
            def configure_optimizers(self):
                optimizer = Adam(...)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": ReduceLROnPlateau(optimizer, ...),
                        "monitor": "metric_to_track",
                        "frequency": "indicates how often the metric is updated"
                        # If "monitor" references validation metrics, then "frequency" should be set to a
                        # multiple of "trainer.check_val_every_n_epoch".
                    },
                }


            # In the case of two optimizers, only one using the ReduceLROnPlateau scheduler
            def configure_optimizers(self):
                optimizer1 = Adam(...)
                optimizer2 = SGD(...)
                scheduler1 = ReduceLROnPlateau(optimizer1, ...)
                scheduler2 = LambdaLR(optimizer2, ...)
                return (
                    {
                        "optimizer": optimizer1,
                        "lr_scheduler": {
                            "scheduler": scheduler1,
                            "monitor": "metric_to_track",
                        },
                    },
                    {"optimizer": optimizer2, "lr_scheduler": scheduler2},
                )

        Metrics can be made available to monitor by simply logging it using
        ``self.log('metric_to_track', metric_val)`` in your :class:`~pytorch_lightning.core.module.LightningModule`.

        Note:
            The ``frequency`` value specified in a dict along with the ``optimizer`` key is an int corresponding
            to the number of sequential batches optimized with the specific optimizer.
            It should be given to none or to all of the optimizers.
            There is a difference between passing multiple optimizers in a list,
            and passing multiple optimizers in dictionaries with a frequency of 1:

                - In the former case, all optimizers will operate on the given batch in each optimization step.
                - In the latter, only one optimizer will operate on the given batch at every step.

            This is different from the ``frequency`` value specified in the ``lr_scheduler_config`` mentioned above.

            .. code-block:: python

                def configure_optimizers(self):
                    optimizer_one = torch.optim.SGD(self.model.parameters(), lr=0.01)
                    optimizer_two = torch.optim.SGD(self.model.parameters(), lr=0.01)
                    return [
                        {"optimizer": optimizer_one, "frequency": 5},
                        {"optimizer": optimizer_two, "frequency": 10},
                    ]

            In this example, the first optimizer will be used for the first 5 steps,
            the second optimizer for the next 10 steps and that cycle will continue.
            If an LR scheduler is specified for an optimizer using the ``lr_scheduler`` key in the above dict,
            the scheduler will only be updated when its optimizer is being used.

        Examples::

            # most cases. no learning rate scheduler
            def configure_optimizers(self):
                return Adam(self.parameters(), lr=1e-3)

            # multiple optimizer case (e.g.: GAN)
            def configure_optimizers(self):
                gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
                return gen_opt, dis_opt

            # example with learning rate schedulers
            def configure_optimizers(self):
                gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
                dis_sch = CosineAnnealing(dis_opt, T_max=10)
                return [gen_opt, dis_opt], [dis_sch]

            # example with step-based learning rate schedulers
            # each optimizer has its own scheduler
            def configure_optimizers(self):
                gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
                gen_sch = {
                    'scheduler': ExponentialLR(gen_opt, 0.99),
                    'interval': 'step'  # called after each training step
                }
                dis_sch = CosineAnnealing(dis_opt, T_max=10) # called every epoch
                return [gen_opt, dis_opt], [gen_sch, dis_sch]

            # example with optimizer frequencies
            # see training procedure in `Improved Training of Wasserstein GANs`, Algorithm 1
            # https://arxiv.org/abs/1704.00028
            def configure_optimizers(self):
                gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
                n_critic = 5
                return (
                    {'optimizer': dis_opt, 'frequency': n_critic},
                    {'optimizer': gen_opt, 'frequency': 1}
                )

        Note:
            Some things to know:

            - Lightning calls ``.backward()`` and ``.step()`` on each optimizer as needed.
            - If learning rate scheduler is specified in ``configure_optimizers()`` with key
              ``"interval"`` (default "epoch") in the scheduler configuration, Lightning will call
              the scheduler's ``.step()`` method automatically in case of automatic optimization.
            - If you use 16-bit precision (``precision=16``), Lightning will automatically handle the optimizers.
            - If you use multiple optimizers, :meth:`training_step` will have an additional ``optimizer_idx`` parameter.
            - If you use :class:`torch.optim.LBFGS`, Lightning handles the closure function automatically for you.
            - If you use multiple optimizers, gradients will be calculated only for the parameters of current optimizer
              at each training step.
            - If you need to control how often those optimizers step or override the default ``.step()`` schedule,
              override the :meth:`optimizer_step` hook.
        """
        rank_zero_warn("`configure_optimizers` must be implemented to be used with the Lightning Trainer")

    def manual_backward(self, loss: Tensor, *args, **kwargs) -> None:
        """Call this directly from your :meth:`training_step` when doing optimizations manually. By using this,
        Lightning can ensure that all the proper scaling gets applied when using mixed precision.

        See :ref:`manual optimization<common/optimization:Manual optimization>` for more examples.

        Example::

            def training_step(...):
                opt = self.optimizers()
                loss = ...
                opt.zero_grad()
                # automatically applies scaling, etc...
                self.manual_backward(loss)
                opt.step()

        Args:
            loss: The tensor on which to compute gradients. Must have a graph attached.
            *args: Additional positional arguments to be forwarded to :meth:`~torch.Tensor.backward`
            **kwargs: Additional keyword arguments to be forwarded to :meth:`~torch.Tensor.backward`
        """
        self._verify_is_manual_optimization("manual_backward")
        self.trainer.strategy.backward(loss, None, None, *args, **kwargs)

    def backward(
        self, loss: Tensor, optimizer: Optional[Optimizer], optimizer_idx: Optional[int], *args, **kwargs
    ) -> None:
        """Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your
        own implementation if you need to.

        Args:
            loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
                holds the normalized value (scaled by 1 / accumulation steps).
            optimizer: Current optimizer being used. ``None`` if using manual optimization.
            optimizer_idx: Index of the current optimizer being used. ``None`` if using manual optimization.

        Example::

            def backward(self, loss, optimizer, optimizer_idx):
                loss.backward()
        """
        loss.backward(*args, **kwargs)

    def toggle_optimizer(self, optimizer: Union[Optimizer, LightningOptimizer], optimizer_idx: int) -> None:
        """Makes sure only the gradients of the current optimizer's parameters are calculated in the training step
        to prevent dangling gradients in multiple-optimizer setup.

        This is only called automatically when automatic optimization is enabled and multiple optimizers are used.
        It works with :meth:`untoggle_optimizer` to make sure ``param_requires_grad_state`` is properly reset.

        Args:
            optimizer: The optimizer to toggle.
            optimizer_idx: The index of the optimizer to toggle.
        """
        # Iterate over all optimizer parameters to preserve their `requires_grad` information
        # in case these are pre-defined during `configure_optimizers`
        param_requires_grad_state = {}
        for opt in self.trainer.optimizers:
            for group in opt.param_groups:
                for param in group["params"]:
                    # If a param already appear in param_requires_grad_state, continue
                    if param in param_requires_grad_state:
                        continue
                    param_requires_grad_state[param] = param.requires_grad
                    param.requires_grad = False

        # Then iterate over the current optimizer's parameters and set its `requires_grad`
        # properties accordingly
        for group in optimizer.param_groups:
            for param in group["params"]:
                param.requires_grad = param_requires_grad_state[param]
        self._param_requires_grad_state = param_requires_grad_state

    def untoggle_optimizer(self, optimizer_idx: int) -> None:
        """Resets the state of required gradients that were toggled with :meth:`toggle_optimizer`.

        This is only called automatically when automatic optimization is enabled and multiple optimizers are used.

        Args:
            optimizer_idx: The index of the optimizer to untoggle.
        """
        for opt_idx, opt in enumerate(self.trainer.optimizers):
            if optimizer_idx != opt_idx:
                for group in opt.param_groups:
                    for param in group["params"]:
                        if param in self._param_requires_grad_state:
                            param.requires_grad = self._param_requires_grad_state[param]
        # save memory
        self._param_requires_grad_state = {}

    def clip_gradients(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ):
        """Handles gradient clipping internally.

        Note:
            Do not override this method. If you want to customize gradient clipping, consider
            using :meth:`configure_gradient_clipping` method.

        Args:
            optimizer: Current optimizer being used.
            gradient_clip_val: The value at which to clip gradients.
            gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
                to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm.
        """
        if gradient_clip_val is None:
            gradient_clip_val = self.trainer.gradient_clip_val or 0.0
        elif self.trainer.gradient_clip_val is not None and self.trainer.gradient_clip_val != gradient_clip_val:
            raise MisconfigurationException(
                f"You have set `Trainer(gradient_clip_val={self.trainer.gradient_clip_val!r})`"
                f" and have passed `clip_gradients(gradient_clip_val={gradient_clip_val!r})`."
                " Please use only one of them."
            )

        if gradient_clip_algorithm is None:
            gradient_clip_algorithm = self.trainer.gradient_clip_algorithm or "norm"
        else:
            gradient_clip_algorithm = gradient_clip_algorithm.lower()
            if (
                self.trainer.gradient_clip_algorithm is not None
                and self.trainer.gradient_clip_algorithm != gradient_clip_algorithm
            ):
                raise MisconfigurationException(
                    f"You have set `Trainer(gradient_clip_algorithm={self.trainer.gradient_clip_algorithm.value!r})`"
                    f" and have passed `clip_gradients(gradient_clip_algorithm={gradient_clip_algorithm!r})"
                    " Please use only one of them."
                )

        if not isinstance(gradient_clip_val, (int, float)):
            raise TypeError(f"`gradient_clip_val` should be an int or a float. Got {gradient_clip_val}.")

        if not GradClipAlgorithmType.supported_type(gradient_clip_algorithm.lower()):
            raise MisconfigurationException(
                f"`gradient_clip_algorithm` {gradient_clip_algorithm} is invalid."
                f" Allowed algorithms: {GradClipAlgorithmType.supported_types()}."
            )

        gradient_clip_algorithm = GradClipAlgorithmType(gradient_clip_algorithm)
        self.trainer.precision_plugin.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        optimizer_idx: int,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ):
        """Perform gradient clipping for the optimizer parameters. Called before :meth:`optimizer_step`.

        Args:
            optimizer: Current optimizer being used.
            optimizer_idx: Index of the current optimizer being used.
            gradient_clip_val: The value at which to clip gradients. By default value passed in Trainer
                will be available here.
            gradient_clip_algorithm: The gradient clipping algorithm to use. By default value
                passed in Trainer will be available here.

        Example::

            # Perform gradient clipping on gradients associated with discriminator (optimizer_idx=1) in GAN
            def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
                if optimizer_idx == 1:
                    # Lightning will handle the gradient clipping
                    self.clip_gradients(
                        optimizer,
                        gradient_clip_val=gradient_clip_val,
                        gradient_clip_algorithm=gradient_clip_algorithm
                    )
                else:
                    # implement your own custom logic to clip gradients for generator (optimizer_idx=0)
        """
        self.clip_gradients(
            optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm
        )

    def lr_scheduler_step(
        self,
        scheduler: LRSchedulerTypeUnion,
        optimizer_idx: int,
        metric: Optional[Any],
    ) -> None:
        r"""
        Override this method to adjust the default way the
        :class:`~pytorch_lightning.trainer.trainer.Trainer` calls each scheduler.
        By default, Lightning calls ``step()`` and as shown in the example
        for each scheduler based on its ``interval``.

        Args:
            scheduler: Learning rate scheduler.
            optimizer_idx: Index of the optimizer associated with this scheduler.
            metric: Value of the monitor used for schedulers like ``ReduceLROnPlateau``.

        Examples::

            # DEFAULT
            def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)

            # Alternative way to update schedulers if it requires an epoch value
            def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
                scheduler.step(epoch=self.current_epoch)

        """
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_idx: int = 0,
        optimizer_closure: Optional[Callable[[], Any]] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        r"""
        Override this method to adjust the default way the :class:`~pytorch_lightning.trainer.trainer.Trainer` calls
        each optimizer.

        By default, Lightning calls ``step()`` and ``zero_grad()`` as shown in the example once per optimizer.
        This method (and ``zero_grad()``) won't be called during the accumulation phase when
        ``Trainer(accumulate_grad_batches != 1)``. Overriding this hook has no benefit with manual optimization.

        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers, this indexes into that list.
            optimizer_closure: The optimizer closure. This closure must be executed as it includes the
                calls to ``training_step()``, ``optimizer.zero_grad()``, and ``backward()``.
            on_tpu: ``True`` if TPU backward is required
            using_native_amp: ``True`` if using native amp
            using_lbfgs: True if the matching optimizer is :class:`torch.optim.LBFGS`

        Examples::

            # DEFAULT
            def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                               optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
                optimizer.step(closure=optimizer_closure)

            # Alternating schedule for optimizer steps (i.e.: GANs)
            def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                               optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
                # update generator opt every step
                if optimizer_idx == 0:
                    optimizer.step(closure=optimizer_closure)

                # update discriminator opt every 2 steps
                if optimizer_idx == 1:
                    if (batch_idx + 1) % 2 == 0 :
                        optimizer.step(closure=optimizer_closure)
                    else:
                        # call the closure by itself to run `training_step` + `backward` without an optimizer step
                        optimizer_closure()

                # ...
                # add as many optimizers as you want

        Here's another example showing how to use this for more advanced things such as
        learning rate warm-up:

        .. code-block:: python

            # learning rate warm-up
            def optimizer_step(
                self,
                epoch,
                batch_idx,
                optimizer,
                optimizer_idx,
                optimizer_closure,
                on_tpu,
                using_native_amp,
                using_lbfgs,
            ):
                # update params
                optimizer.step(closure=optimizer_closure)

                # manually warm up lr without a scheduler
                if self.trainer.global_step < 500:
                    lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr_scale * self.learning_rate

        """
        optimizer.step(closure=optimizer_closure)

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
        """Override this method to change the default behaviour of ``optimizer.zero_grad()``.

        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.

        Examples::

            # DEFAULT
            def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
                optimizer.zero_grad()

            # Set gradients to `None` instead of zero to improve performance.
            def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
                optimizer.zero_grad(set_to_none=True)

        See :meth:`torch.optim.Optimizer.zero_grad` for the explanation of the above example.
        """
        optimizer.zero_grad()

    def tbptt_split_batch(self, batch: Any, split_size: int) -> List[Any]:
        r"""
        When using truncated backpropagation through time, each batch must be split along the
        time dimension. Lightning handles this by default, but for custom behavior override
        this function.

        Args:
            batch: Current batch
            split_size: The size of the split

        Return:
            List of batch splits. Each split will be passed to :meth:`training_step` to enable truncated
            back propagation through time. The default implementation splits root level Tensors and
            Sequences at dim=1 (i.e. time dim). It assumes that each time dim is the same length.

        Examples::

            def tbptt_split_batch(self, batch, split_size):
                splits = []
                for t in range(0, time_dims[0], split_size):
                    batch_split = []
                    for i, x in enumerate(batch):
                        if isinstance(x, torch.Tensor):
                            split_x = x[:, t:t + split_size]
                        elif isinstance(x, collections.abc.Sequence):
                            split_x = [None] * len(x)
                            for batch_idx in range(len(x)):
                              split_x[batch_idx] = x[batch_idx][t:t + split_size]
                        batch_split.append(split_x)
                    splits.append(batch_split)
                return splits

        Note:
            Called in the training loop after
            :meth:`~pytorch_lightning.callbacks.base.Callback.on_train_batch_start`
            if :paramref:`~pytorch_lightning.core.module.LightningModule.truncated_bptt_steps` > 0.
            Each returned batch split is passed separately to :meth:`training_step`.
        """
        time_dims = [len(x[0]) for x in batch if isinstance(x, (Tensor, collections.abc.Sequence))]
        assert len(time_dims) >= 1, "Unable to determine batch time dimension"
        assert all(x == time_dims[0] for x in time_dims), "Batch time dimension length is ambiguous"

        splits = []
        for t in range(0, time_dims[0], split_size):
            batch_split = []
            for i, x in enumerate(batch):
                if isinstance(x, Tensor):
                    split_x = x[:, t : t + split_size]
                elif isinstance(x, collections.abc.Sequence):
                    split_x = [None] * len(x)
                    for batch_idx in range(len(x)):
                        split_x[batch_idx] = x[batch_idx][t : t + split_size]

                batch_split.append(split_x)

            splits.append(batch_split)

        return splits

    def freeze(self) -> None:
        r"""
        Freeze all params for inference.

        Example::

            model = MyLightningModule(...)
            model.freeze()

        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all parameters for training.

        .. code-block:: python

            model = MyLightningModule(...)
            model.unfreeze()
        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    def _verify_is_manual_optimization(self, fn_name):
        if self.automatic_optimization:
            raise MisconfigurationException(
                f"to use {fn_name}, please disable automatic optimization:"
                " set model property `automatic_optimization` as False"
            )

    @classmethod
    def _auto_collect_arguments(cls, frame=None) -> Tuple[Dict, Dict]:
        """Collect all module arguments in the current constructor and all child constructors. The child
        constructors are all the ``__init__`` methods that reach the current class through (chained)
        ``super().__init__()`` calls.

        Args:
            frame: instance frame

        Returns:
            self_arguments: arguments dictionary of the first instance
            parents_arguments: arguments dictionary of the parent's instances
        """
        if not frame:
            frame = inspect.currentframe()

        frame_args = collect_init_args(frame.f_back, [])
        self_arguments = frame_args[-1]

        # set hyper_parameters in child
        self_arguments = self_arguments
        parents_arguments = {}

        # add all arguments from parents
        for args in frame_args[:-1]:
            parents_arguments.update(args)
        return self_arguments, parents_arguments

    @torch.no_grad()
    def to_onnx(self, file_path: Union[str, Path], input_sample: Optional[Any] = None, **kwargs):
        """Saves the model in ONNX format.

        Args:
            file_path: The path of the file the onnx model should be saved to.
            input_sample: An input for tracing. Default: None (Use self.example_input_array)
            **kwargs: Will be passed to torch.onnx.export function.

        Example:
            >>> class SimpleModel(LightningModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.l1 = torch.nn.Linear(in_features=64, out_features=4)
            ...
            ...     def forward(self, x):
            ...         return torch.relu(self.l1(x.view(x.size(0), -1)))

            >>> with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
            ...     model = SimpleModel()
            ...     input_sample = torch.randn((1, 64))
            ...     model.to_onnx(tmpfile.name, input_sample, export_params=True)
            ...     os.path.isfile(tmpfile.name)
            True
        """
        mode = self.training

        if input_sample is None:
            if self.example_input_array is None:
                raise ValueError(
                    "Could not export to ONNX since neither `input_sample` nor"
                    " `model.example_input_array` attribute is set."
                )
            input_sample = self.example_input_array

        input_sample = self._apply_batch_transfer_handler(input_sample)

        if not _TORCH_GREATER_EQUAL_1_10 and "example_outputs" not in kwargs:
            self.eval()
            if isinstance(input_sample, Tuple):
                kwargs["example_outputs"] = self(*input_sample)
            else:
                kwargs["example_outputs"] = self(input_sample)

        torch.onnx.export(self, input_sample, file_path, **kwargs)
        self.train(mode)

    @torch.no_grad()
    def to_torchscript(
        self,
        file_path: Optional[Union[str, Path]] = None,
        method: Optional[str] = "script",
        example_inputs: Optional[Any] = None,
        **kwargs,
    ) -> Union[ScriptModule, Dict[str, ScriptModule]]:
        """By default compiles the whole model to a :class:`~torch.jit.ScriptModule`. If you want to use tracing,
        please provided the argument ``method='trace'`` and make sure that either the `example_inputs` argument is
        provided, or the model has :attr:`example_input_array` set. If you would like to customize the modules that
        are scripted you should override this method. In case you want to return multiple modules, we recommend
        using a dictionary.

        Args:
            file_path: Path where to save the torchscript. Default: None (no file saved).
            method: Whether to use TorchScript's script or trace method. Default: 'script'
            example_inputs: An input to be used to do tracing when method is set to 'trace'.
              Default: None (uses :attr:`example_input_array`)
            **kwargs: Additional arguments that will be passed to the :func:`torch.jit.script` or
              :func:`torch.jit.trace` function.

        Note:
            - Requires the implementation of the
              :meth:`~pytorch_lightning.core.module.LightningModule.forward` method.
            - The exported script will be set to evaluation mode.
            - It is recommended that you install the latest supported version of PyTorch
              to use this feature without limitations. See also the :mod:`torch.jit`
              documentation for supported features.

        Example:
            >>> class SimpleModel(LightningModule):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.l1 = torch.nn.Linear(in_features=64, out_features=4)
            ...
            ...     def forward(self, x):
            ...         return torch.relu(self.l1(x.view(x.size(0), -1)))
            ...
            >>> model = SimpleModel()
            >>> model.to_torchscript(file_path="model.pt")  # doctest: +SKIP
            >>> os.path.isfile("model.pt")  # doctest: +SKIP
            >>> torch.jit.save(model.to_torchscript(file_path="model_trace.pt", method='trace', # doctest: +SKIP
            ...                                     example_inputs=torch.randn(1, 64)))  # doctest: +SKIP
            >>> os.path.isfile("model_trace.pt")  # doctest: +SKIP
            True

        Return:
            This LightningModule as a torchscript, regardless of whether `file_path` is
            defined or not.
        """
        mode = self.training

        self._running_torchscript = True

        if method == "script":
            torchscript_module = torch.jit.script(self.eval(), **kwargs)
        elif method == "trace":
            # if no example inputs are provided, try to see if model has example_input_array set
            if example_inputs is None:
                if self.example_input_array is None:
                    raise ValueError(
                        "Choosing method=`trace` requires either `example_inputs`"
                        " or `model.example_input_array` to be defined."
                    )
                example_inputs = self.example_input_array

            # automatically send example inputs to the right device and use trace
            example_inputs = self._apply_batch_transfer_handler(example_inputs)
            torchscript_module = torch.jit.trace(func=self.eval(), example_inputs=example_inputs, **kwargs)
        else:
            raise ValueError(f"The 'method' parameter only supports 'script' or 'trace', but value given was: {method}")

        self.train(mode)

        if file_path is not None:
            fs = get_filesystem(file_path)
            with fs.open(file_path, "wb") as f:
                torch.jit.save(torchscript_module, f)

        self._running_torchscript = False

        return torchscript_module

    @property
    def use_amp(self) -> bool:
        r"""
        .. deprecated:: v1.6.

            This property was deprecated in v1.6 and will be removed in v1.8.
        """
        if not self._running_torchscript:  # remove with the deprecation removal
            rank_zero_deprecation(
                "`LightningModule.use_amp` was deprecated in v1.6 and will be removed in v1.8."
                " Please use `Trainer.amp_backend`.",
                stacklevel=5,
            )
        return self._use_amp

    @use_amp.setter
    def use_amp(self, use_amp: bool) -> None:
        r"""
        .. deprecated:: v1.6.

            This property was deprecated in v1.6 and will be removed in v1.8.
        """
        if not self._running_torchscript:  # remove with the deprecation removal
            rank_zero_deprecation(
                "`LightningModule.use_amp` was deprecated in v1.6 and will be removed in v1.8."
                " Please use `Trainer.amp_backend`.",
                stacklevel=5,
            )
        self._use_amp = use_amp

    @contextmanager
    def _prevent_trainer_and_dataloaders_deepcopy(self) -> None:
        self._should_prevent_trainer_and_dataloaders_deepcopy = True
        yield
        self._should_prevent_trainer_and_dataloaders_deepcopy = False

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        if self._should_prevent_trainer_and_dataloaders_deepcopy:
            state["_trainer"] = None
            state.pop("train_dataloader", None)
            state.pop("val_dataloader", None)
            state.pop("test_dataloader", None)
            state.pop("predict_dataloader", None)
        return state

    def _register_sharded_tensor_state_dict_hooks_if_available(self) -> None:
        """Adds ShardedTensor state dict hooks if ShardedTensors are supported.

        These hooks ensure that ShardedTensors are included when saving, and are loaded the LightningModule correctly.
        """
        if not _TORCH_GREATER_EQUAL_1_10 or _IS_WINDOWS or not torch.distributed.is_available():
            rank_zero_debug("Could not register sharded tensor state dict hooks")
            return

        if _TORCH_GREATER_EQUAL_1_11:
            from torch.distributed._shard.sharded_tensor import pre_load_state_dict_hook, state_dict_hook
        else:
            from torch.distributed._sharded_tensor import pre_load_state_dict_hook, state_dict_hook

        self._register_state_dict_hook(state_dict_hook)

        if _TORCH_GREATER_EQUAL_1_13:
            self._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)
        else:
            # We need to make sure the self inside the method is a weakref proxy
            self.__class__._register_load_state_dict_pre_hook(weakref.proxy(self), pre_load_state_dict_hook, True)
