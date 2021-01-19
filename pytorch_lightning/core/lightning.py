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

"""nn.Module with additional great features."""

import collections
import copy
import inspect
import os
import re
import tempfile
from abc import ABC
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import ScriptModule, Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from pytorch_lightning import _logger as log
from pytorch_lightning.core.grads import GradInformation
from pytorch_lightning.core.hooks import CheckpointHooks, DataHooks, ModelHooks
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.core.saving import ALLOWED_CONFIG_TYPES, ModelIO, PRIMITIVE_TYPES
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.utilities import rank_zero_warn, TPU_AVAILABLE
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import AttributeDict, collect_init_args, get_init_args

if TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm


class LightningModule(
    ABC,
    DeviceDtypeModuleMixin,
    GradInformation,
    ModelIO,
    ModelHooks,
    DataHooks,
    CheckpointHooks,
    Module,
):
    # Below is for property support of JIT in PyTorch 1.7
    # since none of them is important when using JIT, we are going to ignore them.
    __jit_unused_properties__ = [
        "datamodule",
        "example_input_array",
        "hparams",
        "hparams_initial",
        "on_gpu",
        "current_epoch",
        "global_step",
    ] + DeviceDtypeModuleMixin.__jit_unused_properties__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # see (https://github.com/pytorch/pytorch/blob/3e6bb5233f9ca2c5aa55d9cda22a7ee85439aa6e/
        # torch/nn/modules/module.py#L227)
        torch._C._log_api_usage_once(f"lightning.module.{self.__class__.__name__}")

        self.exp_save_path = None

        self.loaded_optimizer_states_dict = {}

        #: Pointer to the trainer object
        self.trainer = None

        #: Pointer to the logger object
        self.logger = None

        #: True if using dp
        self.use_dp = False

        #: True if using ddp
        self.use_ddp = False

        #: True if using ddp2
        self.use_ddp2 = False

        # True if on tpu
        self.use_tpu = False

        #: True if using amp
        self.use_amp = False

        #: The precision used
        self.precision = 32

        # optionally can be set by user
        self._example_input_array = None
        self._datamodule = None
        self._results: Optional[Result] = None
        self._current_fx_name = ''
        self._running_manual_backward = False
        self._current_hook_fx_name = None
        self._current_dataloader_idx = None
        self._automatic_optimization: bool = True

    def optimizers(self, use_pl_optimizer: bool = True) -> Union[Optimizer, List[Optimizer], List[LightningOptimizer]]:
        if use_pl_optimizer:
            opts = list(self.trainer.lightning_optimizers.values())
        else:
            opts = self.trainer.optimizers

        # single optimizer
        if isinstance(opts, list) and len(opts) == 1 and isinstance(opts[0], Optimizer):
            return opts[0]
        # multiple opts
        return opts

    @property
    def example_input_array(self) -> Any:
        return self._example_input_array

    @property
    def current_epoch(self) -> int:
        """The current epoch"""
        return self.trainer.current_epoch if self.trainer else 0

    @property
    def global_step(self) -> int:
        """Total training batches seen across all epochs"""
        return self.trainer.global_step if self.trainer else 0

    @example_input_array.setter
    def example_input_array(self, example: Any) -> None:
        self._example_input_array = example

    @property
    def datamodule(self) -> Any:
        return self._datamodule

    @datamodule.setter
    def datamodule(self, datamodule: Any) -> None:
        self._datamodule = datamodule

    @property
    def on_gpu(self):
        """
        True if your model is currently running on GPUs.
        Useful to set flags around the LightningModule for different CPU vs GPU behavior.
        """
        return self.device.type == "cuda"

    @property
    def automatic_optimization(self) -> bool:
        """
        If False you are responsible for calling .backward, .step, zero_grad.
        """
        return self._automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, automatic_optimization: bool) -> None:
        self._automatic_optimization = automatic_optimization

    def print(self, *args, **kwargs) -> None:
        r"""
        Prints only from process 0. Use this in any distributed mode to log only once.

        Args:
            *args: The thing to print. Will be passed to Python's built-in print function.
            **kwargs: Will be passed to Python's built-in print function.

        Example:

            .. code-block:: python

                def forward(self, x):
                    self.print(x, 'in forward')

        """
        if self.trainer.is_global_zero:
            print(*args, **kwargs)

    def log(
        self,
        name: str,
        value: Any,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Callable = torch.mean,
        tbptt_reduce_fx: Callable = torch.mean,
        tbptt_pad_token: int = 0,
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_op: Union[Any, str] = 'mean',
        sync_dist_group: Optional[Any] = None,
    ):
        """
        Log a key, value

        Example::

            self.log('train_loss', loss)

        The default behavior per hook is as follows

        .. csv-table:: ``*`` also applies to the test loop
           :header: "LightningMoule Hook", "on_step", "on_epoch", "prog_bar", "logger"
           :widths: 20, 10, 10, 10, 10

           "training_step", "T", "F", "F", "T"
           "training_step_end", "T", "F", "F", "T"
           "training_epoch_end", "F", "T", "F", "T"
           "validation_step*", "F", "T", "F", "T"
           "validation_step_end*", "F", "T", "F", "T"
           "validation_epoch_end*", "F", "T", "F", "T"

        Args:
            name: key name
            value: value name
            prog_bar: if True logs to the progress bar
            logger: if True logs to the logger
            on_step: if True logs at this step. None auto-logs at the training_step but not validation/test_step
            on_epoch: if True logs epoch accumulated metrics. None auto-logs at the val/test step but not training_step
            reduce_fx: reduction function over step values for end of epoch. Torch.mean by default
            tbptt_reduce_fx: function to reduce on truncated back prop
            tbptt_pad_token: token to use for padding
            enable_graph: if True, will not auto detach the graph
            sync_dist: if True, reduces the metric across GPUs/TPUs
            sync_dist_op: the op to sync across GPUs/TPUs
            sync_dist_group: the ddp group
        """
        if self._results is not None:
            # in any epoch end can't log step metrics (only epoch metric)
            if 'epoch_end' in self._current_fx_name and on_step:
                m = f'on_step=True cannot be used on {self._current_fx_name} method'
                raise MisconfigurationException(m)

            if 'epoch_end' in self._current_fx_name and on_epoch is False:
                m = f'on_epoch cannot be False when called from the {self._current_fx_name} method'
                raise MisconfigurationException(m)

            # add log_dict
            # TODO: if logged twice fail with crash

            # set the default depending on the fx_name
            on_step = self.__auto_choose_log_on_step(on_step)
            on_epoch = self.__auto_choose_log_on_epoch(on_epoch)

            if self._current_hook_fx_name is not None:
                self.trainer.logger_connector.check_logging_in_callbacks(
                    self._current_hook_fx_name,
                    on_step=on_step,
                    on_epoch=on_epoch
                )

            # make sure user doesn't introduce logic for multi-dataloaders
            if "/dataloader_idx_" in name:
                raise MisconfigurationException(
                    f"Logged key: {name} should not contain information about dataloader_idx.")

            accelerator = self.trainer.accelerator_backend

            self._results.log(
                name,
                value,
                prog_bar,
                logger,
                on_step,
                on_epoch,
                reduce_fx,
                tbptt_reduce_fx,
                tbptt_pad_token,
                enable_graph,
                sync_dist,
                sync_dist_op,
                sync_dist_group,
                accelerator.sync_tensor,
                self._current_dataloader_idx,
                self.device,
            )

    def log_dict(
        self,
        dictionary: dict,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Callable = torch.mean,
        tbptt_reduce_fx: Callable = torch.mean,
        tbptt_pad_token: int = 0,
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_op: Union[Any, str] = 'mean',
        sync_dist_group: Optional[Any] = None,
    ):
        """
        Log a dictonary of values at once

        Example::

            values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
            self.log_dict(values)

        Args:
            dictionary: key value pairs (str, tensors)
            prog_bar: if True logs to the progress base
            logger: if True logs to the logger
            on_step: if True logs at this step. None auto-logs for training_step but not validation/test_step
            on_epoch: if True logs epoch accumulated metrics. None auto-logs for val/test step but not training_step
            reduce_fx: reduction function over step values for end of epoch. Torch.mean by default
            tbptt_reduce_fx: function to reduce on truncated back prop
            tbptt_pad_token: token to use for padding
            enable_graph: if True, will not auto detach the graph
            sync_dist: if True, reduces the metric across GPUs/TPUs
            sync_dist_op: the op to sync across GPUs/TPUs
            sync_dist_group: the ddp group:
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
                sync_dist_op=sync_dist_op,
                tbptt_pad_token=tbptt_pad_token,
                tbptt_reduce_fx=tbptt_reduce_fx,
            )

    def write_prediction(self, name, value, filename='predictions.pt'):
        self.trainer.evaluation_loop.predictions._add_prediction(name, value, filename)

    def write_prediction_dict(self, predictions_dict, filename='predictions.pt'):
        for k, v in predictions_dict.items():
            self.write_prediction(k, v, filename)

    def __auto_choose_log_on_step(self, on_step):
        if on_step is None:
            if self._current_fx_name in {'training_step', 'training_step_end'}:
                on_step = True
            elif self._current_fx_name in {'evaluation_step', 'evaluation_step_end',
                                           'evaluation_epoch_end', 'training_epoch_end'}:
                on_step = False
            else:
                on_step = False

        return on_step

    def __auto_choose_log_on_epoch(self, on_epoch):
        if on_epoch is None:
            if self._current_fx_name in {'training_step', 'training_step_end'}:
                on_epoch = False
            elif self._current_fx_name in {'evaluation_step', 'evaluation_step_end',
                                           'evaluation_epoch_end', 'training_epoch_end'}:
                on_epoch = True
            else:
                on_epoch = True

        return on_epoch

    def all_gather(self, tensor: Union[torch.Tensor], group: Optional[Any] = None, sync_grads: bool = False):
        r"""
        Allows users to call ``self.all_gather()`` from the LightningModule, thus making
        the ```all_gather``` operation accelerator agnostic.

        ```all_gather``` is a function provided by accelerators to gather a tensor from several
        distributed processes

        Args:
            tensor: tensor of shape (batch, ...)
            group: the process group to gather results from. Defaults to all processes (world)
            sync_grads: flag that allows users to synchronize gradients for all_gather op

        Return:
            A tensor of shape (world_size, batch, ...)
        """
        return self.trainer.accelerator_backend.all_gather(tensor, group=group, sync_grads=sync_grads)

    def forward(self, *args, **kwargs):
        r"""
        Same as :meth:`torch.nn.Module.forward()`, however in Lightning you want this to define
        the operations you want to use for prediction (i.e.: on a server or as a feature extractor).

        Normally you'd call ``self()`` from your :meth:`training_step` method.
        This makes it easy to write a complex system for training with the outputs
        you'd want in a prediction setting.

        You may also find the :func:`~pytorch_lightning.core.decorators.auto_move_data` decorator useful
        when using the module outside Lightning in a production setting.

        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.

        Return:
            Predicted output

        Examples:
            .. code-block:: python

                # example if we were using this model as a feature extractor
                def forward(self, x):
                    feature_maps = self.convnet(x)
                    return feature_maps

                def training_step(self, batch, batch_idx):
                    x, y = batch
                    feature_maps = self(x)
                    logits = self.classifier(feature_maps)

                    # ...
                    return loss

                # splitting it this way allows model to be used a feature extractor
                model = MyModelAbove()

                inputs = server.get_request()
                results = model(inputs)
                server.write_results(results)

                # -------------
                # This is in stark contrast to torch.nn.Module where normally you would have this:
                def forward(self, batch):
                    x, y = batch
                    feature_maps = self.convnet(x)
                    logits = self.classifier(feature_maps)
                    return logits

        """
        return super().forward(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        r"""
        Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.

        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (int): Integer displaying index of this batch
            optimizer_idx (int): When using multiple optimizers, this argument will also be present.
            hiddens(:class:`~torch.Tensor`): Passed in if
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.truncated_bptt_steps` > 0.

        Return:
            Any of.

            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``
            - ``None`` - Training will skip to the next batch

        Note:
            Returning ``None`` is currently not supported for multi-GPU or TPU, or with 16-bit precision enabled.

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
                if optimizer_idx == 1:
                    # do training_step with decoder


        If you add truncated back propagation through time you will also get an additional
        argument with the hidden states of the previous step.

        .. code-block:: python

            # Truncated back-propagation through time
            def training_step(self, batch, batch_idx, hiddens):
                # hiddens are the hidden states from the previous truncated backprop step
                ...
                out, hiddens = self.lstm(data, hiddens)
                ...
                return {'loss': loss, 'hiddens': hiddens}

        Note:
            The loss value shown in the progress bar is smoothed (averaged) over the last values,
            so it differs from the actual loss returned in train/validation step.
        """
        rank_zero_warn(
            "`training_step` must be implemented to be used with the Lightning Trainer"
        )

    def training_step_end(self, *args, **kwargs):
        """
        Use this when training with dp or ddp2 because :meth:`training_step`
        will operate on only part of the batch. However, this is still optional
        and only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be called
            so that you don't have to change your code

        .. code-block:: python

            # pseudocode
            sub_batches = split_batches_for_dp(batch)
            batch_parts_outputs = [training_step(sub_batch) for sub_batch in sub_batches]
            training_step_end(batch_parts_outputs)

        Args:
            batch_parts_outputs: What you return in `training_step` for each batch part.

        Return:
            Anything

        When using dp/ddp2 distributed backends, only a portion of the batch is inside the training_step:

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self(x)

                # softmax uses only a portion of the batch in the denomintaor
                loss = self.softmax(out)
                loss = nce_loss(loss)
                return loss

        If you wish to do something with all the parts of the batch, then use this method to do it:

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self.encoder(x)
                return {'pred': out}

            def training_step_end(self, training_step_outputs):
                gpu_0_pred = training_step_outputs[0]['pred']
                gpu_1_pred = training_step_outputs[1]['pred']
                gpu_n_pred = training_step_outputs[n]['pred']

                # this softmax now uses the full batch
                loss = nce_loss([gpu_0_pred, gpu_1_pred, gpu_n_pred])
                return loss

        See Also:
            See the :ref:`multi_gpu` guide for more details.
        """

    def training_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the training epoch with the outputs of all training steps.
        Use this in case you need to do something with all the outputs for every training_step.

        .. code-block:: python

            # the pseudocode for these calls
            train_outs = []
            for train_batch in train_data:
                out = training_step(train_batch)
                train_outs.append(out)
            training_epoch_end(train_outs)

        Args:
            outputs: List of outputs you defined in :meth:`training_step`, or if there are
                multiple dataloaders, a list containing a list of outputs for each dataloader.

        Return:
            None

        Note:
            If this method is not overridden, this won't be called.

        Example::

            def training_epoch_end(self, training_step_outputs):
                # do something with all training_step outputs
                return result

        With multiple dataloaders, ``outputs`` will be a list of lists. The outer list contains
        one entry per dataloader, while the inner list contains the individual outputs of
        each training step for that dataloader.

        .. code-block:: python

            def training_epoch_end(self, training_step_outputs):
                for out in training_step_outputs:
                    # do something here
        """

    def validation_step(self, *args, **kwargs):
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
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (int): The index of this batch
            dataloader_idx (int): The index of the dataloader that produced this batch
                (only if multiple val dataloaders used)

        Return:
           Any of.

            - Any object or value
            - ``None`` - Validation will skip to the next batch

        .. code-block:: python

            # pseudocode of order
            out = validation_step()
            if defined('validation_step_end'):
                out = validation_step_end(out)
            out = validation_epoch_end(out)


        .. code-block:: python

            # if you have one val dataloader:
            def validation_step(self, batch, batch_idx)

            # if you have multiple val dataloaders:
            def validation_step(self, batch, batch_idx, dataloader_idx)

        Examples:
            .. code-block:: python

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

            If you pass in multiple val dataloaders, :meth:`validation_step` will have an additional argument.

            .. code-block:: python

                # CASE 2: multiple validation dataloaders
                def validation_step(self, batch, batch_idx, dataloader_idx):
                    # dataloader_idx tells you which dataset this is.

        Note:
            If you don't need to validate you don't need to implement this method.

        Note:
            When the :meth:`validation_step` is called, the model has been put in eval mode
            and PyTorch gradients have been disabled. At the end of validation,
            the model goes back to training mode and gradients are enabled.
        """

    def validation_step_end(self, *args, **kwargs):
        """
        Use this when validating with dp or ddp2 because :meth:`validation_step`
        will operate on only part of the batch. However, this is still optional
        and only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be called
            so that you don't have to change your code.

        .. code-block:: python

            # pseudocode
            sub_batches = split_batches_for_dp(batch)
            batch_parts_outputs = [validation_step(sub_batch) for sub_batch in sub_batches]
            validation_step_end(batch_parts_outputs)

        Args:
            batch_parts_outputs: What you return in :meth:`validation_step`
                for each batch part.

        Return:
            None or anything

        .. code-block:: python

            # WITHOUT validation_step_end
            # if used in DP or DDP2, this batch is 1/num_gpus large
            def validation_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self.encoder(x)
                loss = self.softmax(out)
                loss = nce_loss(loss)
                self.log('val_loss', loss)

            # --------------
            # with validation_step_end to do softmax over the full batch
            def validation_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self(x)
                return out

            def validation_step_end(self, val_step_outputs):
                for out in val_step_outputs:
                    # do something with these

        See Also:
            See the :ref:`multi_gpu` guide for more details.
        """

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """
        Called at the end of the validation epoch with the outputs of all validation steps.

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
                        # do something

            With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
            one entry per dataloader, while the inner list contains the individual outputs of
            each validation step for that dataloader.

            .. code-block:: python

                def validation_epoch_end(self, outputs):
                    for dataloader_output_result in outputs:
                        dataloader_outs = dataloader_output_result.dataloader_i_outputs

                    self.log('final_metric', final_value)
        """

    def test_step(self, *args, **kwargs):
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
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (int): The index of this batch.
            dataloader_idx (int): The index of the dataloader that produced this batch
                (only if multiple test dataloaders used).

        Return:
           Any of.

            - Any object or value
            - ``None`` - Testing will skip to the next batch

        .. code-block:: python

            # if you have one test dataloader:
            def test_step(self, batch, batch_idx)

            # if you have multiple test dataloaders:
            def test_step(self, batch, batch_idx, dataloader_idx)

        Examples:
            .. code-block:: python

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

            If you pass in multiple test dataloaders, :meth:`test_step` will have an additional
            argument.

            .. code-block:: python

                # CASE 2: multiple test dataloaders
                def test_step(self, batch, batch_idx, dataloader_idx):
                    # dataloader_idx tells you which dataset this is.

        Note:
            If you don't need to test you don't need to implement this method.

        Note:
            When the :meth:`test_step` is called, the model has been put in eval mode and
            PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
            to training mode and gradients are enabled.
        """

    def test_step_end(self, *args, **kwargs):
        """
        Use this when testing with dp or ddp2 because :meth:`test_step` will operate
        on only part of the batch. However, this is still optional
        and only needed for things like softmax or NCE loss.

        Note:
            If you later switch to ddp or some other mode, this will still be called
            so that you don't have to change your code.

        .. code-block:: python

            # pseudocode
            sub_batches = split_batches_for_dp(batch)
            batch_parts_outputs = [test_step(sub_batch) for sub_batch in sub_batches]
            test_step_end(batch_parts_outputs)

        Args:
            batch_parts_outputs: What you return in :meth:`test_step` for each batch part.

        Return:
            None or anything

        .. code-block:: python

            # WITHOUT test_step_end
            # if used in DP or DDP2, this batch is 1/num_gpus large
            def test_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self(x)
                loss = self.softmax(out)
                self.log('test_loss', loss)

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
                self.log('test_loss', loss)

        See Also:
            See the :ref:`multi_gpu` guide for more details.
        """

    def test_epoch_end(
        self, outputs: List[Any]
    ) -> None:
        """
        Called at the end of a test epoch with the output of all test steps.

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

                    self.log('final_metric', final_value)
        """

    def configure_optimizers(
            self,
    ):
        r"""
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.

        Note:
            The 'frequency' value is an int corresponding to the number of sequential batches
            optimized with the specific optimizer. It should be given to none or to all of the optimizers.
            There is a difference between passing multiple optimizers in a list,
            and passing multiple optimizers in dictionaries with a frequency of 1:
            In the former case, all optimizers will operate on the given batch in each optimization step.
            In the latter, only one optimizer will operate on the given batch at every step.

            The lr_dict is a dictionary which contains the scheduler and its associated configuration.
            The default configuration is shown below.

            .. code-block:: python

                {
                    'scheduler': lr_scheduler, # The LR scheduler instance (required)
                    'interval': 'epoch', # The unit of the scheduler's step size
                    'frequency': 1, # The frequency of the scheduler
                    'reduce_on_plateau': False, # For ReduceLROnPlateau scheduler
                    'monitor': 'val_loss', # Metric for ReduceLROnPlateau to monitor
                    'strict': True, # Whether to crash the training if `monitor` is not found
                    'name': None, # Custom name for LearningRateMonitor to use
                }

            Only the ``scheduler`` key is required, the rest will be set to the defaults above.

        Examples:
            .. code-block:: python

                # most cases
                def configure_optimizers(self):
                    opt = Adam(self.parameters(), lr=1e-3)
                    return opt

                # multiple optimizer case (e.g.: GAN)
                def configure_optimizers(self):
                    generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
                    disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
                    return generator_opt, disriminator_opt

                # example with learning rate schedulers
                def configure_optimizers(self):
                    generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
                    disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
                    discriminator_sched = CosineAnnealing(discriminator_opt, T_max=10)
                    return [generator_opt, disriminator_opt], [discriminator_sched]

                # example with step-based learning rate schedulers
                def configure_optimizers(self):
                    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                    dis_opt = Adam(self.model_disc.parameters(), lr=0.02)
                    gen_sched = {'scheduler': ExponentialLR(gen_opt, 0.99),
                                 'interval': 'step'}  # called after each training step
                    dis_sched = CosineAnnealing(discriminator_opt, T_max=10) # called every epoch
                    return [gen_opt, dis_opt], [gen_sched, dis_sched]

                # example with optimizer frequencies
                # see training procedure in `Improved Training of Wasserstein GANs`, Algorithm 1
                # https://arxiv.org/abs/1704.00028
                def configure_optimizers(self):
                    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                    dis_opt = Adam(self.model_disc.parameters(), lr=0.02)
                    n_critic = 5
                    return (
                        {'optimizer': dis_opt, 'frequency': n_critic},
                        {'optimizer': gen_opt, 'frequency': 1}
                    )

        Note:

            Some things to know:

            - Lightning calls ``.backward()`` and ``.step()`` on each optimizer
              and learning rate scheduler as needed.

            - If you use 16-bit precision (``precision=16``), Lightning will automatically
              handle the optimizers for you.

            - If you use multiple optimizers, :meth:`training_step` will have an additional
              ``optimizer_idx`` parameter.

            - If you use LBFGS Lightning handles the closure function automatically for you.

            - If you use multiple optimizers, gradients will be calculated only
              for the parameters of current optimizer at each training step.

            - If you need to control how often those optimizers step or override the
              default ``.step()`` schedule, override the :meth:`optimizer_step` hook.

            - If you only want to call a learning rate scheduler every ``x`` step or epoch,
              or want to monitor a custom metric, you can specify these in a lr_dict:

              .. code-block:: python

                  {
                      'scheduler': lr_scheduler,
                      'interval': 'step',  # or 'epoch'
                      'monitor': 'val_f1',
                      'frequency': x,
                  }

        """
        rank_zero_warn(
            "`configure_optimizers` must be implemented to be used with the Lightning Trainer"
        )

    def manual_backward(self, loss: Tensor, optimizer: Optimizer, *args, **kwargs) -> None:
        """
        Call this directly from your training_step when doing optimizations manually.
        By using this we can ensure that all the proper scaling when using 16-bit etc has been done for you

        This function forwards all args to the .backward() call as well.

        .. tip:: In manual mode we still automatically clip grads if Trainer(gradient_clip_val=x) is set

        .. tip:: In manual mode we still automatically accumulate grad over batches if
           Trainer(accumulate_grad_batches=x) is set and you use `optimizer.step()`

        Example::

            def training_step(...):
                (opt_a, opt_b) = self.optimizers()
                loss = ...
                # automatically applies scaling, etc...
                self.manual_backward(loss, opt_a)
                opt_a.step()
        """
        # make sure we're using manual opt
        self._verify_is_manual_optimization('manual_backward')

        # backward
        self._running_manual_backward = True
        self.trainer.train_loop.backward(loss, optimizer, -1, *args, **kwargs)
        self._running_manual_backward = False

    def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        """
        Override backward with your own implementation if you need to.

        Args:
            loss: Loss is already scaled by accumulated grads
            optimizer: Current optimizer being used
            optimizer_idx: Index of the current optimizer being used

        Called to perform backward step.
        Feel free to override as needed.
        The loss passed in has already been scaled for accumulated gradients if requested.

        Example::

            def backward(self, loss, optimizer, optimizer_idx):
                loss.backward()

        """
        if self.trainer.train_loop.automatic_optimization or self._running_manual_backward:
            loss.backward(*args, **kwargs)

    def toggle_optimizer(self, optimizer: Optimizer, optimizer_idx: int):
        """
        Makes sure only the gradients of the current optimizer's parameters are calculated
        in the training step to prevent dangling gradients in multiple-optimizer setup.

        .. note:: Only called when using multiple optimizers

        Override for your own behavior

        Args:
            optimizer:
            optimizer_idx:
        """
        for param in self.parameters():
            param.requires_grad = False

        for group in optimizer.param_groups:
            for param in group['params']:
                param.requires_grad = True

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        r"""
        Override this method to adjust the default way the
        :class:`~pytorch_lightning.trainer.trainer.Trainer` calls each optimizer.
        By default, Lightning calls ``step()`` and ``zero_grad()`` as shown in the example
        once per optimizer.

        .. tip:: With `Trainer(enable_pl_optimizer=True)`, you can user `optimizer.step()` directly and it will handle zero_grad, accumulated gradients, AMP, TPU and more automatically for you.

        Warning:
            If you are overriding this method, make sure that you pass the ``optimizer_closure`` parameter
            to ``optimizer.step()`` function as shown in the examples. This ensures that
            ``train_step_and_backward_closure`` is called within
            :meth:`~pytorch_lightning.trainer.training_loop.TrainLoop.run_training_batch`.

        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            optimizer_closure: closure for all optimizers
            on_tpu: true if TPU backward is required
            using_native_amp: True if using native amp
            using_lbfgs: True if the matching optimizer is lbfgs

        Examples:
            .. code-block:: python

                # DEFAULT
                def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
                    optimizer.step(closure=optimizer_closure)

                # Alternating schedule for optimizer steps (i.e.: GANs)
                def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
                    # update generator opt every 2 steps
                    if optimizer_idx == 0:
                        if batch_idx % 2 == 0 :
                            optimizer.step(closure=optimizer_closure)
                            optimizer.zero_grad()

                    # update discriminator opt every 4 steps
                    if optimizer_idx == 1:
                        if batch_idx % 4 == 0 :
                            optimizer.step(closure=optimizer_closure)
                            optimizer.zero_grad()

                    # ...
                    # add as many optimizers as you want


            Here's another example showing how to use this for more advanced things such as
            learning rate warm-up:

            .. code-block:: python

                # learning rate warm-up
                def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
                    # warm up lr
                    if self.trainer.global_step < 500:
                        lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
                        for pg in optimizer.param_groups:
                            pg['lr'] = lr_scale * self.learning_rate

                    # update params
                    optimizer.step(closure=optimizer_closure)
                    optimizer.zero_grad()

        """
        optimizer.step(closure=optimizer_closure)

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int
    ):
        optimizer.zero_grad()

    def tbptt_split_batch(self, batch: Tensor, split_size: int) -> list:
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

        Examples:
            .. code-block:: python

                def tbptt_split_batch(self, batch, split_size):
                  splits = []
                  for t in range(0, time_dims[0], split_size):
                      batch_split = []
                      for i, x in enumerate(batch):
                          if isinstance(x, torch.Tensor):
                              split_x = x[:, t:t + split_size]
                          elif isinstance(x, collections.Sequence):
                              split_x = [None] * len(x)
                              for batch_idx in range(len(x)):
                                  split_x[batch_idx] = x[batch_idx][t:t + split_size]

                          batch_split.append(split_x)

                      splits.append(batch_split)

                  return splits

        Note:
            Called in the training loop after
            :meth:`~pytorch_lightning.callbacks.base.Callback.on_batch_start`
            if :paramref:`~pytorch_lightning.trainer.Trainer.truncated_bptt_steps` > 0.
            Each returned batch split is passed separately to :meth:`training_step`.

        """
        time_dims = [
            len(x[0])
            for x in batch
            if isinstance(x, (torch.Tensor, collections.Sequence))
        ]
        assert len(time_dims) >= 1, "Unable to determine batch time dimension"
        assert all(
            x == time_dims[0] for x in time_dims
        ), "Batch time dimension length is ambiguous"

        splits = []
        for t in range(0, time_dims[0], split_size):
            batch_split = []
            for i, x in enumerate(batch):
                if isinstance(x, torch.Tensor):
                    split_x = x[:, t: t + split_size]
                elif isinstance(x, collections.Sequence):
                    split_x = [None] * len(x)
                    for batch_idx in range(len(x)):
                        split_x[batch_idx] = x[batch_idx][t: t + split_size]

                batch_split.append(split_x)

            splits.append(batch_split)

        return splits

    def summarize(self, mode: Optional[str] = ModelSummary.MODE_DEFAULT) -> Optional[ModelSummary]:
        model_summary = None

        if mode in ModelSummary.MODES:
            model_summary = ModelSummary(self, mode=mode)
            log.info("\n" + str(model_summary))
        elif mode is not None:
            raise MisconfigurationException(
                f"`mode` can be None, {', '.join(ModelSummary.MODES)}, got {mode}"
            )

        return model_summary

    def freeze(self) -> None:
        r"""
        Freeze all params for inference.

        Example:
            .. code-block:: python

                model = MyLightningModule(...)
                model.freeze()

        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        """
        Unfreeze all parameters for training.

        .. code-block:: python

            model = MyLightningModule(...)
            model.unfreeze()

        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        r"""
        Implement this to override the default items displayed in the progress bar.
        By default it includes the average loss value, split index of BPTT (if used)
        and the version of the experiment when using a logger.

        .. code-block::

            Epoch 1:   4%|▎         | 40/1095 [00:03<01:37, 10.84it/s, loss=4.501, v_num=10]

        Here is an example how to override the defaults:

        .. code-block:: python

            def get_progress_bar_dict(self):
                # don't show the version number
                items = super().get_progress_bar_dict()
                items.pop("v_num", None)
                return items

        Return:
            Dictionary with the items to be displayed in the progress bar.
        """
        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.train_loop.running_loss.mean()
        avg_training_loss = None
        if running_train_loss is not None:
            avg_training_loss = running_train_loss.cpu().item()
        elif self.trainer.train_loop.automatic_optimization:
            avg_training_loss = float('NaN')

        tqdm_dict = {}
        if avg_training_loss is not None:
            tqdm_dict["loss"] = f"{avg_training_loss:.3g}"

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict["split_idx"] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            version = self.trainer.logger.version
            # show last 4 places of long version strings
            version = version[-4:] if isinstance(version, str) else version
            tqdm_dict["v_num"] = version

        return tqdm_dict

    def _verify_is_manual_optimization(self, fn_name):
        if self.trainer.train_loop.automatic_optimization:
            raise MisconfigurationException(
                f'to use {fn_name}, please disable automatic optimization:'
                ' set model property `automatic_optimization` as False'
            )

    @classmethod
    def _auto_collect_arguments(cls, frame=None) -> Tuple[Dict, Dict]:
        """
        Collect all module arguments in the current constructor and all child constructors.
        The child constructors are all the ``__init__`` methods that reach the current class through
        (chained) ``super().__init__()`` calls.

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

    def save_hyperparameters(self, *args, frame=None) -> None:
        """Save all model arguments.

        Args:
            args: single object of `dict`, `NameSpace` or `OmegaConf`
             or string names or argumenst from class `__init__`

        >>> from collections import OrderedDict
        >>> class ManuallyArgsModel(LightningModule):
        ...     def __init__(self, arg1, arg2, arg3):
        ...         super().__init__()
        ...         # manually assign arguments
        ...         self.save_hyperparameters('arg1', 'arg3')
        ...     def forward(self, *args, **kwargs):
        ...         ...
        >>> model = ManuallyArgsModel(1, 'abc', 3.14)
        >>> model.hparams
        "arg1": 1
        "arg3": 3.14

        >>> class AutomaticArgsModel(LightningModule):
        ...     def __init__(self, arg1, arg2, arg3):
        ...         super().__init__()
        ...         # equivalent automatic
        ...         self.save_hyperparameters()
        ...     def forward(self, *args, **kwargs):
        ...         ...
        >>> model = AutomaticArgsModel(1, 'abc', 3.14)
        >>> model.hparams
        "arg1": 1
        "arg2": abc
        "arg3": 3.14

        >>> class SingleArgModel(LightningModule):
        ...     def __init__(self, params):
        ...         super().__init__()
        ...         # manually assign single argument
        ...         self.save_hyperparameters(params)
        ...     def forward(self, *args, **kwargs):
        ...         ...
        >>> model = SingleArgModel(Namespace(p1=1, p2='abc', p3=3.14))
        >>> model.hparams
        "p1": 1
        "p2": abc
        "p3": 3.14
        """
        if not frame:
            frame = inspect.currentframe().f_back
        init_args = get_init_args(frame)
        assert init_args, "failed to inspect the self init"
        if not args:
            # take all arguments
            hp = init_args
            self._hparams_name = "kwargs" if hp else None
        else:
            # take only listed arguments in `save_hparams`
            isx_non_str = [i for i, arg in enumerate(args) if not isinstance(arg, str)]
            if len(isx_non_str) == 1:
                hp = args[isx_non_str[0]]
                cand_names = [k for k, v in init_args.items() if v == hp]
                self._hparams_name = cand_names[0] if cand_names else None
            else:
                hp = {arg: init_args[arg] for arg in args if isinstance(arg, str)}
                self._hparams_name = "kwargs"

        # `hparams` are expected here
        if hp:
            self._set_hparams(hp)
        # make deep copy so  there is not other runtime changes reflected
        self._hparams_initial = copy.deepcopy(self._hparams)

    def _set_hparams(self, hp: Union[dict, Namespace, str]) -> None:
        if isinstance(hp, Namespace):
            hp = vars(hp)
        if isinstance(hp, dict):
            hp = AttributeDict(hp)
        elif isinstance(hp, PRIMITIVE_TYPES):
            raise ValueError(f"Primitives {PRIMITIVE_TYPES} are not allowed.")
        elif not isinstance(hp, ALLOWED_CONFIG_TYPES):
            raise ValueError(f"Unsupported config type of {type(hp)}.")

        if isinstance(hp, dict) and isinstance(self.hparams, dict):
            self.hparams.update(hp)
        else:
            self._hparams = hp

    @torch.no_grad()
    def to_onnx(
        self,
        file_path: Union[str, Path],
        input_sample: Optional[Any] = None,
        **kwargs,
    ):
        """
        Saves the model in ONNX format

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

        input_sample = self.transfer_batch_to_device(input_sample)

        if "example_outputs" not in kwargs:
            self.eval()
            kwargs["example_outputs"] = self(input_sample)

        torch.onnx.export(self, input_sample, file_path, **kwargs)
        self.train(mode)

    @torch.no_grad()
    def to_torchscript(
        self,
        file_path: Optional[Union[str, Path]] = None,
        method: Optional[str] = 'script',
        example_inputs: Optional[Any] = None,
        **kwargs,
    ) -> Union[ScriptModule, Dict[str, ScriptModule]]:
        """
        By default compiles the whole model to a :class:`~torch.jit.ScriptModule`.
        If you want to use tracing, please provided the argument `method='trace'` and make sure that either the
        example_inputs argument is provided, or the model has self.example_input_array set.
        If you would like to customize the modules that are scripted you should override this method.
        In case you want to return multiple modules, we recommend using a dictionary.

        Args:
            file_path: Path where to save the torchscript. Default: None (no file saved).
            method: Whether to use TorchScript's script or trace method. Default: 'script'
            example_inputs: An input to be used to do tracing when method is set to 'trace'.
              Default: None (Use self.example_input_array)
            **kwargs: Additional arguments that will be passed to the :func:`torch.jit.script` or
              :func:`torch.jit.trace` function.

        Note:
            - Requires the implementation of the
              :meth:`~pytorch_lightning.core.lightning.LightningModule.forward` method.
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
            >>> torch.jit.save(model.to_torchscript(), "model.pt")  # doctest: +SKIP
            >>> os.path.isfile("model.pt")  # doctest: +SKIP
            >>> torch.jit.save(model.to_torchscript(file_path="model_trace.pt", method='trace', # doctest: +SKIP
            ...                                     example_inputs=torch.randn(1, 64)))  # doctest: +SKIP
            >>> os.path.isfile("model_trace.pt")  # doctest: +SKIP
            True

        Return:
            This LightningModule as a torchscript, regardless of whether file_path is
            defined or not.
        """
        mode = self.training

        if method == 'script':
            torchscript_module = torch.jit.script(self.eval(), **kwargs)
        elif method == 'trace':
            # if no example inputs are provided, try to see if model has example_input_array set
            if example_inputs is None:
                if self.example_input_array is None:
                    raise ValueError(
                        'Choosing method=`trace` requires either `example_inputs`'
                        ' or `model.example_input_array` to be defined'
                    )
                example_inputs = self.example_input_array

            # automatically send example inputs to the right device and use trace
            example_inputs = self.transfer_batch_to_device(example_inputs)
            torchscript_module = torch.jit.trace(func=self.eval(), example_inputs=example_inputs, **kwargs)
        else:
            raise ValueError("The 'method' parameter only supports 'script' or 'trace',"
                             f" but value given was: {method}")

        self.train(mode)

        if file_path is not None:
            torch.jit.save(torchscript_module, file_path)

        return torchscript_module

    @property
    def hparams(self) -> Union[AttributeDict, dict, Namespace]:
        if not hasattr(self, "_hparams"):
            self._hparams = AttributeDict()
        return self._hparams

    @property
    def hparams_initial(self) -> AttributeDict:
        if not hasattr(self, "_hparams_initial"):
            return AttributeDict()
        # prevent any change
        return copy.deepcopy(self._hparams_initial)

    @hparams.setter
    def hparams(self, hp: Union[dict, Namespace, Any]):
        # TODO: remove this method in v1.3.0.
        rank_zero_warn(
            "The setter for self.hparams in LightningModule is deprecated since v1.1.0 and will be"
            " removed in v1.3.0. Replace the assignment `self.hparams = hparams` with "
            " `self.save_hyperparameters()`.",
            DeprecationWarning
        )
        hparams_assignment_name = self.__get_hparams_assignment_variable()
        self._hparams_name = hparams_assignment_name
        self._set_hparams(hp)
        # this resolves case when user does not uses `save_hyperparameters` and do hard assignement in init
        if not hasattr(self, "_hparams_initial"):
            self._hparams_initial = copy.deepcopy(self._hparams)

    def __get_hparams_assignment_variable(self):
        """
        looks at the code of the class to figure out what the user named self.hparams
        this only happens when the user explicitly sets self.hparams
        """
        try:
            class_code = inspect.getsource(self.__class__)
            lines = class_code.split("\n")
            for line in lines:
                line = re.sub(r"\s+", "", line, flags=re.UNICODE)
                if ".hparams=" in line:
                    return line.split("=")[1]
        except Exception:
            return "hparams"

        return None
