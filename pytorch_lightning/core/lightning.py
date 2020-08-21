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

import collections
import inspect
import os
import re
import tempfile
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as torch_distrib
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning import _logger as log
from pytorch_lightning.core.grads import GradInformation
from pytorch_lightning.core.hooks import ModelHooks
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.core.saving import ALLOWED_CONFIG_TYPES, PRIMITIVE_TYPES, ModelIO
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
from pytorch_lightning.utilities.parsing import AttributeDict, collect_init_args, get_init_args
from pytorch_lightning.core.step_result import TrainResult, EvalResult

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


class LightningModule(ABC, DeviceDtypeModuleMixin, GradInformation, ModelIO, ModelHooks, Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.exp_save_path = None

        #: The current epoch
        self.current_epoch = 0

        #: Total training batches seen across all epochs
        self.global_step = 0

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

    @property
    def example_input_array(self) -> Any:
        return self._example_input_array

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
        return self.device.type == 'cuda'

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
            :class:`~pytorch_lightning.core.step_result.TrainResult`

            .. note:: :class:`~pytorch_lightning.core.step_result.TrainResult` is simply a Dict with convenient
                functions for logging, distributed sync and error checking.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.

        Example::

            def training_step(self, batch, batch_idx):
                x, y, z = batch

                # implement your own
                out = self(x)
                loss = self.loss(out, x)

                # TrainResult auto-detaches the loss after the optimization steps are complete
                result = pl.TrainResult(minimize=loss)

        The return object :class:`~pytorch_lightning.core.step_result.TrainResult` controls where to log,
        when to log (step or epoch) and syncing with multiple GPUs.

        .. code-block:: python

            # log to progress bar and logger
            result.log('train_loss', loss, prog_bar=True, logger=True)

            # sync metric value across GPUs in distributed training
            result.log('train_loss_2', loss, sync_dist=True)

            # log to progress bar as well
            result.log('train_loss_2', loss, prog_bar=True)

            # assign arbitrary values
            result.predictions = predictions
            result.some_value = 'some_value'

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

                # TrainResult auto-detaches hiddens
                result = pl.TrainResult(minimize=loss, hiddens=hiddens)
                return result

        Notes:
            The loss value shown in the progress bar is smoothed (averaged) over the last values,
            so it differs from the actual loss returned in train/validation step.
        """
        rank_zero_warn('`training_step` must be implemented to be used with the Lightning Trainer')

    def training_end(self, *args, **kwargs):
        """
        Warnings:
            Deprecated in v0.7.0. Use  :meth:`training_step_end` instead.
        """

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
            :class:`~pytorch_lightning.core.step_result.TrainResult`

            .. note:: :class:`~pytorch_lightning.core.step_result.TrainResult` is simply a Dict with convenient
                functions for logging, distributed sync and error checking.

        When using dp/ddp2 distributed backends, only a portion of the batch is inside the training_step:

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self(x)

                # softmax uses only a portion of the batch in the denomintaor
                loss = self.softmax(out)
                loss = nce_loss(loss)
                return pl.TrainResult(loss)

        If you wish to do something with all the parts of the batch, then use this method to do it:

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self(x)
                result = pl.TrainResult()
                result.out = out

            def training_step_end(self, training_step_outputs):
                # this out is now the full size of the batch
                all_outs = training_step_outputs.out

                # this softmax now uses the full batch
                loss = nce_loss(all_outs)
                result = pl.TrainResult(loss)
                return result

        See Also:
            See the :ref:`multi-gpu-training` guide for more details.
        """

    def training_epoch_end(
            self, outputs: Union[TrainResult, List[TrainResult]]
    ):
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
            :class:`~pytorch_lightning.core.step_result.TrainResult`

        .. note:: :class:`~pytorch_lightning.core.step_result.TrainResult` is simply a Dict with convenient
            functions for logging, distributed sync and error checking.

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

            def training_epoch_end(self, outputs):
                epoch_result = pl.TrainResult()
                for train_result in outputs:
                    all_losses = train_result.minimize
                    # do something with all losses
                return results
        """

    def validation_step(self, *args, **kwargs) -> EvalResult:
        r"""
        Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.

        .. code-block:: python

            # the pseudocode for these calls
            val_outs = []
            for val_batch in val_data:
                out = validation_step(train_batch)
                val_outs.append(out)
                validation_epoch_end(val_outs)

        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (int): The index of this batch
            dataloader_idx (int): The index of the dataloader that produced this batch
                (only if multiple val datasets used)

        Return:
            :class:`~pytorch_lightning.core.step_result.TrainResult`

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
                    result = pl.EvalResult(checkpoint_on=loss)
                    result.log_dict({'val_loss': loss, 'val_acc': val_acc})
                    return result

            If you pass in multiple val datasets, validation_step will have an additional argument.

            .. code-block:: python

                # CASE 2: multiple validation datasets
                def validation_step(self, batch, batch_idx, dataloader_idx):
                    # dataloader_idx tells you which dataset this is.

        Note:
            If you don't need to validate you don't need to implement this method.

        Note:
            When the :meth:`validation_step` is called, the model has been put in eval mode
            and PyTorch gradients have been disabled. At the end of validation,
            the model goes back to training mode and gradients are enabled.
        """

    def validation_step_end(self, *args, **kwargs) -> EvalResult:
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
            :class:`~pytorch_lightning.core.step_result.TrainResult`

        .. code-block:: python

            # WITHOUT validation_step_end
            # if used in DP or DDP2, this batch is 1/num_gpus large
            def validation_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self(x)
                loss = self.softmax(out)
                loss = nce_loss(loss)
                result = pl.EvalResult()
                result.log('val_loss', loss)
                return result

            # --------------
            # with validation_step_end to do softmax over the full batch
            def validation_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self(x)
                result = pl.EvalResult()
                result.out = out
                return result

            def validation_epoch_end(self, output_results):
                # this out is now the full size of the batch
                all_val_step_outs = output_results.out
                loss = nce_loss(all_val_step_outs)

                result = pl.EvalResult(checkpoint_on=loss)
                result.log('val_loss', loss)
                return result

        See Also:
            See the :ref:`multi-gpu-training` guide for more details.
        """

    def validation_end(self, outputs):
        """
        Warnings:
            Deprecated in v0.7.0. Use :meth:`validation_epoch_end` instead.
            Will be removed in 1.0.0.
        """

    def validation_epoch_end(
            self, outputs: Union[EvalResult, List[EvalResult]]
    ) -> EvalResult:
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
            :class:`~pytorch_lightning.core.step_result.TrainResult`

        Note:
            If you didn't define a :meth:`validation_step`, this won't be called.

        - The outputs here are strictly for logging or progress bar.
        - If you don't need to display anything, don't return anything.

        Examples:
            With a single dataloader:

            .. code-block:: python

                def validation_epoch_end(self, val_step_outputs):
                    # do something with the outputs of all val batches
                    all_val_preds = val_step_outputs.predictions

                    val_step_outputs.some_result = calc_all_results(all_val_preds)
                    return val_step_outputs

            With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
            one entry per dataloader, while the inner list contains the individual outputs of
            each validation step for that dataloader.

            .. code-block:: python

                def validation_epoch_end(self, outputs):
                    for dataloader_output_result in outputs:
                        dataloader_outs = dataloader_output_result.dataloader_i_outputs

                    result = pl.EvalResult()
                    result.log('final_metric', final_value)
                    return result
        """

    def test_step(self, *args, **kwargs) -> EvalResult:
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
                (only if multiple test datasets used).

        Return:
            :class:`~pytorch_lightning.core.step_result.TrainResult`

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
                    result = pl.EvalResult(checkpoint_on=loss)
                    result.log_dict({'test_loss': loss, 'test_acc': test_acc})
                    return resultt

            If you pass in multiple validation datasets, :meth:`test_step` will have an additional
            argument.

            .. code-block:: python

                # CASE 2: multiple test datasets
                def test_step(self, batch, batch_idx, dataloader_idx):
                    # dataloader_idx tells you which dataset this is.

        Note:
            If you don't need to validate you don't need to implement this method.

        Note:
            When the :meth:`test_step` is called, the model has been put in eval mode and
            PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
            to training mode and gradients are enabled.
        """

    def test_step_end(self, *args, **kwargs) -> EvalResult:
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
            :class:`~pytorch_lightning.core.step_result.TrainResult`

        .. code-block:: python

            # WITHOUT test_step_end
            # if used in DP or DDP2, this batch is 1/num_gpus large
            def test_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self(x)
                loss = self.softmax(out)
                loss = nce_loss(loss)
                result = pl.EvalResult()
                result.log('test_loss', loss)
                return result

            # --------------
            # with test_step_end to do softmax over the full batch
            def test_step(self, batch, batch_idx):
                # batch is 1/num_gpus big
                x, y = batch

                out = self(x)
                result = pl.EvalResult()
                result.out = out
                return result

            def test_epoch_end(self, output_results):
                # this out is now the full size of the batch
                all_test_step_outs = output_results.out
                loss = nce_loss(all_test_step_outs)

                result = pl.EvalResult(checkpoint_on=loss)
                result.log('test_loss', loss)
                return result

        See Also:
            See the :ref:`multi-gpu-training` guide for more details.
        """

    def test_end(self, outputs):
        """
        Warnings:
             Deprecated in v0.7.0. Use :meth:`test_epoch_end` instead.
             Will be removed in 1.0.0.
        """

    def test_epoch_end(
            self, outputs: Union[EvalResult, List[EvalResult]]
    ) -> EvalResult:

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
            :class:`~pytorch_lightning.core.step_result.TrainResult`

        Note:
            If you didn't define a :meth:`test_step`, this won't be called.

        - The outputs here are strictly for logging or progress bar.
        - If you don't need to display anything, don't return anything.

        Examples:
            With a single dataloader:

            .. code-block:: python

                def test_epoch_end(self, outputs):
                    # do something with the outputs of all test batches
                    all_test_preds = test_step_outputs.predictions

                    test_step_outputs.some_result = calc_all_results(all_test_preds)
                    return test_step_outputs

            With multiple dataloaders, `outputs` will be a list of lists. The outer list contains
            one entry per dataloader, while the inner list contains the individual outputs of
            each test step for that dataloader.

            .. code-block:: python

                def test_epoch_end(self, outputs):
                    for dataloader_output_result in outputs:
                        dataloader_outs = dataloader_output_result.dataloader_i_outputs

                    result = pl.EvalResult()
                    result.log('final_metric', final_value)
                    return results
        """

    def configure_ddp(self, model: 'LightningModule', device_ids: List[int]) -> DistributedDataParallel:
        r"""
        Override to init DDP in your own way or with your own wrapper.
        The only requirements are that:

        1. On a validation batch, the call goes to ``model.validation_step``.
        2. On a training batch, the call goes to ``model.training_step``.
        3. On a testing batch, the call goes to ``model.test_step``.

        Args:
            model: the :class:`LightningModule` currently being optimized.
            device_ids: the list of GPU ids.

        Return:
            DDP wrapped model

        Examples:
            .. code-block:: python

                # default implementation used in Trainer
                def configure_ddp(self, model, device_ids):
                    # Lightning DDP simply routes to test_step, val_step, etc...
                    model = LightningDistributedDataParallel(
                        model,
                        device_ids=device_ids,
                        find_unused_parameters=True
                    )
                    return model

        """
        model = LightningDistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=True)
        return model

    def _init_slurm_connection(self) -> None:
        """"""
        """
        Sets up environment variables necessary for pytorch distributed communications
        based on slurm environment.
        """
        # use slurm job id for the port number
        # guarantees unique ports across jobs from same grid search
        try:
            # use the last 4 numbers in the job id as the id
            default_port = os.environ['SLURM_JOB_ID']
            default_port = default_port[-4:]

            # all ports should be in the 10k+ range
            default_port = int(default_port) + 15000

        except Exception:
            default_port = 12910

        # if user gave a port number, use that one instead
        try:
            default_port = os.environ['MASTER_PORT']
        except Exception:
            os.environ['MASTER_PORT'] = str(default_port)

        # figure out the root node addr
        try:
            root_node = os.environ['SLURM_NODELIST'].split(' ')[0]
        except Exception:
            root_node = '127.0.0.1'

        root_node = self.trainer.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node

    def init_ddp_connection(self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True) -> None:
        """
        Override to define your custom way of setting up a distributed environment.

        Lightning's implementation uses env:// init by default and sets the first node as root
        for SLURM managed cluster.

        Args:
            global_rank: The global process idx.
            world_size: Number of GPUs being use across all nodes. (num_nodes * num_gpus).
            is_slurm_managing_tasks: is cluster managed by SLURM.

        """
        if is_slurm_managing_tasks:
            self._init_slurm_connection()

        if 'MASTER_ADDR' not in os.environ:
            rank_zero_warn("MASTER_ADDR environment variable is not defined. Set as localhost")
            os.environ['MASTER_ADDR'] = '127.0.0.1'
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")

        if 'MASTER_PORT' not in os.environ:
            rank_zero_warn("MASTER_PORT environment variable is not defined. Set as 12910")
            os.environ['MASTER_PORT'] = '12910'
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) != world_size:
            rank_zero_warn(
                f"WORLD_SIZE environment variable ({os.environ['WORLD_SIZE']}) "
                f"is not equal to the computed world size ({world_size}). Ignored."
            )

        torch_backend = "nccl" if self.trainer.on_gpu else "gloo"
        log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank+1}/{world_size}")
        torch_distrib.init_process_group(torch_backend, rank=global_rank, world_size=world_size)

    def configure_sync_batchnorm(self, model: 'LightningModule') -> 'LightningModule':
        """
        Add global batchnorm for a model spread across multiple GPUs and nodes.

        Override to synchronize batchnorm between specific process groups instead
        of the whole world or use a different sync_bn like `apex`'s version.

        Args:
            model: pointer to current :class:`LightningModule`.

        Return:
            LightningModule with batchnorm layers synchronized between process groups
        """
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=None)

        return model

    def configure_apex(
        self, amp: object, model: 'LightningModule', optimizers: List[Optimizer], amp_level: str
    ) -> Tuple['LightningModule', List[Optimizer]]:
        r"""
        Override to init AMP your own way.
        Must return a model and list of optimizers.

        Args:
            amp: pointer to amp library object.
            model: pointer to current :class:`LightningModule`.
            optimizers: list of optimizers passed in :meth:`configure_optimizers`.
            amp_level: AMP mode chosen ('O1', 'O2', etc...)

        Return:
            Apex wrapped model and optimizers

        Examples:
            .. code-block:: python

                # Default implementation used by Trainer.
                def configure_apex(self, amp, model, optimizers, amp_level):
                    model, optimizers = amp.initialize(
                        model, optimizers, opt_level=amp_level,
                    )

                    return model, optimizers
        """
        model, optimizers = amp.initialize(model, optimizers, opt_level=amp_level)

        return model, optimizers

    def configure_optimizers(
        self,
    ) -> Optional[Union[Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]]]:
        r"""
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler' key which value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.

        Note:
            The 'frequency' value is an int corresponding to the number of sequential batches
            optimized with the specific optimizer. It should be given to none or to all of the optimizers.
            There is a difference between passing multiple optimizers in a list,
            and passing multiple optimizers in dictionaries with a frequency of 1:
            In the former case, all optimizers will operate on the given batch in each optimization step.
            In the latter, only one optimizer will operate on the given batch at every step.

            The lr_dict is a dictionary which contains scheduler and its associated configuration.
            It has five keys. The default configuration is shown below.

            .. code-block:: python

                {
                    'scheduler': lr_scheduler, # The LR schduler
                    'interval': 'epoch', # The unit of the scheduler's step size
                    'frequency': 1, # The frequency of the scheduler
                    'reduce_on_plateau': False, # For ReduceLROnPlateau scheduler
                    'monitor': 'val_loss' # Metric to monitor
                }

            If user only provides LR schedulers, then their configuration will set to default as shown above.

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
        rank_zero_warn('`configure_optimizers` must be implemented to be used with the Lightning Trainer')

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        second_order_closure: Optional[Callable] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        r"""
        Override this method to adjust the default way the
        :class:`~pytorch_lightning.trainer.trainer.Trainer` calls each optimizer.
        By default, Lightning calls ``step()`` and ``zero_grad()`` as shown in the example
        once per optimizer.

        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            second_order_closure: closure for second order methods
            on_tpu: true if TPU backward is required
            using_native_amp: True if using native amp
            using_lbfgs: True if the matching optimizer is lbfgs

        Examples:
            .. code-block:: python

                # DEFAULT
                def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                                   second_order_closure, on_tpu, using_native_amp, using_lbfgs):
                    optimizer.step()

                # Alternating schedule for optimizer steps (i.e.: GANs)
                def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                                   second_order_closure, on_tpu, using_native_amp, using_lbfgs):
                    # update generator opt every 2 steps
                    if optimizer_idx == 0:
                        if batch_idx % 2 == 0 :
                            optimizer.step()
                            optimizer.zero_grad()

                    # update discriminator opt every 4 steps
                    if optimizer_idx == 1:
                        if batch_idx % 4 == 0 :
                            optimizer.step()
                            optimizer.zero_grad()

                    # ...
                    # add as many optimizers as you want


            Here's another example showing how to use this for more advanced things such as
            learning rate warm-up:

            .. code-block:: python

                # learning rate warm-up
                def optimizer_step(self, current_epoch, batch_idx, optimizer,
                                    optimizer_idx, second_order_closure, on_tpu, using_native_amp, using_lbfgs):
                    # warm up lr
                    if self.trainer.global_step < 500:
                        lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
                        for pg in optimizer.param_groups:
                            pg['lr'] = lr_scale * self.learning_rate

                    # update params
                    optimizer.step()
                    optimizer.zero_grad()

        Note:
            If you also override the :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_before_zero_grad`
            model hook don't forget to add the call to it before ``optimizer.zero_grad()`` yourself.

        """
        if on_tpu:
            xm.optimizer_step(optimizer)
        elif using_native_amp:
            self.trainer.scaler.step(optimizer)
        elif using_lbfgs:
            optimizer.step(second_order_closure)
        else:
            optimizer.step()

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
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
        time_dims = [len(x[0]) for x in batch if isinstance(x, (torch.Tensor, collections.Sequence))]
        assert len(time_dims) >= 1, "Unable to determine batch time dimension"
        assert all(x == time_dims[0] for x in time_dims), "Batch time dimension length is ambiguous"

        splits = []
        for t in range(0, time_dims[0], split_size):
            batch_split = []
            for i, x in enumerate(batch):
                if isinstance(x, torch.Tensor):
                    split_x = x[:, t : t + split_size]
                elif isinstance(x, collections.Sequence):
                    split_x = [None] * len(x)
                    for batch_idx in range(len(x)):
                        split_x[batch_idx] = x[batch_idx][t : t + split_size]

                batch_split.append(split_x)

            splits.append(batch_split)

        return splits

    def prepare_data(self) -> None:
        """
        Use this to download and prepare data.

        .. warning:: DO NOT set state to the model (use `setup` instead)
            since this is NOT called on every GPU in DDP/TPU

        Example::

            def prepare_data(self):
                # good
                download_data()
                tokenize()
                etc()

                # bad
                self.split = data_split
                self.some_state = some_other_state()

        In DDP prepare_data can be called in two ways (using Trainer(prepare_data_per_node)):

        1. Once per node. This is the default and is only called on LOCAL_RANK=0.
        2. Once in total. Only called on GLOBAL_RANK=0.

        Example::

            # DEFAULT
            # called once per node on LOCAL_RANK=0 of that node
            Trainer(prepare_data_per_node=True)

            # call on GLOBAL_RANK=0 (great for shared file systems)
            Trainer(prepare_data_per_node=False)

        This is called before requesting the dataloaders:

        .. code-block:: python

            model.prepare_data()
                if ddp/tpu: init()
            model.setup(stage)
            model.train_dataloader()
            model.val_dataloader()
            model.test_dataloader()
        """

    def train_dataloader(self) -> DataLoader:
        """
        Implement a PyTorch DataLoader for training.

        Return:
            Single PyTorch :class:`~torch.utils.data.DataLoader`.

        The dataloader you return will not be called every epoch unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to ``True``.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data

        - :meth:`~pytorch_lightning.trainer.Trainer.fit`
        - ...
        - :meth:`prepare_data`
        - :meth:`setup`
        - :meth:`train_dataloader`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Example:
            .. code-block:: python

                def train_dataloader(self):
                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (1.0,))])
                    dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform,
                                    download=True)
                    loader = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=self.batch_size,
                        shuffle=True
                    )
                    return loader

        """
        rank_zero_warn('`train_dataloader` must be implemented to be used with the Lightning Trainer')

    def tng_dataloader(self):  # todo: remove in v1.0.0
        """
        Warnings:
            Deprecated in v0.5.0. Use :meth:`train_dataloader` instead. Will be removed in 1.0.0.
        """
        output = self.train_dataloader()
        rank_zero_warn(
            "`tng_dataloader` has been renamed to `train_dataloader` since v0.5.0."
            " and this method will be removed in v1.0.0",
            DeprecationWarning,
        )
        return output

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        r"""
        Implement one or multiple PyTorch DataLoaders for testing.

        The dataloader you return will not be called every epoch unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to ``True``.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data


        - :meth:`~pytorch_lightning.trainer.Trainer.fit`
        - ...
        - :meth:`prepare_data`
        - :meth:`setup`
        - :meth:`train_dataloader`
        - :meth:`val_dataloader`
        - :meth:`test_dataloader`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Return:
            Single or multiple PyTorch DataLoaders.

        Example:
            .. code-block:: python

                def test_dataloader(self):
                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (1.0,))])
                    dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform,
                                    download=True)
                    loader = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=self.batch_size,
                        shuffle=False
                    )

                    return loader

                # can also return multiple dataloaders
                def test_dataloader(self):
                    return [loader_a, loader_b, ..., loader_n]

        Note:
            If you don't need a test dataset and a :meth:`test_step`, you don't need to implement
            this method.

        Note:
            In the case where you return multiple test dataloaders, the :meth:`test_step`
            will have an argument ``dataloader_idx`` which matches the order here.
        """

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        r"""
        Implement one or multiple PyTorch DataLoaders for validation.

        The dataloader you return will not be called every epoch unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_epoch` to ``True``.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~pytorch_lightning.trainer.Trainer.fit`
        - ...
        - :meth:`prepare_data`
        - :meth:`train_dataloader`
        - :meth:`val_dataloader`
        - :meth:`test_dataloader`

        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Return:
            Single or multiple PyTorch DataLoaders.

        Examples:
            .. code-block:: python

                def val_dataloader(self):
                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (1.0,))])
                    dataset = MNIST(root='/path/to/mnist/', train=False,
                                    transform=transform, download=True)
                    loader = torch.utils.data.DataLoader(
                        dataset=dataset,
                        batch_size=self.batch_size,
                        shuffle=False
                    )

                    return loader

                # can also return multiple dataloaders
                def val_dataloader(self):
                    return [loader_a, loader_b, ..., loader_n]

        Note:
            If you don't need a validation dataset and a :meth:`validation_step`, you don't need to
            implement this method.

        Note:
            In the case where you return multiple validation dataloaders, the :meth:`validation_step`
            will have an argument ``dataloader_idx`` which matches the order here.
        """

    def summarize(self, mode: str = ModelSummary.MODE_DEFAULT) -> ModelSummary:
        model_summary = ModelSummary(self, mode=mode)
        log.info('\n' + str(model_summary))
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

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        r"""
        Called by Lightning to restore your model.
        If you saved something with :meth:`on_save_checkpoint` this is your chance to restore this.

        Args:
            checkpoint: Loaded checkpoint


        Example:
            .. code-block:: python

                def on_load_checkpoint(self, checkpoint):
                    # 99% of the time you don't need to implement this method
                    self.something_cool_i_want_to_save = checkpoint['something_cool_i_want_to_save']

        Note:
            Lightning auto-restores global step, epoch, and train state including amp scaling.
            There is no need for you to restore anything regarding training.
        """

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        r"""
        Called by Lightning when saving a checkpoint to give you a chance to store anything
        else you might want to save.

        Args:
            checkpoint: Checkpoint to be saved

        Example:
            .. code-block:: python

                def on_save_checkpoint(self, checkpoint):
                    # 99% of use cases you don't need to implement this method
                    checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object

        Note:
            Lightning saves all aspects of training (epoch, global step, etc...)
            including amp scaling.
            There is no need for you to store anything about training.

        """

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
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        tqdm_dict = {'loss': '{:.3f}'.format(avg_training_loss)}

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict['split_idx'] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            version = self.trainer.logger.version
            # show last 4 places of long version strings
            version = version[-4:] if isinstance(version, str) else version
            tqdm_dict['v_num'] = version

        return tqdm_dict

    def get_tqdm_dict(self) -> Dict[str, Union[int, str]]:
        """
        Additional items to be displayed in the progress bar.

        Return:
            Dictionary with the items to be displayed in the progress bar.

        Warning:
            Deprecated since v0.7.3.
            Use :meth:`get_progress_bar_dict` instead.
        """
        rank_zero_warn(
            "`get_tqdm_dict` was renamed to `get_progress_bar_dict` in v0.7.3"
            " and this method will be removed in v1.0.0",
            DeprecationWarning,
        )
        return self.get_progress_bar_dict()

    @classmethod
    def _auto_collect_arguments(cls, frame=None) -> Tuple[Dict, Dict]:
        """"""
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

        # set module_arguments in child
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
        assert init_args, 'failed to inspect the self init'
        if not args:
            hp = init_args
            self._hparams_name = 'kwargs' if hp else None
        else:
            isx_non_str = [i for i, arg in enumerate(args) if not isinstance(arg, str)]
            if len(isx_non_str) == 1:
                hp = args[isx_non_str[0]]
                cand_names = [k for k, v in init_args.items() if v == hp]
                self._hparams_name = cand_names[0] if cand_names else None
            else:
                hp = {arg: init_args[arg] for arg in args if isinstance(arg, str)}
                self._hparams_name = 'kwargs'

        # `hparams` are expected here
        if hp:
            self._set_hparams(hp)

    def _set_hparams(self, hp: Union[dict, Namespace, str]) -> None:
        if isinstance(hp, Namespace):
            hp = vars(hp)
        if isinstance(hp, dict):
            hp = AttributeDict(hp)
        elif isinstance(hp, PRIMITIVE_TYPES):
            raise ValueError(f'Primitives {PRIMITIVE_TYPES} are not allowed.')
        elif not isinstance(hp, ALLOWED_CONFIG_TYPES):
            raise ValueError(f'Unsupported config type of {type(hp)}.')

        if isinstance(hp, dict) and isinstance(self.hparams, dict):
            self.hparams.update(hp)
        else:
            self._hparams = hp

    def to_onnx(self, file_path: str, input_sample: Optional[Tensor] = None, **kwargs):
        """Saves the model in ONNX format

        Args:
            file_path: The path of the file the model should be saved to.
            input_sample: A sample of an input tensor for tracing.
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

        if isinstance(input_sample, Tensor):
            input_data = input_sample
        elif self.example_input_array is not None:
            input_data = self.example_input_array
        else:
            raise ValueError('`input_sample` and `example_input_array` tensors are both missing.')

        if 'example_outputs' not in kwargs:
            self.eval()
            kwargs['example_outputs'] = self(input_data)

        torch.onnx.export(self, input_data, file_path, **kwargs)

    @property
    def hparams(self) -> Union[AttributeDict, str]:
        if not hasattr(self, '_hparams'):
            self._hparams = AttributeDict()
        return self._hparams

    @hparams.setter
    def hparams(self, hp: Union[dict, Namespace, Any]):
        hparams_assignment_name = self.__get_hparams_assignment_variable()
        self._hparams_name = hparams_assignment_name
        self._set_hparams(hp)

    def __get_hparams_assignment_variable(self):
        """"""
        """
        looks at the code of the class to figure out what the user named self.hparams
        this only happens when the user explicitly sets self.hparams
        """
        try:
            class_code = inspect.getsource(self.__class__)
            lines = class_code.split('\n')
            for line in lines:
                line = re.sub(r"\s+", "", line, flags=re.UNICODE)
                if '.hparams=' in line:
                    return line.split('=')[1]
        except Exception as e:
            return 'hparams'

        return None
