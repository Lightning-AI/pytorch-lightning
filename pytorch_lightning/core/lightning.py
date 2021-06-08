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
import logging
import numbers
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import ScriptModule, Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from pytorch_lightning.core.base_lightning import RootLightningModule
from pytorch_lightning.core.grads import GradInformation
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.core.saving import ModelIO
from pytorch_lightning.utilities import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection, convert_to_tensors
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
from pytorch_lightning.utilities.distributed import sync_ddp_if_available, tpu_distributed
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from pytorch_lightning.utilities.warnings import WarningCache

warning_cache = WarningCache()
log = logging.getLogger(__name__)


class LightningModule(
    RootLightningModule,
    DeviceDtypeModuleMixin,
    GradInformation,
    ModelIO,
    Module,
):
    # Below is for property support of JIT in PyTorch 1.7
    # since none of these are important when using JIT, we are going to ignore them.
    __jit_unused_properties__ = [
        "datamodule",
        "example_input_array",
        "hparams",
        "hparams_initial",
        "on_gpu",
        "current_epoch",
        "global_step",
        "global_rank",
        "local_rank",
        "logger",
        "model_size",
        "automatic_optimization",
        "truncated_bptt_steps",
    ] + DeviceDtypeModuleMixin.__jit_unused_properties__

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # see (https://github.com/pytorch/pytorch/blob/3e6bb5233f9ca2c5aa55d9cda22a7ee85439aa6e/
        # torch/nn/modules/module.py#L227)
        torch._C._log_api_usage_once(f"lightning.module.{self.__class__.__name__}")

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

    def lr_schedulers(self) -> Optional[Union[Any, List[Any]]]:
        if not self.trainer.lr_schedulers:
            return None

        # ignore other keys "interval", "frequency", etc.
        lr_schedulers = [s["scheduler"] for s in self.trainer.lr_schedulers]

        # single scheduler
        if len(lr_schedulers) == 1:
            return lr_schedulers[0]

        # multiple schedulers
        return lr_schedulers

    @staticmethod
    def _sync(
        value: Union[torch.Tensor, numbers.Number],
        sync_fn: Optional[Callable] = None,
        sync_dist: bool = False,
        sync_dist_op: Union[Any, str] = 'mean',
        sync_dist_group: Optional[Any] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Sync across workers when using distributed training"""
        if isinstance(value, numbers.Number):
            value = torch.tensor(value, device=device, dtype=torch.float)
        sync_fn = sync_fn or sync_ddp_if_available
        dist_available = torch.distributed.is_available() and torch.distributed.is_initialized() or tpu_distributed()
        if not sync_dist or not dist_available:
            return value
        return sync_fn(value, group=sync_dist_group, reduce_op=sync_dist_op)

    def all_gather(
        self,
        data: Union[torch.Tensor, Dict, List, Tuple],
        group: Optional[Any] = None,
        sync_grads: bool = False,
    ):
        r"""
        Allows users to call ``self.all_gather()`` from the LightningModule, thus making
        the ```all_gather``` operation accelerator agnostic.

        ```all_gather``` is a function provided by accelerators to gather a tensor from several
        distributed processes

        Args:
            tensor: int, float, tensor of shape (batch, ...), or a (possibly nested) collection thereof.
            group: the process group to gather results from. Defaults to all processes (world)
            sync_grads: flag that allows users to synchronize gradients for all_gather op

        Return:
            A tensor of shape (world_size, batch, ...), or if the input was a collection
            the output will also be a collection with tensors of this shape.
        """
        group = group if group is not None else torch.distributed.group.WORLD
        all_gather = self.trainer.accelerator.all_gather
        data = convert_to_tensors(data, device=self.device)
        return apply_to_collection(data, torch.Tensor, all_gather, group=group, sync_grads=sync_grads)

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
            batch_idx (int): Integer displaying index of this batch
            optimizer_idx (int): When using multiple optimizers, this argument will also be present.
            hiddens(:class:`~torch.Tensor`): Passed in if
                :paramref:`~pytorch_lightning.core.lightning.LightningModule.truncated_bptt_steps` > 0.

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
        rank_zero_warn("`training_step` must be implemented to be used with the Lightning Trainer")

    def training_step_end(self, *args, **kwargs) -> STEP_OUTPUT:
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
            See the :ref:`advanced/multi_gpu:Multi-GPU training` guide for more details.
        """

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
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
            val_outs = []
            for val_batch in val_data:
                out = validation_step(val_batch)
                if defined('validation_step_end'):
                    out = validation_step_end(out)
                val_outs.append(out)
            val_outs = validation_epoch_end(val_outs)


        .. code-block:: python

            # if you have one val dataloader:
            def validation_step(self, batch, batch_idx)

            # if you have multiple val dataloaders:
            def validation_step(self, batch, batch_idx, dataloader_idx)

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

    def validation_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
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
            See the :ref:`advanced/multi_gpu:Multi-GPU training` guide for more details.
        """

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
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

        If you pass in multiple test dataloaders, :meth:`test_step` will have an additional argument.

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

    def test_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
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
            See the :ref:`advanced/multi_gpu:Multi-GPU training` guide for more details.
        """

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
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

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        """
        Step function called during :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`.
        By default, it calls :meth:`~pytorch_lightning.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
            dataloader_idx: Index of the current dataloader

        Return:
            Predicted output
        """
        return self(batch)

    def configure_callbacks(self):
        """
        Configure model-specific callbacks.
        When the model gets attached, e.g., when ``.fit()`` or ``.test()`` gets called,
        the list returned here will be merged with the list of callbacks passed to the Trainer's ``callbacks`` argument.
        If a callback returned here has the same type as one or several callbacks already present in
        the Trainer's callbacks list, it will take priority and replace them.
        In addition, Lightning will make sure :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`
        callbacks run last.

        Return:
            A list of callbacks which will extend the list of callbacks in the Trainer.

        Example::

            def configure_callbacks(self):
                early_stop = EarlyStopping(monitor"val_acc", mode="max")
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
                (or multiple ``lr_dict``).
            - **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``
                key whose value is a single LR scheduler or ``lr_dict``.
            - **Tuple of dictionaries** as described above, with an optional ``"frequency"`` key.
            - **None** - Fit will run without any optimizer.

        The ``lr_dict`` is a dictionary which contains the scheduler and its associated configuration.
        The default configuration is shown below.

        .. code-block:: python

            lr_dict = {
                # REQUIRED: The scheduler instance
                'scheduler': lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                'interval': 'epoch',
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                'frequency': 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                'monitor': 'val_loss',
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                'strict': True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                'name': None,
            }

        When there are schedulers in which the ``.step()`` method is conditioned on a value, such as the
        :class:`torch.optim.lr_scheduler.ReduceLROnPlateau` scheduler, Lightning requires that the ``lr_dict``
        contains the keyword ``"monitor"`` set to the metric name that the scheduler should be conditioned on.

        .. testcode::

            # The ReduceLROnPlateau scheduler requires a monitor
            def configure_optimizers(self):
                optimizer = Adam(...)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': ReduceLROnPlateau(optimizer, ...),
                        'monitor': 'metric_to_track',
                    }
                }

            # In the case of two optimizers, only one using the ReduceLROnPlateau scheduler
            def configure_optimizers(self):
                optimizer1 = Adam(...)
                optimizer2 = SGD(...)
                scheduler1 = ReduceLROnPlateau(optimizer1, ...)
                scheduler2 = LambdaLR(optimizer2, ...)
                return (
                    {
                        'optimizer': optimizer1,
                        'lr_scheduler': {
                            'scheduler': scheduler1,
                            'monitor': 'metric_to_track',
                        }
                    },
                    {'optimizer': optimizer2, 'lr_scheduler': scheduler2}
                )

        Metrics can be made available to monitor by simply logging it using
        ``self.log('metric_to_track', metric_val)`` in your :class:`~pytorch_lightning.core.lightning.LightningModule`.

        Note:
            The ``frequency`` value specified in a dict along with the ``optimizer`` key is an int corresponding
            to the number of sequential batches optimized with the specific optimizer.
            It should be given to none or to all of the optimizers.
            There is a difference between passing multiple optimizers in a list,
            and passing multiple optimizers in dictionaries with a frequency of 1:

                - In the former case, all optimizers will operate on the given batch in each optimization step.
                - In the latter, only one optimizer will operate on the given batch at every step.

            This is different from the ``frequency`` value specified in the ``lr_dict`` mentioned above.

            .. code-block:: python

                def configure_optimizers(self):
                    optimizer_one = torch.optim.SGD(self.model.parameters(), lr=0.01)
                    optimizer_two = torch.optim.SGD(self.model.parameters(), lr=0.01)
                    return [
                        {'optimizer': optimizer_one, 'frequency': 5},
                        {'optimizer': optimizer_two, 'frequency': 10},
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

            - Lightning calls ``.backward()`` and ``.step()`` on each optimizer and learning rate scheduler as needed.
            - If you use 16-bit precision (``precision=16``), Lightning will automatically handle the optimizers.
            - If you use multiple optimizers, :meth:`training_step` will have an additional ``optimizer_idx`` parameter.
            - If you use :class:`torch.optim.LBFGS`, Lightning handles the closure function automatically for you.
            - If you use multiple optimizers, gradients will be calculated only for the parameters of current optimizer
              at each training step.
            - If you need to control how often those optimizers step or override the default ``.step()`` schedule,
              override the :meth:`optimizer_step` hook.
        """
        rank_zero_warn("`configure_optimizers` must be implemented to be used with the Lightning Trainer")

    def manual_backward(self, loss: Tensor, optimizer: Optional[Optimizer] = None, *args, **kwargs) -> None:
        """
        Call this directly from your training_step when doing optimizations manually.
        By using this we can ensure that all the proper scaling when using 16-bit etc has been done for you.

        This function forwards all args to the .backward() call as well.

        See :ref:`manual optimization<common/optimizers:Manual optimization>` for more examples.

        Example::

            def training_step(...):
                opt = self.optimizers()
                loss = ...
                opt.zero_grad()
                # automatically applies scaling, etc...
                self.manual_backward(loss)
                opt.step()
        """
        if optimizer is not None:
            rank_zero_deprecation(
                "`optimizer` argument to `manual_backward` is deprecated in v1.2 and will be removed in v1.4"
            )

        # make sure we're using manual opt
        self._verify_is_manual_optimization('manual_backward')

        # backward
        self._running_manual_backward = True
        self.trainer.train_loop.backward(loss, optimizer=None, opt_idx=None, *args, **kwargs)
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
        if self.automatic_optimization or self._running_manual_backward:
            loss.backward(*args, **kwargs)

    def toggle_optimizer(self, optimizer: Optimizer, optimizer_idx: int):
        """
        Makes sure only the gradients of the current optimizer's parameters are calculated
        in the training step to prevent dangling gradients in multiple-optimizer setup.

        .. note:: Only called when using multiple optimizers

        Override for your own behavior

        It works with ``untoggle_optimizer`` to make sure param_requires_grad_state is properly reset.

        Args:
            optimizer: Current optimizer used in training_loop
            optimizer_idx: Current optimizer idx in training_loop
        """

        # Iterate over all optimizer parameters to preserve their `requires_grad` information
        # in case these are pre-defined during `configure_optimizers`
        param_requires_grad_state = {}
        for opt in self.optimizers(use_pl_optimizer=False):
            for group in opt.param_groups:
                for param in group['params']:
                    # If a param already appear in param_requires_grad_state, continue
                    if param in param_requires_grad_state:
                        continue
                    param_requires_grad_state[param] = param.requires_grad
                    param.requires_grad = False

        # Then iterate over the current optimizer's parameters and set its `requires_grad`
        # properties accordingly
        for group in optimizer.param_groups:
            for param in group['params']:
                param.requires_grad = param_requires_grad_state[param]
        self._param_requires_grad_state = param_requires_grad_state

    def untoggle_optimizer(self, optimizer_idx: int):
        """
        .. note:: Only called when using multiple optimizers

        Override for your own behavior

        Args:
            optimizer_idx: Current optimizer idx in training_loop
        """
        for opt_idx, opt in enumerate(self.optimizers(use_pl_optimizer=False)):
            if optimizer_idx != opt_idx:
                for group in opt.param_groups:
                    for param in group['params']:
                        if param in self._param_requires_grad_state:
                            param.requires_grad = self._param_requires_grad_state[param]
        # save memory
        self._param_requires_grad_state = dict()

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

        Warning:
            If you are overriding this method, make sure that you pass the ``optimizer_closure`` parameter
            to ``optimizer.step()`` function as shown in the examples. This ensures that
            ``training_step()``, ``optimizer.zero_grad()``, ``backward()`` are called within
            :meth:`~pytorch_lightning.trainer.training_loop.TrainLoop.run_training_batch`.

        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers, this indexes into that list.
            optimizer_closure: Closure for all optimizers
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

        Examples::

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
            if :paramref:`~pytorch_lightning.core.lightning.LightningModule.truncated_bptt_steps` > 0.
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
                    split_x = x[:, t:t + split_size]
                elif isinstance(x, collections.Sequence):
                    split_x = [None] * len(x)
                    for batch_idx in range(len(x)):
                        split_x[batch_idx] = x[batch_idx][t:t + split_size]

                batch_split.append(split_x)

            splits.append(batch_split)

        return splits

    def summarize(self, mode: Optional[str] = ModelSummary.MODE_DEFAULT) -> Optional[ModelSummary]:
        model_summary = None

        if mode in ModelSummary.MODES:
            model_summary = ModelSummary(self, mode=mode)
            log.info("\n" + str(model_summary))
        elif mode is not None:
            raise MisconfigurationException(f"`mode` can be None, {', '.join(ModelSummary.MODES)}, got {mode}")

        return model_summary

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
        """
        Unfreeze all parameters for training.

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
                f'to use {fn_name}, please disable automatic optimization:'
                ' set model property `automatic_optimization` as False'
            )

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

        input_sample = self._apply_batch_transfer_handler(input_sample)

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
                        ' or `model.example_input_array` to be defined.'
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

        return torchscript_module

    @property
    def model_size(self) -> float:
        # todo: think about better way without need to dump model to drive
        tmp_name = f"{uuid.uuid4().hex}.pt"
        torch.save(self.state_dict(), tmp_name)
        size_mb = os.path.getsize(tmp_name) / 1e6
        os.remove(tmp_name)
        return size_mb
