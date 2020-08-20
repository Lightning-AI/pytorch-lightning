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

from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from pytorch_lightning.utilities import move_data_to_device, AMPType

try:
    from apex import amp
except ImportError:
    amp = None


class ModelHooks(Module):

    def setup(self, stage: str):
        """
        Called at the beginning of fit and test.
        This is a good hook when you need to build models dynamically or adjust something about them.
        This hook is called on every process when using DDP.

        Args:
            stage: either 'fit' or 'test'

        Example::

            class LitModel(...):
                def __init__(self):
                    self.l1 = None

                def prepare_data(self):
                    download_data()
                    tokenize()

                    # don't do this
                    self.something = else

                def setup(stage):
                    data = Load_data(...)
                    self.l1 = nn.Linear(28, data.num_classes)

        """

    def teardown(self, stage: str):
        """
        Called at the end of fit and test.

        Args:
            stage: either 'fit' or 'test'
        """

    def on_fit_start(self):
        """
        Called at the very beginning of fit.
        If on DDP it is called on every process
        """

    def on_fit_end(self):
        """
        Called at the very end of fit.
        If on DDP it is called on every process
        """

    def on_train_start(self) -> None:
        """
        Called at the beginning of training before sanity check.
        """
        # do something at the start of training

    def on_train_end(self) -> None:
        """
        Called at the end of training before logger experiment is closed.
        """
        # do something at the end of training

    def on_pretrain_routine_start(self) -> None:
        """
        Called at the beginning of the pretrain routine (between fit and train start).

        - fit
        - pretrain_routine start
        - pretrain_routine end
        - training_start

        """
        # do something at the start of the pretrain routine

    def on_pretrain_routine_end(self) -> None:
        """
        Called at the end of the pretrain routine (between fit and train start).

        - fit
        - pretrain_routine start
        - pretrain_routine end
        - training_start

        """
        # do something at the end of the pretrain routine

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the training loop before anything happens for that batch.

        If you return -1 here, you will skip training for the rest of the current epoch.

        Args:
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """
        # do something when the batch starts

    def on_train_batch_end(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the training loop after the batch.

        Args:
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """
        # do something when the batch ends

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the validation loop before anything happens for that batch.

        Args:
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """
        # do something when the batch starts

    def on_validation_batch_end(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the validation loop after the batch.

        Args:
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """
        # do something when the batch ends

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the test loop before anything happens for that batch.

        Args:
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """
        # do something when the batch starts

    def on_test_batch_end(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the test loop after the batch.

        Args:
            batch: The batched data as it is returned by the training DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader
        """
        # do something when the batch ends

    def on_batch_start(self, batch: Any) -> None:
        """
        Called in the training loop before anything happens for that batch.

        If you return -1 here, you will skip training for the rest of the current epoch.

        Args:
            batch: The batched data as it is returned by the training DataLoader.

        .. warning:: Deprecated in 0.9.0 will remove 1.0.0 (use `on_train_batch_start` instead)
        """
        # do something when the batch starts

    def on_batch_end(self) -> None:
        """
        Called in the training loop after the batch.

        .. warning:: Deprecated in 0.9.0 will remove 1.0.0 (use `on_train_batch_end` instead)
        """
        # do something when the batch ends

    def on_epoch_start(self) -> None:
        """
        Called in the training loop at the very beginning of the epoch.
        """
        # do something when the epoch starts

    def on_epoch_end(self) -> None:
        """
        Called in the training loop at the very end of the epoch.
        """
        # do something when the epoch ends

    def on_train_epoch_start(self) -> None:
        """
        Called in the training loop at the very beginning of the epoch.
        """
        # do something when the epoch starts

    def on_train_epoch_end(self) -> None:
        """
        Called in the training loop at the very end of the epoch.
        """
        # do something when the epoch ends

    def on_validation_epoch_start(self) -> None:
        """
        Called in the validation loop at the very beginning of the epoch.
        """
        # do something when the epoch starts

    def on_validation_epoch_end(self) -> None:
        """
        Called in the validation loop at the very end of the epoch.
        """
        # do something when the epoch ends

    def on_test_epoch_start(self) -> None:
        """
        Called in the test loop at the very beginning of the epoch.
        """
        # do something when the epoch starts

    def on_test_epoch_end(self) -> None:
        """
        Called in the test loop at the very end of the epoch.
        """
        # do something when the epoch ends

    def on_pre_performance_check(self) -> None:
        """
        Called at the very beginning of the validation loop.
        """
        # do something before validation starts

    def on_post_performance_check(self) -> None:
        """
        Called at the very end of the validation loop.
        """
        # do something before validation end

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        """
        Called after optimizer.step() and before optimizer.zero_grad().

        Called in the training loop after taking an optimizer step and before zeroing grads.
        Good place to inspect weight information with weights updated.

        This is where it is called::

            for optimizer in optimizers:
                optimizer.step()
                model.on_before_zero_grad(optimizer) # < ---- called here
                optimizer.zero_grad()

        Args:
            optimizer: The optimizer for which grads should be zeroed.
        """
        # do something with the optimizer or inspect it.

    def on_after_backward(self) -> None:
        """
        Called in the training loop after loss.backward() and before optimizers do anything.
        This is the ideal place to inspect or log gradient information.

        Example::

            def on_after_backward(self):
                # example to inspect gradient information in tensorboard
                if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
                    params = self.state_dict()
                    for k, v in params.items():
                        grads = v
                        name = k
                        self.logger.experiment.add_histogram(tag=name, values=grads,
                                                             global_step=self.trainer.global_step)

        """

    def backward(self, trainer, loss: Tensor, optimizer: Optimizer, optimizer_idx: int) -> None:
        """
        Override backward with your own implementation if you need to.

        Args:
            trainer: Pointer to the trainer
            loss: Loss is already scaled by accumulated grads
            optimizer: Current optimizer being used
            optimizer_idx: Index of the current optimizer being used

        Called to perform backward step.
        Feel free to override as needed.

        The loss passed in has already been scaled for accumulated gradients if requested.

        Example::

            def backward(self, trainer, loss, optimizer, optimizer_idx):
                loss.backward()

        """
        loss.backward()

    def amp_scale_loss(self, unscaled_loss, optimizer, optimizer_idx, amp_backend: AMPType):
        if amp_backend == AMPType.NATIVE:
            scaled_loss = self.trainer.scaler.scale(unscaled_loss)
        else:
            scaled_loss = amp.scale_loss(unscaled_loss, optimizer)

        return scaled_loss

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        """
        Override this hook if your :class:`~torch.utils.data.DataLoader` returns tensors
        wrapped in a custom data structure.

        The data types listed below (and any arbitrary nesting of them) are supported out of the box:

        - :class:`torch.Tensor` or anything that implements `.to(...)`
        - :class:`list`
        - :class:`dict`
        - :class:`tuple`
        - :class:`torchtext.data.batch.Batch`

        For anything else, you need to define how the data is moved to the target device (CPU, GPU, TPU, ...).

        Example::

            def transfer_batch_to_device(self, batch, device)
                if isinstance(batch, CustomBatch):
                    # move all tensors in your custom data structure to the device
                    batch.samples = batch.samples.to(device)
                    batch.targets = batch.targets.to(device)
                else:
                    batch = super().transfer_batch_to_device(data, device)
                return batch

        Args:
            batch: A batch of data that needs to be transferred to a new device.
            device: The target device as defined in PyTorch.

        Returns:
            A reference to the data on the new device.

        Note:
            This hook should only transfer the data and not modify it, nor should it move the data to
            any other device than the one passed in as argument (unless you know what you are doing).

        Note:
            This hook only runs on single GPU training (no data-parallel). If you need multi-GPU support
            for your custom batch objects, you need to define your custom
            :class:`~torch.nn.parallel.DistributedDataParallel` or
            :class:`~pytorch_lightning.overrides.data_parallel.LightningDistributedDataParallel` and
            override :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_ddp`.

        See Also:
            - :func:`~pytorch_lightning.utilities.apply_func.move_data_to_device`
            - :func:`~pytorch_lightning.utilities.apply_func.apply_to_collection`
        """
        return move_data_to_device(batch, device)
