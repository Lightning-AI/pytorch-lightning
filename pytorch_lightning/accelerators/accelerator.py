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
from typing import Any, Callable, Iterable, Optional, Union

import torch
from torch.optim import Optimizer

from pytorch_lightning.core import LightningModule
from pytorch_lightning.plugins.precision import (
    ApexMixedPrecisionPlugin,
    MixedPrecisionPlugin,
    NativeMixedPrecisionPlugin,
    PrecisionPlugin,
)
from pytorch_lightning.plugins.training_type import TrainingTypePlugin
from pytorch_lightning.plugins.training_type.horovod import HorovodPlugin
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.enums import AMPType, LightningEnum


class Accelerator(object):
    """
    The Accelerator Base Class.
    An Accelerator is meant to deal with one type of Hardware.

    Currently there are accelerators for:
    - CPU
    - GPU
    - TPU

    Each Accelerator gets two plugins upon initialization:
    One to handle differences from the training routine and one to handle different precisions.

    """

    def __init__(
        self,
        precision_plugin: PrecisionPlugin,
        training_type_plugin: TrainingTypePlugin,
    ) -> None:
        """

        Args:
            precision_plugin: the plugin to handle precision-specific parts
            training_type_plugin: the plugin to handle different training routines
        """
        self.precision_plugin = precision_plugin
        self.training_type_plugin = training_type_plugin

        self.optimizers = None
        self.lr_schedulers = None
        self.optimizer_frequencies = None

    def setup(self, trainer: "Trainer", model: LightningModule) -> None:
        """
        Connects the plugins to the training process, creates optimizers

        Args:
            trainer: the trainer instance to connect to
            model: the model to train
        """
        self.connect_training_type_plugin(self.training_type_plugin, model)
        self.setup_optimizers(trainer, model)
        self.connect_precision_plugin(self.precision_plugin)

    @property
    def model(self) -> torch.nn.Module:
        """Returns the model. This can also be a wrapped LightningModule.
        For retrieving the pure LightningModule use :attr:`Accelerator.lightning_module`

        """
        return self.training_type_plugin.model

    @model.setter
    def model(self, new_model: torch.nn.Module) -> None:
        self.training_type_plugin.model = new_model

    @property
    def lightning_module(self) -> LightningModule:
        """Returns the pure LightningModule.
        To get the potentially wrapped model use :attr:`Accelerator.model`

        """
        return self.training_type_plugin.lightning_module

    @property
    def root_device(self) -> torch.device:
        return self.training_type_plugin.root_device

    def teardown(self):
        """This method is called to teardown the training process.
        It is the right place to release memory and free other ressources.
        """
        pass

    def batch_to_device(self, batch: Any, device: torch.device) -> Any:
        """Moves the batch to the correct device.
        The returned batch is of the same type as the input batch, just having all tensors on the correct device.

        Args:
            batch: The batch of samples to move to the correct device
            device: The target device
        """
        model = self.lightning_module
        if model is not None:
            return model.transfer_batch_to_device(batch, device)
        return move_data_to_device(batch, device)

    def on_train_start(self):
        """Hook to do something upon the training start"""
        pass

    def training_step(self, args):
        """The actual training step.

        Args:
            args: the arguments for the models training step. Can consist of the following:
                batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                    The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
                batch_idx (int): Integer displaying index of this batch
                optimizer_idx (int): When using multiple optimizers, this argument will also be present.
                hiddens(:class:`~torch.Tensor`): Passed in if
                    :paramref:`~pytorch_lightning.trainer.trainer.Trainer.truncated_bptt_steps` > 0.

        """
        batch = self.to_device(args[0])

        args[0] = batch

        with self.precision_plugin.train_step_context():
            with self.training_type_plugin.train_step_context():
                return self.training_type_plugin.training_step(*args)

    def validation_step(self, args):
        """The actual validation step.

        Args:
            args: the arguments for the models validation step. Can consist of the following:
                batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                    The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
                batch_idx (int): The index of this batch
                dataloader_idx (int): The index of the dataloader that produced this batch
                    (only if multiple val dataloaders used)
        """
        batch = self.to_device(args[0])

        args[0] = batch

        with self.precision_plugin.val_step_context():
            with self.training_type_plugin.val_step_context():
                return self.training_type_plugin.validation_step(*args)

    def test_step(self, args):
        """The actual test step.

        Args:
            args: the arguments for the models test step. Can consist of the following:
                batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                    The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
                batch_idx (int): The index of this batch.
                dataloader_idx (int): The index of the dataloader that produced this batch
                    (only if multiple test dataloaders used).
        """
        batch = self.to_device(args[0])

        args[0] = batch

        with self.precision_plugin.test_step_context():
            with self.training_type_plugin.test_step_context():
                return self.training_type_plugin.test_step(*args)

    def training_step_end(self, output):
        """A hook to do something at the end of the training step

        Args:
            output: the output of the training step
        """
        return output

    def test_step_end(self, output):
        """A hook to do something at the end of the test step

        Args:
            output: the output of the test step
        """
        return output

    def validation_step_end(self, output):
        """A hook to do something at the end of the validation step

        Args:
            output: the output of the validation step
        """
        return output

    def process_dataloader(
        self, dataloader: Union[Iterable, torch.utils.data.DataLoader]
    ) -> Union[Iterable, torch.utils.data.DataLoader]:
        """Wraps the dataloader if necessary

        Args:
            dataloader: iterable. Ideally of type: :class:`torch.utils.data.DataLoader`
        """
        return dataloader

    def backward(
        self,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
        should_accumulate: bool,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forwards backward-calls to the precision plugin.

        Args:
            closure_loss: a tensor holding the loss value to backpropagate
            optimizer: the optimizer to do the step later on.
            opt_idx: the index of the optimizer
            should_accumulate: whether to accumulate gradients
        """
        output = self.precision_plugin.backward(
            self.lightning_module, closure_loss, optimizer, opt_idx, should_accumulate, *args, **kwargs
        )

        # TODO: this is a hack, find a better solution for this (hook?)
        if isinstance(self.training_type_plugin, HorovodPlugin):
            optimizer.synchronize()

        return output

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        current_epoch: int,
        batch_idx: int,
        opt_idx: int,
        lambda_closure: Callable,
    ):
        """performs the actual optimizer step.

        Args:
            optimizer: the optimizer performing the step
            current_epoch: current training epoch
            batch_idx: index of the current batch
            opt_idx: index of the current optimizer
            lambda_closure: closure calculating the loss value

        """
        model_ref = self.lightning_module
        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        native_amp = (
            isinstance(self.precision_plugin, MixedPrecisionPlugin) and self.precision_plugin.backend == AMPType.NATIVE
        )

        self.precision_plugin.pre_optimizer_step(optimizer, opt_idx)
        self.training_type_plugin.pre_optimizer_step(optimizer, opt_idx)

        # model hook
        res = model_ref.optimizer_step(
            epoch=current_epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=opt_idx,
            optimizer_closure=lambda_closure,
            on_tpu=False,  # TPUAccelerator class sets this as True
            using_native_amp=native_amp,
            using_lbfgs=is_lbfgs,
        )

        self.precision_plugin.post_optimizer_step(optimizer, opt_idx)
        self.training_type_plugin.post_optimizer_step(optimizer, opt_idx)
        return res

    def optimizer_zero_grad(
        self, current_epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer, opt_idx: int
    ) -> None:
        """Zeros all model parameter's gradients"""
        model_ref = self.lightning_module
        model_ref.optimizer_zero_grad(current_epoch, batch_idx, optimizer, opt_idx)

    def clip_gradients(self, optimizer: torch.optim.Optimizer, clip_val: Union[int, float]) -> None:
        """clips all the optimizer parameters to the given value"""

        self.precision_plugin.clip_gradients(optimizer, clip_val)

    def on_train_epoch_end(self, outputs) -> None:
        """Hook to do something on the end of an training epoch

        Args:
            outputs: the outputs of the training steps
        """
        pass

    def on_train_end(self) -> None:
        """Hook to do something at the end of the training"""
        pass

    def setup_optimizers(self, trainer: "Trainer", model: LightningModule):
        """creates optimizers and schedulers

        Args:
            trainer: the Trainer, these optimizers should be connected to
            model: the model to be optimized by the created optimizers
        """
        if trainer.testing is True:
            return
        optimizers, lr_schedulers, optimizer_frequencies = trainer.init_optimizers(model)
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.optimizer_frequencies = optimizer_frequencies

    def connect_training_type_plugin(self, plugin: TrainingTypePlugin, model: LightningModule) -> None:
        """Attaches the training type plugin to the accelerator.
        Also transfers ownership of the model to this plugin

        """
        plugin.connect(model)

    def connect_precision_plugin(self, plugin: PrecisionPlugin):
        """Attaches the precision plugin to the accelerator"""
        model, optimizers, schedulers = plugin.connect(self.model, self.optimizers, self.lr_schedulers)
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers

    def to_device(self, batch: Any) -> Any:
        """Pushes the batch to the root device"""
        return self.batch_to_device(batch, self.root_device)

    @property
    def amp_backend(self) -> Optional[LightningEnum]:
        if isinstance(self.precision_plugin, ApexMixedPrecisionPlugin):
            return AMPType.APEX
        elif isinstance(self.precision_plugin, NativeMixedPrecisionPlugin):
            return AMPType.NATIVE
        return None

    @property
    def precision(self) -> int:
        return self.precision_plugin.precision

    @property
    def scaler(self):
        if hasattr(self.precision_plugin, "scaler"):
            return self.precision_plugin.scaler

        return None

    @property
    def rpc_enabled(self) -> bool:
        return self.training_type_plugin.rpc_enabled

    def optimizer_state(self, optimizer: Optimizer) -> dict:
        """
        Returns state of an optimizer. Allows for syncing/collating optimizer state from processes in custom
        plugins.
        """
        if self.training_type_plugin and hasattr(self.training_type_plugin, "optimizer_state"):
            return self.training_type_plugin.optimizer_state(optimizer)
        return optimizer.state_dict()

    def on_save(self, checkpoint):
        return checkpoint
