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
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule
from pytorch_lightning.plugins.precision import ApexMixedPrecisionPlugin, NativeMixedPrecisionPlugin, PrecisionPlugin
from pytorch_lightning.plugins.training_type import TrainingTypePlugin
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available
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

    def setup(self, trainer, model: LightningModule) -> None:
        """
        Connects the plugins to the training process, creates optimizers

        Args:
            trainer: the trainer instance to connect to
            model: the model to train
        """
        self.connect_training_type_plugin(self.training_type_plugin, model)
        self.setup_optimizers(trainer)
        self.connect_precision_plugin(self.precision_plugin)

    def start_training(self, trainer):
        self.training_type_plugin.start_training(trainer)

    def start_testing(self, trainer):
        self.training_type_plugin.start_testing(trainer)

    def start_predicting(self, trainer):
        self.training_type_plugin.start_predicting(trainer)

    def pre_dispatch(self) -> None:
        """Hook to do something before the training/evaluation/prediction starts."""
        self.training_type_plugin.pre_dispatch()
        self.precision_plugin.pre_dispatch()

    def post_dispatch(self) -> None:
        """Hook to do something before the training/evaluation/prediction starts."""
        self.training_type_plugin.post_dispatch()
        self.precision_plugin.post_dispatch()

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

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None) -> Any:
        """Moves the batch to the correct device.
        The returned batch is of the same type as the input batch, just having all tensors on the correct device.

        Args:
            batch: The batch of samples to move to the correct device
            device: The target device
        """
        model = self.lightning_module

        if model is not None:
            return model._apply_batch_transfer_handler(batch, device)

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
        args[0] = self.to_device(args[0])

        with self.precision_plugin.train_step_context(), self.training_type_plugin.train_step_context():
            return self.training_type_plugin.training_step(*args)

    def post_training_step(self):
        self.training_type_plugin.post_training_step()

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

        with self.precision_plugin.val_step_context(), self.training_type_plugin.val_step_context():
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

        with self.precision_plugin.test_step_context(), self.training_type_plugin.test_step_context():
            return self.training_type_plugin.test_step(*args)

    def predict(self, args):
        """The actual predict step.

        Args:
            args: the arguments for the models predict step. Can consist of the following:
                batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                    The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
                batch_idx (int): The index of this batch.
                dataloader_idx (int): The index of the dataloader that produced this batch
                    (only if multiple predict dataloaders used).
        """
        batch = self.to_device(args[0])

        args[0] = batch

        with self.precision_plugin.predict_context(), self.training_type_plugin.predict_context():
            return self.training_type_plugin.predict(*args)

    def training_step_end(self, output):
        """A hook to do something at the end of the training step

        Args:
            output: the output of the training step
        """
        return self.training_type_plugin.training_step_end(output)

    def test_step_end(self, output):
        """A hook to do something at the end of the test step

        Args:
            output: the output of the test step
        """
        return self.training_type_plugin.test_step_end(output)

    def validation_step_end(self, output):
        """A hook to do something at the end of the validation step

        Args:
            output: the output of the validation step
        """
        return self.training_type_plugin.validation_step_end(output)

    def backward(
        self,
        closure_loss: torch.Tensor,
        optimizer: Optimizer,
        optimizer_idx: int,
        should_accumulate: bool,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forwards backward-calls to the precision plugin.

        Args:
            closure_loss: a tensor holding the loss value to backpropagate
            should_accumulate: whether to accumulate gradients
        """
        self.training_type_plugin.pre_backward(closure_loss, should_accumulate, optimizer, optimizer_idx)

        output = self.precision_plugin.backward(
            self.lightning_module, closure_loss, optimizer, optimizer_idx, should_accumulate, *args, **kwargs
        )

        self.training_type_plugin.post_backward(closure_loss, should_accumulate, optimizer, optimizer_idx)

        return output

    def optimizer_step(self, optimizer: Optimizer, opt_idx: int, lambda_closure: Callable, **kwargs):
        """performs the actual optimizer step.

        Args:
            optimizer: the optimizer performing the step
            opt_idx: index of the current optimizer
            lambda_closure: closure calculating the loss value

        """
        make_optimizer_step = self.precision_plugin.pre_optimizer_step(
            self.lightning_module, optimizer, opt_idx, lambda_closure, **kwargs
        )
        if make_optimizer_step:
            self.run_optimizer_step(optimizer, opt_idx, lambda_closure, **kwargs)
        self.precision_plugin.post_optimizer_step(optimizer, opt_idx)
        self.training_type_plugin.post_optimizer_step(optimizer, opt_idx, **kwargs)

    def run_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int, lambda_closure: Callable, **kwargs):
        self.training_type_plugin.optimizer_step(optimizer, lambda_closure=lambda_closure, **kwargs)

    def optimizer_zero_grad(self, current_epoch: int, batch_idx: int, optimizer: Optimizer, opt_idx: int) -> None:
        """Zeros all model parameter's gradients"""
        model_ref = self.lightning_module
        model_ref.optimizer_zero_grad(current_epoch, batch_idx, optimizer, opt_idx)

    def clip_gradients(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
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

    def setup_optimizers(self, trainer):
        """creates optimizers and schedulers

        Args:
            trainer: the Trainer, these optimizers should be connected to
            model: the model to be optimized by the created optimizers
        """
        if trainer.testing:
            return
        optimizers, lr_schedulers, optimizer_frequencies = self.training_type_plugin.init_optimizers(
            trainer=trainer, model=self.lightning_module
        )
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

    def barrier(self, name: Optional[str] = None) -> None:
        self.training_type_plugin.barrier(name=name)

    def broadcast(self, obj: object, src: int = 0) -> object:
        """Broadcasts an object to all processes, such that the src object is broadcast to all other ranks if needed.

        Args:
            obj: Object to broadcast to all process, usually a tensor or collection of tensors.
            src: The source rank of which the object will be broadcast from
        """
        return self.training_type_plugin.broadcast(obj, src)

    def all_gather(self, tensor: Union[torch.Tensor], group: Optional[Any] = None, sync_grads: bool = False):
        """
        Function to gather a tensor from several distributed processes.

        Args:
            tensor: tensor of shape (batch, ...)
            group: the process group to gather results from. Defaults to all processes (world)
            sync_grads: flag that allows users to synchronize gradients for all_gather op
        Return:
            A tensor of shape (world_size, batch, ...)
        """
        return all_gather_ddp_if_available(tensor, group=group, sync_grads=sync_grads)

    def process_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        """Wraps the dataloader if necessary

        Args:
            dataloader: iterable. Ideally of type: :class:`torch.utils.data.DataLoader`
        """
        return self.training_type_plugin.process_dataloader(dataloader)

    @property
    def results(self) -> Any:
        """
        The results of the last training/testing run will be cached within the training type plugin.
        In distributed training, we make sure to transfer the results to the appropriate master process.
        """
        return self.training_type_plugin.results
