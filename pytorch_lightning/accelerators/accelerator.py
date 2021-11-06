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
import contextlib
from abc import abstractmethod
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union

import torch
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision import ApexMixedPrecisionPlugin, NativeMixedPrecisionPlugin, PrecisionPlugin
from pytorch_lightning.plugins.training_type import DataParallelPlugin, TrainingTypePlugin
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from pytorch_lightning.utilities.enums import AMPType, LightningEnum
from pytorch_lightning.utilities.types import _PATH, STEP_OUTPUT


class Accelerator:
    """The Accelerator Base Class. An Accelerator is meant to deal with one type of Hardware.

    Currently there are accelerators for:

    - CPU
    - GPU
    - TPU
    - IPU

    Each Accelerator gets two plugins upon initialization:
    One to handle differences from the training routine and one to handle different precisions.
    """

    def __init__(self, precision_plugin: PrecisionPlugin, training_type_plugin: TrainingTypePlugin) -> None:
        """
        Args:
            precision_plugin: the plugin to handle precision-specific parts
            training_type_plugin: the plugin to handle different training routines
        """
        self.precision_plugin = precision_plugin
        self.training_type_plugin = training_type_plugin

        self.optimizers: List = []
        self.lr_schedulers: List = []
        self.optimizer_frequencies: List = []

    def connect(self, model: "pl.LightningModule") -> None:
        """Transfers ownership of the model to this plugin.

        See deprecation warning below.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.on_train_batch_start` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.connect` is deprecated in v1.5 and will be removed in v1.6. "
            "`connect` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        self.training_type_plugin.connect(model)

    def setup_environment(self) -> None:
        """Setup any processes or distributed connections.

        This is called before the LightningModule/DataModule setup hook which allows the user to access the accelerator
        environment before setup is complete.
        """
        self.training_type_plugin.setup_environment()

    def setup(self, trainer: "pl.Trainer") -> None:
        """Setup plugins for the trainer fit and creates optimizers.

        Args:
            trainer: the trainer instance
        """
        self.setup_training_type_plugin()
        if not self.training_type_plugin.setup_optimizers_in_pre_dispatch:
            self.setup_optimizers(trainer)
        self.setup_precision_plugin()

    def start_training(self, trainer: "pl.Trainer") -> None:
        """
        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.start_training` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.start_training` is deprecated in v1.5 and will be removed in v1.6. "
            "`start_training` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        self.training_type_plugin.start_training(trainer)

    def start_evaluating(self, trainer: "pl.Trainer") -> None:
        """
        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.start_evaluating` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.start_evaluating` is deprecated in v1.5 and will be removed in v1.6. "
            "`start_evaluating` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        self.training_type_plugin.start_evaluating(trainer)

    def start_predicting(self, trainer: "pl.Trainer") -> None:
        """
        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.start_predicting` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.start_predicting` is deprecated in v1.5 and will be removed in v1.6. "
            "`start_predicting` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        self.training_type_plugin.start_predicting(trainer)

    def pre_dispatch(self, trainer: "pl.Trainer") -> None:
        """Hook to do something before the training/evaluation/prediction starts."""
        self._move_optimizer_state()

        self.training_type_plugin.pre_dispatch()
        if self.training_type_plugin.setup_optimizers_in_pre_dispatch:
            self.setup_optimizers(trainer)

        self.precision_plugin.pre_dispatch()

    def _move_optimizer_state(self, device: Optional[torch.device] = None) -> None:
        """Moves the state of the optimizers to the GPU if needed."""
        device = device or self.root_device
        for opt in self.optimizers:
            for p, v in opt.state.items():
                opt.state[p] = apply_to_collection(v, torch.Tensor, move_data_to_device, device)

    def dispatch(self, trainer: "pl.Trainer") -> None:
        """Hook to do something before the training/evaluation/prediction starts."""
        self.training_type_plugin.dispatch(trainer)
        self.precision_plugin.dispatch(trainer)

    def post_dispatch(self, trainer: "pl.Trainer") -> None:
        """Hook to do something after the training/evaluation/prediction starts."""
        self.training_type_plugin.post_dispatch(trainer)
        self.precision_plugin.post_dispatch()

    @property
    def model(self) -> Module:
        """Returns the model.

        This can also be a wrapped LightningModule. For retrieving the pure LightningModule use
        :attr:`Accelerator.lightning_module`
        """
        return self.training_type_plugin.model

    @model.setter
    def model(self, new_model: Module) -> None:
        self.training_type_plugin.model = new_model

    @property
    def lightning_module(self) -> "pl.LightningModule":
        """Returns the pure LightningModule.

        To get the potentially wrapped model use :attr:`Accelerator.model`
        """
        return self.training_type_plugin.lightning_module

    @property
    def root_device(self) -> torch.device:
        """Returns the root device."""
        return self.training_type_plugin.root_device

    def teardown(self) -> None:
        """This method is called to teardown the training process.

        It is the right place to release memory and free other resources.
        """
        self.training_type_plugin.teardown()

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0) -> Any:
        """Moves the batch to the correct device. The returned batch is of the same type as the input batch, just
        having all tensors on the correct device.

        Args:
            batch: The batch of samples to move to the correct device
            device: The target device
            dataloader_idx: The index of the dataloader to which the batch belongs.
        """
        model = self.lightning_module
        device = device or self.root_device

        if model is not None and not isinstance(self.training_type_plugin, DataParallelPlugin):
            # no need to transfer batch to device in DP mode
            return model._apply_batch_transfer_handler(batch, device=device, dataloader_idx=dataloader_idx)

        return move_data_to_device(batch, device)

    def training_step(self, step_kwargs: Dict[str, Union[Any, int]]) -> STEP_OUTPUT:
        """The actual training step.

        See :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` for more details
        """
        with self.precision_plugin.train_step_context():
            return self.training_type_plugin.training_step(*step_kwargs.values())

    def post_training_step(self) -> None:
        """
        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.post_training_step` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.post_training_step` is deprecated in v1.5 and will be removed in v1.6. "
            "`post_training_step` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        self.training_type_plugin.post_training_step()

    def validation_step(self, step_kwargs: Dict[str, Union[Any, int]]) -> Optional[STEP_OUTPUT]:
        """The actual validation step.

        See :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step` for more details
        """
        with self.precision_plugin.val_step_context():
            return self.training_type_plugin.validation_step(*step_kwargs.values())

    def test_step(self, step_kwargs: Dict[str, Union[Any, int]]) -> Optional[STEP_OUTPUT]:
        """The actual test step.

        See :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step` for more details
        """
        with self.precision_plugin.test_step_context():
            return self.training_type_plugin.test_step(*step_kwargs.values())

    def predict_step(self, step_kwargs: Dict[str, Union[Any, int]]) -> STEP_OUTPUT:
        """The actual predict step.

        See :meth:`~pytorch_lightning.core.lightning.LightningModule.predict_step` for more details
        """
        with self.precision_plugin.predict_step_context():
            return self.training_type_plugin.predict_step(*step_kwargs.values())

    def training_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        """A hook to do something at the end of the training step.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.training_step_end` directly.

        Args:
            output: the output of the training step
        """
        rank_zero_deprecation(
            "`Accelerator.training_step_end` is deprecated in v1.5 and will be removed in v1.6. "
            "`training_step_end` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.training_step_end(output)

    def test_step_end(self, output: Optional[STEP_OUTPUT]) -> Optional[STEP_OUTPUT]:
        """A hook to do something at the end of the test step.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.test_step_end` directly.

        Args:
            output: the output of the test step
        """
        rank_zero_deprecation(
            "`Accelerator.test_step_end` is deprecated in v1.5 and will be removed in v1.6. "
            "`test_step_end` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.test_step_end(output)

    def validation_step_end(self, output: Optional[STEP_OUTPUT]) -> Optional[STEP_OUTPUT]:
        """A hook to do something at the end of the validation step.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.validation_step_end` directly.

        Args:
            output: the output of the validation step
        """
        rank_zero_deprecation(
            "`Accelerator.validation_step_end` is deprecated in v1.5 and will be removed in v1.6. "
            "`validation_step_end` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.validation_step_end(output)

    def backward(self, closure_loss: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Forwards backward-calls to the precision plugin.

        Args:
            closure_loss: a tensor holding the loss value to backpropagate
        """
        self.training_type_plugin.pre_backward(closure_loss)
        closure_loss = self.precision_plugin.pre_backward(self.lightning_module, closure_loss)

        self.precision_plugin.backward(self.lightning_module, closure_loss, *args, **kwargs)

        closure_loss = self.precision_plugin.post_backward(self.lightning_module, closure_loss)
        self.training_type_plugin.post_backward(closure_loss)

        return closure_loss

    def optimizer_step(
        self,
        optimizer: Optimizer,
        opt_idx: int,
        closure: Callable[[], Any],
        model: Optional[Union["pl.LightningModule", Module]] = None,
        **kwargs: Any
    ) -> None:
        """performs the actual optimizer step.

        Args:
            optimizer: the optimizer performing the step
            opt_idx: index of the current optimizer
            closure: closure calculating the loss value
            model: reference to the model, optionally defining optimizer step related hooks
            **kwargs: Any extra arguments to ``optimizer.step``
        """
        model = model or self.lightning_module
        self.precision_plugin.optimizer_step(model, optimizer, opt_idx, closure, **kwargs)

    def optimizer_zero_grad(self, current_epoch: int, batch_idx: int, optimizer: Optimizer, opt_idx: int) -> None:
        """Zeros all model parameter's gradients."""
        model_ref = self.lightning_module
        model_ref.optimizer_zero_grad(current_epoch, batch_idx, optimizer, opt_idx)

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        """Creates optimizers and schedulers.

        Args:
            trainer: the Trainer, these optimizers should be connected to
        """
        if trainer.state.fn not in (TrainerFn.FITTING, TrainerFn.TUNING):
            return
        optimizers, lr_schedulers, optimizer_frequencies = self.training_type_plugin.init_optimizers(
            trainer=trainer, model=self.lightning_module
        )
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.optimizer_frequencies = optimizer_frequencies

    def setup_training_type_plugin(self) -> None:
        """Attaches the training type plugin to the accelerator."""
        self.training_type_plugin.setup()

    def setup_precision_plugin(self) -> None:
        """Attaches the precision plugin to the accelerator."""
        model, optimizers, schedulers = self.precision_plugin.connect(self.model, self.optimizers, self.lr_schedulers)
        self.model = model
        self.optimizers = optimizers
        self.lr_schedulers = schedulers

    @property
    def amp_backend(self) -> Optional[LightningEnum]:
        if isinstance(self.precision_plugin, ApexMixedPrecisionPlugin):
            return AMPType.APEX
        if isinstance(self.precision_plugin, NativeMixedPrecisionPlugin):
            return AMPType.NATIVE
        return None

    @property
    def precision(self) -> Union[str, int]:
        return self.precision_plugin.precision

    @property
    def scaler(self) -> Optional["GradScaler"]:
        return getattr(self.precision_plugin, "scaler", None)

    def optimizer_state(self, optimizer: Optimizer) -> Dict[str, Tensor]:
        """Returns state of an optimizer.

        Allows for syncing/collating optimizer state from processes in custom plugins.
        """
        return getattr(self.training_type_plugin, "optimizer_state", lambda x: x.state_dict())(optimizer)

    def lightning_module_state_dict(self) -> Dict[str, Union[Any, Tensor]]:
        """Returns state of model.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.lightning_module_state_dict` directly.

        Allows for syncing/collating model state from processes in custom plugins.
        """
        rank_zero_deprecation(
            "`Accelerator.lightning_module_state_dict` is deprecated in v1.5 and will be removed in v1.6. "
            "`lightning_module_state_dict` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.lightning_module_state_dict()

    def barrier(self, name: Optional[str] = None) -> None:
        """
        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.barrier` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.barrier` is deprecated in v1.5 and will be removed in v1.6. "
            "`Barrier` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        self.training_type_plugin.barrier(name=name)

    def broadcast(self, obj: object, src: int = 0) -> object:
        """Broadcasts an object to all processes, such that the src object is broadcast to all other ranks if
        needed.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.broadcast` directly.

        Args:
            obj: Object to broadcast to all process, usually a tensor or collection of tensors.
            src: The source rank of which the object will be broadcast from
        """
        rank_zero_deprecation(
            "`Accelerator.broadcast` is deprecated in v1.5 and will be removed in v1.6. "
            "`Broadcast` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.broadcast(obj, src)

    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        """Function to gather a tensor from several distributed processes.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.all_gather` directly.

        Args:
            tensor: tensor of shape (batch, ...)
            group: the process group to gather results from. Defaults to all processes (world)
            sync_grads: flag that allows users to synchronize gradients for all_gather op

        Return:
            A tensor of shape (world_size, batch, ...)
        """
        rank_zero_deprecation(
            "`Accelerator.all_gather` is deprecated in v1.5 and will be removed in v1.6. "
            "`all_gather` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.all_gather(tensor, group=group, sync_grads=sync_grads)

    def process_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        """Wraps the dataloader if necessary.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.process_dataloader` directly.

        Args:
            dataloader: iterable. Ideally of type: :class:`torch.utils.data.DataLoader`
        """
        rank_zero_deprecation(
            "`Accelerator.process_dataloader` is deprecated in v1.5 and will be removed in v1.6. "
            "`process_dataloader` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.process_dataloader(dataloader)

    @property
    def results(self) -> Any:
        """The results of the last run will be cached within the training type plugin.

        .. deprecated:: v1.5
            This property is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.results` directly.

        In distributed training, we make sure to transfer the results to the appropriate master process.
        """
        rank_zero_deprecation(
            "`Accelerator.results` is deprecated in v1.5 and will be removed in v1.6. "
            "Accesse results directly from the `TrainingTypePlugin`."
        )
        return self.training_type_plugin.results

    @contextlib.contextmanager
    def model_sharded_context(self) -> Generator[None, None, None]:
        """Provide hook to create modules in a distributed aware context. This is useful for when we'd like to.

        shard the model instantly - useful for extremely large models. Can save memory and
        initialization time.

        Returns:
            Model parallel context.
        """
        with self.training_type_plugin.model_sharded_context():
            yield

    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: _PATH) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.save_checkpoint` directly.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
        """
        rank_zero_deprecation(
            "`Accelerator.save_checkpoint` is deprecated in v1.5 and will be removed in v1.6. "
            "`save_checkpoint` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        self.training_type_plugin.save_checkpoint(checkpoint, filepath)

    @property
    def setup_optimizers_in_pre_dispatch(self) -> bool:
        """Override to delay setting optimizers and schedulers till after dispatch. This is useful when the
        `TrainingTypePlugin` requires operating on the wrapped accelerator model. However this may break certain
        precision plugins such as APEX which require optimizers to be set.

        .. deprecated:: v1.5
            This property is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.setup_optimizers_in_pre_dispatch` directly.

        Returns:
            If True, delay setup optimizers until `pre_dispatch`, else call within `setup`.
        """
        rank_zero_deprecation(
            "`Accelerator.setup_optimizers_in_pre_dispatch` is deprecated in v1.5 and will be removed in v1.6. "
            "Accesse `setup_optimizers_in_pre_dispatch directly` from the `TrainingTypePlugin`."
        )
        return self.training_type_plugin.setup_optimizers_in_pre_dispatch

    @property
    def restore_checkpoint_after_pre_dispatch(self) -> bool:
        """Override to delay restoring from checkpoint till after pre-dispatch. This is useful when the plugin
        requires all the setup hooks to run before loading checkpoint.

        .. deprecated:: v1.5
            This property is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.restore_checkpoint_after_pre_dispatch` directly.

        Returns:
            If true, restore checkpoint after pre_dispatch.
        """
        rank_zero_deprecation(
            "`Accelerator.restore_checkpoint_after_pre_dispatch` is deprecated in v1.5 and will be removed in v1.6."
            " Access `restore_checkpoint_after_pre_dispatch` directly from the `TrainingTypePlugin`."
        )
        return self.training_type_plugin.restore_checkpoint_after_pre_dispatch

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """Gets stats for a given device.

        Args:
            device: device for which to get stats

        Returns:
            Dictionary of device stats
        """
        raise NotImplementedError

    def on_train_start(self) -> None:
        """Called when train begins."""
        return self.training_type_plugin.on_train_start()

    def on_validation_start(self) -> None:
        """Called when validation begins.

        See deprecation warning below.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.on_validation_start` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.on_validation_start` is deprecated in v1.5 and will be removed in v1.6. "
            "`on_validation_start` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.on_validation_start()

    def on_test_start(self) -> None:
        """Called when test begins.

        See deprecation warning below.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.on_test_start` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.on_test_start` is deprecated in v1.5 and will be removed in v1.6. "
            "`on_test_start` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.on_test_start()

    def on_predict_start(self) -> None:
        """Called when predict begins.

        See deprecation warning below.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.on_predict_start` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.on_predict_start` is deprecated in v1.5 and will be removed in v1.6. "
            "`on_predict_start` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.on_predict_start()

    def on_validation_end(self) -> None:
        """Called when validation ends.

        See deprecation warning below.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.on_validation_end` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.on_validation_end` is deprecated in v1.5 and will be removed in v1.6. "
            "`on_validation_end` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.on_validation_end()

    def on_test_end(self) -> None:
        """Called when test end.

        See deprecation warning below.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.on_test_end` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.on_test_end` is deprecated in v1.5 and will be removed in v1.6. "
            "`on_test_end` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.on_test_end()

    def on_predict_end(self) -> None:
        """Called when predict ends.

        See deprecation warning below.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.on_predict_end` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.on_predict_end` is deprecated in v1.5 and will be removed in v1.6. "
            "`on_predict_end` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.on_predict_end()

    def on_train_end(self) -> None:
        """Called when train ends.

        See deprecation warning below.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.on_train_end` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.on_train_end` is deprecated in v1.5 and will be removed in v1.6. "
            "`on_train_end` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.on_train_end()

    # TODO: Update this in v1.7 (deprecation: #9816)
    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called in the training loop before anything happens for that batch.

        See deprecation warning below.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.6.
            Please call `training_type_plugin.on_train_batch_start` directly.
        """
        rank_zero_deprecation(
            "`Accelerator.on_train_batch_start` is deprecated in v1.5 and will be removed in v1.6. "
            "`on_train_batch_start` logic is implemented directly in the `TrainingTypePlugin` implementations."
        )
        return self.training_type_plugin.on_train_batch_start(batch, batch_idx)

    @staticmethod
    @abstractmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
