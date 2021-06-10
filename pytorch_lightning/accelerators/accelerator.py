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
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Generator, Iterable, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision import ApexMixedPrecisionPlugin, NativeMixedPrecisionPlugin, PrecisionPlugin
from pytorch_lightning.plugins.training_type import TrainingTypePlugin
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import _NATIVE_AMP_AVAILABLE, rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from pytorch_lightning.utilities.enums import AMPType, GradClipAlgorithmType, LightningEnum
from pytorch_lightning.utilities.types import STEP_OUTPUT

if _NATIVE_AMP_AVAILABLE:
    from torch.cuda.amp import GradScaler


class Accelerator:
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

        self.optimizers: List = []
        self.lr_schedulers: List = []
        self.optimizer_frequencies: List = []

    def connect(self, model: 'pl.LightningModule') -> None:
        """Transfers ownership of the model to this plugin"""
        self.training_type_plugin.connect(model)

    def setup_environment(self) -> None:
        """
        Setup any processes or distributed connections.
        This is called before the LightningModule/DataModule setup hook
        which allows the user to access the accelerator environment before setup is complete.
        """
        self.training_type_plugin.setup_environment()

    def setup(self, trainer: 'pl.Trainer', model: 'pl.LightningModule') -> None:
        """
        Setup plugins for the trainer fit and creates optimizers.

        Args:
            trainer: the trainer instance
            model: the LightningModule
        """
        self.setup_training_type_plugin(model)
        if not self.training_type_plugin.setup_optimizers_in_pre_dispatch:
            self.setup_optimizers(trainer)
        self.setup_precision_plugin()

    def start_training(self, trainer: 'pl.Trainer') -> None:
        self.training_type_plugin.start_training(trainer)

    def start_evaluating(self, trainer: 'pl.Trainer') -> None:
        self.training_type_plugin.start_evaluating(trainer)

    def start_predicting(self, trainer: 'pl.Trainer') -> None:
        self.training_type_plugin.start_predicting(trainer)

    def pre_dispatch(self, trainer: 'pl.Trainer') -> None:
        """Hook to do something before the training/evaluation/prediction starts."""
        self._move_optimizer_state()

        self.training_type_plugin.pre_dispatch()
        if self.training_type_plugin.setup_optimizers_in_pre_dispatch:
            self.setup_optimizers(trainer)

        self.precision_plugin.pre_dispatch()

    def _move_optimizer_state(self) -> None:
        """ Moves the state of the optimizers to the GPU if needed. """
        for opt in self.optimizers:
            state: DefaultDict = defaultdict(dict)
            for p, v in opt.state.items():
                state[p] = apply_to_collection(v, torch.Tensor, move_data_to_device, self.root_device)
            opt.state = state

    def dispatch(self, trainer: 'pl.Trainer') -> None:
        """Hook to do something before the training/evaluation/prediction starts."""
        self.training_type_plugin.dispatch(trainer)
        self.precision_plugin.dispatch(trainer)

    def post_dispatch(self, trainer: 'pl.Trainer') -> None:
        """Hook to do something after the training/evaluation/prediction starts."""
        self.training_type_plugin.post_dispatch()
        self.precision_plugin.post_dispatch()

    @property
    def model(self) -> Module:
        """
        Returns the model. This can also be a wrapped LightningModule.
        For retrieving the pure LightningModule use :attr:`Accelerator.lightning_module`
        """
        return self.training_type_plugin.model

    @model.setter
    def model(self, new_model: Module) -> None:
        self.training_type_plugin.model = new_model

    @property
    def lightning_module(self) -> 'pl.LightningModule':
        """
        Returns the pure LightningModule.
        To get the potentially wrapped model use :attr:`Accelerator.model`
        """
        return self.training_type_plugin.lightning_module

    @property
    def root_device(self) -> torch.device:
        """Returns the root device"""
        return self.training_type_plugin.root_device

    def teardown(self) -> None:
        """
        This method is called to teardown the training process.
        It is the right place to release memory and free other resources.
        """
        self.training_type_plugin.teardown()

    def batch_to_device(
        self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: Optional[int] = None
    ) -> Any:
        """Moves the batch to the correct device.
        The returned batch is of the same type as the input batch, just having all tensors on the correct device.

        Args:
            batch: The batch of samples to move to the correct device
            device: The target device
            dataloader_idx: The index of the dataloader to which the batch belongs.
        """
        model = self.lightning_module

        if model is not None:
            return model._apply_batch_transfer_handler(batch, device, dataloader_idx)

        return move_data_to_device(batch, device)

    def training_step(
        self,
        step_kwargs: Dict[str, Union[Any, int]],
    ) -> STEP_OUTPUT:
        """The actual training step.

        Args:
            step_kwargs: the arguments for the models training step. Can consist of the following:

                - batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                  The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
                - batch_idx (int): Integer displaying index of this batch
                - optimizer_idx (int): When using multiple optimizers, this argument will also be present.
                - hiddens(:class:`~torch.Tensor`): Passed in if
                  :paramref:`~pytorch_lightning.core.lightning.LightningModule.truncated_bptt_steps` > 0.
        """
        step_kwargs = self.to_device(step_kwargs)

        with self.precision_plugin.train_step_context(), self.training_type_plugin.train_step_context():
            return self.training_type_plugin.training_step(*step_kwargs.values())

    def post_training_step(self) -> None:
        self.training_type_plugin.post_training_step()

    def validation_step(self, step_kwargs: Dict[str, Union[Any, int]]) -> Optional[STEP_OUTPUT]:
        """The actual validation step.

        Args:
            step_kwargs: the arguments for the models validation step. Can consist of the following:

                - batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                  The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
                - batch_idx (int): The index of this batch
                - dataloader_idx (int): The index of the dataloader that produced this batch
                  (only if multiple val dataloaders used)
        """
        step_kwargs = self.to_device(step_kwargs)

        with self.precision_plugin.val_step_context(), self.training_type_plugin.val_step_context():
            return self.training_type_plugin.validation_step(*step_kwargs.values())

    def test_step(self, step_kwargs: Dict[str, Union[Any, int]]) -> Optional[STEP_OUTPUT]:
        """The actual test step.

        Args:
            step_kwargs: the arguments for the models test step. Can consist of the following:

                - batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                  The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
                - batch_idx (int): The index of this batch.
                - dataloader_idx (int): The index of the dataloader that produced this batch
                  (only if multiple test dataloaders used).
        """
        step_kwargs = self.to_device(step_kwargs)

        with self.precision_plugin.test_step_context(), self.training_type_plugin.test_step_context():
            return self.training_type_plugin.test_step(*step_kwargs.values())

    def predict_step(self, step_kwargs: Dict[str, Union[Any, int]]) -> STEP_OUTPUT:
        """The actual predict step.

        Args:
            step_kwargs: the arguments for the models predict step. Can consist of the following:

                - batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                  The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
                - batch_idx (int): The index of this batch.
                - dataloader_idx (int): The index of the dataloader that produced this batch
                  (only if multiple predict dataloaders used).
        """
        step_kwargs = self.to_device(step_kwargs)

        with self.precision_plugin.predict_step_context(), self.training_type_plugin.predict_step_context():
            return self.training_type_plugin.predict_step(*step_kwargs.values())

    def training_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        """A hook to do something at the end of the training step

        Args:
            output: the output of the training step
        """
        return self.training_type_plugin.training_step_end(output)

    def test_step_end(self, output: Optional[STEP_OUTPUT]) -> Optional[STEP_OUTPUT]:
        """A hook to do something at the end of the test step

        Args:
            output: the output of the test step
        """
        return self.training_type_plugin.test_step_end(output)

    def validation_step_end(self, output: Optional[STEP_OUTPUT]) -> Optional[STEP_OUTPUT]:
        """A hook to do something at the end of the validation step

        Args:
            output: the output of the validation step
        """
        return self.training_type_plugin.validation_step_end(output)

    def backward(
        self,
        closure_loss: Tensor,
        optimizer: Optimizer,
        optimizer_idx: int,
        should_accumulate: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
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

    def optimizer_step(self, optimizer: Optimizer, opt_idx: int, lambda_closure: Callable, **kwargs: Any) -> None:
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

    def run_optimizer_step(
        self, optimizer: Optimizer, optimizer_idx: int, lambda_closure: Callable, **kwargs: Any
    ) -> None:
        self.training_type_plugin.optimizer_step(optimizer, lambda_closure=lambda_closure, **kwargs)

    def optimizer_zero_grad(self, current_epoch: int, batch_idx: int, optimizer: Optimizer, opt_idx: int) -> None:
        """Zeros all model parameter's gradients"""
        model_ref = self.lightning_module
        model_ref.optimizer_zero_grad(current_epoch, batch_idx, optimizer, opt_idx)

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float],
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        """clips all the optimizer parameters to the given value"""
        self.precision_plugin.clip_gradients(
            optimizer,
            clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            model=self.model,
        )

    def setup_optimizers(self, trainer: 'pl.Trainer') -> None:
        """
        Creates optimizers and schedulers

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

    def setup_training_type_plugin(self, model: 'pl.LightningModule') -> None:
        """Attaches the training type plugin to the accelerator."""
        self.training_type_plugin.setup(model)

    def setup_precision_plugin(self) -> None:
        """Attaches the precision plugin to the accelerator"""
        model, optimizers, schedulers = self.precision_plugin.connect(self.model, self.optimizers, self.lr_schedulers)
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers

    def to_device(self, step_kwargs: Dict[str, Union[Any, int]]) -> Dict[str, Union[Any, int]]:
        """Pushes the batch to the root device"""
        step_kwargs['batch'] = self.batch_to_device(
            step_kwargs['batch'], self.root_device, dataloader_idx=step_kwargs.get('dataloader_idx', None)
        )
        return step_kwargs

    @property
    def amp_backend(self) -> Optional[LightningEnum]:
        if isinstance(self.precision_plugin, ApexMixedPrecisionPlugin):
            return AMPType.APEX
        elif isinstance(self.precision_plugin, NativeMixedPrecisionPlugin):
            return AMPType.NATIVE
        return None

    @property
    def precision(self) -> Union[str, int]:
        return self.precision_plugin.precision

    @property
    def scaler(self) -> Optional['GradScaler']:
        return getattr(self.precision_plugin, 'scaler', None)

    @property
    def rpc_enabled(self) -> bool:
        return self.training_type_plugin.rpc_enabled

    def optimizer_state(self, optimizer: Optimizer) -> Dict[str, Tensor]:
        """
        Returns state of an optimizer. Allows for syncing/collating optimizer state from processes in custom
        plugins.
        """
        return getattr(self.training_type_plugin, 'optimizer_state', lambda x: x.state_dict())(optimizer)

    def lightning_module_state_dict(self) -> Dict[str, Union[Any, Tensor]]:
        """
        Returns state of model. Allows for syncing/collating model state from processes in custom plugins.
        """
        return self.training_type_plugin.lightning_module_state_dict()

    def on_save(self, checkpoint: Dict[str, Union[Any, Tensor]]) -> Dict[str, Union[Any, Tensor]]:
        return self.training_type_plugin.on_save(checkpoint)

    def barrier(self, name: Optional[str] = None) -> None:
        self.training_type_plugin.barrier(name=name)

    def broadcast(self, obj: object, src: int = 0) -> object:
        """Broadcasts an object to all processes, such that the src object is broadcast to all other ranks if needed.

        Args:
            obj: Object to broadcast to all process, usually a tensor or collection of tensors.
            src: The source rank of which the object will be broadcast from
        """
        return self.training_type_plugin.broadcast(obj, src)

    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        """
        Function to gather a tensor from several distributed processes.

        Args:
            tensor: tensor of shape (batch, ...)
            group: the process group to gather results from. Defaults to all processes (world)
            sync_grads: flag that allows users to synchronize gradients for all_gather op

        Return:
            A tensor of shape (world_size, batch, ...)
        """
        return self.training_type_plugin.all_gather(tensor, group=group, sync_grads=sync_grads)

    def process_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        """Wraps the dataloader if necessary

        Args:
            dataloader: iterable. Ideally of type: :class:`torch.utils.data.DataLoader`
        """
        return self.training_type_plugin.process_dataloader(dataloader)

    def on_reset_train_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        """Called before resetting the train dataloader."""
        return self.training_type_plugin.on_reset_train_dataloader(dataloader)

    def on_reset_val_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        """Called before resetting the val dataloader."""
        return self.training_type_plugin.on_reset_val_dataloader(dataloader)

    def on_reset_test_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        """Called before resetting the test dataloader."""
        return self.training_type_plugin.on_reset_test_dataloader(dataloader)

    def on_reset_predict_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        """Called before resetting the predict dataloader."""
        return self.training_type_plugin.on_reset_predict_dataloader(dataloader)

    @property
    def results(self) -> Any:
        """
        The results of the last run will be cached within the training type plugin.
        In distributed training, we make sure to transfer the results to the appropriate master process.
        """
        return self.training_type_plugin.results

    @contextlib.contextmanager
    def model_sharded_context(self) -> Generator[None, None, None]:
        """
        Provide hook to create modules in a distributed aware context. This is useful for when we'd like to
        shard the model instantly - useful for extremely large models. Can save memory and
        initialization time.

        Returns:
            Model parallel context.
        """
        with self.training_type_plugin.model_sharded_context():
            yield

    # todo: remove in v1.5
    def connect_training_type_plugin(self, plugin: TrainingTypePlugin, model: 'pl.LightningModule') -> None:
        """
        Attaches the training type plugin to the accelerator.
        Also transfers ownership of the model to this plugin

        .. deprecated::v1.3
            Will be removed in v1.5.0.
        """
        rank_zero_warn(
            'Accelerator method `connect_training_type_plugin` was deprecated in v1.3.'
            ' It will be removed in v1.5.'
        )
        self.setup_training_type_plugin(model)

    # todo: remove in v1.5
    def connect_precision_plugin(self, plugin: PrecisionPlugin) -> None:
        """Attaches the precision plugin to the accelerator

        .. deprecated::v1.3
            Will be removed in v1.5.0.
        """
        rank_zero_warn(
            'Accelerator method `connect_precision_plugin` was deprecated in v1.3.'
            ' It will be removed in v1.5.'
        )
        self.setup_precision_plugin()

    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: str) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
        """
        self.training_type_plugin.save_checkpoint(checkpoint, filepath)

    @property
    def call_configure_sharded_model_hook(self) -> bool:
        """
        Allow model parallel hook to be called in suitable environments determined by the training type plugin.
        This is useful for when we want to shard the model once within fit.

        Returns:
            True if we want to call the model parallel setup hook.
        """
        return self.training_type_plugin.call_configure_sharded_model_hook

    @call_configure_sharded_model_hook.setter
    def call_configure_sharded_model_hook(self, mode: bool) -> None:
        self.training_type_plugin.call_configure_sharded_model_hook = mode

    @property
    def setup_optimizers_in_pre_dispatch(self) -> bool:
        """
        Override to delay setting optimizers and schedulers till after dispatch.
        This is useful when the `TrainingTypePlugin` requires operating on the wrapped accelerator model.
        However this may break certain precision plugins such as APEX which require optimizers to be set.

        Returns:
            If True, delay setup optimizers until `pre_dispatch`, else call within `setup`.
        """
        return self.training_type_plugin.setup_optimizers_in_pre_dispatch

    def update_global_step(self, total_batch_idx: int, current_global_step: int) -> int:
        return self.training_type_plugin.update_global_step(total_batch_idx, current_global_step)

    def on_train_epoch_end(self) -> None:
        """Hook to do something on the end of an training epoch."""
        pass

    def on_train_start(self) -> None:
        """Called when train begins."""
        return self.training_type_plugin.on_train_start()

    def on_validation_start(self) -> None:
        """Called when validation begins."""
        return self.training_type_plugin.on_validation_start()

    def on_test_start(self) -> None:
        """Called when test begins."""
        return self.training_type_plugin.on_test_start()

    def on_predict_start(self) -> None:
        """Called when predict begins."""
        return self.training_type_plugin.on_predict_start()

    def on_validation_end(self) -> None:
        """Called when validation ends."""
        return self.training_type_plugin.on_validation_end()

    def on_test_end(self) -> None:
        """Called when test end."""
        return self.training_type_plugin.on_test_end()

    def on_predict_end(self) -> None:
        """Called when predict ends."""
        return self.training_type_plugin.on_predict_end()

    def on_train_end(self) -> None:
        """Called when train ends."""
        return self.training_type_plugin.on_train_end()

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """
        Called in the training loop before anything happens for that batch.
        """
        return self.training_type_plugin.on_train_batch_start(batch, batch_idx, dataloader_idx)
