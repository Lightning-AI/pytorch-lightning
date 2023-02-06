# Copyright The Lightning AI team.
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
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.strategies.launchers.base import _Launcher
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.distributed import ReduceOp
from lightning_fabric.utilities.optimizer import _optimizer_to_device, _optimizers_to_device
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.core.optimizer import _init_optimizers_and_lr_schedulers, LightningOptimizer
from pytorch_lightning.plugins import TorchCheckpointIO
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import (
    LRSchedulerConfig,
    PredictStep,
    STEP_OUTPUT,
    TestStep,
    TrainingStep,
    ValidationStep,
)

TBroadcast = TypeVar("TBroadcast")
TReduce = TypeVar("TReduce")

log = logging.getLogger(__name__)


class Strategy(ABC):
    """Base class for all strategies that change the behaviour of the training, validation and test- loop."""

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ) -> None:
        self._accelerator: Optional["pl.accelerators.Accelerator"] = accelerator
        self._checkpoint_io: Optional[CheckpointIO] = checkpoint_io
        self._precision_plugin: Optional[PrecisionPlugin] = precision_plugin
        self._lightning_module: Optional[pl.LightningModule] = None
        self._model: Optional[Module] = None
        self._launcher: Optional[_Launcher] = None
        self._optimizers: List[Optimizer] = []
        self._lightning_optimizers: Dict[int, LightningOptimizer] = {}
        self.lr_scheduler_configs: List[LRSchedulerConfig] = []
        self.optimizer_frequencies: List[int] = []

    @property
    def launcher(self) -> Optional[_Launcher]:
        return self._launcher

    @property
    def accelerator(self) -> Optional["pl.accelerators.Accelerator"]:
        return self._accelerator

    @accelerator.setter
    def accelerator(self, accelerator: "pl.accelerators.Accelerator") -> None:
        self._accelerator = accelerator

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = TorchCheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = TorchCheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    @property
    def precision_plugin(self) -> PrecisionPlugin:
        return self._precision_plugin if self._precision_plugin is not None else PrecisionPlugin()

    @precision_plugin.setter
    def precision_plugin(self, precision_plugin: Optional[PrecisionPlugin]) -> None:
        self._precision_plugin = precision_plugin

    @property
    def optimizers(self) -> List[Optimizer]:
        return self._optimizers

    @optimizers.setter
    def optimizers(self, optimizers: List[Optimizer]) -> None:
        self._optimizers = optimizers
        self._lightning_optimizers = {
            idx: LightningOptimizer._to_lightning_optimizer(opt, self, idx) for idx, opt in enumerate(self.optimizers)
        }

    def connect(self, model: "pl.LightningModule") -> None:
        """Called by the accelerator to connect the accelerator and the model with this plugin."""
        self._lightning_module = model
        self.model = model

    def _configure_launcher(self) -> None:
        """Attach the launcher based on Strategy."""

    def setup_environment(self) -> None:
        """Setup any processes or distributed connections.

        This is called before the LightningModule/DataModule setup hook which allows the user to access the accelerator
        environment before setup is complete.
        """
        assert self.accelerator is not None
        self.accelerator.setup_device(self.root_device)

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        """Creates optimizers and schedulers.

        Args:
            trainer: the Trainer, these optimizers should be connected to
        """
        if trainer.state.fn != TrainerFn.FITTING:
            return
        assert self.lightning_module is not None
        self.optimizers, self.lr_scheduler_configs, self.optimizer_frequencies = _init_optimizers_and_lr_schedulers(
            self.lightning_module
        )

    def setup(self, trainer: "pl.Trainer") -> None:
        """Setup plugins for the trainer fit and creates optimizers.

        Args:
            trainer: the trainer instance
        """
        assert self.accelerator is not None
        self.accelerator.setup(trainer)
        self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        _optimizers_to_device(self.optimizers, self.root_device)

    def setup_precision_plugin(self) -> None:
        """Attaches the precision plugin to the accelerator."""
        assert self.model is not None
        model, optimizers, lr_scheduler_configs = self.precision_plugin.connect(
            self.model, self.optimizers, self.lr_scheduler_configs
        )
        self.model = model
        self.optimizers = optimizers
        self.lr_scheduler_configs = lr_scheduler_configs

    def optimizer_state(self, optimizer: Optimizer) -> Dict[str, Tensor]:
        """Returns state of an optimizer.

        Allows for syncing/collating optimizer state from processes in custom plugins.
        """
        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer

        if hasattr(optimizer, "consolidate_state_dict"):
            # there are optimizers like PyTorch's ZeroRedundancyOptimizer that shard their
            # states, and to avoid OOM we consolidate the full state on rank 0 only
            optimizer.consolidate_state_dict()
            return optimizer.state_dict() if self.is_global_zero else {}

        # for optimizers that are not sharded, we return the state dict on all ranks
        return optimizer.state_dict()

    def backward(
        self,
        closure_loss: Tensor,
        optimizer: Optional[Optimizer],
        optimizer_idx: Optional[int],
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        r"""Forwards backward-calls to the precision plugin.

        Args:
            closure_loss: a tensor holding the loss value to backpropagate
            optimizer: An optional optimizer that gets passed down to the precision plugin's backward
            optimizer_idx: An optional optimizer index that gets passed down to the precision plugin's backward
            \*args: Positional arguments that get passed down to the precision plugin's backward, intended as arguments
                for the actual function that performs the backward, like :meth:`~torch.Tensor.backward`.
            \**kwargs: Keyword arguments for the same purpose as ``*args``.
        """
        self.pre_backward(closure_loss)
        assert self.lightning_module is not None
        closure_loss = self.precision_plugin.pre_backward(closure_loss, self.lightning_module)

        self.precision_plugin.backward(closure_loss, self.lightning_module, optimizer, optimizer_idx, *args, **kwargs)

        closure_loss = self.precision_plugin.post_backward(closure_loss, self.lightning_module)
        self.post_backward(closure_loss)

        return closure_loss

    def optimizer_step(
        self,
        optimizer: Optimizer,
        opt_idx: int,
        closure: Callable[[], Any],
        model: Optional[Union["pl.LightningModule", Module]] = None,
        **kwargs: Any,
    ) -> Any:
        r"""Performs the actual optimizer step.

        Args:
            optimizer: the optimizer performing the step
            opt_idx: index of the current optimizer
            closure: closure calculating the loss value
            model: reference to the model, optionally defining optimizer step related hooks
            \**kwargs: Keyword arguments to to ``optimizer.step``
        """
        model = model or self.lightning_module
        # TODO(fabric): remove assertion once strategy's optimizer_step typing is fixed
        assert isinstance(model, pl.LightningModule)
        return self.precision_plugin.optimizer_step(
            optimizer, model=model, optimizer_idx=opt_idx, closure=closure, **kwargs
        )

    def _setup_model_and_optimizers(self, model: Module, optimizers: List[Optimizer]) -> Tuple[Module, List[Optimizer]]:
        """Setup a model and multiple optimizers together.

        The returned objects are expected to be in the same order they were passed in. The default implementation will
        call :meth:`_setup_model` and :meth:`_setup_optimizer` on the inputs.
        """
        # TODO: standardize this across all plugins in Lightning and Fabric. Related refactor: #7324
        model = self._setup_model(model)
        optimizers = [self._setup_optimizer(optimizer) for optimizer in optimizers]
        return model, optimizers

    def _setup_model(self, model: Module) -> Module:
        """Performs setup for the model, e.g., by wrapping it by another class."""
        # TODO: standardize this across all plugins in Lightning and Fabric. Related refactor: #7324
        return model

    def _setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Performs setup for the optimizer, e.g., by wrapping it by another class."""
        # TODO: standardize this across all plugins in Lightning and Fabric. Related refactor: #7324
        return optimizer

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0) -> Any:
        """Moves the batch to the correct device.

        The returned batch is of the same type as the input batch, just
        having all tensors on the correct device.

        Args:
            batch: The batch of samples to move to the correct device
            device: The target device
            dataloader_idx: The index of the dataloader to which the batch belongs.
        """
        model = self.lightning_module
        device = device or self.root_device
        if model is not None:
            return model._apply_batch_transfer_handler(batch, device=device, dataloader_idx=dataloader_idx)
        return move_data_to_device(batch, device)

    @property
    @abstractmethod
    def root_device(self) -> torch.device:
        """Returns the root device."""

    @abstractmethod
    def model_to_device(self) -> None:
        """Moves the model to the correct device."""

    @property
    @abstractmethod
    def is_global_zero(self) -> bool:
        """Whether the current process is the rank zero process not only on the local node, but for all nodes."""

    @abstractmethod
    def reduce(
        self,
        tensor: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[Tensor, Any]:
        """Reduces the given tensor (e.g. across GPUs/processes).

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to reduce
            reduce_op: the reduction operation. Defaults to 'mean'.
                Can also be a string 'sum' or ReduceOp.
        """

    @abstractmethod
    def barrier(self, name: Optional[str] = None) -> None:
        """Synchronizes all processes which blocks processes until the whole group enters this function.

        Args:
            name: an optional name to pass into barrier.
        """

    @abstractmethod
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        """Broadcasts an object to all processes.

        Args:
            obj: the object to broadcast
            src: source rank
        """

    @abstractmethod
    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        """Perform an all_gather on all processes.

        Args:
            tensor: the tensor to all_gather
            group: the process group to gather results from
            sync_grads: flag that allows users to synchronize gradients for all_gather op
        """

    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
        """Reduce a boolean decision across all processes."""
        return decision

    def pre_backward(self, closure_loss: Tensor) -> None:
        """Run before precision plugin executes backward."""

    def post_backward(self, closure_loss: Tensor) -> None:
        """Run after precision plugin executes backward."""

    @property
    def model(self) -> Optional[Module]:
        """Returns the potentially wrapped LightningModule."""
        return self._model if self._model is not None else self._lightning_module

    @model.setter
    def model(self, new_model: Optional[Module]) -> None:
        self._model = new_model

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        """Returns the pure LightningModule without potential wrappers."""
        return self._lightning_module

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        torch.cuda.empty_cache()
        return self.checkpoint_io.load_checkpoint(checkpoint_path)

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"])

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        optimizer_states = checkpoint["optimizer_states"]
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)
            _optimizer_to_device(optimizer, self.root_device)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """The actual training step.

        See :meth:`~pytorch_lightning.core.module.LightningModule.training_step` for more details
        """
        with self.precision_plugin.train_step_context():
            assert isinstance(self.model, TrainingStep)
            return self.model.training_step(*args, **kwargs)

    def post_training_step(self) -> None:
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """The actual validation step.

        See :meth:`~pytorch_lightning.core.module.LightningModule.validation_step` for more details
        """
        with self.precision_plugin.val_step_context():
            assert isinstance(self.model, ValidationStep)
            return self.model.validation_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """The actual test step.

        See :meth:`~pytorch_lightning.core.module.LightningModule.test_step` for more details
        """
        with self.precision_plugin.test_step_context():
            assert isinstance(self.model, TestStep)
            return self.model.test_step(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """The actual predict step.

        See :meth:`~pytorch_lightning.core.module.LightningModule.predict_step` for more details
        """
        with self.precision_plugin.predict_step_context():
            assert isinstance(self.model, PredictStep)
            return self.model.predict_step(*args, **kwargs)

    def training_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        return output

    def validation_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        return output

    def test_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        return output

    def process_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """Wraps the dataloader if necessary.

        Args:
            dataloader: iterable. Ideally of type: :class:`torch.utils.data.DataLoader`
        """
        return dataloader

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        """Override to delay restoring from checkpoint till after pre-dispatch. This is useful when the plugin
        requires all the setup hooks to run before loading checkpoint.

        Returns:
            If true, restore checkpoint after pre_dispatch.
        """
        return False

    @property
    def lightning_restore_optimizer(self) -> bool:
        """Override to disable Lightning restoring optimizers/schedulers.

        This is useful for plugins which manage restoring optimizers/schedulers.
        """
        return True

    @property
    def handles_gradient_accumulation(self) -> bool:
        """Whether the plugin handles gradient accumulation internally."""
        return False

    def lightning_module_state_dict(self) -> Dict[str, Any]:
        """Returns model state."""
        assert self.lightning_module is not None
        return self.lightning_module.state_dict()

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
            storage_options: parameter for how to save to storage, passed to ``CheckpointIO`` plugin
        """
        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    def remove_checkpoint(self, filepath: _PATH) -> None:
        """Remove checkpoint filepath from the filesystem.

        Args:
            filepath: Path to checkpoint
        """
        if self.is_global_zero:
            self.checkpoint_io.remove_checkpoint(filepath)

    @contextlib.contextmanager
    def model_sharded_context(self) -> Generator:
        """Provide hook to create modules in a distributed aware context. This is useful for when we'd like to
        shard the model instantly, which is useful for extremely large models which can save memory and
        initialization time.

        Returns: Model parallel context.
        """
        yield

    def teardown(self) -> None:
        """This method is called to teardown the training process.

        It is the right place to release memory and free other resources.
        """
        _optimizers_to_device(self.optimizers, torch.device("cpu"))

        if self.lightning_module is not None:
            log.detail(f"{self.__class__.__name__}: moving model to CPU")
            self.lightning_module.cpu()
        self.precision_plugin.teardown()
        assert self.accelerator is not None
        self.accelerator.teardown()
        self.checkpoint_io.teardown()

    @classmethod
    def register_strategies(cls, strategy_registry: Dict[str, Any]) -> None:
        pass

    def on_train_start(self) -> None:
        """Called when train begins."""
        pass

    def on_validation_start(self) -> None:
        """Called when validation begins."""
        pass

    def on_test_start(self) -> None:
        """Called when test begins."""
        pass

    def on_predict_start(self) -> None:
        """Called when predict begins."""
        pass

    def on_train_end(self) -> None:
        """Called when train ends."""
        pass

    def on_validation_end(self) -> None:
        """Called when validation ends."""
        pass

    def on_test_end(self) -> None:
        """Called when test end."""
        pass

    def on_predict_end(self) -> None:
        """Called when predict ends."""
        pass

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Called in the training loop before anything happens for that batch."""
        pass

    def dispatch(self, trainer: "pl.Trainer") -> None:
        """Hook to do something before the training/evaluation/prediction starts."""
        self.precision_plugin.dispatch(trainer)

    def __getstate__(self) -> Dict:
        # `LightningOptimizer` overrides `self.__class__` so they cannot be pickled
        state = dict(vars(self))  # copy
        state["_lightning_optimizers"] = {}
        return state

    def __setstate__(self, state: Dict) -> None:
        self.__dict__ = state
        self.optimizers = self.optimizers  # re-create the `_lightning_optimizers`
