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
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Iterable, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.plugins import TorchCheckpointIO
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.utilities.distributed import ReduceOp
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT, _PATH, _PREDICT_OUTPUT


class TrainingTypePlugin(ABC):
    """Base class for all training type plugins that change the behaviour of the training, validation and test-
    loop."""

    def __init__(self, checkpoint_io: Optional[CheckpointIO] = None) -> None:
        self._model: Optional[Module] = None
        self._results: Optional[Union[_EVALUATE_OUTPUT, _PREDICT_OUTPUT]] = None
        checkpoint_io = checkpoint_io if checkpoint_io is not None else TorchCheckpointIO()
        self._checkpoint_io = checkpoint_io

    @property
    def checkpoint_io(self) -> CheckpointIO:
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, plugin: CheckpointIO) -> None:
        self._checkpoint_io = plugin

    def connect(self, model: Module) -> None:
        """Called by the accelerator to connect the accelerator and the model with this plugin."""
        self.model = model

    def setup_environment(self) -> None:
        """Setup any processes or distributed connections.

        This is called before the LightningModule/DataModule setup hook which allows the user to access the accelerator
        environment before setup is complete.
        """

    def setup(self) -> None:
        """Called by the accelerator to finish setup."""

    def _setup_model_and_optimizers(self, model: Module, optimizers: List[Optimizer]) -> Tuple[Module, List[Optimizer]]:
        """Setup a model and multiple optimizers together.

        The returned objects are expected to be in the same order they were passed in. The default implementation will
        call :meth:`_setup_model` and :meth:`_setup_optimizer` on the inputs.
        """
        # TODO (@awaelchli): standardize this across all plugins in Lightning and Lite. Related refactor: #7324
        model = self._setup_model(model)
        optimizers = [self._setup_optimizer(optimizer) for optimizer in optimizers]
        return model, optimizers

    def _setup_model(self, model: Module) -> Module:
        """Performs setup for the model, e.g., by wrapping it by another class."""
        # TODO (@awaelchli): standardize this across all plugins in Lightning and Lite. Related refactor: #7324
        return model

    def _setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Performs setup for the optimizer, e.g., by wrapping it by another class."""
        # TODO (@awaelchli): standardize this across all plugins in Lightning and Lite. Related refactor: #7324
        return optimizer

    @property
    @abstractmethod
    def on_gpu(self) -> bool:
        """Returns whether the current process is done on GPU."""

    @property
    @abstractmethod
    def on_tpu(self) -> bool:
        """Returns whether the current process is done on TPU."""

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
        tensor: Union[torch.Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[torch.Tensor, Any]:
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
    def broadcast(self, obj: object, src: int = 0) -> object:
        """Broadcasts an object to all processes.

        Args:
            obj: the object to broadcast
            src: source rank
        """

    @abstractmethod
    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        """Perform an all_gather on all processes.

        Args:
            tensor: the tensor to all_gather
            group: the process group to gather results from
            sync_grads: flag that allows users to synchronize gradients for all_gather op
        """

    def reduce_boolean_decision(self, decision: bool) -> bool:
        """Reduce the early stopping decision across all processes."""
        return decision

    def pre_backward(self, closure_loss: torch.Tensor) -> None:
        """Run before precision plugin executes backward."""

    def post_backward(self, closure_loss: torch.Tensor) -> None:
        """Run after precision plugin executes backward."""

    @property
    def model(self) -> Optional[Module]:
        """Returns the potentially wrapped LightningModule."""
        return self._model

    @model.setter
    def model(self, new_model: Optional[Module]) -> None:
        self._model = new_model

    @property
    def lightning_module(self) -> "pl.LightningModule":
        """Returns the pure LightningModule without potential wrappers."""
        return unwrap_lightning_module(self._model)

    @property
    def results(self) -> Optional[Union[_EVALUATE_OUTPUT, _PREDICT_OUTPUT]]:
        """Enables plugin-agnostic access to the result returned by the training/evaluation/prediction run.

        The result is
        cached instead of returned directly, because some plugins require transmitting the results from one
        multiprocessing context to another in a separate step. For example, the plugins that use the "spawn"
        start-method send the result to the master process through a
        `multiprocessing queue (shared memory) <https://pytorch.org/docs/stable/multiprocessing.html>`_.
        """
        return self._results

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        torch.cuda.empty_cache()
        return self.checkpoint_io.load_checkpoint(checkpoint_path)

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        self.lightning_module.load_state_dict(checkpoint["state_dict"])

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        optimizer_states = checkpoint["optimizer_states"]
        for optimizer, opt_state in zip(self.lightning_module.trainer.accelerator.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)

    def start_training(self, trainer: "pl.Trainer") -> None:
        # double dispatch to initiate the training loop
        self._results = trainer.run_stage()

    def start_evaluating(self, trainer: "pl.Trainer") -> None:
        # double dispatch to initiate the test loop
        self._results = trainer.run_stage()

    def start_predicting(self, trainer: "pl.Trainer") -> None:
        # double dispatch to initiate the predicting loop
        self._results = trainer.run_stage()

    def training_step(self, *args, **kwargs):
        return self.model.training_step(*args, **kwargs)

    def post_training_step(self):
        pass

    def validation_step(self, *args, **kwargs):
        return self.model.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.model.test_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        return self.model.predict_step(*args, **kwargs)

    def training_step_end(self, output):
        return output

    def validation_step_end(self, output):
        return output

    def test_step_end(self, output):
        return output

    def process_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        """Wraps the dataloader if necessary.

        Args:
            dataloader: iterable. Ideally of type: :class:`torch.utils.data.DataLoader`
        """
        return dataloader

    def init_optimizers(self, trainer: "pl.Trainer", model: "pl.LightningModule"):
        return trainer.init_optimizers(model)

    @property
    def setup_optimizers_in_pre_dispatch(self) -> bool:
        """Override to delay setting optimizers and schedulers till after dispatch. This is useful when the
        `TrainingTypePlugin` requires operating on the wrapped accelerator model. However this may break certain
        precision plugins such as APEX which require optimizers to be set.

        Returns:
            If True, delay setup optimizers till pre_dispatch, else call within setup.
        """
        return False

    @property
    def restore_checkpoint_after_pre_dispatch(self) -> bool:
        """Override to delay restoring from checkpoint till after pre-dispatch. This is useful when the plugin
        requires all the setup hooks to run before loading checkpoint.

        Returns:
            If true, restore checkpoint after pre_dispatch.
        """
        return False

    @property
    def lightning_restore_optimizer_and_schedulers(self) -> bool:
        """Override to disable Lightning restoring optimizers/schedulers.

        This is useful for plugins which manage restoring optimizers/schedulers.
        """
        return True

    @property
    def handles_gradient_accumulation(self) -> bool:
        """Whether the plugin handles gradient accumulation internally."""
        return False

    def lightning_module_state_dict(self) -> Dict[str, Union[Any, Tensor]]:
        """Returns model state."""
        model = self.lightning_module
        return model.state_dict()

    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: _PATH) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
        """
        if self.should_rank_save_checkpoint:
            return self.checkpoint_io.save_checkpoint(checkpoint, filepath)

    def remove_checkpoint(self, filepath: _PATH) -> None:
        """Remove checkpoint filepath from the filesystem.

        Args:
            filepath: Path to checkpoint
        """
        if self.should_rank_save_checkpoint:
            return self.checkpoint_io.remove_checkpoint(filepath)

    @contextlib.contextmanager
    def model_sharded_context(self) -> Generator:
        """Provide hook to create modules in a distributed aware context. This is useful for when we'd like to
        shard the model instantly, which is useful for extremely large models which can save memory and
        initialization time.

        Returns: Model parallel context.
        """
        yield

    @abstractmethod
    def teardown(self) -> None:
        """This method is called to teardown the training process.

        It is the right place to release memory and free other resources.
        """

    @classmethod
    def register_plugins(cls, plugin_registry) -> None:
        pass

    @property
    def should_rank_save_checkpoint(self) -> bool:
        """Returns whether the checkpoint should be saved (rank based)"""
        return self.is_global_zero

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

    def on_predict_end(self):
        """Called when predict ends."""
        pass

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Called in the training loop before anything happens for that batch."""
        pass

    def pre_dispatch(self) -> None:
        """Hook to do something before the training/evaluation/prediction starts."""

    def dispatch(self, trainer: "pl.Trainer") -> None:
        """Hook to do something at trainer run_stage starts."""

    def post_dispatch(self, trainer: "pl.Trainer") -> None:
        """Hook to do something after the training/evaluation/prediction finishes."""
