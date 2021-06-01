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
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.plugins.base_plugin import Plugin
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT

TBroadcast = TypeVar("T")


class TrainingTypePlugin(Plugin, ABC):
    """
    Base class for all training type plugins that change the behaviour of the training, validation and test-loop.
    """

    def __init__(self) -> None:
        self._model = None
        self._results: Optional[Union[_EVALUATE_OUTPUT, _PREDICT_OUTPUT]] = None
        self._call_configure_sharded_model_hook = True

    def connect(self, model: Module) -> None:
        """Called by the accelerator to connect the accelerator and the model with this plugin"""
        self.model = model

    def setup_environment(self) -> None:
        """
        Setup any processes or distributed connections.
        This is called before the LightningModule/DataModule setup hook
        which allows the user to access the accelerator environment before setup is complete.
        """

    def setup(self, model: Module) -> None:
        """Called by the accelerator to finish setup."""

    @property
    @abstractmethod
    def on_gpu(self) -> bool:
        """Returns whether the current process is done on GPU"""
        raise NotImplementedError

    @property
    @abstractmethod
    def on_tpu(self) -> bool:
        """Returns whether the current process is done on TPU"""
        raise NotImplementedError

    @property
    @abstractmethod
    def root_device(self) -> torch.device:
        """Returns the root device"""
        raise NotImplementedError

    @abstractmethod
    def model_to_device(self) -> None:
        """Moves the model to the correct device"""

    @property
    @abstractmethod
    def is_global_zero(self) -> bool:
        """Whether the current process is the rank zero process not only on the local node, but for all nodes."""

    @abstractmethod
    def reduce(self, tensor: Union[torch.Tensor, Any], *args: Any, **kwargs: Any) -> Union[torch.Tensor, Any]:
        """
        Reduces the given tensor (e.g. across GPUs/processes).

        Args:
            tensor: the tensor to sync and reduce
            *args: plugin-specific positional arguments
            **kwargs: plugin-specific keyword arguments
        """

    @abstractmethod
    def barrier(self, name: Optional[str] = None) -> None:
        """Forces all possibly joined processes to wait for each other"""

    @abstractmethod
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        """Broadcasts an object to all processes"""

    @abstractmethod
    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        """Perform a all_gather on all processes """

    def reduce_boolean_decision(self, decision: bool) -> bool:
        """Reduce the early stopping decision across all processes"""
        return decision

    def pre_backward(self, closure_loss: torch.Tensor, should_accumulate: bool, optimizer: Optimizer, opt_idx: int):
        """Run before precision plugin executes backward"""

    def post_backward(self, closure_loss: torch.Tensor, should_accumulate: bool, optimizer: Optimizer, opt_idx: int):
        """Run after precision plugin executes backward"""

    def post_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int, **kwargs) -> None:
        """Hook to do something after each optimizer step."""

    @property
    def model(self) -> Module:
        """Returns the potentially wrapped LightningModule"""
        return self._model

    @model.setter
    def model(self, new_model: Module) -> None:
        self._model = new_model

    @property
    def lightning_module(self) -> 'pl.LightningModule':
        """Returns the pure LightningModule without potential wrappers"""
        return unwrap_lightning_module(self._model)

    @property
    def results(self) -> Optional[Union[_EVALUATE_OUTPUT, _PREDICT_OUTPUT]]:
        """
        Enables plugin-agnostic access to the result returned by the training/evaluation/prediction run. The result is
        cached instead of returned directly, because some plugins require transmitting the results from one
        multiprocessing context to another in a separate step. For example, the plugins that use the "spawn"
        start-method send the result to the master process through a
        `multiprocessing queue (shared memory) <https://pytorch.org/docs/stable/multiprocessing.html>`_.
        """
        return self._results

    @property
    def rpc_enabled(self) -> bool:
        return False

    def start_training(self, trainer: 'pl.Trainer') -> None:
        # double dispatch to initiate the training loop
        self._results = trainer.run_stage()

    def start_evaluating(self, trainer: 'pl.Trainer') -> None:
        # double dispatch to initiate the test loop
        self._results = trainer.run_stage()

    def start_predicting(self, trainer: 'pl.Trainer') -> None:
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

    def on_save(self, checkpoint: Dict[str, Union[Any, torch.Tensor]]) -> Dict[str, Union[Any, torch.Tensor]]:
        return checkpoint

    def process_dataloader(self, dataloader: Union[Iterable, DataLoader]) -> Union[Iterable, DataLoader]:
        """Wraps the dataloader if necessary

        Args:
            dataloader: iterable. Ideally of type: :class:`torch.utils.data.DataLoader`
        """
        return dataloader

    def init_optimizers(self, trainer: 'pl.Trainer', model: 'pl.LightningModule'):
        return trainer.init_optimizers(model)

    def optimizer_step(self, optimizer: torch.optim.Optimizer, lambda_closure: Callable, **kwargs):
        optimizer.step(closure=lambda_closure, **kwargs)

    @property
    def setup_optimizers_in_pre_dispatch(self) -> bool:
        """
        Override to delay setting optimizers and schedulers till after dispatch.
        This is useful when the `TrainingTypePlugin` requires operating on the wrapped accelerator model.
        However this may break certain precision plugins such as APEX which require optimizers to be set.
        Returns: If True, delay setup optimizers till pre_dispatch, else call within setup.
        """
        return False

    def restore_model_state_from_ckpt_path(
        self,
        ckpt_path: str,
        map_location: Callable = lambda storage, loc: storage,
    ) -> Tuple[Dict, bool]:
        """
        This function is used to load and restore the model state.

        Args:
            ckpt_path: Path to a checkpoint
            map_location: lambda function to map checkpoint location

        Return
            checkpoint: Return loaded checkpoint
            bool: Wether to load optimizer / lr_schedulers states from checkpoint

        """
        ckpt = pl_load(ckpt_path, map_location=map_location)
        # restore datamodule states
        if self.lightning_module.trainer.datamodule is not None:
            self.lightning_module.trainer.datamodule.on_load_checkpoint(ckpt)

        # hook: give user access to checkpoint if needed.
        self.lightning_module.on_load_checkpoint(ckpt)
        self.lightning_module.load_state_dict(ckpt['state_dict'])
        return ckpt, True

    def update_global_step(self, total_batch_idx: int, current_global_step: int) -> int:
        """
        Provide a hook to count optimizer step calls.

        Args:
            total_batch_idx: Total number of batches seen for training
            current_global_step: Current number of optimizer step calls

        Returns: New optimizer step calls
        """
        return current_global_step + 1

    def lightning_module_state_dict(self) -> Dict[str, Union[Any, Tensor]]:
        """Returns model state."""
        model = self.lightning_module
        return model.state_dict()

    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: str) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
        """
        # dump states as a checkpoint dictionary object
        checkpoint = self.on_save(checkpoint)
        if self.is_global_zero:
            try:
                # write the checkpoint dictionary on the file
                atomic_save(checkpoint, filepath)
            except AttributeError as err:
                key = pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY
                checkpoint.pop(key, None)
                rank_zero_warn(f'Warning, `{key}` dropped from checkpoint. An attribute is not picklable: {err}')
                atomic_save(checkpoint, filepath)

    @contextlib.contextmanager
    def model_sharded_context(self) -> Generator:
        """
        Provide hook to create modules in a distributed aware context. This is useful for when we'd like to
        shard the model instantly, which is useful for extremely large models which can save memory and
        initialization time.

        Returns: Model parallel context.
        """
        yield

    @property
    def call_configure_sharded_model_hook(self) -> bool:
        """
        Allow model parallel hook to be called in suitable environments determined by the training type plugin.
        This is useful for when we want to shard the model once within fit.
        Returns: True if we want to call the model parallel setup hook.
        """
        return self._call_configure_sharded_model_hook

    @call_configure_sharded_model_hook.setter
    def call_configure_sharded_model_hook(self, mode: bool) -> None:
        self._call_configure_sharded_model_hook = mode

    @abstractmethod
    def teardown(self) -> None:
        """
        This method is called to teardown the training process.
        It is the right place to release memory and free other resources.
        """
        raise NotImplementedError

    @classmethod
    def register_plugins(cls, plugin_registry):
        pass

    @property
    def should_rank_save_checkpoint(self) -> bool:
        """Returns whether the checkpoint should be saved (rank based)"""
        return self.is_global_zero
