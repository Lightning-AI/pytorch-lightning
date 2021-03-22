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
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, TYPE_CHECKING, Union

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.plugins.base_plugin import Plugin

if TYPE_CHECKING:
    from pytorch_lightning.trainer.trainer import Trainer


class TrainingTypePlugin(Plugin, ABC):
    """A Plugin to change the behaviour of the training, validation and test-loop."""

    def __init__(self) -> None:
        self._model = None
        self._results = None

    def connect(self, model: 'Module') -> None:
        """Called by the accelerator to connect the accelerator and the model with this plugin"""
        self.model = model

    def setup_environment(self) -> None:
        """
        Setup any processes or distributed connections.
        This is called before the LightningModule/DataModule setup hook
        which allows the user to access the accelerator environment before setup is complete.
        """

    def setup(self, model: 'Module') -> None:
        """Called by the accelerator to finish setup."""

    @property
    @abstractmethod
    def on_gpu(self) -> bool:
        """Returns whether the current process is done on GPU"""

    @property
    @abstractmethod
    def root_device(self) -> torch.device:
        """Returns the root device"""

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
    def broadcast(self, obj: object, src: int = 0) -> object:
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
    def lightning_module(self) -> LightningModule:
        """Returns the pure LightningModule without potential wrappers"""
        return unwrap_lightning_module(self._model)

    @property
    def results(self) -> Any:
        """
        The results of the last training/testing run will be cached here.
        In distributed training, we make sure to transfer the results to the appropriate master process.
        """
        # TODO: improve these docs
        return self._results

    @property
    def rpc_enabled(self) -> bool:
        return False

    def start_training(self, trainer: 'Trainer') -> None:
        # double dispatch to initiate the training loop
        self._results = trainer.run_train()

    def start_evaluating(self, trainer: 'Trainer') -> None:
        # double dispatch to initiate the test loop
        self._results = trainer.run_evaluate()

    def start_predicting(self, trainer: 'Trainer') -> None:
        # double dispatch to initiate the predicting loop
        self._results = trainer.run_predict()

    def training_step(self, *args, **kwargs):
        return self.lightning_module.training_step(*args, **kwargs)

    def post_training_step(self):
        pass

    def validation_step(self, *args, **kwargs):
        return self.lightning_module.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.lightning_module.test_step(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.lightning_module.predict(*args, **kwargs)

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

    def init_optimizers(self, trainer: "Trainer", model: LightningModule):
        return trainer.init_optimizers(model)

    def optimizer_step(self, optimizer: torch.optim.Optimizer, lambda_closure: Callable, **kwargs):
        optimizer.step(closure=lambda_closure, **kwargs)
