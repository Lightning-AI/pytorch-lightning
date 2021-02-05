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
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, TYPE_CHECKING, Union

import torch

from pytorch_lightning import _logger as log
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins.base_plugin import Plugin

if TYPE_CHECKING:
    from pytorch_lightning.trainer.trainer import Trainer


class TrainingTypePlugin(Plugin, ABC):
    """A Plugin to change the behaviour of the training, validation and test-loop."""

    def __init__(self) -> None:
        self._model = None
        self._results = None
        self.global_rank = 0

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
    def reduce(self, output: Union[torch.Tensor, Any], *args: Any, **kwargs: Any) -> Union[torch.Tensor, Any]:
        """Reduces the given output (e.g. across GPUs/Processes)"""

    @abstractmethod
    def barrier(self, name: Optional[str] = None) -> None:
        """Forces all possibly joined processes to wait for each other"""

    @abstractmethod
    def broadcast(self, obj: object, src: int = 0) -> object:
        """Broadcasts an object to all processes"""

    # TODO method this is currently unused. Check after complete refactors are pushed
    def set_nvidia_flags(self, is_slurm_managing_tasks: bool, device_ids: Optional[Sequence]) -> None:
        if device_ids is None:
            return

        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join([str(x) for x in range(torch.cuda.device_count())])
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        if self.lightning_module is not None:
            log.info(f"LOCAL_RANK: {self.lightning_module.trainer.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]")

    def reduce_early_stopping_decision(self, should_stop: bool) -> bool:
        """Reduce the early stopping decision across all possibly spawned processes"""
        return should_stop

    @property
    def model(self) -> torch.nn.Module:
        """Returns the potentially wrapped LightningModule"""
        return self._model

    @model.setter
    def model(self, new_model: torch.nn.Module) -> None:
        self._model = new_model

    @property
    def lightning_module(self) -> Optional[LightningModule]:
        """Returns the pure LightningModule without potential wrappers"""
        return self._model

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
        self._results = trainer.train()

    def start_testing(self, trainer: 'Trainer') -> None:
        # double dispatch to initiate the test loop
        self._results = trainer.run_test()

    def training_step(self, *args, **kwargs):
        return self.lightning_module.training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.lightning_module.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.lightning_module.test_step(*args, **kwargs)
