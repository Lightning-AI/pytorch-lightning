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
from typing import Any, Dict, List, Optional, Union

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class PLStrategyInterface(ABC):
    """Base class for all strategies that change the behaviour of the training, validation and test- loop."""

    @property
    @abstractmethod
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        """Returns the pure LightningModule without potential wrappers."""

    @property
    @abstractmethod
    def model(self) -> Optional[Module]:
        """Returns the potentially wrapped LightningModule."""

    @model.setter
    @abstractmethod
    def model(self, new_model: Optional[Module]) -> None:
        pass

    @property
    @abstractmethod
    def optimizers(self) -> List[Optimizer]:
        pass

    @optimizers.setter
    @abstractmethod
    def optimizers(self, optimizers: List[Optimizer]) -> None:
        pass

    @property
    @abstractmethod
    def restore_checkpoint_after_setup(self) -> bool:
        """Override to delay restoring from checkpoint till after pre-dispatch. This is useful when the plugin
        requires all the setup hooks to run before loading checkpoint.

        Returns:
            If true, restore checkpoint after pre_dispatch.
        """

    @property
    @abstractmethod
    def lightning_restore_optimizer(self) -> bool:
        """Override to disable Lightning restoring optimizers/schedulers.

        This is useful for plugins which manage restoring optimizers/schedulers.
        """

    @property
    @abstractmethod
    def handles_gradient_accumulation(self) -> bool:
        """Whether the plugin handles gradient accumulation internally."""

    @abstractmethod
    def connect(self, model: "pl.LightningModule") -> None:
        """Called by the accelerator to connect the accelerator and the model with this plugin."""

    @abstractmethod
    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        """Creates optimizers and schedulers.

        Args:
            trainer: the Trainer, these optimizers should be connected to
        """

    @abstractmethod
    def setup(self, trainer: "pl.Trainer") -> None:
        """Setup plugins for the trainer fit and creates optimizers.

        Args:
            trainer: the trainer instance
        """

    @abstractmethod
    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """The actual training step.

        See :meth:`~pytorch_lightning.core.module.LightningModule.training_step` for more details
        """

    @abstractmethod
    def post_training_step(self) -> None:
        pass

    @abstractmethod
    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """The actual validation step.

        See :meth:`~pytorch_lightning.core.module.LightningModule.validation_step` for more details
        """

    @abstractmethod
    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """The actual test step.

        See :meth:`~pytorch_lightning.core.module.LightningModule.test_step` for more details
        """

    @abstractmethod
    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """The actual predict step.

        See :meth:`~pytorch_lightning.core.module.LightningModule.predict_step` for more details
        """

    @abstractmethod
    def training_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        pass

    @abstractmethod
    def validation_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        pass

    @abstractmethod
    def test_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        pass

    @abstractmethod
    def lightning_module_state_dict(self) -> Dict[str, Union[Any, Tensor]]:
        """Returns model state."""

    @abstractmethod
    def on_train_start(self) -> None:
        """Called when train begins."""
        pass

    @abstractmethod
    def on_validation_start(self) -> None:
        """Called when validation begins."""
        pass

    @abstractmethod
    def on_test_start(self) -> None:
        """Called when test begins."""
        pass

    @abstractmethod
    def on_predict_start(self) -> None:
        """Called when predict begins."""
        pass

    @abstractmethod
    def on_train_end(self) -> None:
        """Called when train ends."""
        pass

    @abstractmethod
    def on_validation_end(self) -> None:
        """Called when validation ends."""
        pass

    @abstractmethod
    def on_test_end(self) -> None:
        """Called when test end."""
        pass

    @abstractmethod
    def on_predict_end(self) -> None:
        """Called when predict ends."""
        pass

    @abstractmethod
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Called in the training loop before anything happens for that batch."""
        pass

    @abstractmethod
    def dispatch(self, trainer: "pl.Trainer") -> None:
        """Hook to do something before the training/evaluation/prediction starts."""
