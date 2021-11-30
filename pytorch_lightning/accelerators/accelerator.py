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
from typing import Any, Dict, Generator, Optional, Union

import torch
from torch.nn import Module

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.plugins.training_type import TrainingTypePlugin
from pytorch_lightning.utilities.types import STEP_OUTPUT


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

    def __init__(self, precision_plugin: Optional[PrecisionPlugin], training_type_plugin: TrainingTypePlugin) -> None:
        """
        Args:
            precision_plugin: the plugin to handle precision-specific parts

                .. deprecated::
                    The ``precision_plugin`` parameter has been deprecated and will be removed soon.
                    Pass the precision plugin as a parameter to the ``TrainingTypePlugin`` instead.

            training_type_plugin: the plugin to handle different training routines
        """

        self.training_type_plugin = training_type_plugin

        if precision_plugin is not None:
            self.training_type_plugin._precision_plugin = precision_plugin

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
        self.training_type_plugin.setup(trainer)

    def pre_dispatch(self, trainer: "pl.Trainer") -> None:
        """Hook to do something before the training/evaluation/prediction starts."""
        self.training_type_plugin._move_optimizer_state()

        self.training_type_plugin.pre_dispatch()
        if self.training_type_plugin.setup_optimizers_in_pre_dispatch:
            self.training_type_plugin.setup_optimizers(trainer)

        self.training_type_plugin.precision_plugin.pre_dispatch()

    def dispatch(self, trainer: "pl.Trainer") -> None:
        """Hook to do something before the training/evaluation/prediction starts."""
        self.training_type_plugin.dispatch(trainer)
        self.training_type_plugin.precision_plugin.dispatch(trainer)

    def post_dispatch(self, trainer: "pl.Trainer") -> None:
        """Hook to do something after the training/evaluation/prediction starts."""
        self.training_type_plugin.post_dispatch(trainer)
        self.training_type_plugin.precision_plugin.post_dispatch()

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

    def training_step(self, step_kwargs: Dict[str, Union[Any, int]]) -> STEP_OUTPUT:
        """The actual training step.

        See :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` for more details
        """
        with self.training_type_plugin.precision_plugin.train_step_context():
            return self.training_type_plugin.training_step(*step_kwargs.values())

    def validation_step(self, step_kwargs: Dict[str, Union[Any, int]]) -> Optional[STEP_OUTPUT]:
        """The actual validation step.

        See :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step` for more details
        """
        with self.training_type_plugin.precision_plugin.val_step_context():
            return self.training_type_plugin.validation_step(*step_kwargs.values())

    def test_step(self, step_kwargs: Dict[str, Union[Any, int]]) -> Optional[STEP_OUTPUT]:
        """The actual test step.

        See :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step` for more details
        """
        with self.training_type_plugin.precision_plugin.test_step_context():
            return self.training_type_plugin.test_step(*step_kwargs.values())

    def predict_step(self, step_kwargs: Dict[str, Union[Any, int]]) -> STEP_OUTPUT:
        """The actual predict step.

        See :meth:`~pytorch_lightning.core.lightning.LightningModule.predict_step` for more details
        """
        with self.training_type_plugin.precision_plugin.predict_step_context():
            return self.training_type_plugin.predict_step(*step_kwargs.values())

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

    @staticmethod
    @abstractmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
