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
from abc import ABC
from typing import List, Optional, Any, Dict, Callable, Union, Iterable, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from lightning_lite.lite.utilities import move_data_to_device
from lightning_lite.lite.utilities.optimizer import optimizers_to_device
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.plugins import LayerSync
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.interface import PLStrategyInterface
from lightning_lite.lite.strategies.parallel import ParallelStrategy as CoreParallelStrategy
from lightning_lite.lite.accelerators.accelerator import Accelerator as CoreAccelerator
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRSchedulerConfig, TrainingStep, ValidationStep, TestStep, \
    PredictStep


class ParallelStrategy(CoreParallelStrategy, PLStrategyInterface, ABC):
    """Plugin for training with multiple processes in parallel."""

    def __init__(
        self,
        accelerator: Optional["CoreAccelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ):
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision_plugin=precision_plugin)
        self.parallel_devices = parallel_devices
        self.cluster_environment = cluster_environment
        self._layer_sync: Optional[LayerSync] = None

        self._lightning_module: Optional[pl.LightningModule] = None
        self._model: Optional[Module] = None
        self._optimizers: List[Optimizer] = []
        self._lightning_optimizers: Dict[int, LightningOptimizer] = {}
        self.lr_scheduler_configs: List[LRSchedulerConfig] = []
        self.optimizer_frequencies: List[int] = []

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        """Returns the pure LightningModule without potential wrappers."""
        return self._lightning_module

    @property
    def model(self) -> Optional[Module]:
        """Returns the potentially wrapped LightningModule."""
        return self._model if self._model is not None else self._lightning_module

    @model.setter
    def model(self, new_model: Optional[Module]) -> None:
        self._model = new_model

    @property
    def optimizers(self) -> List[Optimizer]:
        return self._optimizers

    @optimizers.setter
    def optimizers(self, optimizers: List[Optimizer]) -> None:
        self._optimizers = optimizers
        self._lightning_optimizers = {
            idx: LightningOptimizer._to_lightning_optimizer(opt, self, idx) for idx, opt in enumerate(self.optimizers)
        }

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        return False

    @property
    def lightning_restore_optimizer(self) -> bool:
        return True

    @property
    def handles_gradient_accumulation(self) -> bool:
        return False

    def connect(self, model: "pl.LightningModule") -> None:
        """Called by the accelerator to connect the accelerator and the model with this plugin."""
        self._lightning_module = model
        self.model = model

    def setup_environment(self) -> None:
        assert self.accelerator is not None
        self.accelerator.setup_environment(self.root_device)

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        if trainer.state.fn not in (TrainerFn.FITTING, TrainerFn.TUNING):
            return
        assert self.lightning_module is not None
        self.optimizers, self.lr_scheduler_configs, self.optimizer_frequencies = _init_optimizers_and_lr_schedulers(
            self.lightning_module
        )

    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)
        self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        optimizers_to_device(self.optimizers, self.root_device)

    def setup_precision_plugin(self) -> None:
        """Attaches the precision plugin to the accelerator."""
        assert self.model is not None
        model, optimizers, lr_scheduler_configs = self.precision_plugin.connect(
            self.model, self.optimizers, self.lr_scheduler_configs
        )
        self.model = model
        self.optimizers = optimizers
        self.lr_scheduler_configs = lr_scheduler_configs

    def run_backward(
            self, tensor: Tensor,
            module: Optional[Module],
            optimizer: Optional[Optimizer],
            optimizer_idx: Optional[int],
            *args: Any,
            **kwargs: Any,
    ) -> None:
        self.pre_backward(tensor, module)
        assert self.lightning_module is not None
        closure_loss = self.precision_plugin.pre_backward(self.lightning_module, tensor)

        self.precision_plugin.backward(self.lightning_module, closure_loss, optimizer, optimizer_idx, *args,
                                       **kwargs)

        closure_loss = self.precision_plugin.post_backward(self.lightning_module, closure_loss)
        self.post_backward(closure_loss, module)

    def optimizer_step(
        self,
        optimizer: Optimizer,
        opt_idx: int,
        closure: Callable[[], Any],
        model: Optional[Module] = None,
        **kwargs: Any,
    ) -> Any:
        model = model or self.lightning_module
        return self.precision_plugin.optimizer_step(model, optimizer, opt_idx, closure, **kwargs)

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None) -> Any:
        model = self.lightning_module
        device = device or self.root_device
        if model is not None:
            return model._apply_batch_transfer_handler(batch, device=device, dataloader_idx=dataloader_idx)
        return move_data_to_device(batch, device)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        with self.precision_plugin.train_step_context():
            assert isinstance(self.model, TrainingStep)
            return self.model.training_step(*args, **kwargs)

    def post_training_step(self) -> None:
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.val_step_context():
            assert isinstance(self.model, ValidationStep)
            return self.model.validation_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.test_step_context():
            assert isinstance(self.model, TestStep)
            return self.model.test_step(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        with self.precision_plugin.predict_step_context():
            assert isinstance(self.model, PredictStep)
            return self.model.predict_step(*args, **kwargs)

    def training_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        return output

    def validation_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        return output

    def test_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        return output

    def lightning_module_state_dict(self) -> Dict[str, Union[Any, Tensor]]:
        assert self.lightning_module is not None
        return self.lightning_module.state_dict()

    def on_train_start(self) -> None:
        pass

    def on_validation_start(self) -> None:
        pass

    def on_test_start(self) -> None:
        pass

    def on_predict_start(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    def on_validation_end(self) -> None:
        pass

    def on_test_end(self) -> None:
        pass

    def on_predict_end(self) -> None:
        pass

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        pass

    def dispatch(self, trainer: "pl.Trainer") -> None:
        self.precision_plugin.dispatch(trainer)

    def reconciliate_processes(self, trace: str) -> None:
        """Function to re-conciliate processes on failure."""

    def __getstate__(self) -> Dict:
        # `LightningOptimizer` overrides `self.__class__` so they cannot be pickled
        state = dict(vars(self))  # copy
        state["_lightning_optimizers"] = {}
        return state

    def __setstate__(self, state: Dict) -> None:
        self.__dict__ = state
        self.optimizers = self.optimizers  # re-create the `_lightning_optimizers`
