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
import json
import os
from typing import Any, Callable, List, Optional, Union

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.utilities import _SHARK_AVAILABLE
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.data import _get_dataloader_init_kwargs
from pytorch_lightning.utilities.enums import PrecisionType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

if _SHARK_AVAILABLE:
    import examples.shark_runner as shark
    import iree.compiler as ireec
    import iree.runtime as ireert
    from torch_mlir_e2e_test.linalg_on_tensors_backends.abc import LinalgOnTensorsBackend
    from torch_mlir_e2e_test.torchscript.configs import LinalgOnTensorsBackendTestConfig
else:
    torch_mlir = None
    iree = None


class LightningSharkModule(_LightningModuleWrapperBase):
    def __init__(self, pl_module: "pl.LightningModule", precision: Union[str, int], device="cpu"):
        super().__init__(pl_module)
        # TODO: support precision
        self.precision = precision
        self.module = pl_module
        self.device = device

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        print("shark swimming")
        return shark.shark_inference(
            module=self.module, input=inputs, device=self.device, dynamic=True, jit_trace=False
        )

    @staticmethod
    def batch_to(data: torch.Tensor) -> torch.Tensor:
        return data


class SharkStrategy(ParallelStrategy):
    """Plugin for training with Shark."""

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        device_iterations: int = 1,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ) -> None:
        """
        Arguments:

            device_iterations: Number of iterations to run on device at once before returning to host.
                This can be used as an optimization to speed up training.
        """
        super().__init__(
            accelerator=accelerator,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        if not _SHARK_AVAILABLE:
            raise MisconfigurationException("Shark is not available, please check dependencies and flags.")

        self.device_iterations = device_iterations

    def setup(self, trainer: "pl.Trainer") -> None:
        # TODO: generalize based on examples of torch_mlir training output,
        # need to find a way to match flat output list of tensors to the list of trainable variables in the model
        self._handle_gradient_accumulation_steps()

        super().setup(trainer)

        self.model = LightningSharkModule(self.lightning_module, self.precision_plugin.precision)

        # TODO: may need to address here creating modules for different parts of training..

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        super().setup_optimizers(trainer)

        if len(self.optimizers) > 1:
            raise MisconfigurationException("Shark currently only supports one optimizer.")

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        return self.model.module if isinstance(self.model, LightningSharkModule) else self.model

    def _handle_gradient_accumulation_steps(self) -> None:
        # TODO: Shark does not handle grad accumulation internally, will need to manage grad accumulation somehow
        ...

    def _prepare_input(self, args: Any):
        # TODO{@dan-garvey): may want to match input againt ClassAnnotator defined in the Accelerator
        def to_tuple(x):
            return tuple(x)

        def to_tensor(x):
            return torch.tensor(x).unsqueeze(0).repeat(self._n_replicate)

        args = apply_to_collection(args, dtype=list, function=to_tuple)
        args = apply_to_collection(args, dtype=(int, float), function=to_tensor)
        return args

    def _step(self, stage: RunningStage, *args: Any, **kwargs: Any):
        args = self._prepare_input(args)
        # self.lightning_module._running_torchscript = True
        out = self.lightning_module.forward(args)
        # self.lightning_module._running_torchscript = False
        return out

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.train_step_context():
            return self._step(RunningStage.TRAINING, *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.val_step_context():
            return self._step(RunningStage.VALIDATING, *args, **kwargs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.test_step_context():
            return self._step(RunningStage.TESTING, *args, **kwargs)

    def predict_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.predict_step_context():
            return self._step(RunningStage.PREDICTING, *args, **kwargs)

    def on_train_start(self):
        # TODO: Support loading a model
        ...

    def on_validation_start(self):
        # TODO: Support loading a model
        ...

    def on_test_start(self):
        # TODO: Support loading a model
        ...

    def on_predict_start(self):
        # TODO: Support loading a model
        ...

    def on_train_end(self):
        # TODO: Support saving a model
        ...

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Updates optimizer stats if LR scheduler modified the optimizer state
        optimizer = self.optimizers[0]

    @property
    def root_device(self) -> torch.device:
        pass

    def model_to_device(self) -> None:
        pass

    @property
    def is_global_zero(self) -> bool:
        return True

    def reduce(self, tensor: Union[torch.Tensor, Any], *args: Any, **kwargs: Any) -> Union[torch.Tensor, Any]:
        return tensor

    def barrier(self, name: Optional[str] = None) -> None:
        pass

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        return tensor

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj
