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
from typing import Any, Callable, Optional

import torch
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.plugins.training_type.single_tpu import SingleTPUPlugin
from pytorch_lightning.plugins.training_type.tpu_spawn import TPUSpawnPlugin
from pytorch_lightning.utilities import _XLA_AVAILABLE
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


class TPUAccelerator(Accelerator):
    """Accelerator for TPU devices."""

    def setup(self, trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
        """
        Raises:
            MisconfigurationException:
                If AMP is used with TPU, or if TPUs are not using a single TPU core or TPU spawn training.
        """
        if isinstance(self.precision_plugin, MixedPrecisionPlugin):
            raise MisconfigurationException(
                "amp + tpu is not supported. Only bfloats are supported on TPU. Consider using TPUHalfPrecisionPlugin"
            )

        if not isinstance(self.training_type_plugin, (SingleTPUPlugin, TPUSpawnPlugin)):
            raise MisconfigurationException("TPUs only support a single tpu core or tpu spawn training.")
        return super().setup(trainer, model)

    def run_optimizer_step(
        self, optimizer: Optimizer, optimizer_idx: int, lambda_closure: Callable, **kwargs: Any
    ) -> None:
        xm.optimizer_step(optimizer, optimizer_args={"closure": lambda_closure, **kwargs})

    def _move_optimizer_state(self, device: Optional[torch.device] = None) -> None:
        """Moves the state of the optimizers to the TPU if needed."""
        # TODO: `self.root_device` would raise error if called outside the spawn process
        # while training on 8 and more cores.
        for opt in self.optimizers:
            for p, v in opt.state.items():
                opt.state[p] = apply_to_collection(v, torch.Tensor, move_data_to_device, self.root_device)
