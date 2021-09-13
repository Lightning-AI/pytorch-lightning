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
from typing import Optional

import torch

from pytorch_lightning.plugins.collective.collective_plugin import Collective
from pytorch_lightning.plugins.collective.single_device_collective import SingleDeviceCollective
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.training_type.training_type_plugin import TrainingTypePlugin
from pytorch_lightning.utilities import _XLA_AVAILABLE


class SingleDevicePlugin(TrainingTypePlugin):
    """Plugin that handles communication on a single device."""

    def __init__(
        self,
        device: torch.device,
        checkpoint_io: Optional[CheckpointIO] = None,
        collective: Optional[Collective] = None,
    ):
        super().__init__(checkpoint_io=checkpoint_io, collective=collective or SingleDeviceCollective())
        self.device: torch.device = device
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1

    @property
    def on_tpu(self) -> bool:
        return self.root_device.type == "xla" and _XLA_AVAILABLE

    @property
    def on_gpu(self) -> bool:
        return self.root_device.type == "cuda" and torch.cuda.is_available()

    @property
    def root_device(self) -> torch.device:
        return self.device

    def model_to_device(self) -> None:
        self._model.to(self.root_device)

    def setup(self) -> None:
        self.model_to_device()

    @property
    def is_global_zero(self) -> bool:
        return True

    def teardown(self) -> None:
        if self.on_gpu:
            # GPU teardown
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()
