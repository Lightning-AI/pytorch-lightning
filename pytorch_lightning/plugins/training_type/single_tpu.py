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
import torch

from pytorch_lightning.plugins.training_type.single_device import SingleDevicePlugin
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities.apply_func import move_data_to_device

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm


class SingleTPUPlugin(SingleDevicePlugin):

    def __init__(self, device: int):

        device = xm.xla_device(device)
        super().__init__(device)

        self.tpu_local_core_rank = 0
        self.tpu_global_core_rank = 0

    @property
    def on_tpu(self) -> bool:
        return True

    def connect(self, model: torch.nn.Module) -> torch.nn.Module:
        self._model = model
        self.model_to_device()
        return self._model

    @property
    def is_distributed(self) -> bool:
        return False

    def model_to_device(self) -> None:
        self._model.to(self.root_device)

    def pre_dispatch(self) -> None:
        if isinstance(self.device, int):
            self.device = xm.xla_device(self.device)

        self.tpu_local_core_rank = xm.get_local_ordinal()
        self.tpu_global_core_rank = xm.get_ordinal()

    def on_save(self, checkpoint: dict) -> dict:
        """
        Move XLA tensors to CPU before saving
        Recommended on XLA Guide:
        https://github.com/pytorch/xla/blob/master/API_GUIDE.md#saving-and-loading-xla-tensors
        """
        return move_data_to_device(checkpoint, torch.device("cpu"))
