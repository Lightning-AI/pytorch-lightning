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

import torch

from pytorch_lightning.core.decorators import parameter_validation
from pytorch_lightning.plugins.training_type.single_device import SingleDevicePlugin
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities.apply_func import move_data_to_device

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm


class SingleTPUPlugin(SingleDevicePlugin):
    """ Plugin for training on a single TPU device. """

    def __init__(self, device: int, debug: bool = False):

        device = xm.xla_device(device)
        super().__init__(device)

        self.debug = debug
        self.tpu_local_core_rank = 0
        self.tpu_global_core_rank = 0

    @property
    def is_distributed(self) -> bool:
        return False

    @parameter_validation
    def model_to_device(self) -> None:
        self.model.to(self.root_device)

    def pre_dispatch(self) -> None:
        if isinstance(self.device, int):
            self.device = xm.xla_device(self.device)

        if self.debug:
            os.environ["PT_XLA_DEBUG"] = str(1)

        self.tpu_local_core_rank = xm.get_local_ordinal()
        self.tpu_global_core_rank = xm.get_ordinal()

    def on_save(self, checkpoint: dict) -> dict:
        """
        Move XLA tensors to CPU before saving
        Recommended on XLA Guide:
        https://github.com/pytorch/xla/blob/master/API_GUIDE.md#saving-and-loading-xla-tensors
        """
        return move_data_to_device(checkpoint, torch.device("cpu"))

    def teardown(self) -> None:
        # TPU teardown
        os.environ.pop("PT_XLA_DEBUG", None)
