# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#

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
import os
from typing import Any, Dict, Optional

from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.training_type.single_device import SingleDevicePlugin
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities import _HPU_AVAILABLE, find_shared_parameters, set_shared_parameters
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import _PATH

class HPUPlugin(SingleDevicePlugin):

    def __init__(
        self,
        device: int,
        checkpoint_io: Optional[CheckpointIO] = None,
        debug: bool = False,
    ):

        device = torch.device("hpu")
        checkpoint_io = checkpoint_io
        super().__init__(device, checkpoint_io=checkpoint_io)

        self.debug = debug

    @property
    def is_distributed(self) -> bool:
        return False

    def setup(self) -> None:
        shared_params = find_shared_parameters(self.model)
        self.model_to_device()
        if is_overridden("on_post_move_to_device", self.lightning_module):
            self.model.on_post_move_to_device()
        else:
            set_shared_parameters(self.model, shared_params)

    def model_to_device(self) -> None:
        self.model.to(self.root_device)

    @property
    def on_hpu(self) -> bool:
        return True

    def pre_dispatch(self) -> None:
        if isinstance(self.device, int):
            self.device = torch.device(self.device)

    def on_save(self, checkpoint: dict) -> dict:
        return move_data_to_device(checkpoint, torch.device("cpu"))
