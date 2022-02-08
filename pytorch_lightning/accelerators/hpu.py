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
import logging
import os
from typing import Any

import torch

import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins import DataParallelPlugin
from pytorch_lightning.plugins.training_type.single_hpu import HPUPlugin
from pytorch_lightning.plugins.precision.hpu_precision import HPUPrecisionPlugin
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from typing import Any, Dict, Union

_log = logging.getLogger(__name__)



class HPUAccelerator(Accelerator):
    """ Accelerator for HPU devices. """

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """HPU device stats aren't supported yet."""
        return {}

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        # TBD: make this configurable
        return 8
        