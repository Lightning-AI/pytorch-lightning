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
from pytorch_lightning.plugins.training_type.hpu import HPUPlugin
from pytorch_lightning.plugins.precision.hpu_precision import HPUPrecisionPlugin
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException

_log = logging.getLogger(__name__)



class HPUAccelerator(Accelerator):
    """ Accelerator for HPU devices. """

    def setup(self, trainer: "pl.Trainer") -> None:
        """
        Raises:
            ValueError:
                If the precision or training type plugin are unsupported.
        """
        if not isinstance(self.precision_plugin, HPUPrecisionPlugin):
            # this configuration should have been avoided in the accelerator connector
            raise ValueError(
                f"The `HPUAccelerator` can only be used with a `HPUPrecisionPlugin`, found: {self.precision_plugin}."
            )
        if not isinstance(self.training_type_plugin, (HPUPlugin, DDPPlugin)):
            raise ValueError(
                "The `HPUAccelerator` can only be used with a `HPUPlugin` or `DDPPlugin,"
                f" found {self.training_type_plugin}."
            )
        return super().setup(trainer)
        