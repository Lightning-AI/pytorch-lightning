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
import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class CPUAccelerator(Accelerator):
    """ Accelerator for CPU devices. """

    def setup(self, trainer: 'pl.Trainer', model: 'pl.LightningModule') -> None:
        """
        Raises:
            MisconfigurationException:
                If AMP is used with CPU, or if the selected device is not CPU.
        """
        if isinstance(self.precision_plugin, MixedPrecisionPlugin):
            raise MisconfigurationException(
                " Mixed precision is currenty only supported with the AMP backend"
                " and AMP + CPU is not supported. Please use a GPU option or"
                " change precision setting."
            )

        if "cpu" not in str(self.root_device):
            raise MisconfigurationException(f"Device should be CPU, got {self.root_device} instead.")

        return super().setup(trainer, model)
