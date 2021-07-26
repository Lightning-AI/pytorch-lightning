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

import torch

import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException

_log = logging.getLogger(__name__)


class GPUAccelerator(Accelerator):
    """Accelerator for GPU devices."""

    def setup_environment(self) -> None:
        super().setup_environment()
        if "cuda" not in str(self.root_device):
            raise MisconfigurationException(f"Device should be GPU, got {self.root_device} instead")
        torch.cuda.set_device(self.root_device)

    def setup(self, trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not GPU.
        """
        self.set_nvidia_flags(trainer.local_rank)
        return super().setup(trainer, model)

    def on_train_start(self) -> None:
        # clear cache before training
        torch.cuda.empty_cache()

    @staticmethod
    def set_nvidia_flags(local_rank: int) -> None:
        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join([str(x) for x in range(torch.cuda.device_count())])
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        _log.info(f"LOCAL_RANK: {local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]")

    def teardown(self) -> None:
        super().teardown()
        self._move_optimizer_state(torch.device("cpu"))
