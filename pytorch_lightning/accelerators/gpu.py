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
import shutil
import subprocess
from typing import Any, Dict, List, Union

import torch

import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_8

_log = logging.getLogger(__name__)


class GPUAccelerator(Accelerator):
    """Accelerator for GPU devices."""

    def setup_environment(self) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not GPU.
        """
        super().setup_environment()
        if "cuda" not in str(self.root_device):
            raise MisconfigurationException(f"Device should be GPU, got {self.root_device} instead")
        torch.cuda.set_device(self.root_device)

    def setup(self, trainer: "pl.Trainer") -> None:
        self.set_nvidia_flags(trainer.local_rank)
        return super().setup(trainer)

    def on_train_start(self) -> None:
        # clear cache before training
        torch.cuda.empty_cache()

    @staticmethod
    def set_nvidia_flags(local_rank: int) -> None:
        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        _log.info(f"LOCAL_RANK: {local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]")

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """Gets stats for the given GPU device."""
        if _TORCH_GREATER_EQUAL_1_8:
            return torch.cuda.memory_stats(device)
        else:
            return self._get_gpu_stats(device)

    def _get_gpu_stats(self, device: torch.device) -> Dict[str, float]:
        nvidia_smi_path = shutil.which("nvidia-smi")
        if nvidia_smi_path is None:
            raise FileNotFoundError("nvidia-smi: command not found")

        gpu_stat_keys = [
            "utilization.gpu",
            "memory.used",
            "memory.free",
            "utilization.memory",
            "fan.speed",
            "temperature.gpu",
            "temperature.memoy",
        ]
        gpu_ids = self._get_gpu_id(device.index)

        gpu_query = ",".join(gpu_stat_keys)
        format = "csv,nounits,noheader"
        result = subprocess.run(
            [nvidia_smi_path, f"--query-gpu={gpu_query}", f"--format={format}", f"--id={gpu_ids}"],
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
            check=True,
        )

        def _to_float(x: str) -> float:
            try:
                return float(x)
            except ValueError:
                return 0.0

        stats = [_to_float(x) for x in result.stdout.strip().split(os.linesep)]
        for key in gpu_stat_keys:
            gpu_stats = {key: stat for _, stat in enumerate(stats)}
        return gpu_stats

    def _get_gpu_id(self, device_id: int) -> List[str]:
        """Get the unmasked real GPU IDs."""
        # All devices if `CUDA_VISIBLE_DEVICES` unset
        default = ",".join(str(i) for i in range(torch.cuda.device_count()))
        cuda_visible_devices: List[str] = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
        return cuda_visible_devices[device_id].strip()

    def teardown(self) -> None:
        super().teardown()
        self._move_optimizer_state(torch.device("cpu"))
