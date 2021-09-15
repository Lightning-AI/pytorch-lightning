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
from typing import List, Optional

import torch

import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_8

_log = logging.getLogger(__name__)


class GPUAccelerator(Accelerator):
    """Accelerator for GPU devices."""

    def setup_environment(self) -> None:
        super().setup_environment()
        if "cuda" not in str(self.root_device):
            raise MisconfigurationException(f"Device should be GPU, got {self.root_device} instead")
        torch.cuda.set_device(self.root_device)

    def setup(self, trainer: "pl.Trainer") -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not GPU.
        """
        self.set_nvidia_flags(trainer.local_rank)

        # The logical device IDs for selected devices
        self._device_ids: List[int] = sorted(set(trainer.data_parallel_device_ids))

        # The unmasked real GPU IDs
        self._gpu_ids: List[int] = self._get_gpu_ids(self._device_ids)

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

    def get_device_stats(self, device: Optional[torch.device] = None) -> None:
        """Gets stats for the given GPU device"""
        if _TORCH_GREATER_EQUAL_1_8:
            return torch.cuda.memory_stats(device=device)
        else:
            gpu_stat_keys = [
                ("utilization.gpu", "%"),
                ("memory.used", "MB"),
                ("memory.free", "MB"),
                ("utilization.memory", "%"),
                ("fan.speed", "%"),
                ("temperature.gpu", "°C"),
                ("temperature.memory", "°C"),
            ]
            gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
            logs = self._parse_gpu_stats(self._device_ids, gpu_stats, gpu_stat_keys)

    def _get_gpu_stats(self, queries: List[str]) -> List[List[float]]:
        if not queries:
            return []

        """Run nvidia-smi to get the gpu stats"""
        gpu_query = ",".join(queries)
        format = "csv,nounits,noheader"
        gpu_ids = ",".join(self._gpu_ids)
        result = subprocess.run(
            [shutil.which("nvidia-smi"), f"--query-gpu={gpu_query}", f"--format={format}", f"--id={gpu_ids}"],
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

        stats = result.stdout.strip().split(os.linesep)
        stats = [[_to_float(x) for x in s.split(", ")] for s in stats]
        return stats

    @staticmethod
    def _get_gpu_ids(device_ids: List[int]) -> List[str]:
        """Get the unmasked real GPU IDs."""
        # All devices if `CUDA_VISIBLE_DEVICES` unset
        default = ",".join(str(i) for i in range(torch.cuda.device_count()))
        cuda_visible_devices: List[str] = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
        return [cuda_visible_devices[device_id].strip() for device_id in device_ids]

    def teardown(self) -> None:
        super().teardown()
        self._move_optimizer_state(torch.device("cpu"))
