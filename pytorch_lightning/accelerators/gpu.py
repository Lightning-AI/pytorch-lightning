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
from typing import Any, Dict, List, Optional, Union

import torch

import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities import device_parser
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _DEVICE

_log = logging.getLogger(__name__)


class GPUAccelerator(Accelerator):
    """Accelerator for GPU devices."""

    def setup_environment(self, root_device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not GPU.
        """
        super().setup_environment(root_device)
        if root_device.type != "cuda":
            raise MisconfigurationException(f"Device should be GPU, got {root_device} instead")
        torch.cuda.set_device(root_device)

    def setup(self, trainer: "pl.Trainer") -> None:
        # TODO refactor input from trainer to local_rank @four4fish
        self.set_nvidia_flags(trainer.local_rank)
        # clear cache before training
        torch.cuda.empty_cache()

    @staticmethod
    def set_nvidia_flags(local_rank: int) -> None:
        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        _log.info(f"LOCAL_RANK: {local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]")

    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        """Gets stats for the given GPU device.

        Args:
            device: GPU device for which to get stats

        Returns:
            A dictionary mapping the metrics to their values.

        Raises:
            FileNotFoundError:
                If nvidia-smi installation not found
        """
        return torch.cuda.memory_stats(device)

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        return device_parser.parse_gpu_ids(devices)

    @staticmethod
    def get_parallel_devices(devices: List[int]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("cuda", i) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return torch.cuda.device_count()

    @staticmethod
    def is_available() -> bool:
        return torch.cuda.device_count() > 0

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "gpu",
            cls,
            description=f"{cls.__class__.__name__}",
        )


def get_nvidia_gpu_stats(device: _DEVICE) -> Dict[str, float]:  # pragma: no-cover
    """Get GPU stats including memory, fan speed, and temperature from nvidia-smi.

    Args:
        device: GPU device for which to get stats

    Returns:
        A dictionary mapping the metrics to their values.

    Raises:
        FileNotFoundError:
            If nvidia-smi installation not found
    """
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        raise FileNotFoundError("nvidia-smi: command not found")

    gpu_stat_metrics = [
        ("utilization.gpu", "%"),
        ("memory.used", "MB"),
        ("memory.free", "MB"),
        ("utilization.memory", "%"),
        ("fan.speed", "%"),
        ("temperature.gpu", "°C"),
        ("temperature.memory", "°C"),
    ]
    gpu_stat_keys = [k for k, _ in gpu_stat_metrics]
    gpu_query = ",".join(gpu_stat_keys)

    index = torch._utils._get_device_index(device)
    gpu_id = _get_gpu_id(index)
    result = subprocess.run(
        [nvidia_smi_path, f"--query-gpu={gpu_query}", "--format=csv,nounits,noheader", f"--id={gpu_id}"],
        encoding="utf-8",
        capture_output=True,
        check=True,
    )

    def _to_float(x: str) -> float:
        try:
            return float(x)
        except ValueError:
            return 0.0

    s = result.stdout.strip()
    stats = [_to_float(x) for x in s.split(", ")]
    gpu_stats = {f"{x} ({unit})": stat for (x, unit), stat in zip(gpu_stat_metrics, stats)}
    return gpu_stats


def _get_gpu_id(device_id: int) -> str:
    """Get the unmasked real GPU IDs."""
    # All devices if `CUDA_VISIBLE_DEVICES` unset
    default = ",".join(str(i) for i in range(torch.cuda.device_count()))
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
    return cuda_visible_devices[device_id].strip()
