# Copyright The Lightning AI team.
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
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Union

import torch

import lightning.pytorch as pl
from lightning.fabric.accelerators.xpu import _check_xpu_math_precision, num_xpu_devices
from lightning.fabric.utilities.device_parser import _parse_gpu_ids
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException

_log = logging.getLogger(__name__)


class XPUAccelerator(Accelerator):
    """Accelerator for Intel XPU devices."""

    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not GPU.
        """
        if device.type != "xpu":
            raise MisconfigurationException(f"Device should be GPU, got {device} instead")
        _check_xpu_math_precision(device)
        torch.xpu.set_device(device)

    def setup(self, trainer: "pl.Trainer") -> None:
        # TODO refactor input from trainer to local_rank @four4fish
        # self.set_intel_flags(trainer.local_rank)
        # clear cache before training
        torch.xpu.empty_cache()

    # @staticmethod
    # def set_intel_flags(local_rank: int) -> None:
    #    # set the correct xpu visible devices (using pci order)
    #    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #    all_gpu_ids = ",".join(str(x) for x in range(num_xpu_devices()))
    #    devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
    #    _log.info(f"LOCAL_RANK: {local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]")

    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        """Gets stats for the given GPU device.

        Args:
            device: GPU device for which to get stats

        Returns:
            A dictionary mapping the metrics to their values.

        Raises:
            FileNotFoundError:
                If xpum-smi installation not found
        """
        return torch.xpu.memory_stats(device)

    def teardown(self) -> None:
        # clean up memory
        torch.xpu.empty_cache()

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        return _parse_gpu_ids(devices, include_xpu=True)

    @staticmethod
    def get_parallel_devices(devices: List[int]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("xpu", i) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_xpu_devices()

    @staticmethod
    def is_available() -> bool:
        return num_xpu_devices() > 0

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "xpu",
            cls,
            description=f"{cls.__class__.__name__}",
        )


def get_intel_gpu_stats(device: _DEVICE) -> Dict[str, float]:  # pragma: no-cover
    """Get GPU stats including memory, fan speed, and temperature from xpum-smi.

    Args:
        device: GPU device for which to get stats

    Returns:
        A dictionary mapping the metrics to their values.

    Raises:
        FileNotFoundError:
            If xpum-smi installation not found
    """
    xpum_smi_path = shutil.which("xpum-smi")
    if xpum_smi_path is None:
        raise FileNotFoundError("xpum-smi: command not found")

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
        [xpum_smi_path, f"--query-gpu={gpu_query}", "--format=csv,nounits,noheader", f"--id={gpu_id}"],
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
    # All devices
    default = ",".join(str(i) for i in range(num_xpu_devices()))
    # cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
    xpu_visible_devices = default.split(",")
    return xpu_visible_devices[device_id].strip()
