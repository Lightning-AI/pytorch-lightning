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
from functools import lru_cache
from typing import Optional, Union

import torch
from typing_extensions import override

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.fabric.utilities.rank_zero import rank_zero_info


class MUSAAccelerator(Accelerator):
    """Accelerator for MUSA devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not of type MUSA.
        """
        if device.type != "musa":
            raise ValueError(f"Device should be MUSA, got {device} instead.")
        _check_musa_matmul_precision(device)
        torch.musa.set_device(device)

    @override
    def teardown(self) -> None:
        _clear_musa_memory()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
        """Accelerator device parsing logic."""
        from lightning.fabric.utilities.device_parser import _parse_gpu_ids

        return _parse_gpu_ids(devices, include_musa=True)

    @staticmethod
    @override
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("musa", i) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_musa_devices()

    @staticmethod
    @override
    def is_available() -> bool:
        return num_musa_devices() > 0

    @staticmethod
    @override
    def name() -> str:
        return "musa"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__name__,
        )


def find_usable_musa_devices(num_devices: int = -1) -> list[int]:
    """Returns a list of all available and usable MUSA GPU devices.

    A GPU is considered usable if we can successfully move a tensor to the device, and this is what this function
    tests for each GPU on the system until the target number of usable devices is found.

    A subset of GPUs on the system might be used by other processes, and if the GPU is configured to operate in
    'exclusive' mode (configurable by the admin), then only one process is allowed to occupy it.

    Args:
        num_devices: The number of devices you want to request. By default, this function will return as many as there
            are usable MUSA GPU devices available.

    Warning:
        If multiple processes call this function at the same time, there can be race conditions in the case where
        both processes determine that the device is unoccupied, leading into one of them crashing later on.

    """
    if num_devices == 0:
        return []
    visible_devices = _get_all_visible_musa_devices()
    if not visible_devices:
        raise ValueError(
            f"You requested to find {num_devices} devices but there are no visible MUSA devices on this machine."
        )
    if num_devices > len(visible_devices):
        raise ValueError(
            f"You requested to find {num_devices} devices but this machine only has {len(visible_devices)} GPUs."
        )

    available_devices = []
    unavailable_devices = []

    for gpu_idx in visible_devices:
        try:
            torch.tensor(0, device=torch.device("musa", gpu_idx))
        except RuntimeError:
            unavailable_devices.append(gpu_idx)
            continue

        available_devices.append(gpu_idx)
        if len(available_devices) == num_devices:
            # exit early if we found the right number of GPUs
            break

    if num_devices != -1 and len(available_devices) != num_devices:
        raise RuntimeError(
            f"You requested to find {num_devices} devices but only {len(available_devices)} are currently available."
            f" The devices {unavailable_devices} are occupied by other processes and can't be used at the moment."
        )
    return available_devices


def _get_all_visible_musa_devices() -> list[int]:
    """Returns a list of all visible MUSA GPU devices.

    Devices masked by the environment variabale ``MUSA_VISIBLE_DEVICES`` won't be returned here. For example, assume you
    have 8 physical GPUs. If ``MUSA_VISIBLE_DEVICES="1,3,6"``, then this function will return the list ``[0, 1, 2]``
    because these are the three visible GPUs after applying the mask ``MUSA_VISIBLE_DEVICES``.

    """
    return list(range(num_musa_devices()))


def num_musa_devices() -> int:
    """Returns the number of available MUSA devices."""
    return torch.musa.device_count()


def is_musa_available() -> bool:
    """Returns a bool indicating if MUSA is currently available."""
    # We set `PYTORCH_NVML_BASED_MUSA_CHECK=1` in lightning.fabric.__init__.py
    return torch.musa.is_available()


def _is_ampere_or_later(device: Optional[torch.device] = None) -> bool:
    major, _ = torch.musa.get_device_capability(device)
    return major >= 8  # Ampere and later leverage tensor cores, where this setting becomes useful


@lru_cache(1)  # show the warning only ever once
def _check_musa_matmul_precision(device: torch.device) -> None:
    if not torch.musa.is_available() or not _is_ampere_or_later(device):
        return
    # check that the user hasn't changed the precision already, this works for both `allow_tf32 = True` and
    # `set_float32_matmul_precision`
    if torch.get_float32_matmul_precision() == "highest":  # default
        rank_zero_info(
            f"You are using a MUSA device ({torch.musa.get_device_name(device)!r}) that has Tensor Cores. To properly"
            " utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off"
            " precision for performance. For more details, read https://pytorch.org/docs/stable/generated/"
            "torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"
        )
    # note: no need change `torch.backends.cudnn.allow_tf32` as it's enabled by default:
    # https://pytorch.org/docs/stable/notes/musa.html#tensorfloat-32-tf32-on-ampere-devices


def _clear_musa_memory() -> None:
    # strangely, the attribute function be undefined when torch.compile is used
    if hasattr(torch._C, "_musa_clearCublasWorkspaces"):
        # https://github.com/pytorch/pytorch/issues/95668
        torch._C._musa_clearMublasWorkspaces()
    torch.musa.empty_cache()
