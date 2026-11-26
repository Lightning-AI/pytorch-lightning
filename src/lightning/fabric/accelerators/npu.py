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
"""Accelerator for Huawei Ascend NPU devices (via ``torch_npu``).

This module implements the platform access layer for the NPU accelerator (B01):
a ``NPUAccelerator`` following the same template as ``CUDAAccelerator`` whose device
APIs are backed by ``torch.npu.*``.

In addition, when running on a machine with an available NPU device, importing this
module triggers ``torch_npu.contrib.transfer_to_npu`` which transparently migrates the
framework's existing ``torch.cuda.*`` code paths (device count, memory stats, streams,
AMP, distributed backend ``nccl`` → ``hccl``, ...) onto ``torch.npu.*`` at runtime. This
is the A-channel mechanical migration: CUDA code paths stay untouched and keep working
for CUDA users, while on NPU machines they resolve to the NPU backend automatically.
"""

import sys
from typing import Any, Optional, Union

import torch
from typing_extensions import override

from lightning_utilities.core.imports import package_available

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning.fabric.utilities.types import _DEVICE

try:  # pragma: no cover - environment dependent
    import torch_npu  # noqa: F401

    # Runtime migration (A-channel mechanical replacement): only active on machines that
    # actually expose an Ascend NPU device, so CUDA-only environments are never affected.
    if torch.npu.is_available():
        from torch_npu.contrib import transfer_to_npu  # noqa: F401

    _TORCH_NPU_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _TORCH_NPU_AVAILABLE = False
    # D2 (W12): do not swallow a *failed* torch_npu init silently. Only warn when torch_npu
    # is actually installed but failed to initialize (typical cause: torch/torch_npu version
    # mismatch) — otherwise `accelerator="auto"` silently falls back to CPU with no signal.
    # `sys.modules.get(...) is not None` is used instead of `in sys.modules` because the
    # import-blocking `sys.modules[name] = None` trick (used to simulate "not installed")
    # leaves the key present; a None value means the package is not importable → stay silent.
    if sys.modules.get("torch_npu") is not None or package_available("torch_npu"):
        rank_zero_warn(
            "`torch_npu` is installed but failed to initialize. The NPU accelerator will be "
            "unavailable and `accelerator='auto'` will fall back to CPU. Check that the "
            "`torch_npu` version matches the installed `torch` version.",
            category=RuntimeWarning,
        )


class NPUAccelerator(Accelerator):
    """Accelerator for Huawei Ascend NPU devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not of type NPU.
        """
        if device.type != "npu":
            raise ValueError(f"Device should be NPU, got {device} instead.")
        torch.npu.set_device(device)

    @override
    def teardown(self) -> None:
        torch.npu.empty_cache()

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Gets stats for the given NPU device (memory info reported by the NPU driver)."""
        free_bytes, total_bytes = torch.npu.mem_get_info(device)
        return {"num_free": free_bytes, "num_total": total_bytes}

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
        """Accelerator device parsing logic (same semantics as :meth:`CUDAAccelerator.parse_devices`)."""
        return _parse_npu_devices(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("npu", i) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_npu_devices()

    @staticmethod
    @override
    def is_available() -> bool:
        return _is_npu_available()

    @staticmethod
    @override
    def name() -> str:
        return "npu"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__name__,
        )


def _is_npu_available() -> bool:
    """Returns a bool indicating if an Ascend NPU is currently available."""
    return _TORCH_NPU_AVAILABLE and getattr(torch, "npu", None) is not None and torch.npu.is_available()


def _parse_npu_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
    """Parses the NPU device ids using the same semantics as ``_parse_gpu_ids`` for CUDA/MPS.

    ``None`` / ``0`` / ``"0"`` / ``"[]"`` mean that no devices are requested and map to ``None``.
    A positive int ``N`` maps to ``[0 .. N)``, ``-1`` to all visible NPU devices, and a sequence
    is validated against the currently available NPU devices.
    """
    if devices is None or (isinstance(devices, int) and devices == 0) or str(devices).strip() in ("0", "[]"):
        return None
    if isinstance(devices, str):
        if devices == "-1":
            devices = -1
        elif "," in devices:
            devices = [int(x.strip()) for x in devices.split(",") if len(x) > 0]
        else:
            devices = int(devices.strip())
    if isinstance(devices, int):
        if devices == -1:
            return list(range(num_npu_devices()))
        return list(range(devices))
    devices = list(devices)
    all_available = set(range(num_npu_devices()))
    for device in devices:
        if device not in all_available:
            raise ValueError(f"You requested npu: {devices}\n But your machine only has: {sorted(all_available)}")
    return devices


def num_npu_devices() -> int:
    """Returns the number of available NPU devices."""
    if getattr(torch, "npu", None) is None:
        return 0
    return torch.npu.device_count()


def _get_all_visible_npu_devices() -> list[int]:
    """Returns a list of all visible NPU devices (masked by ``ASCEND_RT_VISIBLE_DEVICES``)."""
    return list(range(num_npu_devices()))


def find_usable_npu_devices(num_devices: int = -1) -> list[int]:
    """Returns a list of all available and usable Ascend NPU devices.

    A device is considered usable if we can successfully move a tensor to it.

    Args:
        num_devices: The number of devices you want to request. By default, this function will return as many as there
            are usable NPU devices available.

    """
    if num_devices == 0:
        return []
    visible_devices = _get_all_visible_npu_devices()
    if not visible_devices:
        raise ValueError(
            f"You requested to find {num_devices} devices but there are no visible NPU devices on this machine."
        )
    if num_devices > len(visible_devices):
        raise ValueError(
            f"You requested to find {num_devices} devices but this machine only has {len(visible_devices)} NPUs."
        )

    available_devices = []
    for npu_idx in visible_devices:
        try:
            torch.tensor(0, device=torch.device("npu", npu_idx))
        except RuntimeError:
            continue
        available_devices.append(npu_idx)
        if len(available_devices) == num_devices:
            break
    return available_devices
