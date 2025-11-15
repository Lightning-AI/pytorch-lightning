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
import functools
import warnings
from typing import Any, Union

import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.fabric.utilities.imports import _raise_enterprise_not_available

_XLA_AVAILABLE = RequirementCache("torch_xla>=1.13", "torch_xla")
_XLA_GREATER_EQUAL_2_1 = RequirementCache("torch_xla>=2.1")
_XLA_GREATER_EQUAL_2_5 = RequirementCache("torch_xla>=2.5")


class XLAAccelerator(Accelerator):
    """Accelerator for XLA devices, normally TPUs.

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _raise_enterprise_not_available()
        super().__init__(*args, **kwargs)

        from pytorch_lightning_enterprise.accelerators.xla import XLAAccelerator as EnterpriseXLAAccelerator

        self.accelerator_impl = EnterpriseXLAAccelerator(*args, **kwargs)

    @override
    def setup_device(self, device: torch.device) -> None:
        return self.accelerator_impl.setup_device(device)

    @override
    def teardown(self) -> None:
        return self.accelerator_impl.teardown()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Union[int, list[int]]:
        """Accelerator device parsing logic."""
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.accelerators.xla import XLAAccelerator as EnterpriseXLAAccelerator

        return EnterpriseXLAAccelerator.parse_devices(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: Union[int, list[int]]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.accelerators.xla import XLAAccelerator as EnterpriseXLAAccelerator

        return EnterpriseXLAAccelerator.get_parallel_devices(devices)

    @staticmethod
    @override
    # XLA's multiprocessing will pop the TPU_NUM_DEVICES key, so we need to cache it
    # https://github.com/pytorch/xla/blob/v2.0.0/torch_xla/distributed/xla_multiprocessing.py#L280
    @functools.lru_cache(maxsize=1)
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.accelerators.xla import XLAAccelerator as EnterpriseXLAAccelerator

        return EnterpriseXLAAccelerator.auto_device_count()

    @staticmethod
    @override
    @functools.lru_cache(maxsize=1)
    def is_available() -> bool:
        try:
            return XLAAccelerator.auto_device_count() > 0
        except (ValueError, AssertionError, OSError):
            # XLA may raise these exceptions if it's not properly configured. This needs to be avoided for the cases
            # when `torch_xla` is imported but not used
            return False
        except ModuleNotFoundError as e:
            warnings.warn(str(e))
            return False

    @staticmethod
    @override
    def name() -> str:
        return "tpu"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__name__,
        )
