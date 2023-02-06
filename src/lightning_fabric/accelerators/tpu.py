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
import queue as q
import traceback
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from lightning_utilities.core.imports import ModuleAvailableCache

from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.utilities.device_parser import _check_data_type


class TPUAccelerator(Accelerator):
    """Accelerator for TPU devices."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(*args, **kwargs)

    def setup_device(self, device: torch.device) -> None:
        pass

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[Union[int, List[int]]]:
        """Accelerator device parsing logic."""
        return _parse_tpu_devices(devices)

    @staticmethod
    def get_parallel_devices(devices: Union[int, List[int]]) -> List[int]:
        """Gets parallel devices for the Accelerator."""
        if isinstance(devices, int):
            return list(range(devices))
        return devices

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 8

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def is_available() -> bool:
        # check `_XLA_AVAILABLE` again to avoid launching processes
        return bool(_XLA_AVAILABLE) and _is_device_tpu()

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "tpu",
            cls,
            description=cls.__class__.__name__,
        )


# define TPU availability timeout in seconds
TPU_CHECK_TIMEOUT = 60


def _inner_f(queue: Queue, func: Callable, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
    try:
        queue.put(func(*args, **kwargs))
    except Exception:
        traceback.print_exc()
        queue.put(None)


def _multi_process(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Union[bool, Any]:
        queue: Queue = Queue()
        proc = Process(target=_inner_f, args=(queue, func, *args), kwargs=kwargs)
        proc.start()
        proc.join(TPU_CHECK_TIMEOUT)
        try:
            return queue.get_nowait()
        except q.Empty:
            traceback.print_exc()
            return False

    return wrapper


@_multi_process
def _is_device_tpu() -> bool:
    """Check if TPU devices are available. Runs XLA device check within a separate process.

    Return:
        A boolean value indicating if TPU devices are available
    """
    if not _XLA_AVAILABLE:
        return False
    import torch_xla.core.xla_model as xm

    # For the TPU Pod training process, for example, if we have
    # TPU v3-32 with 4 VMs, the world size would be 4 and as
    # we would have to use `torch_xla.distributed.xla_dist` for
    # multiple VMs and TPU_CONFIG won't be available, running
    # `xm.get_xla_supported_devices("TPU")` won't be possible.
    return (xm.xrt_world_size() > 1) or bool(xm.get_xla_supported_devices("TPU"))


_XLA_AVAILABLE = ModuleAvailableCache("torch_xla")


def _tpu_distributed() -> bool:
    if not TPUAccelerator.is_available():
        return False
    import torch_xla.core.xla_model as xm

    return xm.xrt_world_size() > 1


def _parse_tpu_devices(devices: Optional[Union[int, str, List[int]]]) -> Optional[Union[int, List[int]]]:
    """
    Parses the TPU devices given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.Trainer` and :class:`~lightning_fabric.Fabric`.

    Args:
        devices: An int of 1 or string '1' indicates that 1 core with multi-processing should be used
            An int 8 or string '8' indicates that all 8 cores with multi-processing should be used
            A list of ints or a strings containing a list of comma separated integers
            indicates the specific TPU core to use.

    Returns:
        A list of tpu_cores to be used or ``None`` if no TPU cores were requested

    Raises:
        TypeError:
            If TPU devices aren't 1, 8 or [<1-8>]
    """
    _check_data_type(devices)

    if isinstance(devices, str):
        devices = _parse_tpu_devices_str(devices.strip())

    if not _tpu_devices_valid(devices):
        raise TypeError("`devices` can only be 1, 8 or [<1-8>] for TPUs.")

    return devices


def _tpu_devices_valid(devices: Any) -> bool:
    # allow 1 or 8 cores
    if devices in (1, 8, None):
        return True

    # allow picking 1 of 8 indexes
    if isinstance(devices, (list, tuple, set)):
        has_1_tpu_idx = len(devices) == 1
        is_valid_tpu_idx = 1 <= list(devices)[0] <= 8

        is_valid_tpu_core_choice = has_1_tpu_idx and is_valid_tpu_idx
        return is_valid_tpu_core_choice

    return False


def _parse_tpu_devices_str(devices: str) -> Union[int, List[int]]:
    if devices in ("1", "8"):
        return int(devices)
    return [int(x.strip()) for x in devices.split(",") if len(x) > 0]
