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
import functools
import importlib
import queue as q
from multiprocessing import Process, Queue

import torch

TORCHXLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None
if TORCHXLA_AVAILABLE:
    import torch_xla.core.xla_model as xm
else:
    xm = None


def inner_f(queue, func, *args, **kwargs):  # pragma: no cover
    try:
        queue.put(func(*args, **kwargs))
    except Exception:
        import traceback

        traceback.print_exc()
        queue.put(None)


def pl_multi_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        queue = Queue()
        proc = Process(target=inner_f, args=(queue, func, *args), kwargs=kwargs)
        proc.start()
        proc.join(10)
        try:
            return queue.get_nowait()
        except q.Empty:
            return False

    return wrapper


class XLADeviceUtils:
    """Used to detect the type of XLA device"""

    TPU_AVAILABLE = None

    @staticmethod
    def _fetch_xla_device_type(device: torch.device) -> str:
        """
        Returns XLA device type

        Args:
            device: (:class:`~torch.device`): Accepts a torch.device type with a XLA device format i.e xla:0

        Return:
            Returns a str of the device hardware type. i.e TPU
        """
        if xm is not None:
            return xm.xla_device_hw(device)

    @staticmethod
    def _is_device_tpu() -> bool:
        """
        Check if device is TPU

        Return:
            A boolean value indicating if the xla device is a TPU device or not
        """
        if xm is not None:
            device = xm.xla_device()
            device_type = XLADeviceUtils._fetch_xla_device_type(device)
            return device_type == "TPU"

    @staticmethod
    def tpu_device_exists() -> bool:
        """
        Public method to check if TPU is available

        Return:
            A boolean value indicating if a TPU device exists on the system
        """
        if XLADeviceUtils.TPU_AVAILABLE is None and TORCHXLA_AVAILABLE:
            XLADeviceUtils.TPU_AVAILABLE = pl_multi_process(XLADeviceUtils._is_device_tpu)()
        return XLADeviceUtils.TPU_AVAILABLE
