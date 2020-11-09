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

XLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp


def inner_f(_, queue, func, *args):  # pragma: no cover
    try:
        queue.put(func(*args))
    except Exception:
        import traceback

        traceback.print_exc()
        queue.put(None)


def pl_multi_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        queue = Queue()
        xmp.spawn(inner_f,
          args=(queue, func, *args),
          nprocs=1,
          join=True,
          daemon=False,
          start_method='fork')
        return queue.get()

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
        if XLA_AVAILABLE:
            return xm.xla_device_hw(device)

    @staticmethod
    def _is_device_tpu() -> bool:
        """
        Check if device is TPU

        Return:
            A boolean value indicating if the xla device is a TPU device or not
        """
        if XLA_AVAILABLE:
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
        if XLADeviceUtils.TPU_AVAILABLE is None and XLA_AVAILABLE:
            XLADeviceUtils.TPU_AVAILABLE = pl_multi_process(XLADeviceUtils._is_device_tpu)()
        return XLADeviceUtils.TPU_AVAILABLE
