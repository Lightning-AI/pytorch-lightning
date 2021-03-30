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
import os
import queue as q
import traceback

import torch.multiprocessing as mp

from pytorch_lightning.utilities.imports import _XLA_AVAILABLE

if _XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

#: define waiting time got checking TPU available in sec
TPU_CHECK_TIMEOUT = 60


def inner_f(index, queue, func, *args):  # pragma: no cover
    queue.put(func(index, *args))


def pl_multi_process(func):

    @functools.wraps(func)
    def wrapper(*args):
        smp = mp.get_context("spawn")
        queue = smp.Queue()
        cxt = xmp.spawn(inner_f, args=(queue, func, *args), join=False)

        # errors in the subprocesses are caught and saved in the error_queues
        # inside the context, but we don't bother to check them.
        if not cxt.join(TPU_CHECK_TIMEOUT):
            for proc in cxt.processes:
                if proc.is_alive():
                    proc.terminate()
                proc.join()

        try:
            return queue.get_nowait()
        except q.Empty:
            return None

    return wrapper


class XLADeviceUtils:
    """Used to detect the type of XLA device"""

    _TPU_AVAILABLE = False

    @staticmethod
    def _is_device_tpu() -> bool:
        """
        Check if device is TPU

        Return:
            A boolean value indicating if the xla device is a TPU device or not
        """
        if not _XLA_AVAILABLE:
            return False

        try:
            device = xm.xla_device()
            device_type = XLADeviceUtils._fetch_xla_device_type(device)
            return device_type == "TPU"

        # Missing XLA Configuration
        except RuntimeError as e:
            traceback.print_exc()
            return False

    @staticmethod
    def xla_available() -> bool:
        """
        Check if XLA library is installed

        Return:
            A boolean value indicating if a XLA is installed
        """
        return _XLA_AVAILABLE

    @staticmethod
    def tpu_device_exists() -> bool:
        """
        Runs XLA device check within a separate process

        Return:
            A boolean value indicating if a TPU device exists on the system
        """
        if os.getenv("PL_TPU_AVAILABLE", '0') == "1":
            XLADeviceUtils._TPU_AVAILABLE = True

        if XLADeviceUtils.xla_available() and not XLADeviceUtils._TPU_AVAILABLE:

            XLADeviceUtils._TPU_AVAILABLE = bool(pl_multi_process(XLADeviceUtils._is_device_tpu)())

            if XLADeviceUtils._TPU_AVAILABLE:
                os.environ["PL_TPU_AVAILABLE"] = '1'

        return XLADeviceUtils._TPU_AVAILABLE
