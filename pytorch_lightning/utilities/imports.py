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
"""General utilities"""
import importlib
import operator
import platform
import sys
from importlib.util import find_spec

import torch
from packaging.version import Version
from pkg_resources import DistributionNotFound


def _module_available(module_path: str) -> bool:
    """
    Check if a path is available in your environment

    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False


def _compare_version(package: str, op, version) -> bool:
    """
    Compare package version with some requirements

    >>> _compare_version("torch", operator.ge, "0.1")
    True
    """
    try:
        pkg = importlib.import_module(package)
    except (ModuleNotFoundError, DistributionNotFound):
        return False
    try:
        pkg_version = Version(pkg.__version__)
    except TypeError:
        # this is mock by sphinx, so it shall return True ro generate all summaries
        return True
    return op(pkg_version, Version(version))


_IS_WINDOWS = platform.system() == "Windows"
_IS_INTERACTIVE = hasattr(sys, "ps1")  # https://stackoverflow.com/a/64523765
_TORCH_LOWER_EQUAL_1_4 = _compare_version("torch", operator.le, "1.5.0")
_TORCH_GREATER_EQUAL_1_5 = _compare_version("torch", operator.ge, "1.5.0")
_TORCH_GREATER_EQUAL_1_6 = _compare_version("torch", operator.ge, "1.6.0")
_TORCH_GREATER_EQUAL_1_7 = _compare_version("torch", operator.ge, "1.7.0")
_TORCH_GREATER_EQUAL_1_8 = _compare_version("torch", operator.ge, "1.8.0")
_TORCH_GREATER_EQUAL_1_8_1 = _compare_version("torch", operator.ge, "1.8.1")
_TORCH_GREATER_EQUAL_1_9 = _compare_version("torch", operator.ge, "1.9.0")

_APEX_AVAILABLE = _module_available("apex.amp")
_BOLTS_AVAILABLE = _module_available('pl_bolts')
_DEEPSPEED_AVAILABLE = not _IS_WINDOWS and _module_available('deepspeed')
_FAIRSCALE_AVAILABLE = _TORCH_GREATER_EQUAL_1_6 and not _IS_WINDOWS and _module_available('fairscale.nn')
_FAIRSCALE_PIPE_AVAILABLE = _FAIRSCALE_AVAILABLE and _compare_version("fairscale", operator.le, "0.1.3")
_FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE = _FAIRSCALE_AVAILABLE and _compare_version("fairscale", operator.ge, "0.3.3")
_FAIRSCALE_FULLY_SHARDED_AVAILABLE = _FAIRSCALE_AVAILABLE and _compare_version("fairscale", operator.ge, "0.3.4")
_GROUP_AVAILABLE = not _IS_WINDOWS and _module_available('torch.distributed.group')
_HOROVOD_AVAILABLE = _module_available("horovod.torch")
_HYDRA_AVAILABLE = _module_available("hydra")
_HYDRA_EXPERIMENTAL_AVAILABLE = _module_available("hydra.experimental")
_KINETO_AVAILABLE = _TORCH_GREATER_EQUAL_1_8_1 and torch.profiler.kineto_available()
_NATIVE_AMP_AVAILABLE = _module_available("torch.cuda.amp") and hasattr(torch.cuda.amp, "autocast")
_OMEGACONF_AVAILABLE = _module_available("omegaconf")
_RPC_AVAILABLE = not _IS_WINDOWS and _module_available('torch.distributed.rpc')
_TORCH_QUANTIZE_AVAILABLE = bool([eg for eg in torch.backends.quantized.supported_engines if eg != 'none'])
_TORCHTEXT_AVAILABLE = _module_available("torchtext")
_TORCHVISION_AVAILABLE = _module_available('torchvision')
_TORCHMETRICS_LOWER_THAN_0_3 = _compare_version("torchmetrics", operator.lt, "0.3.0")
_TORCHMETRICS_GREATER_EQUAL_0_3 = _compare_version("torchmetrics", operator.ge, "0.3.0")
_XLA_AVAILABLE = _module_available("torch_xla")

from pytorch_lightning.utilities.xla_device import XLADeviceUtils  # noqa: E402

_TPU_AVAILABLE = XLADeviceUtils.tpu_device_exists()
