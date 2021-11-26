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
"""General utilities."""
import importlib
import operator
import os
import platform
import sys
from importlib.util import find_spec
from typing import Callable

import pkg_resources
import torch
from packaging.version import Version
from pkg_resources import DistributionNotFound


def _module_available(module_path: str) -> bool:
    """Check if a path is available in your environment.

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


def _compare_version(package: str, op: Callable, version: str, use_base_version: bool = False) -> bool:
    """Compare package version with some requirements.

    >>> _compare_version("torch", operator.ge, "0.1")
    True
    """
    try:
        pkg = importlib.import_module(package)
    except (ModuleNotFoundError, DistributionNotFound):
        return False
    try:
        if hasattr(pkg, "__version__"):
            pkg_version = Version(pkg.__version__)
        else:
            # try pkg_resources to infer version
            pkg_version = Version(pkg_resources.get_distribution(package).version)
    except TypeError:
        # this is mocked by Sphinx, so it should return True to generate all summaries
        return True
    if use_base_version:
        pkg_version = Version(pkg_version.base_version)
    return op(pkg_version, Version(version))


_IS_WINDOWS = platform.system() == "Windows"
_IS_INTERACTIVE = hasattr(sys, "ps1")  # https://stackoverflow.com/a/64523765
_TORCH_GREATER_EQUAL_1_7 = _compare_version("torch", operator.ge, "1.7.0")
_TORCH_GREATER_EQUAL_1_8 = _compare_version("torch", operator.ge, "1.8.0")
_TORCH_GREATER_EQUAL_1_8_1 = _compare_version("torch", operator.ge, "1.8.1")
_TORCH_GREATER_EQUAL_1_9 = _compare_version("torch", operator.ge, "1.9.0")
_TORCH_GREATER_EQUAL_1_10 = _compare_version("torch", operator.ge, "1.10.0")
# _TORCH_GREATER_EQUAL_DEV_1_11 = _compare_version("torch", operator.ge, "1.11.0", use_base_version=True)

_APEX_AVAILABLE = _module_available("apex.amp")
_DEEPSPEED_AVAILABLE = _module_available("deepspeed")
_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and _module_available("fairscale.nn")
_FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE = _FAIRSCALE_AVAILABLE and _compare_version("fairscale", operator.ge, "0.3.3")
_FAIRSCALE_FULLY_SHARDED_AVAILABLE = _FAIRSCALE_AVAILABLE and _compare_version("fairscale", operator.ge, "0.3.4")
_GROUP_AVAILABLE = not _IS_WINDOWS and _module_available("torch.distributed.group")
_HOROVOD_AVAILABLE = _module_available("horovod.torch")
_HYDRA_AVAILABLE = _module_available("hydra")
_HYDRA_EXPERIMENTAL_AVAILABLE = _module_available("hydra.experimental")
_JSONARGPARSE_AVAILABLE = _module_available("jsonargparse")
_KINETO_AVAILABLE = _TORCH_GREATER_EQUAL_1_8_1 and torch.profiler.kineto_available()
_NEPTUNE_AVAILABLE = _module_available("neptune")
_NEPTUNE_GREATER_EQUAL_0_9 = _NEPTUNE_AVAILABLE and _compare_version("neptune", operator.ge, "0.9.0")
_OMEGACONF_AVAILABLE = _module_available("omegaconf")
_POPTORCH_AVAILABLE = _module_available("poptorch")
_RICH_AVAILABLE = _module_available("rich") and _compare_version("rich", operator.ge, "10.2.2")
_TORCH_QUANTIZE_AVAILABLE = bool([eg for eg in torch.backends.quantized.supported_engines if eg != "none"])
_TORCHTEXT_AVAILABLE = _module_available("torchtext")
_TORCHVISION_AVAILABLE = _module_available("torchvision")
_XLA_AVAILABLE: bool = _module_available("torch_xla")

from pytorch_lightning.utilities.xla_device import XLADeviceUtils  # noqa: E402

_TPU_AVAILABLE = XLADeviceUtils.tpu_device_exists()

if _POPTORCH_AVAILABLE:
    import poptorch

    _IPU_AVAILABLE = poptorch.ipuHardwareIsAvailable()
else:
    _IPU_AVAILABLE = False


# experimental feature within PyTorch Lightning.
def _fault_tolerant_training() -> bool:
    return _TORCH_GREATER_EQUAL_1_7 and int(os.getenv("PL_FAULT_TOLERANT_TRAINING", 0))
