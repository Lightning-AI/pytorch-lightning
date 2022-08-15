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
import platform
import sys
from importlib.util import find_spec
from typing import Callable

import pkg_resources
import torch
from packaging.version import Version
from pkg_resources import DistributionNotFound


def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> _package_available('os')
    True
    >>> _package_available('bla')
    False
    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


def _module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.

    >>> _module_available('os')
    True
    >>> _module_available('os.bla')
    False
    >>> _module_available('bla.bla')
    False
    """
    module_names = module_path.split(".")
    if not _package_available(module_names[0]):
        return False
    try:
        importlib.import_module(module_path)
    except ImportError:
        return False
    return True


def _compare_version(package: str, op: Callable, version: str, use_base_version: bool = False) -> bool:
    """Compare package version with some requirements.

    >>> _compare_version("torch", operator.ge, "0.1")
    True
    >>> _compare_version("does_not_exist", operator.ge, "0.0")
    False
    """
    try:
        pkg = importlib.import_module(package)
    except (ImportError, DistributionNotFound):
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


class _RequirementAvailable:
    """Boolean-like class for check of requirement with extras and version specifiers.

    >>> _RequirementAvailable("torch>=0.1")
    Requirement 'torch>=0.1' met
    >>> bool(_RequirementAvailable("torch>=0.1"))
    True
    >>> bool(_RequirementAvailable("torch>100.0"))
    False
    """

    def __init__(self, requirement: str) -> None:
        self.requirement = requirement

    def _check_requirement(self) -> None:
        if not hasattr(self, "available"):
            try:
                pkg_resources.require(self.requirement)
                self.available = True
                self.message = f"Requirement {self.requirement!r} met"
            except Exception as ex:
                self.available = False
                self.message = f"Requirement {self.requirement!r} not met, {ex.__class__.__name__}: {ex}"

    def __bool__(self) -> bool:
        self._check_requirement()
        return self.available

    def __str__(self) -> str:
        self._check_requirement()
        return self.message

    def __repr__(self) -> str:
        return self.__str__()


_IS_WINDOWS = platform.system() == "Windows"
_IS_INTERACTIVE = hasattr(sys, "ps1")  # https://stackoverflow.com/a/64523765
_PYTHON_GREATER_EQUAL_3_8_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 8)
_PYTHON_GREATER_EQUAL_3_10_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 10)
_TORCH_GREATER_EQUAL_1_9_1 = _compare_version("torch", operator.ge, "1.9.1")
_TORCH_GREATER_EQUAL_1_10 = _compare_version("torch", operator.ge, "1.10.0")
_TORCH_LESSER_EQUAL_1_10_2 = _compare_version("torch", operator.le, "1.10.2")
_TORCH_GREATER_EQUAL_1_11 = _compare_version("torch", operator.ge, "1.11.0")
_TORCH_GREATER_EQUAL_1_12 = _compare_version("torch", operator.ge, "1.12.0")
_TORCH_GREATER_EQUAL_1_13 = _compare_version("torch", operator.ge, "1.13.0", use_base_version=True)

_APEX_AVAILABLE = _module_available("apex.amp")
_DALI_AVAILABLE = _module_available("nvidia.dali")
_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and _module_available("fairscale.nn")
_FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE = _FAIRSCALE_AVAILABLE and _compare_version("fairscale", operator.ge, "0.3.3")
_FAIRSCALE_FULLY_SHARDED_AVAILABLE = _FAIRSCALE_AVAILABLE and _compare_version("fairscale", operator.ge, "0.3.4")
_GROUP_AVAILABLE = not _IS_WINDOWS and _module_available("torch.distributed.group")
_HABANA_FRAMEWORK_AVAILABLE = _package_available("habana_frameworks")
_HIVEMIND_AVAILABLE = _package_available("hivemind")
_HOROVOD_AVAILABLE = _module_available("horovod.torch")
_HYDRA_AVAILABLE = _package_available("hydra")
_HYDRA_EXPERIMENTAL_AVAILABLE = _module_available("hydra.experimental")
_KINETO_AVAILABLE = torch.profiler.kineto_available()
_OMEGACONF_AVAILABLE = _package_available("omegaconf")
_POPTORCH_AVAILABLE = _package_available("poptorch")
_PSUTIL_AVAILABLE = _package_available("psutil")
_RICH_AVAILABLE = _package_available("rich") and _compare_version("rich", operator.ge, "10.2.2")
_TORCH_QUANTIZE_AVAILABLE = bool([eg for eg in torch.backends.quantized.supported_engines if eg != "none"])
_TORCHTEXT_AVAILABLE = _package_available("torchtext")
_TORCHTEXT_LEGACY: bool = _TORCHTEXT_AVAILABLE and _compare_version("torchtext", operator.lt, "0.11.0")
_TORCHVISION_AVAILABLE = _package_available("torchvision")
_XLA_AVAILABLE: bool = _package_available("torch_xla")


from pytorch_lightning.utilities.xla_device import XLADeviceUtils  # noqa: E402

_TPU_AVAILABLE = XLADeviceUtils.tpu_device_exists()

if _POPTORCH_AVAILABLE:
    import poptorch

    _IPU_AVAILABLE = poptorch.ipuHardwareIsAvailable()
else:
    _IPU_AVAILABLE = False

if _HABANA_FRAMEWORK_AVAILABLE:
    from habana_frameworks.torch.utils.library_loader import is_habana_avaialble

    _HPU_AVAILABLE = is_habana_avaialble()
else:
    _HPU_AVAILABLE = False


# experimental feature within PyTorch Lightning.
def _fault_tolerant_training() -> bool:
    from pytorch_lightning.utilities.enums import _FaultTolerantMode

    return _FaultTolerantMode.detect_current_mode().is_enabled
