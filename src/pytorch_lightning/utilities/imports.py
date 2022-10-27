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
import operator
import platform
import sys

import torch
from lightning_utilities.core.imports import compare_version, module_available, package_available, RequirementCache

_IS_WINDOWS = platform.system() == "Windows"
_PYTHON_GREATER_EQUAL_3_8_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 8)
_PYTHON_GREATER_EQUAL_3_10_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 10)
_TORCH_GREATER_EQUAL_1_9_1 = compare_version("torch", operator.ge, "1.9.1")
_TORCH_GREATER_EQUAL_1_10 = compare_version("torch", operator.ge, "1.10.0")
_TORCH_LESSER_EQUAL_1_10_2 = compare_version("torch", operator.le, "1.10.2")
_TORCH_GREATER_EQUAL_1_11 = compare_version("torch", operator.ge, "1.11.0")
_TORCH_GREATER_EQUAL_1_12 = compare_version("torch", operator.ge, "1.12.0")
_TORCH_GREATER_EQUAL_1_13 = compare_version("torch", operator.ge, "1.13.0", use_base_version=True)

_APEX_AVAILABLE = module_available("apex.amp")
_DALI_AVAILABLE = module_available("nvidia.dali")
_HABANA_FRAMEWORK_AVAILABLE = package_available("habana_frameworks")
_HIVEMIND_AVAILABLE = package_available("hivemind")
_HOROVOD_AVAILABLE = module_available("horovod.torch")
_KINETO_AVAILABLE = torch.profiler.kineto_available()
_OMEGACONF_AVAILABLE = package_available("omegaconf")
_POPTORCH_AVAILABLE = package_available("poptorch")
_PSUTIL_AVAILABLE = package_available("psutil")
_RICH_AVAILABLE = package_available("rich") and compare_version("rich", operator.ge, "10.2.2")
_TORCH_QUANTIZE_AVAILABLE = bool([eg for eg in torch.backends.quantized.supported_engines if eg != "none"])
_TORCHVISION_AVAILABLE = RequirementCache("torchvision")

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
