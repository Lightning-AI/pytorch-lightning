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
"""General utilities."""
import operator
import sys

import torch
from lightning_utilities.core.imports import compare_version, package_available, RequirementCache

_PYTHON_GREATER_EQUAL_3_8_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 8)
_PYTHON_GREATER_EQUAL_3_10_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 10)
_PYTHON_GREATER_EQUAL_3_11_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 11)
# duplicated from fabric because HPU is patching it below
_TORCH_GREATER_EQUAL_1_13 = compare_version("torch", operator.ge, "1.13.0")
_TORCHMETRICS_GREATER_EQUAL_0_9_1 = RequirementCache("torchmetrics>=0.9.1")

_KINETO_AVAILABLE = torch.profiler.kineto_available()
_OMEGACONF_AVAILABLE = package_available("omegaconf")
_POPTORCH_AVAILABLE = package_available("poptorch")
_PSUTIL_AVAILABLE = package_available("psutil")
_RICH_AVAILABLE = package_available("rich") and compare_version("rich", operator.ge, "10.2.2")
_TORCHVISION_AVAILABLE = RequirementCache("torchvision")
_LIGHTNING_COLOSSALAI_AVAILABLE = RequirementCache("lightning-colossalai")

if _POPTORCH_AVAILABLE:
    import poptorch

    _IPU_AVAILABLE = poptorch.ipuHardwareIsAvailable()
else:
    _IPU_AVAILABLE = False
