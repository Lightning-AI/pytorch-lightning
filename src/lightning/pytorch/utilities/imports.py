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
import sys

import torch
from lightning_utilities.core.imports import package_available, RequirementCache
from lightning_utilities.core.rank_zero import rank_zero_warn

_PYTHON_GREATER_EQUAL_3_11_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 11)
_TORCHMETRICS_GREATER_EQUAL_0_9_1 = RequirementCache("torchmetrics>=0.9.1")
_TORCHMETRICS_GREATER_EQUAL_0_11 = RequirementCache("torchmetrics>=0.11.0")  # using new API with task

_KINETO_AVAILABLE = torch.profiler.kineto_available()
_OMEGACONF_AVAILABLE = package_available("omegaconf")
_TORCHVISION_AVAILABLE = RequirementCache("torchvision")
_LIGHTNING_COLOSSALAI_AVAILABLE = RequirementCache("lightning-colossalai")
_LIGHTNING_BAGUA_AVAILABLE = RequirementCache("lightning-bagua")

_LIGHTNING_GRAPHCORE_AVAILABLE = False
if RequirementCache("lightning-graphcore"):
    try:
        __import__("lightning_graphcore")
        _LIGHTNING_GRAPHCORE_AVAILABLE = True
    except ImportError as err:
        rank_zero_warn(f"Import of Graphcore package failed for some compatibility issues: \n{err}")

_LIGHTNING_HABANA_AVAILABLE = False
if RequirementCache("lightning-habana"):
    try:
        __import__("lightning_habana")
        _LIGHTNING_HABANA_AVAILABLE = True
    except ImportError as err:
        rank_zero_warn(f"Import of Habana package failed for some compatibility issues: \n{err}")
