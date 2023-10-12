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
import functools
import sys

from lightning_utilities.core.imports import RequirementCache, package_available
from lightning_utilities.core.rank_zero import rank_zero_warn

_PYTHON_GREATER_EQUAL_3_11_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 11)
_TORCHMETRICS_GREATER_EQUAL_0_9_1 = RequirementCache("torchmetrics>=0.9.1")
_TORCHMETRICS_GREATER_EQUAL_0_11 = RequirementCache("torchmetrics>=0.11.0")  # using new API with task
_TORCHMETRICS_GREATER_EQUAL_1_0_0 = RequirementCache("torchmetrics>=1.0.0")

_OMEGACONF_AVAILABLE = package_available("omegaconf")
_TORCHVISION_AVAILABLE = RequirementCache("torchvision")
_LIGHTNING_COLOSSALAI_AVAILABLE = RequirementCache("lightning-colossalai")
_LIGHTNING_BAGUA_AVAILABLE = RequirementCache("lightning-bagua")


@functools.lru_cache(maxsize=128)
def _try_import_module(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    # added also AttributeError fro case of impoerts like pl.LightningModule
    except (ImportError, AttributeError) as err:
        rank_zero_warn(f"Import of {module_name} package failed for some compatibility issues: \n{err}")
        return False


@functools.lru_cache(maxsize=1)
def _lightning_graphcore_available() -> bool:
    # This is defined as a function instead of a constant to avoid circular imports, because `lightning_graphcore`
    # also imports Lightning
    return bool(RequirementCache("lightning-graphcore")) and _try_import_module("lightning_graphcore")


@functools.lru_cache(maxsize=1)
def _lightning_habana_available() -> bool:
    # This is defined as a function instead of a constant to avoid circular imports, because `lightning_habana`
    # also imports Lightning
    return bool(RequirementCache("lightning-habana")) and _try_import_module("lightning_habana")
