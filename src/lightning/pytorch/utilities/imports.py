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

from lightning.pytorch.utilities.rank_zero import rank_zero_warn

_PYTHON_GREATER_EQUAL_3_11_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 11)
_TORCHMETRICS_GREATER_EQUAL_0_8_0 = RequirementCache("torchmetrics>=0.8.0")
_TORCHMETRICS_GREATER_EQUAL_0_9_1 = RequirementCache("torchmetrics>=0.9.1")
_TORCHMETRICS_GREATER_EQUAL_0_11 = RequirementCache("torchmetrics>=0.11.0")  # using new API with task
_TORCHMETRICS_GREATER_EQUAL_1_0_0 = RequirementCache("torchmetrics>=1.0.0")

_OMEGACONF_AVAILABLE = package_available("omegaconf")
_TORCHVISION_AVAILABLE = RequirementCache("torchvision")


@functools.lru_cache(maxsize=128)
def _try_import_module(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    # Also on AttributeError for failed imports like pl.LightningModule
    except (ImportError, AttributeError) as err:
        rank_zero_warn(f"Import of {module_name} package failed for some compatibility issues:\n{err}")
        return False


_LIGHTNING_HABANA_AVAILABLE = RequirementCache("lightning-habana>=1.2.0")


def _habana_available_and_importable() -> bool:
    # This is defined as a function instead of a constant to avoid circular imports, because `lightning_habana`
    # also imports Lightning
    return bool(_LIGHTNING_HABANA_AVAILABLE) and _try_import_module("lightning_habana")
