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
from typing import Any, Dict, List, Tuple

from lightning_utilities.core.imports import RequirementCache

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning.fabric.utilities.testing import _RunIf as FabricRunIf
from lightning.pytorch.accelerators.cpu import _PSUTIL_AVAILABLE
from lightning.pytorch.callbacks.progress.rich_progress import _RICH_AVAILABLE
from lightning.pytorch.core.module import _ONNX_AVAILABLE
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE

_SKLEARN_AVAILABLE = RequirementCache("scikit-learn")


def _RunIf(
    *,
    rich: bool = False,
    omegaconf: bool = False,
    psutil: bool = False,
    sklearn: bool = False,
    onnx: bool = False,
    **kwargs: Any
) -> Tuple[List[str], Dict[str, bool]]:
    """
    Args:
        rich: Require that willmcgugan/rich is installed.
        omegaconf: Require that omry/omegaconf is installed.
        psutil: Require that psutil is installed.
        sklearn: Require that scikit-learn is installed.
        onnx: Require that onnx is installed.
    """

    reasons, kwargs = FabricRunIf(**kwargs)

    if rich and not _RICH_AVAILABLE:
        reasons.append("Rich")

    if omegaconf and not _OMEGACONF_AVAILABLE:
        reasons.append("omegaconf")

    if psutil and not _PSUTIL_AVAILABLE:
        reasons.append("psutil")

    if sklearn and not _SKLEARN_AVAILABLE:
        reasons.append("scikit-learn")

    if onnx and _TORCH_GREATER_EQUAL_2_0 and not _ONNX_AVAILABLE:
        reasons.append("onnx")

    return reasons, kwargs
