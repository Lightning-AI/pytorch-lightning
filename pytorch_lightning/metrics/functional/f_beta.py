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
import torch
from torchmetrics.functional import f1 as _f1
from torchmetrics.functional import fbeta as _fbeta

from pytorch_lightning.utilities.deprecation import deprecated


@deprecated(target=_fbeta, ver_deprecate="1.3.0", ver_remove="1.5.0")
def fbeta(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    beta: float = 1.0,
    threshold: float = 0.5,
    average: str = "micro",
    multilabel: bool = False
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.accuracy`. Will be removed in v1.5.0.
    """


@deprecated(target=_f1, ver_deprecate="1.3.0", ver_remove="1.5.0")
def f1(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    threshold: float = 0.5,
    average: str = "micro",
    multilabel: bool = False
) -> torch.Tensor:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.f1`. Will be removed in v1.5.0.
    """
