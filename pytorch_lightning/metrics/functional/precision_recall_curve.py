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
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torchmetrics.functional import precision_recall_curve as _precision_recall_curve

from pytorch_lightning.utilities.deprecation import deprecated


@deprecated(target=_precision_recall_curve, ver_deprecate="1.3.0", ver_remove="1.5.0")
def precision_recall_curve(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    sample_weights: Optional[Sequence] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor],
                                                                  List[torch.Tensor]], ]:
    """
    .. deprecated::
        Use :func:`torchmetrics.functional.accuracy`. Will be removed in v1.5.0.
    """
