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
from typing import Sequence, Tuple, Optional

import torch

def _auroc_update():
    pass


def _auroc_compute() -> torch.Tensor:
    pass
    
    
def auroc(
        preds: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[Sequence] = None,
        num_classes: Optional[int] = None,
        average: Optional[str] = 'macro',
        max_fpr: Optional[float] = None,
        multi_class: str = 'raise'
) -> torch.Tensor:
    """ Compute `Area Under the Receiver Operating Characteristic Curve (ROC AUC) 
    <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Further_interpretations>`_
    
    Example (binary case):
        
        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> auroc(pred, target, pos_label=1)
        tensor(1.)
        
    Example (multiclass case):
        
        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> auroc(pred, target, num_classes=5)
        [tensor(1.), tensor(1.), tensor(0.2500), tensor(0.2500), tensor(nan)]
    
    """
    preds, target = _auroc_update(preds, target)
    #Table_of_confusion