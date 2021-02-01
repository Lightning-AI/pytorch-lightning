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
from typing import Any, Callable, Optional

import torch

from pytorch_lightning.metrics.functional.auc import _auc_compute, _auc_update
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.utilities import rank_zero_warn


class AUC(Metric):
    r"""
    Computes Area Under the Curve (AUC) using the trapezoidal rule

    Forward accepts two input tensors that should be 1D and have the same number
    of elements

    Args:
        reorder: AUC expects its first input to be sorted. If this is not the case,
            setting this argument to ``True`` will use a stable sorting algorithm to
            sort the input in decending order
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather
    """

    def __init__(
        self,
        reorder: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.reorder = reorder

        self.add_state("x", default=[], dist_reduce_fx=None)
        self.add_state("y", default=[], dist_reduce_fx=None)

        rank_zero_warn(
            'Metric `AUC` will save all targets and predictions in buffer.'
            ' For large datasets this may lead to large memory footprint.'
        )

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            x: Predictions from model (probabilities, or labels)
            y: Ground truth labels
        """
        x, y = _auc_update(x, y)

        self.x.append(x)
        self.y.append(y)

    def compute(self) -> torch.Tensor:
        """
        Computes AUC based on inputs passed in to ``update`` previously.
        """
        x = torch.cat(self.x, dim=0)
        y = torch.cat(self.y, dim=0)
        return _auc_compute(x, y, reorder=self.reorder)
