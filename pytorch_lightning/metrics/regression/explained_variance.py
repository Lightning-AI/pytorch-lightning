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
from typing import Any, Callable, Optional

from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.metrics.functional.explained_variance import (
    _explained_variance_update,
    _explained_variance_compute,
)


class ExplainedVariance(Metric):
    """
    Computes explained variance.

    Forward accepts

    - ``preds`` (float tensor): ``(N,)`` or ``(N, ...)`` (multioutput)
    - ``target`` (long tensor): ``(N,)`` or ``(N, ...)`` (multioutput)

    In the case of multioutput, as default the variances will be uniformly
    averaged over the additional dimensions. Please see argument `multioutput`
    for changing this behavior.

    Args:
        multioutput:
            Defines aggregation in the case of multiple output scores. Can be one
            of the following strings (default is `'uniform_average'`.):

            * `'raw_values'` returns full set of scores
            * `'uniform_average'` scores are uniformly averaged
            * `'variance_weighted'` scores are weighted by their individual variances

        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example:

        >>> from pytorch_lightning.metrics import ExplainedVariance
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> explained_variance = ExplainedVariance()
        >>> explained_variance(preds, target)
        tensor(0.9572)

        >>> target = torch.tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = torch.tensor([[0, 2], [-1, 2], [8, -5]])
        >>> explained_variance = ExplainedVariance(multioutput='raw_values')
        >>> explained_variance(preds, target)
        tensor([0.9677, 1.0000])
    """

    def __init__(
        self,
        multioutput: str = 'uniform_average',
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
        allowed_multioutput = ('raw_values', 'uniform_average', 'variance_weighted')
        if multioutput not in allowed_multioutput:
            raise ValueError(
                f'Invalid input to argument `multioutput`. Choose one of the following: {allowed_multioutput}'
            )
        self.multioutput = multioutput
        self.add_state("y", default=[], dist_reduce_fx=None)
        self.add_state("y_pred", default=[], dist_reduce_fx=None)

        rank_zero_warn(
            'Metric `ExplainedVariance` will save all targets and'
            ' predictions in buffer. For large datasets this may lead'
            ' to large memory footprint.'
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _explained_variance_update(preds, target)
        self.y_pred.append(preds)
        self.y.append(target)

    def compute(self):
        """
        Computes explained variance over state.
        """
        preds = torch.cat(self.y_pred, dim=0)
        target = torch.cat(self.y, dim=0)
        return _explained_variance_compute(preds, target, self.multioutput)
