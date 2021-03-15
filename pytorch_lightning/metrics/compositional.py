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
from typing import Callable, Union

import torch
from torchmetrics.metric import CompositionalMetric as _CompositionalMetric

from pytorch_lightning.metrics import Metric
from pytorch_lightning.utilities import rank_zero_warn


class CompositionalMetric(_CompositionalMetric):
    r"""
    This implementation refers to :class:`~torchmetrics.metric.CompositionalMetric`.

    .. warning:: This metric is deprecated, use ``torchmetrics.metric.CompositionalMetric``. Will be removed in v1.5.0.
    """

    def __init__(
        self,
        operator: Callable,
        metric_a: Union[Metric, int, float, torch.Tensor],
        metric_b: Union[Metric, int, float, torch.Tensor, None],
    ):
        rank_zero_warn(
            "This `Metric` was deprecated since v1.3.0 in favor of `torchmetrics.Metric`."
            " It will be removed in v1.5.0", DeprecationWarning
        )
        super().__init__(operator=operator, metric_a=metric_a, metric_b=metric_b)
