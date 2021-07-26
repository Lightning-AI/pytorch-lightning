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
from torchmetrics import Metric
from torchmetrics.metric import CompositionalMetric as _CompositionalMetric

from pytorch_lightning.metrics.utils import deprecated_metrics, void


class CompositionalMetric(_CompositionalMetric):
    @deprecated_metrics(target=_CompositionalMetric)
    def __init__(
        self,
        operator: Callable,
        metric_a: Union[Metric, int, float, torch.Tensor],
        metric_b: Union[Metric, int, float, torch.Tensor, None],
    ):
        """
        .. deprecated::
            Use :class:`torchmetrics.metric.CompositionalMetric`. Will be removed in v1.5.0.
        """
        void(operator, metric_a, metric_b)
