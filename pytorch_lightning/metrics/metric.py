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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torchmetrics import Metric as __Metric
from torchmetrics import MetricCollection as __MetricCollection

from pytorch_lightning.utilities.distributed import rank_zero_warn


class Metric(__Metric):
    r"""
    This implementation refers to :class:`~torchmetrics.Metric`.

    .. warning:: This metric is deprecated, use ``torchmetrics.Metric``. Will be removed in v1.5.0.
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        rank_zero_warn(
            "This `Metric` was deprecated since v1.3.0 in favor of `torchmetrics.Metric`."
            " It will be removed in v1.5.0", DeprecationWarning
        )
        super(Metric, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )


class MetricCollection(__MetricCollection):
    r"""
    This implementation refers to :class:`~torchmetrics.MetricCollection`.

    .. warning:: This metric is deprecated, use ``torchmetrics.MetricCollection``. Will be removed in v1.5.0.
    """

    def __init__(self, metrics: Union[List[Metric], Tuple[Metric], Dict[str, Metric]]):
        rank_zero_warn(
            "This `MetricCollection` was deprecated since v1.3.0 in favor of `torchmetrics.MetricCollection`."
            " It will be removed in v1.5.0", DeprecationWarning
        )
        super().__init__(metrics=metrics)
