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

from deprecate import deprecated
from torchmetrics import Metric as _Metric
from torchmetrics.collections import MetricCollection as _MetricCollection

from pytorch_lightning.metrics.utils import _DEPRECATION_ARGS


class Metric(_Metric):

    @deprecated(target=_Metric, **_DEPRECATION_ARGS)
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        r"""
        .. deprecated::
            Use :class:`torchmetrics.Metric`. Will be removed in v1.5.0.
        """


class MetricCollection(_MetricCollection):

    @deprecated(target=_MetricCollection, **_DEPRECATION_ARGS)
    def __init__(self, metrics: Union[List[Metric], Tuple[Metric], Dict[str, Metric]]):
        """
        .. deprecated::
            Use :class:`torchmetrics.MetricCollection`. Will be removed in v1.5.0.
        """
