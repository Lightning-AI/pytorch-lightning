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
from typing import Any, Optional

from torchmetrics import F1 as _F1
from torchmetrics import FBeta as _FBeta

from pytorch_lightning.metrics.utils import deprecated_metrics, void


class FBeta(_FBeta):
    @deprecated_metrics(target=_FBeta, args_mapping={"multilabel": None})
    def __init__(
        self,
        num_classes: int,
        beta: float = 1.0,
        threshold: float = 0.5,
        average: str = "micro",
        multilabel: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        """
        This implementation refers to :class:`~torchmetrics.FBeta`.

        .. deprecated::
            Use :class:`~torchmetrics.FBeta`. Will be removed in v1.5.0.
        """
        _ = num_classes, beta, threshold, average, multilabel, compute_on_step, dist_sync_on_step, process_group


class F1(_F1):
    @deprecated_metrics(target=_F1, args_mapping={"multilabel": None})
    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        average: str = "micro",
        multilabel: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        """
        This implementation refers to :class:`~torchmetrics.F1`.

        .. deprecated::
            Use :class:`~torchmetrics.F1`. Will be removed in v1.5.0.
        """
        void(num_classes, threshold, average, multilabel, compute_on_step, dist_sync_on_step, process_group)
