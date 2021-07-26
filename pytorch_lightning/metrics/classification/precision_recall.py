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

from torchmetrics import Precision as _Precision
from torchmetrics import Recall as _Recall

from pytorch_lightning.metrics.utils import deprecated_metrics, void


class Precision(_Precision):
    @deprecated_metrics(target=_Precision, args_mapping={"multilabel": None, "is_multiclass": None})
    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "micro",
        multilabel: bool = False,
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        is_multiclass: Optional[bool] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        """
        This implementation refers to :class:`~torchmetrics.Precision`.

        .. deprecated::
            Use :class:`~torchmetrics.Precision`. Will be removed in v1.5.0.
        """
        _ = (
            num_classes,
            threshold,
            average,
            multilabel,
            mdmc_average,
            ignore_index,
            top_k,
            is_multiclass,
            compute_on_step,
            dist_sync_on_step,
            process_group,
            dist_sync_fn,
        )


class Recall(_Recall):
    @deprecated_metrics(target=_Recall, args_mapping={"multilabel": None, "is_multiclass": None})
    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "micro",
        multilabel: bool = False,
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        is_multiclass: Optional[bool] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        """
        This implementation refers to :class:`~torchmetrics.Recall`.

        .. deprecated::
            Use :class:`~torchmetrics.Recall`. Will be removed in v1.5.0.
        """
        void(
            num_classes,
            threshold,
            average,
            multilabel,
            mdmc_average,
            ignore_index,
            top_k,
            is_multiclass,
            compute_on_step,
            dist_sync_on_step,
            process_group,
            dist_sync_fn,
        )
