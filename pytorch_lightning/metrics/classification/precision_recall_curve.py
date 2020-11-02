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
from typing import Optional, Any

import torch

from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional.precision_recall_curve import (
    _precision_recall_curve_update,
    _precision_recall_curve_compute
)


class PrecisionRecallCurve(Metric):
    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        multilabel: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.num_classes = num_classes
        self.threshold = threshold
        self.multilabel = multilabel

        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("predicted_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        _precision_recall_curve_update(preds, target)

    def compute(self):
        return _precision_recall_curve_compute()