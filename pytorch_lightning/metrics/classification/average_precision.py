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
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.metrics.functional.average_precision import (
    _average_precision_update,
    _average_precision_compute
)


class AveragePrecision(Metric):
    def __init__(
        self,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
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
        self.pos_label = pos_label

        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

        rank_zero_warn(
            'Metric `AveragePrecision` will save all targets and'
            ' predictions in buffer. For large datasets this may lead'
            ' to large memory footprint.'
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target, num_classes, pos_label = _average_precision_update(
            preds,
            target,
            self.num_classes,
            self.pos_label
        )
        self.preds.append(preds)
        self.target.append(target)
        self.num_classes = num_classes
        self.pos_label = pos_label

    def compute(self):
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        return _average_precision_compute(preds, target, self.num_classes, self.pos_label)
