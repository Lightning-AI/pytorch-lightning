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

from pytorch_lightning.metrics.classification.stat_scores import StatScores


class SubsetAccuracy(StatScores):
    def __init__(
        self,
        threshold: float = 0.5,
        logits: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            threshold=threshold,
            logits=logits,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            average="samples",
        )

    def compute(self):
        all_correct = ~(self.fp + self.fn > 0)
        return all_correct.float().mean()


class HammingLoss(StatScores):
    def __init__(
        self,
        threshold: float = 0.5,
        logits: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            threshold=threshold,
            logits=logits,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            average="micro",
        )

    def compute(self):
        return (self.fp.float() + self.fn) / (self.tp + self.fp + self.tn + self.fn)