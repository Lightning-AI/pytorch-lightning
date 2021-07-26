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

from torchmetrics import ROC as _ROC

from pytorch_lightning.metrics.utils import deprecated_metrics, void


class ROC(_ROC):
    @deprecated_metrics(target=_ROC)
    def __init__(
        self,
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        """
        This implementation refers to :class:`~torchmetrics.ROC`.

        .. deprecated::
            Use :class:`~torchmetrics.ROC`. Will be removed in v1.5.0.
        """
        void(num_classes, pos_label, compute_on_step, dist_sync_on_step, process_group)
