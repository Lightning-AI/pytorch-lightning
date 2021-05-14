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

from abc import ABC

from pytorch_lightning.utilities.distributed import rank_zero_deprecation
from pytorch_lightning.utilities.metrics import metrics_to_scalars as new_metrics_to_scalars


class TrainerLoggingMixin(ABC):
    """
    TODO: Remove this class in v1.5.

    Use the utilities from ``pytorch_lightning.utilities.metrics`` instead.
    """

    def metrics_to_scalars(self, metrics: dict) -> dict:
        rank_zero_deprecation(
            "Internal: TrainerLoggingMixin.metrics_to_scalars is deprecated in v1.3"
            " and will be removed in v1.5."
            " Use `pytorch_lightning.utilities.metrics.metrics_to_scalars` instead."
        )
        return new_metrics_to_scalars(metrics)
