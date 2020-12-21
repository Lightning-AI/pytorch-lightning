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
import numbers
from typing import Any

import torch

from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.utilities import TPU_AVAILABLE


class MetricsHolder:

    """
    This class hold metris and triggers convert when trying to access them
    """

    def __init__(self):
        self.metrics = {}

    def update(self, metrics):
        self.metrics.update(metrics)

    def reset(self, metrics):
        self.metrics = metrics

    def convert(self, use_tpu: bool, device: torch.device):
        for key, value in self.metrics.items():
            self.metrics[key] = self._convert(value, use_tpu, device)

    def _convert(self, current: Any, use_tpu: bool, device: torch.device):
        if current is not None:
            if isinstance(current, Metric):
                current = current.compute()
            elif isinstance(current, numbers.Number):
                current = torch.tensor(current, device=device, dtype=torch.float)

        if use_tpu and TPU_AVAILABLE:
            current = current.cpu()

        return current
