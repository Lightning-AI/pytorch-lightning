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


class MetricsHolder:
    """
    This class acts as a dictonary holder.
    It holds metrics and implements conversion functions.
    Those functions will be triggered within LoggerConnector
    when the property is being requested from the user.
    """

    def __init__(self, to_float: bool = False):
        self.metrics = {}
        self._to_float = to_float

    def update(self, metrics):
        self.metrics.update(metrics)

    def pop(self, key, default):
        return self.metrics.pop(key, default)

    def reset(self, metrics):
        self.metrics = metrics

    def convert(self, use_tpu: bool, device: torch.device):
        for key, value in self.metrics.items():
            self.metrics[key] = self._convert(value, use_tpu, device)

    def _convert(self, current: Any, use_tpu: bool, device: torch.device):
        if self._to_float:
            return self._convert_to_float(current, use_tpu, device)
        return self._convert_to_tensor(current, use_tpu, device)

    def _convert_to_float(self, current, use_tpu: bool, device: torch.device):
        if isinstance(current, Metric):
            current = current.compute().detach()

        if isinstance(current, torch.Tensor):
            current = float(current.item())

        elif isinstance(current, int):
            current = float(current)

        return current

    def _convert_to_tensor(self, current: Any, use_tpu: bool, device: torch.device):
        if current is not None:
            if isinstance(current, Metric):
                current = current.compute().detach()

            elif isinstance(current, numbers.Number):
                if device is None:
                    current = torch.tensor(current, dtype=torch.float)
                else:
                    current = torch.tensor(current, device=device, dtype=torch.float)

        if isinstance(current, torch.Tensor) and current.device.type == "xla":
            current = current.cpu()

        return current
