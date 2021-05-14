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
from typing import Dict, Optional

import torch
from torchmetrics import Metric

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _METRIC


class MetricsHolder:
    """
    This class acts as a dictionary holder.
    It holds metrics and implements conversion functions.
    Those functions will be triggered within LoggerConnector
    when the property is being requested from the user.
    """

    def __init__(self, to_float: bool = False) -> None:
        self.metrics: Dict[str, _METRIC] = {}
        self._to_float = to_float

    def update(self, metrics: dict) -> None:
        self.metrics.update(metrics)

    def pop(self, key: str, default: _METRIC) -> _METRIC:
        return self.metrics.pop(key, default)

    def reset(self, metrics: Dict[str, _METRIC]) -> None:
        self.metrics = metrics

    def convert(self, device: Optional[torch.device]) -> None:
        for key, value in self.metrics.items():
            if self._to_float:
                if isinstance(value, torch.Tensor) and value.numel() != 1:
                    raise MisconfigurationException(
                        f"The metric `{key}` does not contain a single element"
                        f" thus it cannot be converted to float. Found `{value}`"
                    )
                converted = self._convert_to_float(value)
            else:
                converted = self._convert_to_tensor(value, device)
            self.metrics[key] = converted

    @staticmethod
    def _convert_to_float(current: _METRIC) -> float:
        if isinstance(current, Metric):
            current = current.compute().detach()

        if isinstance(current, torch.Tensor):
            current = float(current.item())

        elif isinstance(current, int):
            current = float(current)

        return current

    @staticmethod
    def _convert_to_tensor(current: _METRIC, device: Optional[torch.device]) -> torch.Tensor:
        if isinstance(current, Metric):
            current = current.compute().detach()

        elif isinstance(current, numbers.Number):
            current = torch.tensor(current, device=device, dtype=torch.float)

        if isinstance(current, torch.Tensor) and current.device.type == "xla":
            current = current.cpu()

        return current
