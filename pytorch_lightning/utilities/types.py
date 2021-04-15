from typing import Any, Union

import torch
from torchmetrics import Metric

_METRIC_TYPE = Union[Metric, torch.Tensor, int, float]
