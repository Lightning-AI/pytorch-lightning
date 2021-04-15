from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader, IterableDataset
from torchmetrics import Metric

_METRIC_TYPE = Union[Metric, torch.Tensor, int, float]
_STEP_OUTPUT_TYPE = Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]
