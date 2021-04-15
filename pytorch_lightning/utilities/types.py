from typing import Dict, Iterator, Optional, Union

import torch
from torchmetrics import Metric

_METRIC = Union[Metric, torch.Tensor, int, float]
_STEP_OUTPUT = Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]
_PARAMETERS = Iterator[torch.nn.Parameter]
