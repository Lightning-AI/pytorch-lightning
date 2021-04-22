from typing import Any, Iterator, List, Mapping, Union

import torch
from torchmetrics import Metric
"""
Convention:
 - Do not include any `_TYPE` suffix
 - Types used in public hooks (as those in the `LightningModule` and `Callback`) should be public (no trailing `_`)
"""
_METRIC = Union[Metric, torch.Tensor, int, float]
STEP_OUTPUT = Union[torch.Tensor, Mapping[str, Any]]
EPOCH_OUTPUT = List[STEP_OUTPUT]
_PARAMETERS = Iterator[torch.nn.Parameter]
