"""
Convention:
 - Do not include any `_TYPE` suffix
 - Types used in public hooks (as those in the `LightningModule` and `Callback`) should be public (no trailing `_`)
"""

from typing import Any, Dict, Iterator, List, Sequence, Union

import torch
from torchmetrics import Metric

_METRIC = Union[Metric, torch.Tensor, int, float]
STEP_OUTPUT = Union[torch.Tensor, Dict[str, Any]]
EPOCH_OUTPUT = List[STEP_OUTPUT]
_PARAMETERS = Iterator[torch.nn.Parameter]
BATCH = Union[Dict[str, Union[torch.Tensor, Any]], Sequence[Union[torch.Tensor, Any]], torch.Tensor, Any]
