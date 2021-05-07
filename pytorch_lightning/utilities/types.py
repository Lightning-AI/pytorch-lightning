from typing import Any, Dict, Iterator, List, Sequence, Union

import torch
from torch.utils.data import DataLoader
from torchmetrics import Metric
"""
Convention:
 - Do not include any `_TYPE` suffix
 - Types used in public hooks (as those in the `LightningModule` and `Callback`) should be public (no trailing `_`)
"""
_METRIC = Union[Metric, torch.Tensor, int, float]
STEP_OUTPUT = Union[torch.Tensor, Dict[str, Any]]
EPOCH_OUTPUT = List[STEP_OUTPUT]
_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader
_PREDICT_OUTPUT = Union[List[Any], List[List[Any]]]
_PARAMETERS = Iterator[torch.nn.Parameter]
# TODO: expand allowed train_dataloaders types once fixed.
TRAIN_DATALOADERS = Union[DataLoader, Sequence[DataLoader], Dict[str, DataLoader], 'CombinedLoader']
EVAL_DATALOADERS = Union[DataLoader, Sequence[DataLoader]]
