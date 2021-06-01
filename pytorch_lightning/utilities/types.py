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
"""
Convention:
 - Do not include any `_TYPE` suffix
 - Types used in public hooks (as those in the `LightningModule` and `Callback`) should be public (no trailing `_`)
"""
from numbers import Number
from typing import Any, Dict, Iterator, List, Union

import torch
from torchmetrics import Metric

_METRIC = Union[Metric, torch.Tensor, Number]
# real type is `Union[_METRIC, Dict[str, '_METRIC_COLLECTION']]` but Sphinx fails with `RecursionError`
_METRIC_COLLECTION = Union[_METRIC, Dict[str, _METRIC]]
STEP_OUTPUT = Union[torch.Tensor, Dict[str, Any]]
EPOCH_OUTPUT = List[STEP_OUTPUT]
_EVALUATE_OUTPUT = List[Dict[str, float]]  # 1 dict per DataLoader
_PREDICT_OUTPUT = Union[List[Any], List[List[Any]]]
_PARAMETERS = Iterator[torch.nn.Parameter]
