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

from typing import Iterable

import torch
from torch.optim import Optimizer

from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from pytorch_lightning.utilities.types import _DEVICE


def optimizers_to_device(optimizers: Iterable[Optimizer], device: _DEVICE) -> None:
    """Moves optimizer states for a sequence of optimizers to the device."""
    for opt in optimizers:
        optimizer_to_device(opt, device)


def optimizer_to_device(optimizer: Optimizer, device: _DEVICE) -> None:
    """Moves the state of a single optimizer to the device."""
    for p, v in optimizer.state.items():
        optimizer.state[p] = apply_to_collection(v, torch.Tensor, move_data_to_device, device)
