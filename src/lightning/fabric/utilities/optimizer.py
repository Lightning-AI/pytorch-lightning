# Copyright The Lightning AI team.
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

from collections.abc import Iterable, MutableMapping

from torch import Tensor
from torch.optim import Optimizer

from lightning.fabric.utilities.apply_func import apply_to_collection, move_data_to_device
from lightning.fabric.utilities.types import _DEVICE


def _optimizers_to_device(optimizers: Iterable[Optimizer], device: _DEVICE) -> None:
    """Moves optimizer states for a sequence of optimizers to the device."""
    for opt in optimizers:
        _optimizer_to_device(opt, device)


def _optimizer_to_device(optimizer: Optimizer, device: _DEVICE) -> None:
    """Moves the state of a single optimizer to the device."""
    for p, v in optimizer.state.items():
        if not isinstance(v, MutableMapping):
            # Support for custom optimizers
            optimizer.state[p] = apply_to_collection(v, Tensor, move_data_to_device, device, allow_frozen=True)
            continue
        for key, val in v.items():
            # The 'step' parameter needs to remain unmoved (possibly on the CPU) since that is where the optimizer
            # needs it. See https://github.com/pytorch/pytorch/issues/74424
            if key != "step":
                v[key] = move_data_to_device(val, device)
