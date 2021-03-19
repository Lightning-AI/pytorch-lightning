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
from functools import wraps, partial
from typing import Any, Callable, Sequence, Tuple, TYPE_CHECKING

import torch

from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.apply_func import apply_to_collection

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim import Optimizer


class DoublePrecisionPlugin(PrecisionPlugin):
    """Plugin for training with double (`torch.float64`) precision."""

    precision: int = 64

    @staticmethod
    def _to_double_precision(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_floating_point():
            return tensor.to(dtype=torch.float64)
        return tensor

    @staticmethod
    def _teardown(model: 'Module', old_forward: Callable) -> None:
        model.forward = old_forward

    def _patch_forward(self, model: 'Module') -> None:
        old_forward = model.forward

        @wraps(old_forward)
        def new_forward(*args, **kwargs) -> Any:
            return old_forward(*apply_to_collection(args, torch.Tensor, DoublePrecisionPlugin._to_double_precision),
                               **apply_to_collection(kwargs, torch.Tensor, DoublePrecisionPlugin._to_double_precision))

        model.forward = new_forward
        self.post_dispatch = partial(DoublePrecisionPlugin._teardown, model, old_forward)

    def connect(
        self,
        model: 'Module',
        optimizers: Sequence['Optimizer'],
        lr_schedulers: Sequence[Any],
    ) -> Tuple['Module', Sequence['Optimizer'], Sequence[Any]]:
        """Converts the model to double precision and wraps the forward to convert incoming floating point data.
        Does not alter `optimizers` or `lr_schedulers`."""
        model = model.to(dtype=torch.float64)
        self._patch_forward(model)
        return model, optimizers, lr_schedulers
