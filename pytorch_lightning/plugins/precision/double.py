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
from functools import wraps
from typing import Any, List, Sequence, Tuple, TYPE_CHECKING

import torch

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.apply_func import apply_to_collection

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim import Optimizer


class _DoublePrecisionPatch:
    """Class to handle patching of methods in the ``LightningModule`` and subsequent teardown."""

    def __init__(self, model: 'Module', method_name: str, old_method: Any) -> None:
        self.model = model
        self.method_name = method_name
        self.old_method = old_method

    def teardown(self) -> None:
        setattr(self.model, self.method_name, self.old_method)

    @staticmethod
    def _to_double_precision(data: torch.Tensor) -> torch.Tensor:
        if data.is_floating_point():
            return data.double()
        return data

    @staticmethod
    def _move_float_tensors_to_double(collection: Any) -> Any:
        return apply_to_collection(collection, torch.Tensor, function=_DoublePrecisionPatch._to_double_precision)

    @classmethod
    def patch(cls, model: 'Module', method_name: str) -> '_DoublePrecisionPatch':
        old_method = getattr(model, method_name)

        @wraps(old_method)
        def new_method(*args: Any, **kwargs: Any) -> Any:
            return old_method(
                *_DoublePrecisionPatch._move_float_tensors_to_double(args),
                **_DoublePrecisionPatch._move_float_tensors_to_double(kwargs)
            )

        setattr(model, method_name, new_method if callable(old_method) else old_method)
        return cls(model, method_name, old_method)


class DoublePrecisionPlugin(PrecisionPlugin):
    """Plugin for training with double (``torch.float64``) precision."""

    precision: int = 64

    def __init__(self) -> None:
        super().__init__()
        self.patches: List[_DoublePrecisionPatch] = []

    def connect(
        self,
        model: 'Module',
        optimizers: Sequence['Optimizer'],
        lr_schedulers: Sequence[Any],
    ) -> Tuple['Module', Sequence['Optimizer'], Sequence[Any]]:
        """Converts the model to double precision and wraps the `training_step`, `validation_step`, `test_step`,
        `predict_step`, and `forward` methods to convert incoming floating point data to double. Does not alter
        `optimizers` or `lr_schedulers`."""
        model = model.to(dtype=torch.float64)
        if isinstance(model, LightningModule):
            self.patches.append(_DoublePrecisionPatch.patch(model, 'training_step'))
            self.patches.append(_DoublePrecisionPatch.patch(model, 'validation_step'))
            self.patches.append(_DoublePrecisionPatch.patch(model, 'test_step'))
            self.patches.append(_DoublePrecisionPatch.patch(model, 'predict_step'))
        self.patches.append(_DoublePrecisionPatch.patch(model, 'forward'))

        return super().connect(model, optimizers, lr_schedulers)

    def post_dispatch(self) -> None:
        while len(self.patches) > 0:
            self.patches.pop().teardown()
