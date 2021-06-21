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
from contextlib import contextmanager
from typing import Any, Generator, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.base import _LightningPrecisionModuleWrapperBase
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.apply_func import apply_to_collection


class LightningDoublePrecisionModule(_LightningPrecisionModuleWrapperBase):
    """
    LightningModule wrapper which converts incoming floating point data in ``*_step`` and ``forward`` to double
    (``torch.float64``) precision.

    Args:
        pl_module: the model to wrap
    """

    def __init__(self, pl_module: LightningModule):
        super().__init__(pl_module)

    @staticmethod
    def _to_double_precision(data: torch.Tensor) -> torch.Tensor:
        if data.is_floating_point():
            return data.double()
        return data

    @staticmethod
    def _move_float_tensors_to_double(collection: Any) -> Any:
        return apply_to_collection(
            collection,
            torch.Tensor,
            LightningDoublePrecisionModule._to_double_precision,
        )

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.training_step(
            *LightningDoublePrecisionModule._move_float_tensors_to_double(args),
            **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs),
        )

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.validation_step(
            *LightningDoublePrecisionModule._move_float_tensors_to_double(args),
            **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs),
        )

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.test_step(
            *LightningDoublePrecisionModule._move_float_tensors_to_double(args),
            **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs),
        )

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.predict_step(
            *LightningDoublePrecisionModule._move_float_tensors_to_double(args),
            **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs),
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(
            *LightningDoublePrecisionModule._move_float_tensors_to_double(args),
            **LightningDoublePrecisionModule._move_float_tensors_to_double(kwargs),
        )


class DoublePrecisionPlugin(PrecisionPlugin):
    """ Plugin for training with double (``torch.float64``) precision. """

    precision: int = 64

    def connect(
        self,
        model: nn.Module,
        optimizers: List[Optimizer],
        lr_schedulers: List[Any],
    ) -> Tuple[nn.Module, List['Optimizer'], List[Any]]:
        """Converts the model to double precision and wraps it in a ``LightningDoublePrecisionModule`` to convert
        incoming floating point data to double (``torch.float64``) precision. Does not alter `optimizers` or
        `lr_schedulers`.
        """
        model = model.to(dtype=torch.float64)
        model = LightningDoublePrecisionModule(model)

        return super().connect(model, optimizers, lr_schedulers)

    @contextmanager
    def train_step_context(self) -> Generator[None, None, None]:
        """
        A context manager to change the default tensor type.
        See: :meth:`torch.set_default_tensor_type`
        """
        torch.set_default_tensor_type(torch.DoubleTensor)
        yield
        torch.set_default_tensor_type(torch.FloatTensor)

    @contextmanager
    def val_step_context(self) -> Generator[None, None, None]:
        """
        A context manager to change the default tensor type.
        See: :meth:`torch.set_default_tensor_type`
        """
        torch.set_default_tensor_type(torch.DoubleTensor)
        yield
        torch.set_default_tensor_type(torch.FloatTensor)

    @contextmanager
    def test_step_context(self) -> Generator[None, None, None]:
        """
        A context manager to change the default tensor type.
        See: :meth:`torch.set_default_tensor_type`
        """
        torch.set_default_tensor_type(torch.DoubleTensor)
        yield
        torch.set_default_tensor_type(torch.FloatTensor)

    @contextmanager
    def predict_step_context(self) -> Generator[None, None, None]:
        """
        A context manager to change the default tensor type.
        See: :meth:`torch.set_default_tensor_type`
        """
        torch.set_default_tensor_type(torch.DoubleTensor)
        yield
        torch.set_default_tensor_type(torch.FloatTensor)
