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
import torch.nn
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.overrides.base import _LightningPrecisionModuleWrapperBase
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.apply_func import apply_to_collection


class LightningPrecisionModule(_LightningPrecisionModuleWrapperBase):
    """LightningModule wrapper which converts incoming data in ``*_step`` and ``forward`` to a specific
    precision."""

    def __init__(self, pl_module: "pl.LightningModule", dtype: torch.dtype) -> None:
        """Wraps the user's LightningModule.

        Requires overriding all ``*_step`` methods and ``forward`` so that it can safely be wrapped by a
        ``_LightningModuleWrapperBase`` and a ``*DataParallel``.
        """
        super().__init__(pl_module)
        self.__dtype = dtype

    @staticmethod
    def _to(data: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return data.to(dtype)

    def _move_tensors(self, collection: Any) -> Any:
        return apply_to_collection(collection, torch.Tensor, LightningPrecisionModule._to, self.__dtype)

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.training_step(*self._move_tensors(args), **self._move_tensors(kwargs))

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.validation_step(*self._move_tensors(args), **self._move_tensors(kwargs))

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.test_step(*self._move_tensors(args), **self._move_tensors(kwargs))

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.predict_step(*self._move_tensors(args), **self._move_tensors(kwargs))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*self._move_tensors(args), **self._move_tensors(kwargs))


class DtypePrecisionPlugin(PrecisionPlugin):
    """Plugin for training with double a specific :class:`torch.dtype`."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.__dtype = dtype

    def connect(
        self, model: Module, optimizers: List[Optimizer], lr_schedulers: List[Any]
    ) -> Tuple[Module, List[Optimizer], List[Any]]:
        """Wraps the model it in a ``LightningPrecisionModule`` to convert incoming data to a specific
        precision."""
        model = LightningPrecisionModule(model, self.__dtype)
        return super().connect(model, optimizers, lr_schedulers)

    @contextmanager
    def autodtype(self) -> Generator[None, None, None]:
        """A context manager to change the default tensor type.

        See: :meth:`torch.set_default_dtype`
        """
        previous = torch.get_default_dtype()
        torch.set_default_dtype(self.__dtype)
        try:
            yield
        finally:
            # make sure the default dtype is restored. otherwise, the new dtype can leak if the program fails
            torch.set_default_dtype(previous)

    def forward_context(self) -> Generator[None, None, None]:
        return self.autodtype()


class DoublePrecisionPlugin(DtypePrecisionPlugin):
    """Plugin for training with double (``torch.float64``) precision."""

    precision: int = 64

    def __init__(self):
        super().__init__(torch.double)
