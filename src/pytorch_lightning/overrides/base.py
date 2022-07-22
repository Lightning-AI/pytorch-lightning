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
from typing import Any, Union

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

import pytorch_lightning as pl
from pytorch_lightning.core.mixins import DeviceDtypeModuleMixin


class _LightningPrecisionModuleWrapperBase(DeviceDtypeModuleMixin, torch.nn.Module):
    def __init__(self, pl_module: "pl.LightningModule") -> None:
        """Wraps the user's LightningModule. Requires overriding all ``*_step`` methods and ``forward`` so that it
        can safely be wrapped by a ``_LightningModuleWrapperBase`` and a ``*DataParallel``.

        Args:
            pl_module: the model to wrap
        """
        super().__init__()
        self.module = pl_module

        # set the parameters_to_ignore from LightningModule.
        _ddp_params_and_buffers_to_ignore = getattr(pl_module, "_ddp_params_and_buffers_to_ignore", [])
        self._ddp_params_and_buffers_to_ignore = [f"module.{p}" for p in _ddp_params_and_buffers_to_ignore]

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _LightningModuleWrapperBase(DeviceDtypeModuleMixin, torch.nn.Module):
    def __init__(self, pl_module: Union["pl.LightningModule", _LightningPrecisionModuleWrapperBase]) -> None:
        """Wraps the user's LightningModule and redirects the forward call to the appropriate method, either
        ``training_step``, ``validation_step``, ``test_step``, or ``predict_step``.

        Inheriting classes may also modify the inputs or outputs of forward.

        Args:
            pl_module: the model to wrap
        """
        super().__init__()
        self.module = pl_module

        # set the parameters_to_ignore from LightningModule.
        _ddp_params_and_buffers_to_ignore = getattr(pl_module, "_ddp_params_and_buffers_to_ignore", [])
        self._ddp_params_and_buffers_to_ignore = [f"module.{p}" for p in _ddp_params_and_buffers_to_ignore]

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        pl_module = unwrap_lightning_module(self.module)
        trainer = pl_module._trainer

        if trainer is not None:
            if trainer.training:
                output = self.module.training_step(*inputs, **kwargs)
                # In manual_optimization, we need to prevent DDP reducer as
                # it is done manually in `LightningModule.manual_backward`
                # `require_backward_grad_sync` will be reset in the
                # ddp_strategy `post_training_step` hook
                if not pl_module.automatic_optimization:
                    trainer.model.require_backward_grad_sync = False  # type: ignore[assignment]
                return output
            if trainer.testing:
                return self.module.test_step(*inputs, **kwargs)
            if trainer.sanity_checking or trainer.validating:
                return self.module.validation_step(*inputs, **kwargs)
            if trainer.predicting:
                return self.module.predict_step(*inputs, **kwargs)
        return self.module(*inputs, **kwargs)


def unwrap_lightning_module(wrapped_model: nn.Module) -> "pl.LightningModule":
    """Recursively unwraps a :class:`~pytorch_lightning.core.module.LightningModule` by following the ``.module``
    attributes on the wrapper.

    Raises:
        TypeError: If the unwrapping leads to a module that is not a LightningModule and that cannot be unwrapped
            further.
    """
    model = wrapped_model
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model = unwrap_lightning_module(model.module)
    if isinstance(model, (_LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase)):
        model = unwrap_lightning_module(model.module)
    if not isinstance(model, pl.LightningModule):
        raise TypeError(f"Unwrapping the module did not yield a `LightningModule`, got {type(model)} instead.")
    return model
