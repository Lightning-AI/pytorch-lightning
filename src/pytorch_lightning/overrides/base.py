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
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

import pytorch_lightning as pl
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


class _LightningPrecisionModuleWrapperBase(_DeviceDtypeModuleMixin, torch.nn.Module):
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


class _LightningModuleWrapperBase(_DeviceDtypeModuleMixin, torch.nn.Module):
    def __init__(
        self, forward_module: Optional[Union["pl.LightningModule", _LightningPrecisionModuleWrapperBase]]
    ) -> None:
        """Wraps the user's LightningModule and redirects the forward call to the appropriate method, either
        ``training_step``, ``validation_step``, ``test_step``, or ``predict_step``.

        Inheriting classes may also modify the inputs or outputs of forward.

        Args:
            forward_module: The module to wrap. If it's not a LightningModule, it must have an attribute ``.module``
                pointing to a LightningModule reference.
        """
        super().__init__()
        if not isinstance(forward_module, pl.LightningModule) and (
            not isinstance(getattr(forward_module, "module", None), pl.LightningModule)
        ):
            raise ValueError(
                "`forward_module` must be a `LightningModule` instance or have an attribute `.module` pointing to one,"
                f" got: {forward_module.__class__.__qualname__}"
            )
        # TODO: In v2.0.0, remove the Optional type from forward_module and remove the assertion
        assert forward_module is not None
        self._forward_module = forward_module

        # set the parameters_to_ignore from LightningModule.
        _ddp_params_and_buffers_to_ignore = getattr(self._forward_module, "_ddp_params_and_buffers_to_ignore", [])
        self._ddp_params_and_buffers_to_ignore = [f"module.{p}" for p in _ddp_params_and_buffers_to_ignore]

    @property
    def lightning_module(self) -> "pl.LightningModule":
        if isinstance(self._forward_module, pl.LightningModule):
            return self._forward_module
        return self._forward_module.module

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        pl_module = self.lightning_module
        trainer = pl_module._trainer

        if trainer is not None:
            if trainer.training:
                output = self._forward_module.training_step(*inputs, **kwargs)
                # In manual_optimization, we need to prevent DDP reducer as
                # it is done manually in `LightningModule.manual_backward`
                # `require_backward_grad_sync` will be reset in the
                # ddp_strategy `post_training_step` hook
                if not pl_module.automatic_optimization:
                    assert trainer.model is not None
                    trainer.model.require_backward_grad_sync = False  # type: ignore[assignment]
                return output
            if trainer.testing:
                return self._forward_module.test_step(*inputs, **kwargs)
            if trainer.sanity_checking or trainer.validating:
                return self._forward_module.validation_step(*inputs, **kwargs)
            if trainer.predicting:
                return self._forward_module.predict_step(*inputs, **kwargs)
        return self._forward_module(*inputs, **kwargs)

    @classmethod
    def _validate_init_arguments(
        cls,
        pl_module: Optional[Union["pl.LightningModule", _LightningPrecisionModuleWrapperBase]] = None,
        forward_module: Optional[Union["pl.LightningModule", _LightningPrecisionModuleWrapperBase]] = None,
    ) -> None:
        # TODO: In v2.0.0, remove this method and mark the forward_module init argument in all subclasses as required
        if pl_module is not None:
            rank_zero_deprecation(
                f"The argument `pl_module` in `{cls.__name__}` is deprecated in v1.8.0 and will be removed in"
                " v2.0.0. Please use `forward_module` instead."
            )
        elif forward_module is None:
            raise ValueError("Argument `forward_module` is required.")


def unwrap_lightning_module(wrapped_model: nn.Module, _suppress_warning: bool = False) -> "pl.LightningModule":
    """Recursively unwraps a :class:`~pytorch_lightning.core.module.LightningModule` by following the ``.module``
    attributes on the wrapper.

    .. deprecated:: v1.8.0
        The function ``unwrap_lightning_module`` is deprecated in v1.8.0 and will be removed in v2.0.0. Access the
        ``LightningModule`` directly through the strategy attribute ``Strategy.lightning_module``.

    Raises:
        TypeError: If the unwrapping leads to a module that is not a LightningModule and that cannot be unwrapped
            further.
    """
    if not _suppress_warning:
        rank_zero_deprecation(
            "The function `unwrap_lightning_module` is deprecated in v1.8.0 and will be removed in v2.0.0. Access the"
            " `LightningModule` directly through the strategy attribute `Strategy.lightning_module`."
        )
    model = wrapped_model
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model = unwrap_lightning_module(model.module)
    if isinstance(model, _LightningModuleWrapperBase):
        model = model.lightning_module
    if isinstance(model, _LightningPrecisionModuleWrapperBase):
        model = model.module
    if not isinstance(model, pl.LightningModule):
        raise TypeError(f"Unwrapping the module did not yield a `LightningModule`, got {type(model)} instead.")
    return model
