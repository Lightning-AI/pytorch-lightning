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
import contextlib
from collections.abc import Generator
from functools import partial
from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins import Precision as FabricPrecision
from lightning.fabric.utilities.types import Steppable
from lightning.pytorch.core.hooks import CheckpointHooks
from lightning.pytorch.trainer import call
from lightning.pytorch.utilities import GradClipAlgorithmType


class Precision(FabricPrecision, CheckpointHooks):
    """Base class for all plugins handling the precision-specific parts of the training.

    The class attribute precision must be overwritten in child classes. The default value reflects fp32 training.

    """

    def connect(
        self, model: Module, optimizers: list[Optimizer], lr_schedulers: list[Any]
    ) -> tuple[Module, list[Optimizer], list[Any]]:
        """Connects this plugin to the accelerator and the training process."""
        return model, optimizers, lr_schedulers

    @override
    def pre_backward(self, tensor: Tensor, module: "pl.LightningModule") -> Tensor:  # type: ignore[override]
        trainer = module.trainer
        call._call_callback_hooks(trainer, "on_before_backward", tensor)
        call._call_lightning_module_hook(trainer, "on_before_backward", tensor)
        return tensor

    @override
    def backward(  # type: ignore[override]
        self,
        tensor: Tensor,
        model: "pl.LightningModule",
        optimizer: Optional[Steppable],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""Performs the actual backpropagation.

        Args:
            tensor: the loss value obtained from the closure
            model: the model to be optimized
            optimizer: current optimizer being used. ``None`` if using manual optimization
            \*args: Positional arguments intended for the actual function that performs the backward, like
                :meth:`~torch.Tensor.backward`.
            \**kwargs: Keyword arguments for the same purpose as ``*args``.

        """
        model.backward(tensor, *args, **kwargs)

    @override
    def post_backward(self, tensor: Tensor, module: "pl.LightningModule") -> Tensor:  # type: ignore[override]
        # once backward has been applied, release graph
        closure_loss = tensor.detach()
        trainer = module.trainer
        call._call_callback_hooks(trainer, "on_after_backward")
        call._call_lightning_module_hook(trainer, "on_after_backward")
        return closure_loss

    def _after_closure(self, model: "pl.LightningModule", optimizer: Steppable) -> None:
        """Utility to share some code after the closure has been run."""
        trainer = model.trainer
        call._call_callback_hooks(trainer, "on_before_optimizer_step", optimizer)
        call._call_lightning_module_hook(trainer, "on_before_optimizer_step", optimizer)
        self._clip_gradients(
            model,
            optimizer,
            trainer.gradient_clip_val,
            gradient_clip_algorithm=trainer.gradient_clip_algorithm,
        )

    def _wrap_closure(
        self,
        model: "pl.LightningModule",
        optimizer: Steppable,
        closure: Callable[[], Any],
    ) -> Any:
        """This double-closure allows makes sure the ``closure`` is executed before the ``on_before_optimizer_step``
        hook is called.

        The closure (generally) runs ``backward`` so this allows inspecting gradients in this hook. This structure is
        consistent with the ``Precision`` subclasses that cannot pass ``optimizer.step(closure)`` directly.

        """
        closure_result = closure()
        self._after_closure(model, optimizer)
        return closure_result

    @override
    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Steppable,
        model: "pl.LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        """Hook to run the optimizer step."""
        closure = partial(self._wrap_closure, model, optimizer, closure)
        return optimizer.step(closure=closure, **kwargs)

    def _clip_gradients(
        self,
        model: Union["pl.LightningModule", Module],
        optimizer: Steppable,
        clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[GradClipAlgorithmType] = None,
    ) -> None:
        if not isinstance(model, pl.LightningModule) or not model.automatic_optimization:
            # the configuration validator disallows clipping on manual
            return

        call._call_lightning_module_hook(
            model.trainer,
            "configure_gradient_clipping",
            optimizer,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        """Clips the gradients."""
        if clip_val <= 0:
            return
        if gradient_clip_algorithm == GradClipAlgorithmType.VALUE:
            self.clip_grad_by_value(optimizer, clip_val)
        elif gradient_clip_algorithm == GradClipAlgorithmType.NORM:
            self.clip_grad_by_norm(optimizer, clip_val)

    def clip_grad_by_value(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        """Clip gradients by value."""
        parameters = self.main_params(optimizer)
        torch.nn.utils.clip_grad_value_(parameters, clip_value=clip_val)

    def clip_grad_by_norm(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        """Clip gradients by norm."""
        parameters = self.main_params(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, clip_val)

    @contextlib.contextmanager
    def train_step_context(self) -> Generator[None, None, None]:
        """A contextmanager for the training step."""
        with self.forward_context():
            yield

    @contextlib.contextmanager
    def val_step_context(self) -> Generator[None, None, None]:
        """A contextmanager for the validation step."""
        with self.forward_context():
            yield

    @contextlib.contextmanager
    def test_step_context(self) -> Generator[None, None, None]:
        """A contextmanager for the test step."""
        with self.forward_context():
            yield

    @contextlib.contextmanager
    def predict_step_context(self) -> Generator[None, None, None]:
        """A contextmanager for the predict step."""
        with self.forward_context():
            yield
