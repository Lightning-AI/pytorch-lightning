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
from functools import partial
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from lightning_fabric.plugins import Precision as FabricPrecision
from lightning_fabric.utilities.types import Steppable
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.utilities import grad_norm, GradClipAlgorithmType


class PrecisionPlugin(FabricPrecision, CheckpointHooks):
    """Base class for all plugins handling the precision-specific parts of the training.

    The class attribute precision must be overwritten in child classes. The default value reflects fp32 training.
    """

    def connect(
        self, model: Module, optimizers: List[Optimizer], lr_schedulers: List[Any]
    ) -> Tuple[Module, List[Optimizer], List[Any]]:
        """Connects this plugin to the accelerator and the training process."""
        return model, optimizers, lr_schedulers

    def pre_backward(self, tensor: Tensor, module: "pl.LightningModule") -> Tensor:  # type: ignore[override]
        module.trainer._call_callback_hooks("on_before_backward", tensor)
        module.trainer._call_lightning_module_hook("on_before_backward", tensor)
        return tensor

    def backward(  # type: ignore[override]
        self,
        tensor: Tensor,
        model: "pl.LightningModule",
        optimizer: Optional[Steppable],
        optimizer_idx: Optional[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""Performs the actual backpropagation.

        Args:
            tensor: the loss value obtained from the closure
            model: the model to be optimized
            optimizer: current optimizer being used. ``None`` if using manual optimization
            optimizer_idx: the index of the current optimizer. ``None`` if using manual optimization
            \*args: Positional arguments intended for the actual function that performs the backward, like
                :meth:`~torch.Tensor.backward`.
            \**kwargs: Keyword arguments for the same purpose as ``*args``.
        """
        model.backward(tensor, optimizer, optimizer_idx, *args, **kwargs)

    def post_backward(self, tensor: Tensor, module: "pl.LightningModule") -> Tensor:  # type: ignore[override]
        # once backward has been applied, release graph
        closure_loss = tensor.detach()
        module.trainer._call_callback_hooks("on_after_backward")
        module.trainer._call_lightning_module_hook("on_after_backward")
        return closure_loss

    def _after_closure(self, model: "pl.LightningModule", optimizer: Steppable, optimizer_idx: int) -> None:
        """Utility to share some code after the closure has been run."""
        trainer = model.trainer
        trainer._call_callback_hooks("on_before_optimizer_step", optimizer, optimizer_idx)
        trainer._call_lightning_module_hook("on_before_optimizer_step", optimizer, optimizer_idx)
        # TODO: this is done for the entire model but should be changed to per-optimizer
        if optimizer_idx == 0:
            self._track_grad_norm(trainer)
        self._clip_gradients(
            model,
            optimizer,
            optimizer_idx,
            trainer.gradient_clip_val,
            gradient_clip_algorithm=trainer.gradient_clip_algorithm,
        )

    def _wrap_closure(
        self,
        model: "pl.LightningModule",
        optimizer: Optimizer,
        optimizer_idx: int,
        closure: Callable[[], Any],
    ) -> Any:
        """This double-closure allows makes sure the ``closure`` is executed before the
        ``on_before_optimizer_step`` hook is called.

        The closure (generally) runs ``backward`` so this allows inspecting gradients in this hook. This structure is
        consistent with the ``PrecisionPlugin`` subclasses that cannot pass ``optimizer.step(closure)`` directly.
        """
        closure_result = closure()
        self._after_closure(model, optimizer, optimizer_idx)
        return closure_result

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Steppable,
        model: "pl.LightningModule",
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        """Hook to run the optimizer step."""
        closure = partial(self._wrap_closure, model, optimizer, optimizer_idx, closure)
        return optimizer.step(closure=closure, **kwargs)

    def _track_grad_norm(self, trainer: "pl.Trainer") -> None:
        if trainer.track_grad_norm == -1:
            return

        kwargs = {}
        if len(trainer.loggers) == 1:
            kwargs["group_separator"] = trainer.loggers[0].group_separator

        grad_norm_dict = grad_norm(trainer.lightning_module, trainer.track_grad_norm, **kwargs)
        if grad_norm_dict:
            prev_fx = trainer.lightning_module._current_fx_name
            trainer.lightning_module._current_fx_name = "on_before_optimizer_step"
            trainer.lightning_module.log_grad_norm(grad_norm_dict)
            trainer.lightning_module._current_fx_name = prev_fx

    def _clip_gradients(
        self,
        model: Union["pl.LightningModule", Module],
        optimizer: Steppable,
        optimizer_idx: int,
        clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[GradClipAlgorithmType] = None,
    ) -> None:
        if not isinstance(model, pl.LightningModule) or not model.automatic_optimization:
            # the configuration validator disallows clipping on manual
            return

        model.trainer._call_lightning_module_hook(
            "configure_gradient_clipping",
            optimizer,
            optimizer_idx,
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

    def dispatch(self, trainer: "pl.Trainer") -> None:
        """Hook to do something when ``Strategy.dispatch()`` gets called."""

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
