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
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable, Literal, Optional, Union, cast

import torch
from torch import Tensor
from torch.optim import LBFGS, Optimizer
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins.precision.amp import _optimizer_handles_unscaling
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_4
from lightning.fabric.utilities.types import Optimizable
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.pytorch.utilities import GradClipAlgorithmType
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class _AutocastClearCacheOnExit:
    """Proxy a grad-disabling context manager and clear the autocast cache when it exits."""

    def __init__(self, context_manager: Any, *, clear_cache: bool) -> None:
        self._context_manager = context_manager
        self._clear_cache = clear_cache

    def __enter__(self) -> Any:
        return self._context_manager.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        out = self._context_manager.__exit__(exc_type, exc, tb)
        if self._clear_cache:
            torch.clear_autocast_cache()
        return out

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        return self._context_manager(func)


class MixedPrecision(Precision):
    """Plugin for Automatic Mixed Precision (AMP) training with ``torch.autocast``.

    Args:
        precision: Whether to use ``torch.float16`` (``'16-mixed'``) or ``torch.bfloat16`` (``'bf16-mixed'``).
        device: The device for ``torch.autocast``.
        scaler: An optional :class:`torch.cuda.amp.GradScaler` to use.

    """

    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed"],
        device: str,
        scaler: Optional["torch.amp.GradScaler"] = None,
    ) -> None:
        if precision not in ("16-mixed", "bf16-mixed"):
            raise ValueError(
                f"`Passed `{type(self).__name__}(precision={precision!r})`."
                f" Precision must be '16-mixed' or 'bf16-mixed'."
            )

        self.precision = precision
        if scaler is None and self.precision == "16-mixed":
            scaler = torch.amp.GradScaler(device=device) if _TORCH_GREATER_EQUAL_2_4 else torch.cuda.amp.GradScaler()
        if scaler is not None and self.precision == "bf16-mixed":
            raise MisconfigurationException(f"`precision='bf16-mixed'` does not use a scaler, found {scaler}.")
        self.device = device
        self.scaler = scaler

    @override
    def pre_backward(self, tensor: Tensor, module: "pl.LightningModule") -> Tensor:  # type: ignore[override]
        if self.scaler is not None:
            tensor = self.scaler.scale(tensor)
        return super().pre_backward(tensor, module)

    @override
    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: "pl.LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if self.scaler is None:
            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(optimizer, model=model, closure=closure, **kwargs)
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException("AMP and the LBFGS optimizer are not compatible.")
        closure_result = closure()

        # If backward was skipped in automatic optimization (return None), unscaling is not needed
        skip_unscaling = closure_result is None and model.automatic_optimization

        if not _optimizer_handles_unscaling(optimizer) and not skip_unscaling:
            # Unscaling needs to be performed here in case we are going to apply gradient clipping.
            # Optimizers that perform unscaling in their `.step()` method are not supported (e.g., fused Adam).
            # Note: `unscale` happens after the closure is executed, but before the `on_before_optimizer_step` hook.
            self.scaler.unscale_(optimizer)  # type: ignore[arg-type]

        self._after_closure(model, optimizer)

        # in manual optimization, the closure does not return a value
        if not skip_unscaling:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            step_output = self.scaler.step(optimizer, **kwargs)  # type: ignore[arg-type]
            self.scaler.update()
            return step_output
        return closure_result

    @override
    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        if clip_val > 0 and _optimizer_handles_unscaling(optimizer):
            raise RuntimeError(
                f"The current optimizer, {type(optimizer).__qualname__}, does not allow for gradient clipping"
                " because it performs unscaling of gradients internally. HINT: Are you using a 'fused' optimizer?"
            )
        super().clip_gradients(optimizer=optimizer, clip_val=clip_val, gradient_clip_algorithm=gradient_clip_algorithm)

    def autocast_context_manager(self) -> torch.autocast:
        dtype = torch.bfloat16 if self.precision == "bf16-mixed" else torch.half
        return torch.autocast(self.device, dtype=dtype, cache_enabled=False)

    @override
    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast and clear cached casts after nested grad-disabling contexts exit."""
        original_no_grad = torch.no_grad
        original_inference_mode = torch.inference_mode

        def _clear_cache_on_exit(
            context_factory: Callable[..., Any], *, clear_cache: Callable[..., bool]
        ) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> _AutocastClearCacheOnExit:
                return _AutocastClearCacheOnExit(
                    context_factory(*args, **kwargs),
                    clear_cache=clear_cache(*args, **kwargs),
                )

            return wrapper

        try:
            # Lightning wraps the whole step in a persistent autocast context. If a nested `no_grad` or
            # `inference_mode` block creates cached casts there, later grad-enabled forwards in the same step can
            # incorrectly reuse them. Clear the autocast cache when such nested contexts exit, while keeping the
            # default cached path for normal training.
            torch_module = cast(Any, torch)
            torch_module.no_grad = _clear_cache_on_exit(original_no_grad, clear_cache=lambda *args, **kwargs: True)
            torch_module.inference_mode = _clear_cache_on_exit(
                original_inference_mode,
                clear_cache=lambda *args, **kwargs: bool(args[0] if args else kwargs.get("mode", True)),
            )
            dtype = torch.bfloat16 if self.precision == "bf16-mixed" else torch.half
            with torch.autocast(self.device, dtype=dtype):
                yield
        finally:
            torch_module = cast(Any, torch)
            torch_module.no_grad = original_no_grad
            torch_module.inference_mode = original_inference_mode

    @override
    def state_dict(self) -> dict[str, Any]:
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict)

    @override
    def compute_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.precision == "bf16-mixed" else torch.half
