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
from typing import Any, Callable, Dict, Generator, Union

import torch
from torch.optim import LBFGS, Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision.mixed import MixedPrecisionPlugin
from pytorch_lightning.utilities import (
    _NATIVE_AMP_AVAILABLE,
    _TORCH_BFLOAT_AVAILABLE,
    _TORCH_CPU_AMP_AVAILABLE,
    AMPType,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class NativeMixedPrecisionPlugin(MixedPrecisionPlugin):
    """Plugin for native mixed precision training with :mod:`torch.cuda.amp`.

    Args:
        precision: Whether to use torch.float16 (16) or torch.bfloat16 (bf16).
    """

    def __init__(self, precision: Union[int, str] = 16, use_cpu: bool = False) -> None:
        super().__init__()

        if not _NATIVE_AMP_AVAILABLE:
            raise MisconfigurationException(
                "You have asked for native AMP but your PyTorch version does not support it."
                " Consider upgrading with `pip install torch>=1.6`."
            )

        if use_cpu and not _TORCH_CPU_AMP_AVAILABLE:
            raise MisconfigurationException(
                "You have asked for native AMP on CPU, but AMP is only available on GPU for PyTorch 1.9 "
                "and lower. To use native AMP on CPU, install PyTorch 1.10 or later."
            )

        self.use_cpu = use_cpu
        self._dtype = self._select_precision_dtype(precision)
        self.backend = AMPType.NATIVE
        if not self.is_bfloat16:
            self.scaler = torch.cuda.amp.GradScaler()

    def _select_precision_dtype(self, precision: Union[int, str] = 16) -> torch.dtype:
        if precision == "bf16":
            if not _TORCH_BFLOAT_AVAILABLE:
                raise MisconfigurationException(
                    "To use bfloat16 with native amp you must install torch greater or equal to 1.10."
                )
            return torch.bfloat16
        elif self.use_cpu:
            raise MisconfigurationException(
                "CPU native amp only supports bfloat16. Please pass precision='bf16' to the Trainer."
            )
        return torch.float16

    @property
    def is_bfloat16(self) -> bool:
        return self._dtype == torch.bfloat16

    def pre_backward(self, model: "pl.LightningModule", closure_loss: torch.Tensor) -> torch.Tensor:
        if self.is_bfloat16:
            return super().pre_backward(model, closure_loss)
        closure_loss = self.scaler.scale(closure_loss)
        return super().pre_backward(model, closure_loss)

    def pre_optimizer_step(
        self,
        model: "pl.LightningModule",
        optimizer: Optimizer,
        optimizer_idx: int,
        lambda_closure: Callable,
        **kwargs: Any,
    ) -> bool:
        if self.is_bfloat16:
            # skip scaler logic, as bfloat16 does not require scaler
            return super().pre_optimizer_step(model, optimizer, optimizer_idx, lambda_closure, **kwargs)
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"native PyTorch amp and lbfgs are not compatible (optimizer {optimizer_idx})."
                " To request, please file a Github issue in PyTorch and tag @mcarilli"
            )
        result = True
        # FIXME: is this correct for manual?
        if model.automatic_optimization:
            result = lambda_closure()
        self.scaler.unscale_(optimizer)
        super().pre_optimizer_step(model, optimizer, optimizer_idx, lambda_closure, **kwargs)
        # lambda_closure returning None indicates that backward has been skipped
        if result is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        return False

    def autocast_context_manager(self) -> torch.cuda.amp.autocast:
        if self.use_cpu:
            return torch.cpu.amp.autocast(dtype=self._dtype)  # Only reached in pytorch==1.10 where this is ok. skipcq
        if self.is_bfloat16:
            return torch.cuda.amp.autocast(dtype=self._dtype)  # Only reached in pytorch==1.10 where this is ok. skipcq
        return torch.cuda.amp.autocast()

    @contextmanager
    def train_step_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield

    @contextmanager
    def val_step_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield

    @contextmanager
    def test_step_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield

    @contextmanager
    def predict_step_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "native_amp_scaling_state" in checkpoint and not self.is_bfloat16:
            self.scaler.load_state_dict(checkpoint["native_amp_scaling_state"])

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if not self.is_bfloat16:
            checkpoint["native_amp_scaling_state"] = self.scaler.state_dict()
