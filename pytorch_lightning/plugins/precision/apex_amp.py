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
from typing import Any, Callable, Dict, Optional, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision.mixed import MixedPrecisionPlugin
from pytorch_lightning.utilities import _APEX_AVAILABLE, AMPType
from pytorch_lightning.utilities.types import _PARAMETERS

if _APEX_AVAILABLE:
    from apex import amp


class ApexMixedPrecisionPlugin(MixedPrecisionPlugin):
    """Mixed Precision Plugin based on Nvidia/Apex (https://github.com/NVIDIA/apex)"""

    def __init__(self, amp_level: str = "O2") -> None:
        super().__init__()
        self.backend = AMPType.APEX
        self.amp_level = amp_level
        self._connected = False

    def master_params(self, optimizer: Optimizer) -> _PARAMETERS:
        return amp.master_params(optimizer)

    def dispatch(self, trainer: "pl.Trainer") -> None:
        if not self._connected:
            accelerator = trainer.accelerator
            _, accelerator.optimizers = amp.initialize(
                trainer.lightning_module, accelerator.optimizers, opt_level=self.amp_level
            )
            self._connected = True
        return super().dispatch(trainer)

    def backward(
        self,
        model: "pl.LightningModule",
        closure_loss: Tensor,
        optimizer: Optional[Optimizer],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run before precision plugin executes backward

        Args:
            model: the model to be optimized
            closure_loss: the loss value obtained from the closure
            optimizer: current optimizer being used. ``None`` if using manual optimization
        """
        opt = optimizer or model.trainer.optimizers
        with amp.scale_loss(closure_loss, opt) as closure_loss:
            super().backward(model, closure_loss, optimizer, *args, **kwargs)

    @staticmethod
    def reinit_scheduler_properties(optimizers: Sequence[Optimizer], schedulers: Sequence[Any]) -> None:
        """Reinitializes schedulers with correct properties"""
        # Reinitialize optimizer.step properties added by schedulers
        for scheduler in schedulers:
            scheduler = scheduler["scheduler"]
            state = None

            for optimizer in optimizers:
                # check that we dont mix users optimizers and schedulers
                if scheduler.optimizer == optimizer:
                    # Find the mro belonging to the base lr scheduler class
                    for i, mro in enumerate(scheduler.__class__.__mro__):
                        if mro in (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            state = scheduler.state_dict()
                            scheduler.__class__.__mro__[i].__init__(scheduler, optimizer)
                            scheduler.load_state_dict(state)
                            break

                if state is not None:
                    break

    def pre_optimizer_step(
        self,
        model: "pl.LightningModule",
        optimizer: Optimizer,
        optimizer_idx: int,
        lambda_closure: Callable,
        **kwargs: Any,
    ) -> bool:
        """Hook to do something before each optimizer step."""
        super().pre_optimizer_step(model, optimizer, optimizer_idx, lambda_closure, **kwargs)
        # the following should be in a `optimizer_step` hook but we don't have one in the precision plugin.
        lambda_closure()  # APEX amp does not support closures
        optimizer.step(**kwargs)
        return False

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "amp_scaling_state" in checkpoint:
            amp.load_state_dict(checkpoint["amp_scaling_state"])

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["amp_scaling_state"] = amp.state_dict()
