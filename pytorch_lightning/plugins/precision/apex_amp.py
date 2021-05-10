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
from typing import Any, Callable, ContextManager, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
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
        model: LightningModule,
        closure_loss: Tensor,
        optimizer: Optimizer,
        opt_idx: int,
        should_accumulate: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """performs the actual backpropagation

        Args:
            model: the model to be optimized
            closure_loss: the loss value obtained from the closure
            optimizer: the optimizer to perform the step lateron
            opt_idx: the optimizer index
            should_accumulate: whether to accumulate gradients or not

        """
        opt = model.trainer.optimizers if optimizer is None else optimizer
        scaled_loss: ContextManager[Tensor] = amp.scale_loss(closure_loss, opt)

        # enter apex context
        closure_loss = scaled_loss.__enter__()

        # do backward pass
        # TODO: not entirely sure, why we need this
        if model is not None and isinstance(model, LightningModule):
            model.backward(closure_loss, optimizer, opt_idx, **kwargs)

            # TODO: avoid dev_debugger and track these calls with mock
            model.trainer.dev_debugger.track_event('AMP', str(AMPType.APEX))

        else:
            closure_loss.backward(*args, **kwargs)

        # exit amp context
        error = scaled_loss.__exit__(None, None, None)
        if error:
            raise Exception("apex unscale error")

        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()
        return closure_loss

    @staticmethod
    def reinit_scheduler_properties(optimizers: Sequence[Optimizer], schedulers: Sequence[Any]) -> None:
        """Reinitializes schedulers with correct properties"""
        # Reinitialize optimizer.step properties added by schedulers
        for scheduler in schedulers:
            scheduler = scheduler['scheduler']
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
        pl_module: LightningModule,
        optimizer: Optimizer,
        optimizer_idx: int,
        lambda_closure: Callable,
        **kwargs: Any,
    ) -> bool:
        """
        always called before the optimizer step.
        """
        # apex amp does not support closures.
        lambda_closure()

        if not pl_module.automatic_optimization:
            pl_module.trainer.call_hook("on_after_backward")

        optimizer.step(**kwargs)
        return False
