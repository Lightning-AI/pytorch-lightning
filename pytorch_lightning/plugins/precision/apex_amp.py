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
from typing import Any, Callable, ContextManager, List, Sequence, Tuple, Type

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from pytorch_lightning.core import LightningModule
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

    def master_params(self, optimizer: Optimizer) -> _PARAMETERS:
        return amp.master_params(optimizer)

    def connect(
        self,
        model: Module,
        optimizers: List[Optimizer],
        lr_schedulers: List[Any],
    ) -> Tuple[Module, List[Optimizer], List[Any]]:
        """Connects the precision plugin to the training process,
        configures apex and reinits the schedulers
        """
        if model.device.type != "cuda":
            return model, optimizers, lr_schedulers
        model, optimizers = self.configure_apex(amp, model, list(optimizers), self.amp_level)
        self.reinit_scheduler_properties(optimizers, lr_schedulers)
        return model, optimizers, lr_schedulers

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

    def configure_apex(
        self,
        amp: Type,
        model: Module,
        optimizers: List[Optimizer],
        amp_level: str,
    ) -> Tuple[Module, List[Optimizer]]:
        r"""
        Override to init AMP your own way.
        Must return a model and list of optimizers.

        Args:
            amp: pointer to amp library object.
            model: pointer to current :class:`torch.nn.Module`.
            optimizers: list of optimizers passed in :meth:`configure_optimizers`.
            amp_level: AMP mode chosen ('O1', 'O2', etc...)

        Return:
            Apex wrapped model and optimizers

        Examples:
            .. code-block:: python

                # Default implementation used by Trainer.
                def configure_apex(self, amp, model, optimizers, amp_level):
                    model, optimizers = amp.initialize(
                        model, optimizers, opt_level=amp_level,
                    )

                    return model, optimizers
        """
        model, optimizers = amp.initialize(model, optimizers, opt_level=amp_level)
        return model, optimizers

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
