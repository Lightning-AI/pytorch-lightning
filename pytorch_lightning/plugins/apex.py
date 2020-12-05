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
import math
from typing import List, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities import APEX_AVAILABLE, AMPType
from pytorch_lightning.utilities.distributed import rank_zero_warn

if APEX_AVAILABLE:
    from apex import amp


class ApexPlugin(PrecisionPlugin):

    def __init__(self, trainer=None):
        self.trainer = trainer

    def connect(self, model, optimizers):
        model, optimizers = self.configure_apex(amp, model, optimizers, self.trainer.amp_level)
        self.trainer.reinit_scheduler_properties(optimizers, self.trainer.lr_schedulers)
        return model, optimizers

    def training_step(self, fx, args):
        output = fx(args)
        return output

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        closure_loss = amp.scale_loss(closure_loss, optimizer)

        # enter apex context
        self.trainer.dev_debugger.track_event('AMP', str(AMPType.APEX))
        context = closure_loss
        closure_loss = closure_loss.__enter__()

        # do backward pass
        if self.trainer.train_loop.automatic_optimization:
            model = self.trainer.get_model()
            model.backward(closure_loss, optimizer, opt_idx)
        else:
            closure_loss.backward(*args, **kwargs)

        # exit amp context
        a, b, c = None, None, None
        error = context.__exit__(a, b, c)
        if error:
            rank_zero_warn(a, b, c)
            raise Exception('apex unscale error')

        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()
        return closure_loss

    def configure_apex(
        self,
        amp: object,
        model: LightningModule,
        optimizers: List[Optimizer],
        amp_level: str,
    ) -> Tuple[LightningModule, List[Optimizer]]:
        r"""
        Override to init AMP your own way.
        Must return a model and list of optimizers.

        Args:
            amp: pointer to amp library object.
            model: pointer to current :class:`LightningModule`.
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

    def clip_gradients(self, grad_clip_val: Union[int, float], optimizer: Optimizer, norm_type: float):
        """
        This code is a modification of :meth:`torch.nn.utils.clip_grad_norm_` using a higher epsilon for fp16 weights.
        This is important when setting amp_level to O2, and the master weights are in fp16.
        Args:
            grad_clip_val: Maximum norm of gradients.
            optimizer: Optimizer with gradients that will be clipped.
            norm_type: (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        """
        model = self.trainer.get_model()
        parameters = model.parameters()
        max_norm = float(grad_clip_val)

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]

        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        clip_coef = max_norm / (total_norm + self.norm_clipping_epsilon)
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef.to(p.grad.device))

    @property
    def norm_clipping_epsilon(self):
        return 1e-5

    def optimizer_step(self, trainer, optimizer, closure):
        # apex amp does not yet support closures.
        # TODO: pass the closure to the step ASAP
        with trainer.profiler.profile("closure"):
            closure()

        if not self.trainer.train_loop.automatic_optimization:
            trainer.call_hook("on_after_backward")

        with trainer.profiler.profile("optimizer_step"):
            optimizer.step()
