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
from typing import Union

import torch
from torch.optim import Optimizer

from pytorch_lightning.plugins.precision_plugin import PrecisionPlugin


class NativeAMPPlugin(PrecisionPlugin):

    def __init__(self, trainer=None):
        """
        Integrates native amp into Lightning's internals.
        """
        self.trainer = trainer

    def connect(self, model, optimizers):
        return model, optimizers

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        closure_loss = self.trainer.scaler.scale(closure_loss)

        automatic_optimization = self.trainer.train_loop.automatic_optimization

        # do backward pass
        if automatic_optimization:
            model = self.trainer.get_model()
            model.backward(closure_loss, optimizer, opt_idx)
        else:
            closure_loss.backward(*args, **kwargs)

        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()

        # unscale gradient to allow analyze within `on_after_backward`
        if not self.trainer.train_loop.should_accumulate() and automatic_optimization:
            self.trainer.scaler.unscale_(optimizer)

        return closure_loss

    def training_step(self, fx, args):
        with torch.cuda.amp.autocast():
            output = fx(*args)
        return output

    def clip_gradients(self, grad_clip_val: Union[int, float], optimizer: Optimizer, norm_type: float):
        model = self.trainer.get_model()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_val, norm_type=norm_type)
