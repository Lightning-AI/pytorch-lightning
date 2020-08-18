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

import torch
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities import AMPType

try:
    from apex import amp
except ImportError:
    amp = None


class GPUBackend(object):
    amp_backend: AMPType

    def __init__(self, trainer):
        self.trainer = trainer

    def setup(self, model):

        # call setup
        self.trainer.call_setup_hook(model)

        torch.cuda.set_device(self.trainer.root_gpu)
        model.cuda(self.trainer.root_gpu)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        optimizers, lr_schedulers, optimizer_frequencies = self.trainer.init_optimizers(model)
        self.trainer.optimizers = optimizers
        self.trainer.lr_schedulers = lr_schedulers
        self.trainer.optimizer_frequencies = optimizer_frequencies

        if self.trainer.amp_backend == AMPType.APEX:
            model = self._setup_nvidia_apex(model)
        return model

    def train(self, model):
        results = self.trainer.run_pretrain_routine(model)
        return results

    def _setup_nvidia_apex(self, model: LightningModule):
        model, optimizers = model.configure_apex(amp, model, self.trainer.optimizers, self.trainer.amp_level)
        self.trainer.optimizers = optimizers
        self.trainer.reinit_scheduler_properties(self.trainer.optimizers, self.trainer.lr_schedulers)
        return model
