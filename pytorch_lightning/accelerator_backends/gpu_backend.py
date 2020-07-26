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

try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True


class GPUBackend(object):

    def __init__(self, trainer):
        self.trainer = trainer

    def setup(self, model):

        # call setup
        if not self.trainer.testing:
            self.trainer.setup('fit')
            model.setup('fit')

        model.cuda(self.trainer.root_gpu)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        optimizers, lr_schedulers, optimizer_frequencies = self.trainer.init_optimizers(model)
        self.trainer.optimizers = optimizers
        self.trainer.lr_schedulers = lr_schedulers
        self.trainer.optimizer_frequencies = optimizer_frequencies

        # TODO: remove with dropping NVIDIA AMP support
        native_amp_available = hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast")
        if self.trainer.use_amp and not native_amp_available:
            model = self._setup_nvidia_apex(model)
        return model

    def train(self, model):
        results = self.trainer.run_pretrain_routine(model)
        return results

    def _setup_nvidia_apex(self, model):
        model, optimizers = model.configure_apex(amp, model, self.trainer.optimizers, self.trainer.amp_level)
        self.trainer.optimizers = optimizers
        self.trainer.reinit_scheduler_properties(self.trainer.optimizers, self.trainer.lr_schedulers)
        return model
