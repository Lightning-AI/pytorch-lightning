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

from pytorch_lightning.utilities.exceptions import MisconfigurationException


class CPUBackend(object):

    def __init__(self, trainer):
        self.trainer = trainer

    def setup(self, model):
        # run through amp wrapper
        if self.trainer.amp_backend:
            raise MisconfigurationException('amp + cpu is not supported.  Please use a GPU option')

        # call setup after the ddp process has connected
        self.trainer.call_setup_hook(model)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        optimizers, lr_schedulers, optimizer_frequencies = self.trainer.init_optimizers(model)
        self.trainer.optimizers = optimizers
        self.trainer.lr_schedulers = lr_schedulers
        self.trainer.optimizer_frequencies = optimizer_frequencies

    def train(self, model):
        results = self.trainer.run_pretrain_routine(model)
        return results
