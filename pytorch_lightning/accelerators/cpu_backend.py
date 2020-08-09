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
from pytorch_lightning import LightningModule
from pytorch_lightning.accelerators.base import LightningBackend
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class CPUBackend(LightningBackend):

    def setup(self, model: LightningModule):
        super().setup(model)
        # run through amp wrapper
        if self._trainer.amp_backend:
            raise MisconfigurationException('amp + cpu is not supported.  Please use a GPU option')

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        optimizers, lr_schedulers, optimizer_frequencies = self._trainer.init_optimizers(model)
        self._trainer.optimizers = optimizers
        self._trainer.lr_schedulers = lr_schedulers
        self._trainer.optimizer_frequencies = optimizer_frequencies

    def train(self):
        results = self._trainer.run_pretrain_routine(self._model)
        return results
