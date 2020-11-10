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

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.distributed.dist import LightningDistributed


class GPUAccelerator(Accelerator):
    amp_backend: AMPType

    def __init__(self, trainer, cluster_environment=None):
        """
        Runs training using a single GPU

        Example::

            # default
            trainer = Trainer(accelerator=GPUAccelerator())

        """
        super().__init__(trainer, cluster_environment)
        self.dist = LightningDistributed()
        self.nickname = None

    def setup(self, model):

        # call setup
        self.trainer.call_setup_hook(model)

        torch.cuda.set_device(self.trainer.root_gpu)
        model.cuda(self.trainer.root_gpu)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.setup_optimizers(model)

        # 16-bit
        model = self.trainer.precision_connector.connect(model)

        self.trainer.model = model

    def train(self):
        model = self.trainer.model

        # set up training routine
        self.trainer.train_loop.setup_training(model)

        # train or test
        results = self.train_or_test()
        return results

    def training_step(self, args):
        if self.trainer.amp_backend == AMPType.NATIVE:
            with torch.cuda.amp.autocast():
                output = self.__training_step(args)
        else:
            output = self.__training_step(args)

        return output

    def __training_step(self, args):
        batch = args[0]
        batch = self.to_device(batch)
        args[0] = batch
        output = self.trainer.model.training_step(*args)
        return output

    def validation_step(self, args):
        if self.trainer.amp_backend == AMPType.NATIVE:
            with torch.cuda.amp.autocast():
                output = self.__validation_step(args)
        else:
            output = self.__validation_step(args)

        return output

    def __validation_step(self, args):
        batch = args[0]
        batch = self.to_device(batch)
        args[0] = batch
        output = self.trainer.model.validation_step(*args)
        return output

    def test_step(self, args):
        if self.trainer.amp_backend == AMPType.NATIVE:
            with torch.cuda.amp.autocast():
                output = self.__test_step(args)
        else:
            output = self.__test_step(args)

        return output

    def __test_step(self, args):
        batch = args[0]
        batch = self.to_device(batch)
        args[0] = batch
        output = self.trainer.model.test_step(*args)
        return output

    def to_device(self, batch):
        gpu_id = 0
        if isinstance(self.trainer.data_parallel_device_ids, list):
            gpu_id = self.trainer.data_parallel_device_ids[0]

        # Don't copy the batch since there is a single gpu that the batch could
        # be referenced from and if there are multiple optimizers the batch will
        # wind up copying it to the same device repeatedly.
        return self.batch_to_device(batch, gpu_id)
