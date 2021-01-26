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
from typing import Optional, Union

import torch
from torch import optim

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.cluster_environments import ClusterEnvironment
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.distributed import LightningDistributed
from pytorch_lightning.overrides.data_parallel import LightningDataParallel
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class DataParallelAccelerator(Accelerator):

    def __init__(self, trainer, cluster_environment: Optional[ClusterEnvironment] = None):
        """
        Runs training using DP via manual start (not HPC cluster)

        Example::

            # default
            trainer = Trainer(accelerator=DataParallelAccelerator())

        """
        super().__init__(trainer, cluster_environment)
        self.model_autocast_original_forward = None
        self.dist = LightningDistributed()
        self.nickname = 'dp'

    def setup(self, model):
        # call setup after the ddp process has connected
        self.trainer.call_setup_hook(model)

        # put model on correct device
        model.cuda(self.trainer.root_gpu)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.setup_optimizers(model)

        # init torch data parallel
        model = self.__init_torch_data_parallel(model)

        # hack forward to do autocast for the user
        self.model_autocast_original_forward = model.forward

        # init half precision
        if self.trainer.amp_backend:
            model = self.__init_half_precision(model)

        self.trainer.model = model

    def __init_torch_data_parallel(self, model):
        # create list of device ids
        device_ids = self.trainer.data_parallel_device_ids
        if isinstance(device_ids, int):
            device_ids = list(range(device_ids))

        # set dp device
        torch.cuda.set_device(self.trainer.root_gpu)
        model = LightningDataParallel(model, device_ids=device_ids)
        return model

    def __init_half_precision(self, model):
        if self.trainer.amp_backend == AMPType.NATIVE:
            self.__init_native_amp(model)
        else:
            model = self.__init_nvidia_apex(model)
        return model

    def __init_native_amp(self, model):
        model.forward = torch.cuda.amp.autocast()(model.forward)

    def __init_nvidia_apex(self, model):
        # check for this bug (amp + dp + !01 doesn't work)
        # https://github.com/NVIDIA/apex/issues/227
        if self.trainer.amp_level == 'O2':
            raise MisconfigurationException(
                f'Amp level {self.trainer.amp_level} with DataParallel is not supported.'
                f' See this note from NVIDIA for more info: https://github.com/NVIDIA/apex/issues/227.'
                f' We recommend you switch to ddp if you want to use amp')
        else:
            model = self.trainer.precision_connector.connect(model)

        return model

    def teardown(self):
        # replace the original fwd function
        self.trainer.model.forward = self.model_autocast_original_forward
        self.barrier()

    def _step(self, args):
        if self.trainer.amp_backend == AMPType.NATIVE:
            with torch.cuda.amp.autocast():
                output = self.trainer.model(*args)
        else:
            output = self.trainer.model(*args)
        return output

    def training_step(self, args):
        return self._step(args)

    def validation_step(self, args):
        return self._step(args)

    def test_step(self, args):
        return self._step(args)

    def training_step_end(self, output):
        if isinstance(output, Result):
            output.dp_reduce()
        elif isinstance(output, torch.Tensor):
            output = output.mean()
        return output

    def validation_step_end(self, output):
        if isinstance(output, Result):
            output.dp_reduce()
        elif isinstance(output, torch.Tensor):
            output = output.mean()
        return output

    def test_step_end(self, output):
        if isinstance(output, Result):
            output.dp_reduce()
        elif isinstance(output, torch.Tensor):
            output = output.mean()
        return output

    def get_reference_model(self, model) -> LightningModule:
        if isinstance(model, LightningDataParallel):
            return model.module
        return model

    @property
    def require_distributed_sampler(self):
        return False
