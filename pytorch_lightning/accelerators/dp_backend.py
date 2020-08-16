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
from torch import optim

from pytorch_lightning.overrides.data_parallel import LightningDataParallel
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException

try:
    from apex import amp
except ImportError:
    amp = None


class DataParallelBackend(object):

    def __init__(self, trainer):
        self.trainer = trainer
        self.model_autocast_original_forward = None

    def setup(self, model):
        # call setup after the ddp process has connected
        self.trainer.call_setup_hook(model)

        # put model on correct device
        model.cuda(self.trainer.root_gpu)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        optimizers, lr_schedulers, optimizer_frequencies = self.trainer.init_optimizers(model)
        self.trainer.optimizers = optimizers
        self.trainer.lr_schedulers = lr_schedulers
        self.trainer.optimizer_frequencies = optimizer_frequencies

        # hack forward to do autocast for the user
        self.model_autocast_original_forward = model.forward

        # init half precision
        if self.trainer.amp_backend:
            model = self.__init_half_precision(model)

        # init torch data parallel
        model = self.__init_torch_data_parallel(model)

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
            model, optimizers = model.configure_apex(amp, model, self.trainer.optimizers, self.trainer.amp_level)
            self.reinit_scheduler_properties(optimizers, self.trainer.lr_schedulers)

        return model

    def train(self):
        model = self.trainer.model
        results = self.trainer.run_pretrain_routine(model)
        return results

    def teardown(self):

        # replace the original fwd function
        self.trainer.model.forward = self.model_autocast_original_forward

    def reinit_scheduler_properties(self, optimizers: list, schedulers: list):
        """
        Reinitialize optimizer.step properties added by schedulers
        """
        for scheduler in schedulers:
            scheduler = scheduler['scheduler']

            for optimizer in optimizers:
                # check that we dont mix users optimizers and schedulers
                if scheduler.optimizer == optimizer:
                    # Find the mro belonging to the base lr scheduler class
                    for i, mro in enumerate(scheduler.__class__.__mro__):
                        is_regular_scheduler = optim.lr_scheduler._LRScheduler
                        is_lr_reduce_on_plateau = optim.lr_scheduler.ReduceLROnPlateau
                        if is_regular_scheduler or is_lr_reduce_on_plateau:
                            idx = i
                            state = scheduler.state_dict()
                        else:
                            state = None

                scheduler.__class__.__mro__[idx].__init__(scheduler, optimizer)
                if state is not None:
                    scheduler.load_state_dict(state)
