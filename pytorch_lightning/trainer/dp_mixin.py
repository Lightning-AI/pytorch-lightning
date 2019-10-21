import os
import re
import signal
import pdb
from subprocess import call

import torch
import torch.distributed as dist
from pytorch_lightning.pt_overrides.override_data_parallel import (
    LightningDistributedDataParallel, LightningDataParallel)
from pytorch_lightning.utilities.debugging import MisconfigurationException


class TrainerDPMixin(object):

    def __dp_train(self, model):

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        model.cuda(self.root_gpu)

        # check for this bug (amp + dp + !01 doesn't work)
        # https://github.com/NVIDIA/apex/issues/227
        if self.use_dp and self.use_amp:
            m = f"""
            Amp level {self.amp_level} with DataParallel is not supported.
            See this note from NVIDIA for more info: https://github.com/NVIDIA/apex/issues/227.
            We recommend you switch to ddp if you want to use amp
            """
            raise MisconfigurationException(m)

        # create list of device ids
        device_ids = self.data_parallel_device_ids
        if type(device_ids) is int:
            device_ids = list(range(device_ids))

        model = LightningDataParallel(model, device_ids=device_ids)

        self.__run_pretrain_routine(model)
