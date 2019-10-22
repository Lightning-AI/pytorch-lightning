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

try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class TrainerDPMixin(object):
    def copy_trainer_model_properties(self, model):
        if isinstance(model, LightningDataParallel):
            ref_model = model.module
        elif isinstance(model, LightningDistributedDataParallel):
            ref_model = model.module
        else:
            ref_model = model

        for m in [model, ref_model]:
            m.trainer = self
            m.on_gpu = self.on_gpu
            m.use_dp = self.use_dp
            m.use_ddp2 = self.use_ddp2
            m.use_ddp = self.use_ddp
            m.use_amp = self.use_amp
            m.testing = self.testing
            m.single_gpu = self.single_gpu

    def transfer_batch_to_gpu(self, batch, gpu_id):
        # base case: object can be directly moved using `cuda` or `to`
        if callable(getattr(batch, 'cuda', None)):
            return batch.cuda(gpu_id)

        elif callable(getattr(batch, 'to', None)):
            return batch.to(torch.device('cuda', gpu_id))

        # when list
        elif isinstance(batch, list):
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return batch

        # when tuple
        elif isinstance(batch, tuple):
            batch = list(batch)
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return tuple(batch)

        # when dict
        elif isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.transfer_batch_to_gpu(v, gpu_id)

            return batch

        # nothing matches, return the value as is without transform
        return batch

    def single_gpu_train(self, model):
        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        model.cuda(self.root_gpu)

        if self.use_amp:
            # An example
            model, optimizers = amp.initialize(
                model, self.optimizers, opt_level=self.amp_level,
            )
            self.optimizers = optimizers

        self.run_pretrain_routine(model)

    def dp_train(self, model):

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

        self.run_pretrain_routine(model)
