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

"""
Lightning can automate saving and loading checkpoints
=====================================================

Checkpointing is enabled by default to the current working directory.
To change the checkpoint path pass in::

    Trainer(default_root_dir='/your/path/to/save/checkpoints')


To modify the behavior of checkpointing pass in your own callback.

.. code-block:: python

    from pytorch_lightning.callbacks import ModelCheckpoint

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    trainer = Trainer(checkpoint_callback=checkpoint_callback)


Restoring training session
--------------------------

You might want to not only load a model but also continue training it. Use this method to
restore the trainer state as well. This will continue from the epoch and global step you last left off.
However, the dataloaders will start from the first batch again (if you shuffled it shouldn't matter).

Lightning will restore the session if you pass a logger with the same version and there's a saved checkpoint.

.. code-block:: python

    from pytorch_lightning import Trainer

    trainer = Trainer(
        resume_from_checkpoint=PATH
    )

    # this fit call loads model weights and trainer state
    # the trainer continues seamlessly from where you left off
    # without having to do anything else.
    trainer.fit(model)


The trainer restores:

- global_step
- current_epoch
- All optimizers
- All lr_schedulers
- Model weights

You can even change the logic of your model as long as the weights and "architecture" of
the system isn't different. If you add a layer, for instance, it might not work.

At a rough level, here's what happens inside Trainer :py:mod:`pytorch_lightning.base_module.saving.py`:

.. code-block:: python

    self.global_step = checkpoint['global_step']
    self.current_epoch = checkpoint['epoch']

    # restore the optimizers
    optimizer_states = checkpoint['optimizer_states']
    for optimizer, opt_state in zip(self.optimizers, optimizer_states):
        optimizer.load_state_dict(opt_state)

    # restore the lr schedulers
    lr_schedulers = checkpoint['lr_schedulers']
    for scheduler, lrs_state in zip(self.lr_schedulers, lr_schedulers):
        scheduler['scheduler'].load_state_dict(lrs_state)

    # uses the model you passed into trainer
    model.load_state_dict(checkpoint['state_dict'])

"""

import io
import os
import re
import signal
from abc import ABC
from subprocess import call

import torch
import torch.distributed as torch_distrib

import pytorch_lightning
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.overrides.data_parallel import LightningDataParallel, LightningDistributedDataParallel
from pytorch_lightning.utilities import AMPType, rank_zero_warn
from pytorch_lightning.utilities.cloud_io import atomic_save, get_filesystem
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.upgrade_checkpoint import KEYS_MAPPING as DEPRECATED_CHECKPOINT_KEYS
from pytorch_lightning.accelerators.base_backend import Accelerator


class TrainerIOMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    model: LightningModule
    on_gpu: bool
    root_gpu: ...
    resume_from_checkpoint: ...
    use_ddp: bool
    use_ddp2: bool
    use_horovod: bool
    checkpoint_callback: ...
    global_rank: int
    weights_save_path: str
    logger: LightningLoggerBase
    early_stop_callback: ...
    lr_schedulers: ...
    optimizers: ...
    on_tpu: bool
    num_training_batches: int
    accumulate_grad_batches: int
    scaler: ...
    use_tpu: bool
    amp_backend: AMPType
    accelerator_backend: Accelerator

    def get_model(self):
        is_dp_module = isinstance(self.model, (LightningDistributedDataParallel, LightningDataParallel))
        model = self.model.module if is_dp_module else self.model
        return model

    # --------------------
    # HPC SIGNAL HANDLING
    # --------------------
    def register_slurm_signal_handlers(self):
        # see if we're using slurm (not interactive)
        on_slurm = False
        try:
            job_name = os.environ['SLURM_JOB_NAME']
            if job_name != 'bash':
                on_slurm = True
        except Exception:
            pass

        if on_slurm:
            log.info('Set SLURM handle signals.')
            signal.signal(signal.SIGUSR1, self.sig_handler)
            signal.signal(signal.SIGTERM, self.term_handler)

    def sig_handler(self, signum, frame):  # pragma: no-cover
        if self.is_global_zero:
            # save weights
            log.info('handling SIGUSR1')
            self.hpc_save(self.weights_save_path, self.logger)

            # find job id
            job_id = os.environ['SLURM_JOB_ID']
            cmd = ['scontrol', 'requeue', job_id]

            # requeue job
            log.info(f'requeing job {job_id}...')
            result = call(cmd)

            # print result text
            if result == 0:
                log.info(f'requeued exp {job_id}')
            else:
                log.warning('requeue failed...')

            # close experiment to avoid issues
            self.logger.close()

    def term_handler(self, signum, frame):
        # save
        log.info("bypassing sigterm")
