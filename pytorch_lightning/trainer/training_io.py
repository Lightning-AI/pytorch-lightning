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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.overrides.data_parallel import (
    LightningDistributedDataParallel,
    LightningDataParallel,
)
from pytorch_lightning.utilities import rank_zero_warn, AMPType
from pytorch_lightning.utilities.cloud_io import gfile, makedirs
from pytorch_lightning.utilities.cloud_io import load as pl_load, atomic_save

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True

try:
    from apex import amp
except ImportError:
    amp = None

try:
    import horovod.torch as hvd
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True

try:
    from omegaconf import Container
except ImportError:
    OMEGACONF_AVAILABLE = False
else:
    OMEGACONF_AVAILABLE = True


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

    def get_model(self):
        is_dp_module = isinstance(self.model, (LightningDistributedDataParallel, LightningDataParallel))
        model = self.model.module if is_dp_module else self.model
        return model

    # --------------------
    # CHECK-POINTING
    # --------------------
    def restore_weights(self, model: LightningModule):
        """
        We attempt to restore weights in this order:
        1. HPC weights.
        2. if no HPC weights restore checkpoint_path weights
        3. otherwise don't restore weights
        """
        # clear cache before restore
        if self.on_gpu:
            torch.cuda.empty_cache()

        # if script called from hpc resubmit, load weights
        did_restore_hpc_weights = self.restore_hpc_weights_if_needed(model)

        # clear cache after restore
        if self.on_gpu:
            torch.cuda.empty_cache()

        if not did_restore_hpc_weights:
            if self.resume_from_checkpoint is not None:
                self.restore(self.resume_from_checkpoint, on_gpu=self.on_gpu)

        # wait for all models to restore weights
        if self.use_ddp or self.use_ddp2:
            # wait for all processes to catch up
            torch_distrib.barrier()

        # wait for all models to restore weights
        if self.on_tpu and XLA_AVAILABLE:
            # wait for all processes to catch up
            torch_xla.core.xla_model.rendezvous("pl.TrainerIOMixin.restore_weights")

        elif self.use_horovod:
            # wait for all processes to catch up
            hvd.join()

        # clear cache after restore
        if self.on_gpu:
            torch.cuda.empty_cache()

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

    # --------------------
    # MODEL SAVE CHECKPOINT
    # --------------------

    def save_checkpoint(self, filepath, weights_only: bool = False):
        checkpoint = self.dump_checkpoint(weights_only)

        if self.is_global_zero:
            # do the actual save
            try:
                atomic_save(checkpoint, filepath)
            except AttributeError as err:
                if LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:
                    del checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
                rank_zero_warn(
                    'Warning, `module_arguments` dropped from checkpoint.' f' An attribute is not picklable {err}'
                )
                atomic_save(checkpoint, filepath)

    def restore(self, checkpoint_path: str, on_gpu: bool):
        """
        Restore training state from checkpoint.
        Also restores all training state like:
        - epoch
        - callbacks
        - schedulers
        - optimizer
        """

        # if on_gpu:
        #     checkpoint = torch.load(checkpoint_path)
        # else:
        # load on CPU first
        checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

        # load model state
        model = self.get_model()

        # load the state_dict on the model automatically
        model.load_state_dict(checkpoint['state_dict'])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        if on_gpu:
            model.cuda(self.root_gpu)

        # restore amp scaling
        if self.amp_backend == AMPType.NATIVE and 'native_amp_scaling_state' in checkpoint:
            self.scaler.load_state_dict(checkpoint['native_amp_scaling_state'])
        elif self.amp_backend == AMPType.APEX and 'amp_scaling_state' in checkpoint:
            amp.load_state_dict(checkpoint['amp_scaling_state'])

        # load training state (affects trainer only)
        self.restore_training_state(checkpoint)

    def dump_checkpoint(self, weights_only: bool = False) -> dict:
        """Creating model checkpoint.

        Args:
            weights_only: saving model weights only

        Return:
             structured dictionary
        """
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'global_step': self.global_step + 1,
            'pytorch-lightning_version': pytorch_lightning.__version__,
        }

        if not weights_only:

            # TODO support more generic way for callbacks to persist a state_dict in a checkpoint
            checkpoint_callbacks = [c for c in self.callbacks if isinstance(c, ModelCheckpoint)]
            early_stopping_callbacks = [c for c in self.callbacks if isinstance(c, EarlyStopping)]

            if checkpoint_callbacks:
                # we add the official checkpoint callback to the end of the list
                # extra user provided callbacks will not be persisted yet
                checkpoint[ModelCheckpoint.CHECKPOINT_STATE_BEST_SCORE] = self.checkpoint_callback.best_model_score
                checkpoint[ModelCheckpoint.CHECKPOINT_STATE_BEST_PATH] = self.checkpoint_callback.best_model_path

            if early_stopping_callbacks and checkpoint_callbacks:
                # we add the official early stopping callback to the end of the list
                # extra user provided callbacks will not be persisted yet
                checkpoint['early_stop_callback_state_dict'] = early_stopping_callbacks[-1].state_dict()

            # save optimizers
            optimizer_states = []
            for i, optimizer in enumerate(self.optimizers):
                optimizer_states.append(optimizer.state_dict())
            checkpoint['optimizer_states'] = optimizer_states

            # save lr schedulers
            lr_schedulers = []
            for scheduler in self.lr_schedulers:
                lr_schedulers.append(scheduler['scheduler'].state_dict())
            checkpoint['lr_schedulers'] = lr_schedulers

            # save native amp scaling
            if self.amp_backend == AMPType.NATIVE and not self.use_tpu and self.scaler is not None:
                checkpoint['native_amp_scaling_state'] = self.scaler.state_dict()
            elif self.amp_backend == AMPType.APEX:
                checkpoint['amp_scaling_state'] = amp.state_dict()

        # add the module_arguments and state_dict from the model
        model = self.get_model()

        checkpoint['state_dict'] = model.state_dict()

        if model.hparams:
            if hasattr(model, '_hparams_name'):
                checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_NAME] = model._hparams_name
            # add arguments to the checkpoint
            checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY] = model.hparams
            if OMEGACONF_AVAILABLE:
                if isinstance(model.hparams, Container):
                    checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_TYPE] = type(model.hparams)

        # give the model a chance to add a few things
        model.on_save_checkpoint(checkpoint)

        return checkpoint

    # --------------------
    # HPC IO
    # --------------------
    def restore_hpc_weights_if_needed(self, model: LightningModule):
        """If there is a set of hpc weights, use as signal to restore model."""
        did_restore = False

        # look for hpc weights
        folderpath = str(self.weights_save_path)
        if gfile.exists(folderpath):
            files = gfile.listdir(folderpath)
            hpc_weight_paths = [x for x in files if 'hpc_ckpt' in x]

            # if hpc weights exist restore model
            if len(hpc_weight_paths) > 0:
                self.hpc_load(folderpath, self.on_gpu)
                did_restore = True
        return did_restore

    def restore_training_state(self, checkpoint):
        """
        Restore trainer state.
        Model will get its change to update
        :param checkpoint:
        :return:
        """
        if 'optimizer_states' not in checkpoint or 'lr_schedulers' not in checkpoint:
            raise KeyError(
                'Trying to restore training state but checkpoint contains only the model.'
                ' This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`.'
            )

        # TODO support more generic way for callbacks to load callback state_dicts
        checkpoint_callbacks = [c for c in self.callbacks if isinstance(c, ModelCheckpoint)]
        early_stopping_callbacks = [c for c in self.callbacks if isinstance(c, EarlyStopping)]

        if checkpoint_callbacks:
            if ModelCheckpoint.CHECKPOINT_STATE_BEST_SCORE in checkpoint:
                checkpoint_callbacks[-1].best_model_score = checkpoint[ModelCheckpoint.CHECKPOINT_STATE_BEST_SCORE]
            else:
                # Old naming until version 0.7.6
                rank_zero_warn(
                    'Loading a checkpoint created with an old version of Lightning; '
                    'this will not be supported in the future.'
                )
                checkpoint_callbacks[-1].best_model_score = checkpoint['checkpoint_callback_best']
            checkpoint_callbacks[-1].best_model_path = checkpoint[ModelCheckpoint.CHECKPOINT_STATE_BEST_PATH]

        if early_stopping_callbacks:
            state = checkpoint['early_stop_callback_state_dict']
            early_stopping_callbacks[-1].load_state_dict(state)

        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']

        # Division deals with global step stepping once per accumulated batch
        # Inequality deals with different global step for odd vs even num_training_batches
        n_accum = 1 if self.accumulate_grad_batches is None else self.accumulate_grad_batches
        expected_steps = self.num_training_batches / n_accum
        if self.num_training_batches != 0 and self.global_step % expected_steps > 1:
            rank_zero_warn(
                "You're resuming from a checkpoint that ended mid-epoch. "
                "This can cause unreliable results if further training is done, "
                "consider using an end of epoch checkpoint. "
            )

        # restore the optimizers
        optimizer_states = checkpoint['optimizer_states']
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)

            # move optimizer to GPU 1 weight at a time
            # avoids OOM
            if self.root_gpu is not None:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.root_gpu)

        # restore the lr schedulers
        lr_schedulers = checkpoint['lr_schedulers']
        for scheduler, lrs_state in zip(self.lr_schedulers, lr_schedulers):
            scheduler['scheduler'].load_state_dict(lrs_state)

    # ----------------------------------
    # PRIVATE OPS
    # ----------------------------------
    def hpc_save(self, folderpath: str, logger):
        # make sure the checkpoint folder exists
        folderpath = str(folderpath)  # because the tests pass a path object
        if not gfile.exists(folderpath):
            makedirs(folderpath)

        # save logger to make sure we get all the metrics
        logger.save()

        ckpt_number = self.max_ckpt_in_folder(folderpath) + 1

        if not gfile.exists(folderpath):
            makedirs(folderpath)
        filepath = os.path.join(folderpath, f'hpc_ckpt_{ckpt_number}.ckpt')

        # give model a chance to do something on hpc_save
        model = self.get_model()
        checkpoint = self.dump_checkpoint()

        model.on_hpc_save(checkpoint)

        # do the actual save
        # TODO: fix for anything with multiprocess DP, DDP, DDP2
        try:
            atomic_save(checkpoint, filepath)
        except AttributeError as err:
            if LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:
                del checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
            rank_zero_warn(
                'warning, `module_arguments` dropped from checkpoint.' f' An attribute is not picklable {err}'
            )
            atomic_save(checkpoint, filepath)

        return filepath

    def hpc_load(self, folderpath, on_gpu):
        filepath = '{}/hpc_ckpt_{}.ckpt'.format(folderpath, self.max_ckpt_in_folder(folderpath))

        # load on CPU first
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

        # load model state
        model = self.get_model()

        # load the state_dict on the model automatically
        model.load_state_dict(checkpoint['state_dict'])

        # restore amp scaling
        if self.amp_backend == AMPType.NATIVE and 'native_amp_scaling_state' in checkpoint:
            self.scaler.load_state_dict(checkpoint['native_amp_scaling_state'])
        elif self.amp_backend == AMPType.APEX and 'amp_scaling_state' in checkpoint:
            amp.load_state_dict(checkpoint['amp_scaling_state'])

        if self.root_gpu is not None:
            model.cuda(self.root_gpu)

        # load training state (affects trainer only)
        self.restore_training_state(checkpoint)

        # call model hook
        model.on_hpc_load(checkpoint)

        log.info(f'restored hpc model from: {filepath}')

    def max_ckpt_in_folder(self, path, name_key='ckpt_'):
        files = gfile.listdir(str(path))
        files = [x for x in files if name_key in x]
        if len(files) == 0:
            return 0

        ckpt_vs = []
        for name in files:
            name = name.split(name_key)[-1]
            name = re.sub('[^0-9]', '', name)
            ckpt_vs.append(int(name))

        return max(ckpt_vs)
