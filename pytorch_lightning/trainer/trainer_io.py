"""
Lightning can automate saving and loading checkpoints
=====================================================

Checkpointing is enabled by default to the current working directory.
To change the checkpoint path pass in::

    Trainer(default_save_path='/your/path/to/save/checkpoints')


To modify the behavior of checkpointing pass in your own callback.

.. code-block:: python

    from pytorch_lightning.callbacks import ModelCheckpoint

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_best_only=True,
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
    from pytorch_lightning.logging import TestTubeLogger

    logger = TestTubeLogger(
        save_dir='./savepath',
        version=1  # An existing version with a saved checkpoint
    )
    trainer = Trainer(
        logger=logger,
        default_save_path='./savepath'
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

At a rough level, here's what happens inside Trainer :py:mod:`pytorch_lightning.base_module.model_saving.py`:

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
        scheduler.load_state_dict(lrs_state)

    # uses the model you passed into trainer
    model.load_state_dict(checkpoint['state_dict'])

"""

import os
import re
import signal
import warnings
from subprocess import call
import logging
from abc import ABC

import torch
import torch.distributed as dist

from pytorch_lightning.overrides.data_parallel import (
    LightningDistributedDataParallel,
    LightningDataParallel,
)


class TrainerIOMixin(ABC):

    def __init__(self):
        # this is just a summary on variables used in this abstract class,
        #  the proper values/initialisation should be done in child class
        self.model = None
        self.on_gpu = None
        self.root_gpu = None
        self.resume_from_checkpoint = None
        self.use_ddp = None
        self.use_ddp2 = None
        self.checkpoint_callback = None
        self.proc_rank = None
        self.weights_save_path = None
        self.logger = None
        self.early_stop_callback = None
        self.lr_schedulers = None
        self.optimizers = None

    def get_model(self):
        is_dp_module = isinstance(self.model, (LightningDistributedDataParallel,
                                               LightningDataParallel))
        model = self.model.module if is_dp_module else self.model
        return model

    # --------------------
    # CHECK-POINTING
    # --------------------
    def restore_weights(self, model):
        """
        To restore weights we have two cases.
        First, attempt to restore hpc weights. If successful, don't restore
        other weights.

        Otherwise, try to restore actual weights
        :param model:
        :return:
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
            else:
                # restore weights if same exp version
                self.restore_state_if_checkpoint_exists(model)

        # wait for all models to restore weights
        if self.use_ddp or self.use_ddp2:
            # wait for all processes to catch up
            dist.barrier()

        # clear cache after restore
        if self.on_gpu:
            torch.cuda.empty_cache()

    def restore_state_if_checkpoint_exists(self, model):
        did_restore = False

        # do nothing if there's not dir or callback
        no_ckpt_callback = (self.checkpoint_callback is None) or (not self.checkpoint_callback)
        if no_ckpt_callback or not os.path.exists(self.checkpoint_callback.filepath):
            return did_restore

        # restore trainer state and model if there is a weight for this experiment
        last_epoch = -1
        last_ckpt_name = None

        # find last epoch
        checkpoints = os.listdir(self.checkpoint_callback.filepath)
        for name in checkpoints:
            # ignore hpc ckpts
            if 'hpc_' in name:
                continue

            if '.ckpt' in name:
                epoch = name.split('epoch_')[1]
                epoch = int(re.sub('[^0-9]', '', epoch))

                if epoch > last_epoch:
                    last_epoch = epoch
                    last_ckpt_name = name

        # restore last checkpoint
        if last_ckpt_name is not None:
            last_ckpt_path = os.path.join(self.checkpoint_callback.filepath, last_ckpt_name)
            self.restore(last_ckpt_path, self.on_gpu)
            logging.info(f'model and trainer restored from checkpoint: {last_ckpt_path}')
            did_restore = True

        return did_restore

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
        except Exception as e:
            pass

        if on_slurm:
            logging.info('set slurm handle signals')
            signal.signal(signal.SIGUSR1, self.sig_handler)
            signal.signal(signal.SIGTERM, self.term_handler)

    def sig_handler(self, signum, frame):
        if self.proc_rank == 0:
            # save weights
            logging.info('handling SIGUSR1')
            self.hpc_save(self.weights_save_path, self.logger)

            # find job id
            job_id = os.environ['SLURM_JOB_ID']
            cmd = 'scontrol requeue {}'.format(job_id)

            # requeue job
            logging.info('\nrequeing job {job_id}...')
            result = call(cmd, shell=True)

            # print result text
            if result == 0:
                logging.info('requeued exp {job_id}')
            else:
                logging.info('requeue failed...')

            # close experiment to avoid issues
            self.logger.close()

    def term_handler(self, signum, frame):
        # save
        logging.info("bypassing sigterm")

    # --------------------
    # MODEL SAVE CHECKPOINT
    # --------------------
    def save_checkpoint(self, filepath):
        checkpoint = self.dump_checkpoint()

        # do the actual save
        try:
            torch.save(checkpoint, filepath)
        except AttributeError:
            if 'hparams' in checkpoint:
                del checkpoint['hparams']

            torch.save(checkpoint, filepath)

    def restore(self, checkpoint_path, on_gpu):

        # if on_gpu:
        #     checkpoint = torch.load(checkpoint_path)
        # else:
        # load on CPU first
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # load model state
        model = self.get_model()

        # load the state_dict on the model automatically
        model.load_state_dict(checkpoint['state_dict'])
        if on_gpu:
            model.cuda(self.root_gpu)

        # load training state (affects trainer only)
        self.restore_training_state(checkpoint)

    def dump_checkpoint(self):

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step
        }

        if self.checkpoint_callback is not None and self.checkpoint_callback is not False:
            checkpoint['checkpoint_callback_best'] = self.checkpoint_callback.best

        if self.early_stop_callback is not None and self.checkpoint_callback is not False:
            checkpoint['early_stop_callback_wait'] = self.early_stop_callback.wait
            checkpoint['early_stop_callback_patience'] = self.early_stop_callback.patience

        # save optimizers
        optimizer_states = []
        for i, optimizer in enumerate(self.optimizers):
            optimizer_states.append(optimizer.state_dict())

        checkpoint['optimizer_states'] = optimizer_states

        # save lr schedulers
        lr_schedulers = []
        for i, scheduler in enumerate(self.lr_schedulers):
            lr_schedulers.append(scheduler.state_dict())

        checkpoint['lr_schedulers'] = lr_schedulers

        # add the hparams and state_dict from the model
        model = self.get_model()
        checkpoint['state_dict'] = model.state_dict()
        if hasattr(model, "hparams"):
            checkpoint['hparams'] = vars(model.hparams)
        else:
            warnings.warn(
                "Did not find hyperparameters at model.hparams. Saving checkpoint without"
                " hyperparameters"
            )

        # give the model a chance to add a few things
        model.on_save_checkpoint(checkpoint)

        return checkpoint

    # --------------------
    # HPC IO
    # --------------------
    def restore_hpc_weights_if_needed(self, model):
        """
        If there is a set of hpc weights, use as signal to restore model
        :param model:
        :return:
        """
        did_restore = False

        # look for hpc weights
        folderpath = self.weights_save_path
        if os.path.exists(folderpath):
            files = os.listdir(folderpath)
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
        if self.checkpoint_callback is not None and self.checkpoint_callback is not False:
            self.checkpoint_callback.best = checkpoint['checkpoint_callback_best']

        if self.early_stop_callback is not None and self.early_stop_callback is not False:
            self.early_stop_callback.wait = checkpoint['early_stop_callback_wait']
            self.early_stop_callback.patience = checkpoint['early_stop_callback_patience']

        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']

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
            scheduler.load_state_dict(lrs_state)

    # ----------------------------------
    # PRIVATE OPS
    # ----------------------------------
    def hpc_save(self, folderpath, logger):
        # make sure the checkpoint folder exists
        os.makedirs(folderpath, exist_ok=True)

        # save logger to make sure we get all the metrics
        logger.save()

        ckpt_number = self.max_ckpt_in_folder(folderpath) + 1

        if not os.path.exists(folderpath):
            os.makedirs(folderpath, exist_ok=True)
        filepath = '{}/hpc_ckpt_{}.ckpt'.format(folderpath, ckpt_number)

        # give model a chance to do something on hpc_save
        model = self.get_model()
        checkpoint = self.dump_checkpoint()

        model.on_hpc_save(checkpoint)

        # do the actual save
        # TODO: fix for anything with multiprocess DP, DDP, DDP2
        try:
            torch.save(checkpoint, filepath)
        except AttributeError:
            if 'hparams' in checkpoint:
                del checkpoint['hparams']

            torch.save(checkpoint, filepath)

        return filepath

    def hpc_load(self, folderpath, on_gpu):
        filepath = '{}/hpc_ckpt_{}.ckpt'.format(folderpath, self.max_ckpt_in_folder(folderpath))

        # load on CPU first
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

        # load model state
        model = self.get_model()

        # load the state_dict on the model automatically
        model.load_state_dict(checkpoint['state_dict'])

        if self.root_gpu is not None:
            model.cuda(self.root_gpu)

        # load training state (affects trainer only)
        self.restore_training_state(checkpoint)

        # call model hook
        model.on_hpc_load(checkpoint)

        logging.info(f'restored hpc model from: {filepath}')

    def max_ckpt_in_folder(self, path, name_key='ckpt_'):
        files = os.listdir(path)
        files = [x for x in files if name_key in x]
        if len(files) == 0:
            return 0

        ckpt_vs = []
        for name in files:
            name = name.split(name_key)[-1]
            name = re.sub('[^0-9]', '', name)
            ckpt_vs.append(int(name))

        return max(ckpt_vs)


def load_hparams_from_tags_csv(tags_csv):
    from argparse import Namespace
    import pandas as pd

    tags_df = pd.read_csv(tags_csv)
    dic = tags_df.to_dict(orient='records')

    ns_dict = {row['key']: convert(row['value']) for row in dic}

    ns = Namespace(**ns_dict)
    return ns


def convert(val):
    constructors = [int, float, str]

    if type(val) is str:
        if val.lower() == 'true':
            return True
        if val.lower() == 'false':
            return False

    for c in constructors:
        try:
            return c(val)
        except ValueError:
            pass
    return val
