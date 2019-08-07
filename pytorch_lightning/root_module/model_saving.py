import os
import re

import torch

from pytorch_lightning.pt_overrides.override_data_parallel import (
    LightningDistributedDataParallel, LightningDataParallel)


class ModelIO(object):

    def on_load_checkpoint(self, checkpoint):
        """
        Do something with the checkpoint
        Gives model a chance to load something before state_dict is restored
        :param checkpoint:
        :return:
        """
        pass

    def on_save_checkpoint(self, checkpoint):
        """
        Give the model a chance to add something to the checkpoint.
        state_dict is already there
        """
        pass

    # -------------------------
    # OPTIONAL HOOKS
    # -------------------------
    def on_hpc_save(self, checkpoint):
        """
        Hook to do whatever you need right before Slurm manager saves the model
        :return:
        """
        pass

    def on_hpc_load(self, checkpoint):
        """
        Hook to do whatever you need right before Slurm manager loads the model
        :return:
        """
        pass


class TrainerIO(object):

    def __get_model(self):
        is_dp_module = isinstance(self.model, (LightningDistributedDataParallel,
                                               LightningDataParallel))
        model = self.model.module if is_dp_module else self.model
        return model

    # --------------------
    # MODEL SAVE CHECKPOINT
    # --------------------
    def save_checkpoint(self, filepath):
        checkpoint = self.dump_checkpoint()

        # do the actual save
        torch.save(checkpoint, filepath)

    def restore(self, checkpoint_path, on_gpu):

        if on_gpu:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # load training state (affects trainer only)
        self.restore_training_state(checkpoint)

        # load model state
        model = self.__get_model()

        # load the state_dict on the model automatically
        model.load_state_dict(checkpoint['state_dict'])

    def dump_checkpoint(self):

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step
        }

        if self.checkpoint_callback is not None:
            checkpoint['checkpoint_callback_best'] = self.checkpoint_callback.best

        if self.early_stop_callback is not None:
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

        # add the state_dict from the model
        model = self.__get_model()
        checkpoint['state_dict'] = model.state_dict()

        # give the model a chance to add a few things
        model.on_save_checkpoint(checkpoint)

        return checkpoint

    # --------------------
    # HPC IO
    # --------------------
    def enable_auto_hpc_walltime_manager(self):
        if self.cluster is None:
            return

        # allow test tube to handle model check pointing automatically
        # only if proc 0 so we don't trigger world_size resubmits
        if self.proc_rank == 0:
            self.cluster.set_checkpoint_save_function(
                self.hpc_save,
                kwargs={
                    'folderpath': self.checkpoint_callback.filepath,
                    'experiment': self.experiment
                }
            )

        self.cluster.set_checkpoint_load_function(
            self.hpc_load,
            kwargs={
                'folderpath': self.checkpoint_callback.filepath,
                'on_gpu': self.on_gpu
            }
        )

    def restore_training_state(self, checkpoint):
        """
        Restore trainer state.
        Model will get its change to update
        :param checkpoint:
        :return:
        """
        if self.checkpoint_callback is not None:
            self.checkpoint_callback.best = checkpoint['checkpoint_callback_best']

        if self.early_stop_callback is not None:
            self.early_stop_callback.wait = checkpoint['early_stop_callback_wait']
            self.early_stop_callback.patience = checkpoint['early_stop_callback_patience']

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

    # ----------------------------------
    # PRIVATE OPS
    # ----------------------------------
    def hpc_save(self, folderpath, experiment):
        # make sure the checkpoint folder exists
        os.makedirs(folderpath, exist_ok=True)

        # save exp to make sure we get all the metrics
        experiment.save()

        # close experiment to avoid issues
        experiment.close()

        ckpt_number = self.max_ckpt_in_folder(folderpath) + 1

        if not os.path.exists(folderpath):
            os.makedirs(folderpath, exist_ok=True)
        filepath = '{}/hpc_ckpt_{}.ckpt'.format(folderpath, ckpt_number)

        # give model a chance to do something on hpc_save
        model = self.__get_model()
        checkpoint = self.dump_checkpoint()

        model.on_hpc_save(checkpoint)

        # do the actual save
        torch.save(checkpoint, filepath)

        return filepath

    def hpc_load(self, folderpath, on_gpu):
        filepath = '{}/hpc_ckpt_{}.ckpt'.format(folderpath, self.max_ckpt_in_folder(folderpath))

        if on_gpu:
            checkpoint = torch.load(filepath)
        else:
            checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

        # load training state (affects trainer only)
        self.restore_training_state(checkpoint)

        # load model state
        model = self.__get_model()

        # load the state_dict on the model automatically
        model.load_state_dict(checkpoint['state_dict'])

        # call model hook
        model.on_hpc_load(checkpoint)

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
