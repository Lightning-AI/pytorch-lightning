import torch
import os
import re
import pdb
from pytorch_lightning.pt_overrides.override_data_parallel import LightningDistributedDataParallel

class ModelIO(object):

    def load_model_specific(self, checkpoint):
        """
        Do something with the checkpoint
        :param checkpoint:
        :return:
        """
        raise NotImplementedError

    def get_save_dict(self):
        """
        Return specific things for the model
        :return:
        """
        raise NotImplementedError

    # -------------------------
    # OPTIONAL HOOKS
    # -------------------------
    def on_hpc_save(self):
        """
        Hook to do whatever you need right before Slurm manager saves the model
        :return:
        """
        pass

    def on_hpc_load(self):
        """
        Hook to do whatever you need right before Slurm manager loads the model
        :return:
        """
        pass


class TrainerIO(object):

    # --------------------
    # MODEL SAVE CHECKPOINT
    # --------------------
    def save_checkpoint(self, filepath):
        checkpoint = self.dump_checkpoint()

        # do the actual save
        torch.save(checkpoint, filepath)

    def dump_checkpoint(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'checkpoint_callback_best': self.checkpoint_callback.best,
            'early_stop_callback_wait': self.early_stop_callback.wait,
            'early_stop_callback_patience': self.early_stop_callback.patience,
            'global_step': self.global_step
        }

        optimizer_states = []
        for i, optimizer in enumerate(self.optimizers):
            optimizer_states.append(optimizer.state_dict())

        checkpoint['optimizer_states'] = optimizer_states

        # request what to save from the model
        model = self.model.module if type(self.model) is LightningDistributedDataParallel else self.model
        checkpoint_dict = model.get_save_dict()

        # merge trainer and model saving items
        checkpoint.update(checkpoint_dict)
        return checkpoint

    # --------------------
    # HPC IO
    # --------------------
    def enable_auto_hpc_walltime_manager(self):
        if self.cluster is None:
            return

        # allow test tube to handle model check pointing automatically
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
        self.checkpoint_callback.best = checkpoint['checkpoint_callback_best']
        self.early_stop_callback.wait = checkpoint['early_stop_callback_wait']
        self.early_stop_callback.patience = checkpoint['early_stop_callback_patience']
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']

        # restore the optimizers
        optimizer_states = checkpoint['optimizer_states']
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)

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
        self.on_hpc_save()

        # request what to save from the model
        checkpoint_dict = self.dump_checkpoint()

        # do the actual save
        torch.save(checkpoint_dict, filepath)

    def hpc_load(self, folderpath, on_gpu):
        filepath = '{}/hpc_ckpt_{}.ckpt'.format(folderpath, self.max_ckpt_in_folder(folderpath))

        if on_gpu:
            checkpoint = torch.load(filepath)
        else:
            checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

        # load training state
        self.restore_training_state(checkpoint)

        # load model state
        model = self.model.module if type(self.model) is LightningDataParallel else self.model
        model.load_model_specific(checkpoint)

        # call model hook
        self.on_hpc_load()

    def max_ckpt_in_folder(self, path):
        files = os.listdir(path)
        files = [x for x in files if 'ckpt_' in x]
        if len(files) == 0:
            return 0

        ckpt_vs = []
        for name in files:
            name = name.split('ckpt_')[-1]
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
