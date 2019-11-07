import os
import warnings
import collections
from argparse import Namespace

import torch
import torch.distributed as dist

from pytorch_lightning.root_module.decorators import data_loader
from pytorch_lightning.root_module.grads import GradInformation
from pytorch_lightning.root_module.hooks import ModelHooks
from pytorch_lightning.root_module.memory import ModelSummary
from pytorch_lightning.root_module.model_saving import ModelIO
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv
import logging
from pytorch_lightning.pt_overrides.override_data_parallel import LightningDistributedDataParallel


class LightningModule(GradInformation, ModelIO, ModelHooks):

    def __init__(self, *args, **kwargs):
        super(LightningModule, self).__init__(*args, **kwargs)

        self.dtype = torch.FloatTensor
        self.exp_save_path = None
        self.current_epoch = 0
        self.global_step = 0
        self.loaded_optimizer_states_dict = {}
        self.trainer = None
        self.logger = None
        self.example_input_array = None

        # track if gpu was requested for checkpointing
        self.on_gpu = False
        self.use_dp = False
        self.use_ddp = False
        self.use_ddp2 = False
        self.use_amp = False

    def forward(self, *args, **kwargs):
        """
        Expand model in into whatever you need.
        Also need to return the target
        :param x:
        :return:
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        """
        return loss, dict with metrics for tqdm
        :param called with batch, batch_nb
        additional: optimizer_i if multiple optimizers used
        :return: dict with loss key and optional log, progress keys
                if implementing training_step, return whatever you need in that step
        """
        raise NotImplementedError

    def training_end(self, *args, **kwargs):
        """
        return loss, dict with metrics for tqdm
        :param called with outputs of training_step
        :return: dict with loss key and optional log, progress keys
        """
        pass

    def validation_step(self, *args, **kwargs):
        """
        return whatever outputs will need to be aggregated in validation_end
        OPTIONAL
        :param called with batch, batch_nb
        additional: dataset_i if multiple val datasets used
        :return:
        """
        pass

    def test_step(self, *args, **kwargs):
        """
        return whatever outputs will need to be aggregated in test_end
        OPTIONAL
        :param called with batch, batch_nb
        additional: dataset_i if multiple val datasets used
        :return:
        """
        pass

    def validation_end(self, outputs):
        """
        Outputs has the appended output after each validation step
        OPTIONAL
        :param outputs:
        :return: dic_with_metrics for tqdm
        """
        pass

    def test_end(self, outputs):
        """
        Outputs has the appended output after each test step
        OPTIONAL
        :param outputs:
        :return: dic_with_metrics for tqdm
        """
        pass

    def configure_ddp(self, model, device_ids):
        """
        Override to init DDP in a different way or use your own wrapper.
        Must return model.
        :param model:
        :param device_ids:
        :return: DDP wrapped model
        """
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )
        return model

    def init_ddp_connection(self, proc_rank, world_size):
        """
        Connect all procs in the world using the env:// init
        Use the first node as the root address
        """

        # use slurm job id for the port number
        # guarantees unique ports across jobs from same grid search
        try:
            # use the last 4 numbers in the job id as the id
            default_port = os.environ['SLURM_JOB_ID']
            default_port = default_port[-4:]

            # all ports should be in the 10k+ range
            default_port = int(default_port) + 15000

        except Exception as e:
            default_port = 12910

        # if user gave a port number, use that one instead
        try:
            default_port = os.environ['MASTER_PORT']
        except Exception:
            os.environ['MASTER_PORT'] = str(default_port)

        # figure out the root node addr
        try:
            root_node = os.environ['SLURM_NODELIST'].split(' ')[0]
        except Exception:
            root_node = '127.0.0.2'

        root_node = self.trainer.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        dist.init_process_group('nccl', rank=proc_rank, world_size=world_size)

    def configure_apex(self, amp, model, optimizers, amp_level):
        """
        Override to init AMP your own way
        Must return a model and list of optimizers
        :param amp:
        :param model:
        :param optimizers:
        :param amp_level:
        :return: Apex wrapped model and optimizers
        """
        model, optimizers = amp.initialize(
            model, optimizers, opt_level=amp_level,
        )

        return model, optimizers

    def configure_optimizers(self):
        """
        Return a list of optimizers and a list of schedulers (could be empty)
        :return:
        """
        raise NotImplementedError

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        """
        Do something instead of the standard optimizer behavior
        :param epoch_nb:
        :param batch_nb:
        :param optimizer:
        :param optimizer_i:
        :param second_order_closure: closure for second order methods
        :return:
        """
        if isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(second_order_closure)
        else:
            optimizer.step()

        # clear gradients
        optimizer.zero_grad()

    def tbptt_split_batch(self, batch, split_size):
        """
        Return list of batch splits. Each split will be passed to forward_step to enable truncated
        back propagation through time. The default implementation splits root level Tensors and
        Sequences at dim=1 (i.e. time dim). It assumes that each time dim is the same length.
        :return:
        """
        time_dims = [len(x[0]) for x in batch if isinstance(
            x, torch.Tensor) or isinstance(x, collections.Sequence)]
        assert len(time_dims) >= 1, "Unable to determine batch time dimension"
        assert all(x == time_dims[0] for x in time_dims), "Batch time dimension length is ambiguous"

        splits = []
        for t in range(0, time_dims[0], split_size):
            batch_split = []
            for i, x in enumerate(batch):
                if isinstance(x, torch.Tensor):
                    split_x = x[:, t:t + split_size]
                elif isinstance(x, collections.Sequence):
                    split_x = [None] * len(x)
                    for batch_idx in range(len(x)):
                        split_x[batch_idx] = x[batch_idx][t:t + split_size]

                batch_split.append(split_x)

            splits.append(batch_split)

        return splits

    @data_loader
    def tng_dataloader(self):
        """
        Implement a PyTorch DataLoader
        * Deprecated in v0.5.0. use train_dataloader instead. *
        :return:
        """
        raise NotImplementedError

    @data_loader
    def train_dataloader(self):
        """
        Implement a PyTorch DataLoader
        :return:
        """
        #
        try:
            output = self.tng_dataloader()
            warnings.warn("tng_dataloader has been renamed to train_dataloader since v0.5.0",
                          DeprecationWarning)
            return output
        except NotImplementedError:
            raise NotImplementedError

    @data_loader
    def test_dataloader(self):
        """
        Implement a PyTorch DataLoader
        :return:
        """
        return None

    @data_loader
    def val_dataloader(self):
        """
        Implement a PyTorch DataLoader
        :return:
        """
        return None

    @classmethod
    def load_from_metrics(cls, weights_path, tags_csv):
        """
        Primary way of loading model from csv weights path
        :param weights_path:
        :param tags_csv:
        :param map_location: dic for mapping storage {'cuda:1':'cuda:0'}
        :return:
        """
        hparams = load_hparams_from_tags_csv(tags_csv)
        hparams.__setattr__('on_gpu', False)

        # load on CPU only to avoid OOM issues
        # then its up to user to put back on GPUs
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

        # load the state_dict on the model automatically
        model = cls(hparams)
        model.load_state_dict(checkpoint['state_dict'])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        """
        Primary way of loading model from a checkpoint
        :param checkpoint_path:
        :param map_location: dic for mapping storage {'cuda:1':'cuda:0'}
        :return:
        """

        # load on CPU only to avoid OOM issues
        # then its up to user to put back on GPUs
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        try:
            ckpt_hparams = checkpoint['hparams']
        except KeyError:
            raise IOError(
                "Checkpoint does not contain hyperparameters. Are your model hyperparameters stored"
                "in self.hparams?"
            )
        hparams = Namespace(**ckpt_hparams)

        # load the state_dict on the model automatically
        model = cls(hparams)
        model.load_state_dict(checkpoint['state_dict'])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model

    def summarize(self, mode):
        model_summary = ModelSummary(self, mode=mode)
        logging.info(model_summary)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

        self.train()
