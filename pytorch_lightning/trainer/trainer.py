"""
The trainer handles all the logic for running a val loop, training loop, distributing, etc.. .
"""

import os
import re
import warnings

import numpy as np
import tqdm
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim.optimizer import Optimizer

from pytorch_lightning.root_module.root_module import LightningModule
from pytorch_lightning.root_module import memory
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.trainer.trainer_io import TrainerIO
from pytorch_lightning.pt_overrides.override_data_parallel import (
    LightningDistributedDataParallel, LightningDataParallel)
from pytorch_lightning.callbacks import GradientAccumulationScheduler, \
    ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.debugging import MisconfigurationException
import pdb
from pytorch_lightning.trainer import ignored_warnings


try:
    from apex import amp
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


def reduce_distributed_output(output, nb_gpus):
    if nb_gpus <= 1:
        return output

    # when using DP, we get one output per gpu
    # average outputs and return
    if type(output) is torch.Tensor:
        return output.mean()

    for k, v in output.items():
        # recurse on nested dics
        if isinstance(output[k], dict):
            output[k] = reduce_distributed_output(output[k], nb_gpus)

        # reduce only metrics that have the same nb of gpus
        elif output[k].size(0) == nb_gpus:
            reduced = torch.mean(output[k])
            output[k] = reduced
    return output


class Trainer(TrainerIO):

    def __init__(self,
                 logger=None,
                 checkpoint_callback=None,
                 early_stop_callback=None,
                 default_save_path=None,
                 gradient_clip_val=0,
                 process_position=0,
                 nb_gpu_nodes=1,
                 gpus=None,
                 log_gpu_memory=None,
                 show_progress_bar=True,
                 overfit_pct=0.0,
                 track_grad_norm=-1,
                 check_val_every_n_epoch=1,
                 fast_dev_run=False,
                 accumulate_grad_batches=1,
                 max_nb_epochs=1000,
                 min_nb_epochs=1,
                 train_percent_check=1.0,
                 val_percent_check=1.0,
                 test_percent_check=1.0,
                 val_check_interval=1.0,
                 log_save_interval=100,
                 row_log_interval=10,
                 distributed_backend=None,
                 use_amp=False,
                 print_nan_grads=False,
                 print_weights_summary=True,
                 weights_save_path=None,
                 amp_level='O2',
                 nb_sanity_val_steps=5):
        """

        :param logger: Logger for experiment tracking
        :param checkpoint_callback: Callback for checkpointing
        :param early_stop_callback: Callback for early stopping
        :param default_save_path: Default path for logs+weights if no logger/ckpt_callback passed
        :param gradient_clip_val: int. 0 means don't clip.
        :param process_position: shown in the tqdm bar
        :param nb_gpu_nodes: number of GPU nodes
        :param gpus: int. (ie: 2 gpus) OR list to specify which GPUs [0, 1] or '0,1'
        :param log_gpu_memory: str. None, 'min_max', 'all'
        :param show_progress_bar: Bool. If true shows tqdm bar
        :param overfit_pct: float. uses this much of all datasets
        :param track_grad_norm: int. -1 no tracking. Otherwise tracks that norm
        :param check_val_every_n_epoch: int. check val every n train epochs
        :param fast_dev_run: Bool. runs full iteration over everything to find bugs
        :param accumulate_grad_batches: int. Accumulates grads every k batches
        :param max_nb_epochs: int.
        :param min_nb_epochs: int.
        :param train_percent_check: int. How much of train set to check
        :param val_percent_check: int. How much of val set to check
        :param test_percent_check: int. How much of test set to check
        :param val_check_interval: int. Check val this frequently within a train epoch
        :param log_save_interval: int. Writes logs to disk this often
        :param row_log_interval: int. How often to add logging rows
        :param distributed_backend: str. Options: 'dp', 'ddp', 'ddp2'.
        :param use_amp: Bool. If true uses apex for 16bit precision
        :param print_nan_grads: Bool. Prints nan gradients
        :param print_weights_summary: Bool. Prints summary of weights
        :param weights_save_path: Bool. Where to save weights if on cluster
        :param amp_level: str. Check nvidia docs for level
        :param nb_sanity_val_steps: int. How many val steps before a full train loop.
        """
        # Transfer params
        self.nb_gpu_nodes = nb_gpu_nodes
        self.log_gpu_memory = log_gpu_memory
        self.gradient_clip_val = gradient_clip_val
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.enable_early_stop = early_stop_callback is not None
        self.track_grad_norm = track_grad_norm
        self.fast_dev_run = fast_dev_run
        self.on_gpu = gpus is not None and torch.cuda.is_available()
        self.process_position = process_position
        self.print_weights_summary = print_weights_summary
        self.max_nb_epochs = max_nb_epochs
        self.min_nb_epochs = min_nb_epochs
        self.nb_sanity_val_steps = nb_sanity_val_steps
        self.print_nan_grads = print_nan_grads

        # set default save path if user didn't provide one
        self.default_save_path = default_save_path
        if self.default_save_path is None:
            self.default_save_path = os.getcwd()

        # training bookeeping
        self.total_batch_nb = 0
        self.running_loss = []
        self.avg_loss = 0
        self.batch_nb = 0
        self.tqdm_metrics = {}
        self.nb_val_batches = 0
        self.nb_training_batches = 0
        self.nb_test_batches = 0
        self.get_train_dataloader = None
        self.get_test_dataloaders = None
        self.get_val_dataloaders = None

        # training state
        self.model = None
        self.testing = False
        self.lr_schedulers = []
        self.optimizers = None
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0

        # configure early stop callback
        # creates a default one if none passed in
        self.early_stop_callback = early_stop_callback
        if self.early_stop_callback is None:
            self.early_stop = EarlyStopping(
                monitor='val_loss',
                patience=3,
                verbose=True,
                mode='min'
            )

        # configure logger
        self.logger = logger
        if self.logger is None:
            self.logger = TestTubeLogger(
                save_dir=self.default_save_path,
                version=self.slurm_job_id,
                name='lightning_logs'
            )
        self.logger.rank = 0

        # configure checkpoint callback
        self.checkpoint_callback = checkpoint_callback
        if self.checkpoint_callback is None:
            if isinstance(logger, TestTubeLogger):
                ckpt_path = '{}/{}/{}'.format(self.default_save_path, self.logger.name,
                                              self.logger.version)
            else:
                ckpt_path = self.default_save_path

            self.checkpoint_callback = ModelCheckpoint(
                filepath=ckpt_path
            )

        # configure weights save path
        self.__configure_weights_path(checkpoint_callback, weights_save_path)

        # accumulated grads
        self.__configure_accumulated_gradients(accumulate_grad_batches)

        # allow int, string and gpu list
        self.data_parallel_device_ids = self.__parse_gpu_ids(gpus)
        self.root_gpu = self.__set_root_gpu(self.data_parallel_device_ids)

        # distributed backend choice
        self.use_ddp = False
        self.use_ddp2 = False
        self.use_dp = False
        self.single_gpu = False
        self.distributed_backend = distributed_backend
        self.__set_distributed_mode(distributed_backend, nb_gpu_nodes)

        # init flags for SLURM+ddp to work
        self.proc_rank = 0
        self.world_size = 1
        self.node_rank = 0
        self.__configure_slurm_ddp(nb_gpu_nodes)

        # nvidia setup
        self.__set_nvidia_flags(self.is_slurm_managing_tasks, self.data_parallel_device_ids)

        # can't init progress bar here because starting a new process
        # means the progress_bar won't survive pickling
        self.show_progress_bar = show_progress_bar

        # logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval
        self.row_log_interval = row_log_interval

        # how much of the data to use
        self.__determine_data_use_amount(train_percent_check, val_percent_check,
                                         test_percent_check, overfit_pct)

        # 16 bit mixed precision training using apex
        self.amp_level = amp_level
        self.__init_amp(use_amp)

    @property
    def slurm_job_id(self):
        try:
            job_id = os.environ['SLURM_JOB_ID']
            job_id = int(job_id)
        except Exception as e:
            job_id = None
        return job_id

    def __configure_weights_path(self, checkpoint_callback, weights_save_path):
        """
        Weight path set in this priority:
        Checkpoint_callback's path (if passed in).
        User provided weights_saved_path
        Otherwise use os.getcwd()
        """
        self.weights_save_path = weights_save_path

        if self.checkpoint_callback is not None:
            self.checkpoint_callback.save_function = self.save_checkpoint

            # if checkpoint callback used, then override the weights path
            self.weights_save_path = self.checkpoint_callback.filepath

        # if weights_save_path is still none here, set to current workingdir
        if self.weights_save_path is None:
            self.weights_save_path = self.default_save_path

    def __init_amp(self, use_amp):
        self.use_amp = use_amp and APEX_AVAILABLE
        if self.use_amp:
            print('using 16bit precision')

        if use_amp and not APEX_AVAILABLE:  # pragma: no cover
            msg = """
            You set use_amp=True but do not have apex installed.
            Install apex first using this guide and rerun with use_amp=True:
            https://github.com/NVIDIA/apex#linux

            this run will NOT use 16 bit precision
            """
            raise ModuleNotFoundError(msg)

    def __configure_accumulated_gradients(self, accumulate_grad_batches):
        self.accumulate_grad_batches = None

        if isinstance(accumulate_grad_batches, dict):
            self.accumulation_scheduler = GradientAccumulationScheduler(accumulate_grad_batches)
        elif isinstance(accumulate_grad_batches, int):
            schedule = {1: accumulate_grad_batches}
            self.accumulation_scheduler = GradientAccumulationScheduler(schedule)
        else:
            raise TypeError("Gradient accumulation supports only int and dict types")

    def __parse_gpu_ids(self, gpus):
        """
        :param gpus: Int, string or list of ids
        :return:
        """
        # if gpus = -1 then use all available devices
        # otherwise, split the string using commas
        if gpus is not None:
            if type(gpus) is list:
                gpus = gpus
            elif type(gpus) is str:
                if gpus == '-1':
                    gpus = list(range(0, torch.cuda.device_count()))
                else:
                    gpus = [int(x.strip()) for x in gpus.split(',')]
            elif type(gpus) is int:
                gpus = gpus
            else:
                raise Exception('gpus has to be a string, int or list of ints')

        return gpus

    def __set_root_gpu(self, gpus):
        if gpus is None:
            return None

        # set root gpu
        root_gpu = 0
        if type(gpus) is list:
            root_gpu = gpus[0]

        return root_gpu

    @property
    def num_gpus(self):
        gpus = self.data_parallel_device_ids
        if gpus is None:
            return 0

        if type(gpus) is list:
            return len(gpus)
        if type(gpus) is int:
            return gpus

        m = 'gpus must be int, none or list of ints'
        raise MisconfigurationException(m)

    def __set_distributed_mode(self, distributed_backend, nb_gpu_nodes):
        # skip for CPU
        if self.num_gpus == 0:
            return

        # single GPU case
        # in single gpu case we allow ddp so we can train on multiple
        # nodes, 1 gpu per node
        if self.num_gpus == 1:
            self.single_gpu = True

            if distributed_backend is not None:
                self.use_dp = distributed_backend == 'dp'
                self.use_ddp = distributed_backend == 'ddp'
                self.use_ddp2 = distributed_backend == 'ddp2'

                # disable single gpu when using ddp2
                if self.use_ddp2:
                    self.single_gpu = False

        # multiple GPU case
        elif self.num_gpus > 1:
            if distributed_backend is not None:
                # DP, DDP case
                self.use_dp = distributed_backend == 'dp'
                self.use_ddp = distributed_backend == 'ddp'
                self.use_ddp2 = distributed_backend == 'ddp2'

            elif distributed_backend is None:
                m = 'When using multiple GPUs set ' \
                    'Trainer(distributed_backend=dp) (or ddp)'
                raise MisconfigurationException(m)

        # use ddp automatically if nb_gpu_nodes > 1
        if nb_gpu_nodes > 1 and self.use_dp:  # pragma: no cover
            self.use_ddp = True
            self.use_dp = False
            w = 'DataParallel does not support nb_gpu_nodes > 1. ' \
                'Switching to DistributedDataParallel for you. ' \
                'To silence this warning set distributed_backend=ddp'
            warnings.warn(w)

        print('gpu available: {}, used: {}'.format(torch.cuda.is_available(), self.on_gpu))

    def __configure_slurm_ddp(self, nb_gpu_nodes):
        self.is_slurm_managing_tasks = False

        # extract SLURM flag vars
        # whenever we have the correct number of tasks, we let slurm manage processes
        # otherwise we launch the required number of processes
        if self.use_ddp:
            self.nb_requested_gpus = self.num_gpus * nb_gpu_nodes
            self.nb_slurm_tasks = 0
            try:
                self.nb_slurm_tasks = int(os.environ['SLURM_NTASKS'])
                self.is_slurm_managing_tasks = self.nb_slurm_tasks == self.nb_requested_gpus

                # in interactive mode we don't manage tasks
                job_name = os.environ['SLURM_JOB_NAME']
                if job_name == 'bash':
                    self.is_slurm_managing_tasks = False

            except Exception:
                # likely not on slurm, so set the slurm managed flag to false
                self.is_slurm_managing_tasks = False

        # used for tests only, set this flag to simulate slurm managing a task
        try:
            should_fake = int(os.environ['FAKE_SLURM_MANAGING_TASKS'])
            if should_fake:
                self.is_slurm_managing_tasks = True
        except Exception as e:
            pass

    def __set_nvidia_flags(self, is_slurm_managing_tasks, data_parallel_device_ids):
        if data_parallel_device_ids is None:
            return

        # set the correct cuda visible devices (using pci order)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        # when slurm is managing the task it sets the visible devices
        if not is_slurm_managing_tasks:
            if type(data_parallel_device_ids) is int:
                id_str = ','.join(str(x) for x in list(range(data_parallel_device_ids)))
                os.environ["CUDA_VISIBLE_DEVICES"] = id_str
            else:
                gpu_str = ','.join([str(x) for x in data_parallel_device_ids])
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

        print(f'VISIBLE GPUS: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    @property
    def data_parallel(self):
        return self.use_dp or self.use_ddp or self.use_ddp2

    def __determine_data_use_amount(self, train_percent_check, val_percent_check,
                                    test_percent_check, overfit_pct):
        """
        Use less data for debugging purposes
        """
        self.train_percent_check = train_percent_check
        self.val_percent_check = val_percent_check
        self.test_percent_check = test_percent_check
        if overfit_pct > 0:
            self.train_percent_check = overfit_pct
            self.val_percent_check = overfit_pct
            self.test_percent_check = overfit_pct

    def __get_model(self):
        return self.model.module if self.data_parallel else self.model

    def __is_function_implemented(self, f_name):
        model = self.__get_model()
        f_op = getattr(model, f_name, None)
        return callable(f_op)

    def __is_overriden(self, f_name):
        model = self.__get_model()
        super_object = LightningModule

        # when code pointers are different, it was overriden
        is_overriden = getattr(model, f_name).__code__ is not getattr(super_object, f_name).__code__
        return is_overriden

    @property
    def __training_tqdm_dict(self):
        tqdm_dict = {
            'loss': '{0:.3f}'.format(self.avg_loss),
            'epoch': '{}'.format(self.current_epoch),
            'batch_nb': '{}'.format(self.batch_nb),
        }

        if self.logger is not None and self.logger.version is not None:
            tqdm_dict['v_nb'] = self.logger.version

        tqdm_dict.update(self.tqdm_metrics)

        if self.on_gpu:
            tqdm_dict['gpu'] = '{}'.format(torch.cuda.current_device())

        return tqdm_dict

    @property
    def training_tqdm_dict(self):
        """
        Read-only for tqdm metrics
        :return:
        """
        return self.__training_tqdm_dict

    def __layout_bookeeping(self):

        # determine number of training batches
        self.nb_training_batches = len(self.get_train_dataloader())
        self.nb_training_batches = int(self.nb_training_batches * self.train_percent_check)

        # determine number of validation batches
        # val datasets could be none, 1 or 2+
        if self.get_val_dataloaders() is not None:
            self.nb_val_batches = sum(len(dataloader) for dataloader in self.get_val_dataloaders())
            self.nb_val_batches = int(self.nb_val_batches * self.val_percent_check)
            self.nb_val_batches = max(1, self.nb_val_batches)

        # determine number of test batches
        if self.get_test_dataloaders() is not None:
            self.nb_test_batches = sum(
                len(dataloader) for dataloader in self.get_test_dataloaders()
            )
            self.nb_test_batches = int(self.nb_test_batches * self.test_percent_check)
            self.nb_test_batches = max(1, self.nb_test_batches)

        # determine when to check validation
        self.val_check_batch = int(self.nb_training_batches * self.val_check_interval)
        self.val_check_batch = max(1, self.val_check_batch)

    def __add_tqdm_metrics(self, metrics):
        for k, v in metrics.items():
            if type(v) is torch.Tensor:
                v = v.item()

            self.tqdm_metrics[k] = v

    def __evaluation_forward(self, model, batch, batch_idx, dataloader_idx, test=False):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        if test and len(self.get_test_dataloaders()) > 1:
            args.append(dataloader_idx)

        elif not test and len(self.get_val_dataloaders()) > 1:
            args.append(dataloader_idx)

        # handle DP, DDP forward
        if self.use_ddp or self.use_dp or self.use_ddp2:
            output = model(*args)
            return output

        # single GPU
        if self.single_gpu:
            # for single GPU put inputs on gpu manually
            root_gpu = 0
            if type(self.data_parallel_device_ids) is list:
                root_gpu = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, root_gpu)
            args[0] = batch

        # CPU
        if test:
            output = model.test_step(*args)
        else:
            output = model.validation_step(*args)

        return output

    def evaluate(self, model, dataloaders, max_batches, test=False):
        """
        Run evaluation code
        :param model: PT model
        :param dataloaders: list of PT dataloaders
        :param max_batches: Scalar
        :param test: boolean
        :return:
        """
        # enable eval mode
        model.zero_grad()
        model.eval()

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        # bookkeeping
        outputs = []

        # run training
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dl_outputs = []
            for batch_idx, batch in enumerate(dataloader):

                if batch is None:  # pragma: no cover
                    continue

                # stop short when on fast_dev_run (sets max_batch=1)
                if batch_idx >= max_batches:
                    break

                # -----------------
                # RUN EVALUATION STEP
                # -----------------
                output = self.__evaluation_forward(model,
                                                   batch,
                                                   batch_idx,
                                                   dataloader_idx,
                                                   test)

                # track outputs for collation
                dl_outputs.append(output)

                # batch done
                if self.show_progress_bar:
                    self.progress_bar.update(1)
            outputs.append(dl_outputs)

        eval_results = {}

        # with a single dataloader don't pass an array
        if len(dataloaders) == 1:
            outputs = outputs[0]

        # give model a chance to do something with the outputs (and method defined)
        model = self.__get_model()
        if test and self.__is_overriden('test_end'):
            eval_results = model.test_end(outputs)
        elif self.__is_overriden('validation_end'):
            eval_results = model.validation_end(outputs)

        # enable train mode again
        model.train()

        # enable gradients to save memory
        torch.set_grad_enabled(True)

        return eval_results

    def get_dataloaders(self, model):
        """
        Dataloaders are provided by the model
        :param model:
        :return:
        """
        self.get_train_dataloader = model.train_dataloader
        self.get_test_dataloaders = model.test_dataloader
        self.get_val_dataloaders = model.val_dataloader

        # call warnings from proc zero only which triggers dataloaders
        # if those have to download data it will only happen on proc 0
        if self.proc_rank == 0:
            on_ddp = self.use_ddp or self.use_ddp2
            if on_ddp and not isinstance(self.get_train_dataloader().sampler, DistributedSampler):
                msg = """
                You're using multiple gpus and multiple nodes without using a DistributedSampler
                to assign a subset of your data to each process. To silence this warning, pass a
                DistributedSampler to your DataLoader.

                ie: this:
                dataset = myDataset()
                dataloader = Dataloader(dataset)

                becomes:
                dataset = myDataset()
                dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                dataloader = Dataloader(dataset, sampler=dist_sampler)

                If you want each process to load the full dataset, ignore this warning.
                """
                warnings.warn(msg)

            if on_ddp and self.get_val_dataloaders() is not None:
                for dataloader in self.get_val_dataloaders():
                    if not isinstance(dataloader.sampler, DistributedSampler):
                        msg = """
                        Your val_dataloader(s) don't use DistributedSampler.

                        You're using multiple gpus and multiple nodes without using a
                        DistributedSampler to assign a subset of your data to each process.
                        To silence this warning, pass a DistributedSampler to your DataLoader.

                        ie: this:
                        dataset = myDataset()
                        dataloader = Dataloader(dataset)

                        becomes:
                        dataset = myDataset()
                        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                        dataloader = Dataloader(dataset, sampler=dist_sampler)

                        If you want each process to load the full dataset, ignore this warning.
                        """
                        warnings.warn(msg)
                        break

            if on_ddp and self.get_test_dataloaders() is not None:
                for dataloader in self.get_test_dataloaders():
                    if not isinstance(dataloader.sampler, DistributedSampler):
                        msg = """
                        Your test_dataloader(s) don't use DistributedSampler.

                        You're using multiple gpus and multiple nodes without using a
                        DistributedSampler to assign a subset of your data to each process.
                        To silence this warning, pass a DistributedSampler to your DataLoader.

                        ie: this:
                        dataset = myDataset()
                        dataloader = Dataloader(dataset)

                        becomes:
                        dataset = myDataset()
                        dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                        dataloader = Dataloader(dataset, sampler=dist_sampler)

                        If you want each process to load the full dataset, ignore this warning.
                        """
                        warnings.warn(msg)
                        break

        if self.use_ddp or self.use_ddp2:
            # wait for all processes to catch up
            dist.barrier()

            # load each dataloader
            self.get_train_dataloader()
            self.get_test_dataloaders()
            self.get_val_dataloaders()

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    def fit(self, model):
        # when using multi-node or DDP within a node start each module in a separate process
        if self.use_ddp2:
            task = int(os.environ['SLURM_LOCALID'])
            self.ddp_train(task, model)

        elif self.use_ddp:
            if self.is_slurm_managing_tasks:
                task = int(os.environ['SLURM_LOCALID'])
                self.ddp_train(task, model)
            else:
                mp.spawn(self.ddp_train, nprocs=self.num_gpus, args=(model, ))

        # 1 gpu or dp option triggers training using DP module
        # easier to avoid NCCL issues
        elif self.use_dp:
            self.__dp_train(model)

        elif self.single_gpu:
            self.__single_gpu_train(model)

        # ON CPU
        else:
            # run through amp wrapper
            if self.use_amp:
                raise MisconfigurationException('amp + cpu is not supported.'
                                                ' Please use a GPU option')

            # CHOOSE OPTIMIZER
            # allow for lr schedulers as well
            self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

            self.__run_pretrain_routine(model)

        # return 1 when finished
        # used for testing or when we need to know that training succeeded
        return 1

    def init_optimizers(self, optimizers):

        # single optimizer
        if isinstance(optimizers, Optimizer):
            return [optimizers], []

        # two lists
        elif len(optimizers) == 2 and isinstance(optimizers[0], list):
            optimizers, lr_schedulers = optimizers
            return optimizers, lr_schedulers

        # single list or tuple
        elif isinstance(optimizers, list) or isinstance(optimizers, tuple):
            return optimizers, []

    def __single_gpu_train(self, model):
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

        self.__run_pretrain_routine(model)

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

    def ddp_train(self, gpu_nb, model):
        """
        Entry point into a DP thread
        :param gpu_nb:
        :param model:
        :param cluster_obj:
        :return:
        """
        # node rank using relative slurm id
        # otherwise default to node rank 0
        try:
            node_id = os.environ['SLURM_NODEID']
            self.node_rank = int(node_id)
        except Exception:
            self.node_rank = 0

        # show progressbar only on progress_rank 0
        self.show_progress_bar = self.show_progress_bar and self.node_rank == 0 and gpu_nb == 0

        # determine which process we are and world size
        if self.use_ddp:
            self.proc_rank = self.node_rank * self.num_gpus + gpu_nb
            self.world_size = self.nb_gpu_nodes * self.num_gpus

        elif self.use_ddp2:
            self.proc_rank = self.node_rank
            self.world_size = self.nb_gpu_nodes

        # let the exp know the rank to avoid overwriting logs
        if self.logger is not None:
            self.logger.rank = self.proc_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.__init_tcp_connection()

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        # MODEL
        # copy model to each gpu
        if self.distributed_backend == 'ddp':
            torch.cuda.set_device(gpu_nb)
        model.cuda(gpu_nb)

        # set model properties before going into wrapper
        model.trainer = self
        model.on_gpu = self.on_gpu
        model.use_dp = self.use_dp
        model.use_ddp2 = self.use_ddp2
        model.use_ddp = self.use_ddp
        model.use_amp = self.use_amp
        model.testing = self.testing

        # override root GPU
        self.root_gpu = gpu_nb

        # AMP
        # run through amp wrapper before going to distributed DP
        if self.use_amp:
            # An example
            model, optimizers = amp.initialize(
                model, self.optimizers, opt_level=self.amp_level,
            )
            self.optimizers = optimizers

        # DDP2 uses all GPUs on the machine
        if self.distributed_backend == 'ddp':
            device_ids = [gpu_nb]
        elif self.use_ddp2:
            device_ids = None

        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )

        # continue training routine
        self.__run_pretrain_routine(model)

    def __init_tcp_connection(self):
        """
        Connect all procs in the world using the env:// init
        Use the first node as the root address
        :param port:
        :param tries:
        :return:
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

        root_node = self.resolve_root_node_address(root_node)
        os.environ['MASTER_ADDR'] = root_node
        dist.init_process_group("nccl", rank=self.proc_rank, world_size=self.world_size)

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name = root_node.split('[')[0]
            number = root_node.split(',')[0]
            if '-' in number:
                number = number.split('-')[0]

            number = re.sub('[^0-9]', '', number)
            root_node = name + number

        return root_node

    def __run_pretrain_routine(self, model):
        """
        Sanity check a few things before starting actual training
        :param model:
        :return:
        """
        ref_model = model
        if self.data_parallel:
            ref_model = model.module

        # give model convenience properties
        ref_model.trainer = self

        # set local properties on the model
        ref_model.on_gpu = self.on_gpu
        ref_model.single_gpu = self.single_gpu
        ref_model.use_dp = self.use_dp
        ref_model.use_ddp = self.use_ddp
        ref_model.use_ddp2 = self.use_ddp2
        ref_model.use_amp = self.use_amp
        ref_model.testing = self.testing

        # register auto-resubmit when on SLURM
        self.register_slurm_signal_handlers()

        # transfer data loaders from model
        self.get_dataloaders(ref_model)

        # init training constants
        self.__layout_bookeeping()

        # print model summary
        if self.proc_rank == 0 and self.print_weights_summary:
            ref_model.summarize()

        # link up experiment object
        if self.logger is not None:
            ref_model.logger = self.logger

            # save exp to get started
            if hasattr(ref_model, "hparams"):
                self.logger.log_hyperparams(ref_model.hparams)
            self.logger.save()

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # restore training and model before hpc call
        self.restore_weights(model)

        # progress bar init
        if self.show_progress_bar:
            self.progress_bar = tqdm.tqdm(0, position=self.process_position)

        # when testing requested only run test and return
        if self.testing:
            self.__run_evaluation(test=True)
            return

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        ref_model.on_sanity_check_start()
        if self.get_val_dataloaders() is not None and self.nb_sanity_val_steps > 0:
            # reset progress_bar limit for sanity check
            if self.show_progress_bar:
                self.progress_bar.reset(self.nb_sanity_val_steps)

            self.evaluate(model, self.get_val_dataloaders(), self.nb_sanity_val_steps, self.testing)

        # ---------------------------
        # CORE TRAINING LOOP
        # ---------------------------
        self.__train()

    def __train(self):
        # run all epochs
        for epoch_nb in range(self.current_epoch, self.max_nb_epochs):
            # set seed for distributed sampler (enables shuffling for each epoch)
            if self.use_ddp and hasattr(self.get_train_dataloader().sampler, 'set_epoch'):
                self.get_train_dataloader().sampler.set_epoch(epoch_nb)

            # get model
            model = self.__get_model()

            # update training progress in trainer and model
            model.current_epoch = epoch_nb
            self.current_epoch = epoch_nb
            self.total_batches = self.nb_training_batches + self.nb_val_batches
            self.batch_loss_value = 0  # accumulated grads

            # init progress_bar when requested
            if self.show_progress_bar:
                self.progress_bar.reset(self.total_batches)

            # changing gradient according accumulation_scheduler
            self.accumulation_scheduler.on_epoch_begin(epoch_nb, self)

            # -----------------
            # RUN TNG EPOCH
            # -----------------
            self.run_training_epoch()

            # update LR schedulers
            if self.lr_schedulers is not None:
                for lr_scheduler in self.lr_schedulers:
                    lr_scheduler.step(self.current_epoch)

            # early stopping
            met_min_epochs = epoch_nb > self.min_nb_epochs
            if self.enable_early_stop and met_min_epochs:
                should_stop = self.early_stop_callback.on_epoch_end(epoch=epoch_nb,
                                                                    logs=self.__training_tqdm_dict)
                # stop training
                stop = should_stop and met_min_epochs
                if stop:
                    return

    def run_training_epoch(self):
        # before epoch hook
        if self.__is_function_implemented('on_epoch_start'):
            model = self.__get_model()
            model.on_epoch_start()

        # run epoch
        for batch_nb, batch in enumerate(self.get_train_dataloader()):
            self.batch_nb = batch_nb
            self.global_step += 1

            model = self.__get_model()
            model.global_step = self.global_step

            # stop when the flag is changed or we've gone past the amount
            #  requested in the batches
            self.total_batch_nb += 1
            met_batch_limit = batch_nb > self.nb_training_batches
            if met_batch_limit:
                break

            # ---------------
            # RUN TRAIN STEP
            # ---------------
            output = self.__run_training_batch(batch, batch_nb)
            batch_result, grad_norm_dic, batch_step_metrics = output
            early_stop_epoch = batch_result == -1

            # ---------------
            # RUN VAL STEP
            # ---------------
            is_val_check_batch = (batch_nb + 1) % self.val_check_batch == 0
            can_check_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
            if self.fast_dev_run or is_val_check_batch or early_stop_epoch:
                if can_check_epoch:
                    self.__run_evaluation(test=self.testing)

            # when batch should be saved
            if (batch_nb + 1) % self.log_save_interval == 0 or early_stop_epoch:
                if self.proc_rank == 0 and self.logger is not None:
                    self.logger.save()

            # when metrics should be logged
            if batch_nb % self.row_log_interval == 0 or early_stop_epoch:

                # logs user requested information to logger
                self.__log_metrics(batch_step_metrics, grad_norm_dic)

            # end epoch early
            if early_stop_epoch:
                break

        # epoch end hook
        if self.__is_function_implemented('on_epoch_end'):
            model = self.__get_model()
            model.on_epoch_end()

    def __log_metrics(self, metrics, grad_norm_dic):
        """
        Logs the metric dict passed in
        :param metrics:
        :param grad_norm_dic:
        :return:
        """
        # added metrics by Lightning for convenience
        metrics['epoch'] = self.current_epoch

        # add gpu memory
        if self.on_gpu and self.log_gpu_memory:
            mem_map = memory.get_memory_profile(self.log_gpu_memory)
            metrics.update(mem_map)

        # add norms
        metrics.update(grad_norm_dic)

        # turn all tensors to scalars
        scalar_metrics = self.__metrics_to_scalars(metrics)

        # log actual metrics
        if self.proc_rank == 0 and self.logger is not None:
            self.logger.log_metrics(scalar_metrics, step_num=self.global_step)
            self.logger.save()

    def test(self, model=None):
        if model is not None:
            self.testing = True
            self.fit(model)
        else:
            self.__run_evaluation(test=True)

    def __metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.__metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def __log_vals_blacklist(self):
        """avoid logging some vals lightning uses to maintain state"""
        blacklist = {'batch_nb', 'v_nb', 'gpu'}
        return blacklist

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

    def __training_forward(self, batch, batch_nb, opt_idx):
        """
        Handle forward for each training case (distributed, single gpu, etc...)
        :param batch:
        :param batch_nb:
        :return:
        """
        # ---------------
        # FORWARD
        # ---------------
        # enable not needing to add opt_idx to training_step
        args = [batch, batch_nb]
        if len(self.optimizers) > 1:
            args.append(opt_idx)

        if self.use_ddp or self.use_ddp2:
            output = self.model(*args)
        elif self.use_dp:
            output = self.model(*args)
        elif self.single_gpu:
            gpu_id = 0
            if type(self.data_parallel_device_ids) is list:
                gpu_id = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, gpu_id)
            args[0] = batch
            output = self.model.training_step(*args)

        else:
            output = self.model.training_step(*args)

        # format and reduce outputs accordingly
        loss, progress_bar_metrics, log_metrics = self.__process_output(output, train=True)
        return loss, progress_bar_metrics, log_metrics

    def __process_output(self, output, train=False):
        """
        Reduces output according to the training mode.
        Separates loss from logging and tqdm metrics
        :param output:
        :return:
        """
        try:
            progress_output = output['progress_bar']

            # reduce progress metrics for tqdm when using dp
            if train and self.use_dp or self.use_ddp2:
                nb_gpus = self.num_gpus
                progress_output = reduce_distributed_output(progress_output, nb_gpus)

            progress_bar_metrics = progress_output
        except Exception:
            progress_bar_metrics = {}

        # extract metrics to log to experiment
        try:
            log_output = output['log']

            # reduce progress metrics for tqdm when using dp
            if train and self.use_dp or self.use_ddp2:
                nb_gpus = self.num_gpus
                log_output = reduce_distributed_output(log_output, nb_gpus)

            log_metrics = log_output
        except Exception:
            log_metrics = {}

        # ---------------
        # EXTRACT LOSS
        # ---------------
        # if output dict doesn't have the keyword loss
        # then assume the output=loss if scalar
        loss = None
        if train:
            try:
                loss = output['loss']
            except Exception:
                if type(output) is torch.Tensor:
                    loss = output
                else:
                    raise RuntimeError(
                        'No `loss` value in the dictionary returned from `model.training_step()`.'
                    )

            # when using dp need to reduce the loss
            if self.use_dp or self.use_ddp2:
                loss = reduce_distributed_output(loss, self.num_gpus)

        return loss, progress_bar_metrics, log_metrics

    def __clip_gradients(self):
        if self.gradient_clip_val > 0:
            model = self.__get_model()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)

    def __print_nan_grads(self):
        model = self.__get_model()
        for param in model.parameters():
            if torch.isnan(param.grad.float()).any():
                print(param, param.grad)

    def __run_training_batch(self, batch, batch_nb):
        # track grad norms
        grad_norm_dic = {}

        # track metrics to log
        all_log_metrics = []

        if batch is None:
            return 0, grad_norm_dic

        # hook
        if self.__is_function_implemented('on_batch_start'):
            model_ref = self.__get_model()
            response = model_ref.on_batch_start(batch)

            if response == -1:
                return -1, grad_norm_dic

        if self.show_progress_bar:
            self.progress_bar.update(1)

        # call training_step once per optimizer
        for opt_idx, optimizer in enumerate(self.optimizers):

            # wrap the forward step in a closure so second order methods work
            def optimizer_closure():
                # forward pass
                output = self.__training_forward(batch, batch_nb, opt_idx)
                closure_loss, progress_bar_metrics, log_metrics = output

                # track progress bar metrics
                self.__add_tqdm_metrics(progress_bar_metrics)
                all_log_metrics.append(log_metrics)

                # accumulate loss
                # (if accumulate_grad_batches = 1 no effect)
                closure_loss = closure_loss / self.accumulate_grad_batches

                # backward pass
                if self.use_amp:
                    with amp.scale_loss(closure_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    closure_loss.backward()

                # insert after step hook
                if self.__is_function_implemented('on_after_backward'):
                    model_ref = self.__get_model()
                    model_ref.on_after_backward()

                return closure_loss

            # calculate loss
            loss = optimizer_closure()

            # nan grads
            if self.print_nan_grads:
                self.__print_nan_grads()

            # track total loss for logging (avoid mem leaks)
            self.batch_loss_value += loss.item()

            # gradient update with accumulated gradients
            if (self.batch_nb + 1) % self.accumulate_grad_batches == 0:

                # track gradient norms when requested
                if batch_nb % self.row_log_interval == 0:
                    if self.track_grad_norm > 0:
                        model = self.__get_model()
                        grad_norm_dic = model.grad_norm(self.track_grad_norm)

                # clip gradients
                self.__clip_gradients()

                # calls .step(), .zero_grad()
                # override function to modify this behavior
                model = self.__get_model()
                model.optimizer_step(self.current_epoch, batch_nb,
                                     optimizer, opt_idx, optimizer_closure)

                # calculate running loss for display
                self.running_loss.append(self.batch_loss_value)
                self.batch_loss_value = 0
                self.avg_loss = np.mean(self.running_loss[-100:])

                # update progress bar
                if self.show_progress_bar:
                    # add model specific metrics
                    tqdm_metrics = self.__training_tqdm_dict
                    self.progress_bar.set_postfix(**tqdm_metrics)

        # activate batch end hook
        if self.__is_function_implemented('on_batch_end'):
            model = self.__get_model()
            model.on_batch_end()

        # collapse all metrics into one dict
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}
        return 0, grad_norm_dic, all_log_metrics

    def __run_evaluation(self, test=False):
        # when testing make sure user defined a test step
        can_run_test_step = False
        if test:
            can_run_test_step = self.__is_overriden('test_step') and self.__is_overriden('test_end')
            if not can_run_test_step:
                m = '''You called .test() without defining a test step or test_end.
                Please define and try again'''
                raise MisconfigurationException(m)

        # validate only if model has validation_step defined
        # test only if test_step or validation_step are defined
        run_val_step = self.__is_overriden('validation_step')

        if run_val_step or can_run_test_step:

            # hook
            model = self.__get_model()
            model.on_pre_performance_check()

            # select dataloaders
            dataloaders = self.get_val_dataloaders()
            max_batches = self.nb_val_batches

            # calculate max batches to use
            if test:
                dataloaders = self.get_test_dataloaders()
                max_batches = self.nb_test_batches

            # cap max batches to 1 when using fast_dev_run
            if self.fast_dev_run:
                max_batches = 1

            # run evaluation
            eval_results = self.evaluate(self.model,
                                         dataloaders,
                                         max_batches,
                                         test)
            _, progress_bar_metrics, log_metrics = self.__process_output(eval_results)

            # add metrics to prog bar
            self.__add_tqdm_metrics(progress_bar_metrics)

            # log metrics
            self.__log_metrics(log_metrics, {})

            # hook
            model.on_post_performance_check()

            if self.show_progress_bar:
                # add model specific metrics
                tqdm_metrics = self.__training_tqdm_dict
                self.progress_bar.set_postfix(**tqdm_metrics)

        # model checkpointing
        if self.proc_rank == 0 and self.checkpoint_callback is not None and not test:
            print('save callback...')
            self.checkpoint_callback.on_epoch_end(epoch=self.current_epoch,
                                                  logs=self.__training_tqdm_dict)
