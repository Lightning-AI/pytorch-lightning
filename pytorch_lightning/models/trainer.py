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

from pytorch_lightning.root_module.root_module import LightningModule
from pytorch_lightning.root_module.memory import get_gpu_memory_map
from pytorch_lightning.root_module.model_saving import TrainerIO
from pytorch_lightning.pt_overrides.override_data_parallel import (
    LightningDistributedDataParallel, LightningDataParallel)
from pytorch_lightning.utilities.debugging import MisconfigurationException

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
                 experiment=None,
                 early_stop_callback=None,
                 checkpoint_callback=None,
                 gradient_clip=0,
                 cluster=None,
                 process_position=0,
                 current_gpu_name=0,
                 nb_gpu_nodes=1,
                 gpus=None,
                 progress_bar=True,
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
                 val_check_interval=0.95,
                 log_save_interval=100,
                 add_log_row_interval=10,
                 distributed_backend='dp',
                 use_amp=False,
                 print_nan_grads=False,
                 print_weights_summary=True,
                 amp_level='O2',
                 nb_sanity_val_steps=5):
        """

        :param experiment: Test-tube experiment
        :param early_stop_callback: from pytorch_lightning import EarlyStopping
        :param checkpoint_callback: from pytorch_lightning import Checkpoint
        :param gradient_clip:
        :param cluster:
        :param process_position:
        :param current_gpu_name:
        :param nb_gpu_nodes:
        :param gpus:
        :param progress_bar:
        :param overfit_pct:
        :param track_grad_norm:
        :param check_val_every_n_epoch:
        :param fast_dev_run:
        :param accumulate_grad_batches:
        :param max_nb_epochs:
        :param min_nb_epochs:
        :param train_percent_check:
        :param val_percent_check:
        :param test_percent_check:
        :param val_check_interval:
        :param log_save_interval:
        :param add_log_row_interval:
        :param distributed_backend:
            'do' to use DistributedParallel, 'dp' to use DistributedDataParallel, 'n' to use none
        :param use_amp:
        :param print_nan_grads:
        :param print_weights_summary:
        :param amp_level:
        :param nb_sanity_val_steps:
        """
        # Transfer params
        self.nb_gpu_nodes = nb_gpu_nodes
        self.gradient_clip = gradient_clip
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.enable_early_stop = early_stop_callback is not None
        self.track_grad_norm = track_grad_norm
        self.fast_dev_run = fast_dev_run
        self.on_gpu = gpus is not None and torch.cuda.is_available()
        self.progress_bar = progress_bar
        self.experiment = experiment
        self.exp_save_path = None
        if self.experiment is not None:
            self.exp_save_path = experiment.get_data_path(experiment.name, experiment.version)
        self.cluster = cluster
        self.process_position = process_position
        self.current_gpu_name = current_gpu_name
        self.print_weights_summary = print_weights_summary
        self.checkpoint_callback = checkpoint_callback

        if self.checkpoint_callback is not None:
            self.checkpoint_callback.save_function = self.save_checkpoint

        self.early_stop = early_stop_callback
        self.model = None
        self.max_nb_epochs = max_nb_epochs
        self.accumulate_grad_batches = accumulate_grad_batches
        self.early_stop_callback = early_stop_callback
        self.min_nb_epochs = min_nb_epochs
        self.nb_sanity_val_steps = nb_sanity_val_steps
        self.lr_schedulers = []
        self.amp_level = amp_level
        self.print_nan_grads = print_nan_grads
        self.data_parallel_device_ids = None
        self.world_size = 1
        self.node_rank = 0
        self.use_ddp = False
        self.use_dp = False
        self.single_gpu = False

        # training bookeeping
        self.total_batch_nb = 0
        self.running_loss = []
        self.avg_loss = 0
        self.batch_nb = 0
        self.tqdm_metrics = {}
        self.nb_val_batches = None
        self.nb_tng_batches = None
        self.nb_test_batches = None

        # gpus come in as a string.
        # if gpus = -1 then use all available devices
        # otherwise, split the string using commas
        if gpus is not None:
            if type(gpus) is list:
                self.data_parallel_device_ids = gpus
            elif type(gpus) is str:
                if gpus == '-1':
                    self.data_parallel_device_ids = list(range(0, torch.cuda.device_count()))
                else:
                    self.data_parallel_device_ids = [int(x.strip()) for x in gpus.split(',')]
            else:
                raise Exception('gpus has to be a string or list of ids')

            # set the correct cuda visible devices (using pci order)
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in
                                                           self.data_parallel_device_ids])
            print('VISIBLE GPUS: %r' % os.environ["CUDA_VISIBLE_DEVICES"])

        # make DP and DDP mutually exclusive
        # single GPU will also use DP with devices=[0]
        requested_gpus = self.data_parallel_device_ids is not None
        if requested_gpus and len(self.data_parallel_device_ids) > 0:
            self.use_dp = distributed_backend == 'dp'
            self.use_ddp = distributed_backend == 'ddp'

            # use ddp automatically if nb_gpu_nodes > 1
            if nb_gpu_nodes > 1 and self.use_dp:  # pragma: no cover
                self.use_ddp = True
                self.use_dp = False
                w = 'DataParallel does not support nb_gpu_nodes > 1. ' \
                    'Switching to DistributedDataParallel for you. ' \
                    'To silence this warning set distributed_backend=ddp'
                warnings.warn(w)

        # remove dp and ddp when requesting single gpu
        if self.data_parallel_device_ids is not None and len(self.data_parallel_device_ids) == 1:
            self.use_ddp = False
            self.use_dp = False
            self.single_gpu = True

        # extract SLURM flag vars
        # whenever we have the correct number of tasks, we let slurm manage processes
        # otherwise we launch the required number of processes
        if self.use_ddp:
            self.nb_requested_gpus = len(self.data_parallel_device_ids) * self.nb_gpu_nodes
            self.nb_slurm_tasks = 0
            try:
                self.nb_slurm_tasks = int(os.environ['SLURM_NTASKS'])
                self.is_slurm_managing_tasks = self.nb_slurm_tasks == self.nb_requested_gpus
            except Exception:
                # likely not on slurm, so set the slurm managed flag to false
                self.is_slurm_managing_tasks = False

        # process info
        self.proc_rank = 0

        # training state
        self.optimizers = None
        self.prog_bar = None
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0

        # logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval
        self.add_log_row_interval = add_log_row_interval

        # dataloaders
        self.tng_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None

        # how much of the data to use
        self.__determine_data_use_amount(train_percent_check, val_percent_check,
                                         test_percent_check, overfit_pct)
        print('gpu available: {}, used: {}'.format(torch.cuda.is_available(), self.on_gpu))

        # 16 bit mixed precision training using apex
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

    def restore_state_if_existing_checkpoint(self):
        # restore trainer state and model if there is a weight for this experiment
        last_epoch = -1
        last_ckpt_name = None

        # do nothing if there's not dir or callback
        no_ckpt_callback = self.checkpoint_callback is None
        if no_ckpt_callback or not os.path.exists(self.checkpoint_callback.filepath):
            return

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
            print(f'model and trainer restored from checkpoint: {last_ckpt_path}')

    @property
    def data_parallel(self):
        return self.use_dp or self.use_ddp

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
        super_object = super(model.__class__, model)

        # when code pointers are different, it was overriden
        is_overriden = getattr(model, f_name).__code__ is not getattr(super_object, f_name).__code__
        return is_overriden

    @property
    def __tng_tqdm_dic(self):
        tqdm_dic = {
            'tng_loss': '{0:.3f}'.format(self.avg_loss),
            'epoch': '{}'.format(self.current_epoch),
            'batch_nb': '{}'.format(self.batch_nb),
        }

        if self.experiment is not None:
            tqdm_dic['v_nb'] = self.experiment.version

        tqdm_dic.update(self.tqdm_metrics)

        if self.on_gpu:
            tqdm_dic['gpu'] = '{}'.format(self.current_gpu_name)

        return tqdm_dic

    @property
    def tng_tqdm_dic(self):
        """
        Read-only for tqdm metrics
        :return:
        """
        return self.__tng_tqdm_dic

    def __layout_bookeeping(self):

        # determine number of training batches
        self.nb_tng_batches = len(self.tng_dataloader)
        self.nb_tng_batches = int(self.nb_tng_batches * self.train_percent_check)

        # determine number of validation batches
        # val datasets could be none, 1 or 2+
        self.nb_val_batches = 0
        if self.val_dataloader is not None:
            self.nb_val_batches = sum(len(dataloader) for dataloader in self.val_dataloader)

        self.nb_val_batches = int(self.nb_val_batches * self.val_percent_check)
        self.nb_val_batches = max(1, self.nb_val_batches)
        self.nb_val_batches = self.nb_val_batches

        # determine number of test batches
        self.nb_test_batches = len(self.test_dataloader) if self.test_dataloader is not None else 0
        self.nb_test_batches = int(self.nb_test_batches * self.test_percent_check)

        # determine when to check validation
        self.val_check_batch = int(self.nb_tng_batches * self.val_check_interval)

    def __add_tqdm_metrics(self, metrics):
        for k, v in metrics.items():
            if type(v) is torch.Tensor:
                v = v.item()

            self.tqdm_metrics[k] = v

    def validate(self, model, dataloader, max_batches, dataloader_i):
        """
        Run validation code
        :param model: PT model
        :param dataloader: PT dataloader
        :param max_batches: Scalar
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
        for batch_i, data_batch in enumerate(dataloader):

            if data_batch is None:  # pragma: no cover
                continue

            # stop short when on fast dev run
            if max_batches is not None and batch_i >= max_batches:
                break

            # -----------------
            # RUN VALIDATION STEP
            # -----------------
            if self.use_ddp:
                output = model(data_batch, batch_i, dataloader_i)
            elif self.use_dp:
                output = model(data_batch, batch_i, dataloader_i)
            elif self.single_gpu:
                # put inputs on gpu manually
                gpu_id = self.data_parallel_device_ids[0]
                for i, x in enumerate(data_batch):
                    if isinstance(x, torch.Tensor):
                        data_batch[i] = x.cuda(gpu_id)

                # do non dp, ddp step
                output = model.validation_step(data_batch, batch_i, dataloader_i)

            else:
                output = model.validation_step(data_batch, batch_i, dataloader_i)

            outputs.append(output)

            # batch done
            if self.progress_bar and self.prog_bar is not None:
                self.prog_bar.update(1)

        # give model a chance to do something with the outputs (and method defined)
        val_results = {}
        if self.__is_overriden('validation_end'):
            if self.data_parallel:
                val_results = model.module.validation_end(outputs)
            else:
                val_results = model.validation_end(outputs)

        # enable train mode again
        model.train()

        # enable gradients to save memory
        torch.set_grad_enabled(True)

        return val_results

    def get_dataloaders(self, model):
        """
        Dataloaders are provided by the model
        :param model:
        :return:
        """
        self.tng_dataloader = model.tng_dataloader

        self.test_dataloader = model.test_dataloader
        self.val_dataloader = model.val_dataloader

        # handle returning an actual dataloader instead of a list of loaders
        have_val_loaders = self.val_dataloader is not None
        if have_val_loaders and not isinstance(self.val_dataloader, list):
            self.val_dataloader = [self.val_dataloader]

        if self.use_ddp and not isinstance(self.tng_dataloader.sampler, DistributedSampler):
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

        if self.use_ddp and\
                not all(isinstance(dataloader, DistributedSampler)
                        for dataloader in self.val_dataloader):
            msg = """
You're val_dataloader(s) are not all DistributedSamplers.
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

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    def fit(self, model):

        # when using multi-node or DDP within a node start each module in a separate process
        if self.use_ddp:
            # must copy only the meta of the exp so it survives pickle/unpickle
            #  when going to new process
            if self.experiment is not None:
                self.experiment = self.experiment.get_meta_copy()

            if self.is_slurm_managing_tasks:
                task = int(os.environ['SLURM_LOCALID'])
                self.ddp_train(task, model)
            else:
                msg = """
You requested %(nb_gpus)s GPUs but launched %(nb_tasks)s slurm tasks.
We will launch %(nb_gpus)s processes for you.
We recommend you let slurm manage the processes by setting: --ntasks-per-node=%(nb_gpus)s
If you're not using SLURM, ignore this message!
""" % {'nb_gpus': self.nb_requested_gpus, 'nb_tasks': self.nb_slurm_tasks}
                warnings.warn(msg)
                mp.spawn(self.ddp_train, nprocs=len(self.data_parallel_device_ids), args=(model, ))

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
            self.optimizers = model.configure_optimizers()
            if len(self.optimizers) == 2:
                self.optimizers, self.lr_schedulers = self.optimizers

            self.__run_pretrain_routine(model)

        # return 1 when finished
        # used for testing or when we need to know that training succeeded
        return 1

    def __single_gpu_train(self, model):
        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers = model.configure_optimizers()
        if len(self.optimizers) == 2:
            self.optimizers, self.lr_schedulers = self.optimizers

        model.cuda(self.data_parallel_device_ids[0])

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
        self.optimizers = model.configure_optimizers()
        if len(self.optimizers) == 2:
            self.optimizers, self.lr_schedulers = self.optimizers

        model.cuda(self.data_parallel_device_ids[0])

        # check for this bug (amp + dp + !01 doesn't work)
        # https://github.com/NVIDIA/apex/issues/227
        if self.use_dp and self.use_amp:
            m = """
Amp level %r with DataParallel is not supported.
See this note from NVIDIA for more info: https://github.com/NVIDIA/apex/issues/227.
We recommend you switch to ddp if you want to use amp
""" % self.amp_level
            raise MisconfigurationException(m)

        model = LightningDataParallel(model, device_ids=self.data_parallel_device_ids)

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

        # recover original exp before went into process
        # init in write mode only on proc 0
        if self.experiment is not None:
            self.experiment.debug = self.proc_rank > 0
            self.experiment = self.experiment.get_non_ddp_exp()

        # show progbar only on prog_rank 0
        self.prog_bar = self.prog_bar and self.node_rank == 0 and gpu_nb == 0

        # determine which process we are and world size
        self.proc_rank = self.node_rank * len(self.data_parallel_device_ids) + gpu_nb
        self.world_size = self.nb_gpu_nodes * len(self.data_parallel_device_ids)

        # let the exp know the rank to avoid overwriting logs
        if self.experiment is not None:
            self.experiment.rank = self.proc_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        self.__init_tcp_connection()

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        self.optimizers = model.configure_optimizers()
        if len(self.optimizers) == 2:
            self.optimizers, self.lr_schedulers = self.optimizers

        # MODEL
        # copy model to each gpu
        torch.cuda.set_device(gpu_nb)
        model.cuda(gpu_nb)

        # AMP
        # run through amp wrapper before going to distributed DP
        if self.use_amp:
            # An example
            model, optimizers = amp.initialize(
                model, self.optimizers, opt_level=self.amp_level,
            )
            self.optimizers = optimizers

        model = LightningDistributedDataParallel(model, device_ids=[gpu_nb],
                                                 find_unused_parameters=True)

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
        # sets the appropriate port
        try:
            port = os.environ['MASTER_PORT']
        except Exception:
            port = 12910
            os.environ['MASTER_PORT'] = str(port)

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

        ref_model.trainer = self

        # set local properties on the model
        ref_model.on_gpu = self.on_gpu

        # transfer data loaders from model
        self.get_dataloaders(ref_model)

        # init training constants
        self.__layout_bookeeping()

        # print model summary
        if self.proc_rank == 0 and self.print_weights_summary:
            ref_model.summarize()

        # give model convenience properties
        ref_model.trainer = self

        if self.experiment is not None:
            ref_model.experiment = self.experiment

        # save exp to get started
        if self.proc_rank == 0 and self.experiment is not None:
            self.experiment.save()

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # restore training and model before hpc call
        self.restore_state_if_existing_checkpoint()

        # enable cluster checkpointing
        # also restores training state
        # hpc checkpoint overrides any other checkpoints loaded before
        if self.cluster is not None:  # pragma: no cover
            self.enable_auto_hpc_walltime_manager()

        # run tiny validation (if validation defined) to make sure program won't crash during val
        ref_model.on_sanity_check_start()
        if self.val_dataloader is not None:
            for ds_i, dataloader in enumerate(self.val_dataloader):
                self.validate(model, dataloader, self.nb_sanity_val_steps, ds_i)

        # ---------------------------
        # CORE TRAINING LOOP
        # ---------------------------
        self.__train()

    def __train(self):
        # run all epochs
        for epoch_nb in range(self.current_epoch, self.max_nb_epochs):
            # get model
            model = self.__get_model()

            # update training progress in trainer and model
            model.current_epoch = epoch_nb
            self.current_epoch = epoch_nb
            self.total_batches = self.nb_tng_batches + self.nb_val_batches
            self.batch_loss_value = 0  # accumulated grads

            # init progbar when requested
            if self.progress_bar:
                self.prog_bar = tqdm.tqdm(range(self.total_batches),
                                          position=self.process_position)

            # -----------------
            # RUN TNG EPOCH
            # -----------------
            self.run_tng_epoch()

            # update LR schedulers
            if self.lr_schedulers is not None:
                for lr_scheduler in self.lr_schedulers:
                    lr_scheduler.step()

            # early stopping
            met_min_epochs = epoch_nb > self.min_nb_epochs
            if self.enable_early_stop and met_min_epochs:
                should_stop = self.early_stop_callback.on_epoch_end(epoch=epoch_nb,
                                                                    logs=self.__tng_tqdm_dic)
                # stop training
                stop = should_stop and met_min_epochs
                if stop:
                    return

    def run_tng_epoch(self):
        # before epoch hook
        if self.__is_function_implemented('on_epoch_start'):
            model = self.__get_model()
            model.on_epoch_start()

        # run epoch
        for batch_nb, data_batch in enumerate(self.tng_dataloader):
            self.batch_nb = batch_nb
            self.global_step += 1

            model = self.__get_model()
            model.global_step = self.global_step

            # stop when the flag is changed or we've gone past the amount
            #  requested in the batches
            self.total_batch_nb += 1
            met_batch_limit = batch_nb > self.nb_tng_batches
            if met_batch_limit:
                break

            # ---------------
            # RUN TRAIN STEP
            # ---------------
            batch_result = self.__run_tng_batch(data_batch, batch_nb)
            early_stop_epoch = batch_result == -1

            # ---------------
            # RUN VAL STEP
            # ---------------
            is_val_check_batch = (batch_nb + 1) % self.val_check_batch == 0
            if self.fast_dev_run or is_val_check_batch or early_stop_epoch:
                self.__run_validation()

            # when batch should be saved
            if (batch_nb + 1) % self.log_save_interval == 0 or early_stop_epoch:
                if self.proc_rank == 0 and self.experiment is not None:
                    self.experiment.save()

            # when metrics should be logged
            if batch_nb % self.add_log_row_interval == 0 or early_stop_epoch:
                # count items in memory
                # nb_params, nb_tensors = count_mem_items()

                model = self.__get_model()
                metrics = self.__tng_tqdm_dic

                # add gpu memory
                if self.on_gpu:
                    mem_map = get_gpu_memory_map()
                    metrics.update(mem_map)

                # add norms
                if self.track_grad_norm > 0:
                    model = self.__get_model()
                    grad_norm_dic = model.grad_norm(self.track_grad_norm)
                    metrics.update(grad_norm_dic)

                if self.__is_function_implemented('on_tng_metrics'):
                    model.on_tng_metrics(metrics)

                # log metrics
                scalar_metrics = self.__metrics_to_scalars(
                    metrics, blacklist=self.__log_vals_blacklist())
                if self.proc_rank == 0 and self.experiment is not None:
                    self.experiment.log(scalar_metrics, global_step=self.global_step)
                    self.experiment.save()

            # end epoch early
            if early_stop_epoch:
                break

        # epoch end hook
        if self.__is_function_implemented('on_epoch_end'):
            model = self.__get_model()
            model.on_epoch_end()

    def __metrics_to_scalars(self, metrics, blacklist=set()):
        new_metrics = {}
        for k, v in metrics.items():
            if type(v) is torch.Tensor:
                v = v.item()

            if type(v) is dict:
                v = self.__metrics_to_scalars(v)

            if k not in blacklist:
                new_metrics[k] = float(v)

        return new_metrics

    def __log_vals_blacklist(self):
        """avoid logging some vals lightning uses to maintain state"""
        blacklist = {'batch_nb', 'v_nb', 'gpu'}
        return blacklist

    def __run_tng_batch(self, data_batch, batch_nb):
        if data_batch is None:
            return 0

        # hook
        if self.__is_function_implemented('on_batch_start'):
            model_ref = self.__get_model()
            response = model_ref.on_batch_start(data_batch)

            if response == -1:
                return -1

        if self.progress_bar:
            self.prog_bar.update(1)

        # forward pass
        # return a scalar value and a dic with tqdm metrics
        if self.use_ddp:
            output = self.model(data_batch, batch_nb)
        elif self.use_dp:
            output = self.model(data_batch, batch_nb)
        elif self.single_gpu:
            gpu_id = self.data_parallel_device_ids[0]
            for i, x in enumerate(data_batch):
                if isinstance(x, torch.Tensor):
                    data_batch[i] = x.cuda(gpu_id)
            output = self.model.training_step(data_batch, batch_nb)

        else:
            output = self.model.training_step(data_batch, batch_nb)

        try:
            prog_output = output['prog']

            # reduce prog metrics for tqdm when using dp
            if self.use_dp:
                nb_gpus = len(self.data_parallel_device_ids)
                prog_output = reduce_distributed_output(prog_output, nb_gpus)

            model_specific_tqdm_metrics_dic = prog_output
        except Exception:
            model_specific_tqdm_metrics_dic = {}

        # if output dict doesn't have the keyword loss
        # then assume the output=loss if scalar
        try:
            loss = output['loss']
        except Exception:
            if type(output) is torch.Tensor:
                loss = output

        # when using dp need to reduce the loss
        if self.use_dp:
            loss = reduce_distributed_output(loss, len(self.data_parallel_device_ids))

        self.__add_tqdm_metrics(model_specific_tqdm_metrics_dic)

        # accumulate loss (if accumulate_grad_batches = 1 no effect)
        loss = loss / self.accumulate_grad_batches

        # backward pass
        if self.use_amp:
            # scale loss when using amp
            for optimizer in self.optimizers:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            loss.backward()

        # insert after step hook
        if self.__is_function_implemented('on_after_backward'):
            model_ref = self.__get_model()
            response = model_ref.on_after_backward()

        if self.print_nan_grads:
            model = self.__get_model()
            for param in model.parameters():
                print(param.grad.float().sum())

        # track total loss for logging (avoid mem leaks)
        self.batch_loss_value += loss.item()

        # gradient update with accumulated gradients
        if (self.batch_nb + 1) % self.accumulate_grad_batches == 0:
            # clip gradients
            if self.gradient_clip > 0:
                model = self.__get_model()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)

            # update gradients across all optimizers
            for optimizer in self.optimizers:
                optimizer.step()

                # insert after step hook
                if self.__is_function_implemented('on_before_zero_grad'):
                    model_ref = self.__get_model()
                    response = model_ref.on_before_zero_grad(optimizer)

                # clear gradients
                optimizer.zero_grad()

            # calculate running loss for display
            self.running_loss.append(self.batch_loss_value)
            self.batch_loss_value = 0
            self.avg_loss = np.mean(self.running_loss[-100:])

            # update progbar
            if self.progress_bar:
                # add model specific metrics
                tqdm_metrics = self.__tng_tqdm_dic
                self.prog_bar.set_postfix(**tqdm_metrics)

        # activate batch end hook
        if self.__is_function_implemented('on_batch_end'):
            model = self.__get_model()
            model.on_batch_end()

        return 0

    def __run_validation(self):
        # decide if can check epochs
        can_check_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
        if self.fast_dev_run:
            print('skipping to check performance bc of --fast_dev_run')
        elif not can_check_epoch:
            return

        # validate only if model has validation_step defined
        if self.__is_overriden('validation_step'):

            # hook
            if self.__is_function_implemented('on_pre_performance_check'):
                model = self.__get_model()
                model.on_pre_performance_check()

            # use full val set on end of epoch
            # use a small portion otherwise
            max_batches = None if not self.fast_dev_run else 1
            for ds_i, dataloader in enumerate(self.val_dataloader):
                val_out_metrics = self.validate(self.model, dataloader, max_batches, ds_i)
                self.__add_tqdm_metrics(val_out_metrics)

            # hook
            if self.__is_function_implemented('on_post_performance_check'):
                model = self.__get_model()
                model.on_post_performance_check()

            if self.progress_bar:
                # add model specific metrics
                tqdm_metrics = self.__tng_tqdm_dic
                self.prog_bar.set_postfix(**tqdm_metrics)

        # model checkpointing
        if self.proc_rank == 0 and self.checkpoint_callback is not None:
            print('save callback...')
            self.checkpoint_callback.on_epoch_end(epoch=self.current_epoch,
                                                  logs=self.__tng_tqdm_dic)
