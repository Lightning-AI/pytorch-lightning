"""
The trainer handles all the logic for running a val loop, training loop, distributing, etc.. .
"""

import os
import sys
import warnings
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch.optim.optimizer import Optimizer

from pytorch_lightning.trainer.auto_mix_precision import TrainerAMPMixin
from pytorch_lightning.trainer.callback_config import TrainerCallbackConfigMixin
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.distrib_data_parallel import TrainerDDPMixin
from pytorch_lightning.trainer.distrib_parts import (
    TrainerDPMixin,
    parse_gpu_ids,
    determine_root_gpu_device
)
from pytorch_lightning.trainer.evaluation_loop import TrainerEvaluationLoopMixin
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from pytorch_lightning.trainer.training_loop import TrainerTrainLoopMixin
from pytorch_lightning.trainer.training_io import TrainerIOMixin
from pytorch_lightning.trainer.training_tricks import TrainerTrainingTricksMixin
from pytorch_lightning.utilities.debugging import MisconfigurationException

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class Trainer(TrainerIOMixin,
              TrainerDPMixin,
              TrainerDDPMixin,
              TrainerLoggingMixin,
              TrainerModelHooksMixin,
              TrainerTrainingTricksMixin,
              TrainerDataLoadingMixin,
              TrainerAMPMixin,
              TrainerEvaluationLoopMixin,
              TrainerTrainLoopMixin,
              TrainerCallbackConfigMixin,
              ):

    def __init__(
            self,
            logger=True,
            checkpoint_callback=True,
            early_stop_callback=True,
            default_save_path=None,
            gradient_clip_val=0,
            gradient_clip=None,  # backward compatible, todo: remove in v0.8.0
            process_position=0,
            nb_gpu_nodes=None,  # backward compatible, todo: remove in v0.8.0
            num_nodes=1,
            gpus=None,
            log_gpu_memory=None,
            show_progress_bar=True,
            overfit_pct=0.0,
            track_grad_norm=-1,
            check_val_every_n_epoch=1,
            fast_dev_run=False,
            accumulate_grad_batches=1,
            max_nb_epochs=None,  # backward compatible, todo: remove in v0.8.0
            min_nb_epochs=None,  # backward compatible, todo: remove in v0.8.0
            max_epochs=1000,
            min_epochs=1,
            train_percent_check=1.0,
            val_percent_check=1.0,
            test_percent_check=1.0,
            val_check_interval=1.0,
            log_save_interval=100,
            row_log_interval=10,
            add_row_log_interval=None,  # backward compatible, todo: remove in v0.8.0
            distributed_backend=None,
            use_amp=False,
            print_nan_grads=False,
            weights_summary='full',
            weights_save_path=None,
            amp_level='O1',
            nb_sanity_val_steps=None,  # backward compatible, todo: remove in v0.8.0
            num_sanity_val_steps=5,
            truncated_bptt_steps=None,
            resume_from_checkpoint=None,
    ):
        """

        :param logger: Logger for experiment tracking
        :param checkpoint_callback: Callback for checkpointing
        :param early_stop_callback: Callback for early stopping
        :param str default_save_path: Default path for logs+weights if no logger/ckpt_callback passed
        :param int gradient_clip_val: 0 means don't clip.
        :param int gradient_clip: 0 means don't clip. Deprecated.
        :param process_position: shown in the tqdm bar
        :param int num_nodes: number of GPU nodes
        :param list|str|int gpus: int. (ie: 2 gpus) OR list to specify which GPUs [0, 1] OR '0,1'
            OR '-1' / -1 to use all available gpus
        :param str log_gpu_memory: None, 'min_max', 'all'
        :param bool show_progress_bar: If true shows tqdm bar
        :param float overfit_pct: uses this much of all datasets
        :param int track_grad_norm: -1 no tracking. Otherwise tracks that norm
        :param int check_val_every_n_epoch: check val every n train epochs
        :param bool fast_dev_run: runs full iteration over everything to find bugs
        :param int accumulate_grad_batches: Accumulates grads every k batches
        :param int max_epochs:
        :param int min_epochs:
        :param int train_percent_check: How much of train set to check
        :param int val_percent_check: How much of val set to check
        :param int test_percent_check: How much of test set to check
        :param float|int val_check_interval: If float, % of tng epoch. If int, check every n batch
        :param int log_save_interval: Writes logs to disk this often
        :param int row_log_interval: How often to add logging rows
        :param int add_row_log_interval: How often to add logging rows. Deprecated.
        :param str distributed_backend: Options: 'dp', 'ddp', 'ddp2'.
        :param bool use_amp: If true uses apex for 16bit precision
        :param bool print_nan_grads: Prints nan gradients
        :param str weights_summary: Options: 'full', 'top', None to not print.
        :param bool weights_save_path: Where to save weights if on cluster
        :param str amp_level: Check nvidia docs for level
        :param int num_sanity_val_steps: How many val steps before a full train loop.
        :param int truncated_bptt_steps: Enables multiple backward passes for each batch.

        .. warning:: Following arguments become deprecated and they will be removed in v0.8.0:
            - `gradient_clip`,
            - `nb_gpu_nodes`,
            - `max_nb_epochs`,
            - `min_nb_epochs`,
            - `add_row_log_interval`,
            - `nb_sanity_val_steps`

        """
        # Transfer params
        if nb_gpu_nodes is not None:  # Backward compatibility
            warnings.warn("`nb_gpu_nodes` has renamed to `num_nodes` since v0.5.0"
                          " and will be removed in v0.8.0", DeprecationWarning)
            if not num_nodes:  # in case you did not set the proper value
                num_nodes = nb_gpu_nodes
        self.num_gpu_nodes = num_nodes
        self.log_gpu_memory = log_gpu_memory
        if gradient_clip is not None:  # Backward compatibility
            warnings.warn("`gradient_clip` has renamed to `gradient_clip_val` since v0.5.0"
                          " and will be removed in v0.8.0", DeprecationWarning)
            if not gradient_clip_val:  # in case you did not set the proper value
                gradient_clip_val = gradient_clip
        self.gradient_clip_val = gradient_clip_val
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.track_grad_norm = track_grad_norm
        self.on_gpu = True if (gpus and torch.cuda.is_available()) else False
        self.process_position = process_position
        self.weights_summary = weights_summary
        if max_nb_epochs is not None:  # Backward compatibility
            warnings.warn("`max_nb_epochs` has renamed to `max_epochs` since v0.5.0"
                          " and will be removed in v0.8.0", DeprecationWarning)
            if not max_epochs:  # in case you did not set the proper value
                max_epochs = max_nb_epochs
        self.max_epochs = max_epochs
        if min_nb_epochs is not None:  # Backward compatibility
            warnings.warn("`min_nb_epochs` has renamed to `min_epochs` since v0.5.0"
                          " and will be removed in v0.8.0", DeprecationWarning)
            if not min_epochs:  # in case you did not set the proper value
                min_epochs = min_nb_epochs
        self.min_epochs = min_epochs
        if nb_sanity_val_steps is not None:  # Backward compatibility
            warnings.warn("`nb_sanity_val_steps` has renamed to `num_sanity_val_steps` since v0.5.0"
                          " and will be removed in v0.8.0", DeprecationWarning)
            if not num_sanity_val_steps:  # in case you did not set the proper value
                num_sanity_val_steps = nb_sanity_val_steps
        self.num_sanity_val_steps = num_sanity_val_steps
        self.print_nan_grads = print_nan_grads
        self.truncated_bptt_steps = truncated_bptt_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.shown_warnings = set()

        self.fast_dev_run = fast_dev_run
        if self.fast_dev_run:
            self.num_sanity_val_steps = 1
            self.max_epochs = 1
            m = '''
            Running in fast_dev_run mode: will run a full train,
            val loop using a single batch
            '''
            logging.info(m)

        # set default save path if user didn't provide one
        self.default_save_path = default_save_path
        if self.default_save_path is None:
            self.default_save_path = os.getcwd()

        # training bookeeping
        self.total_batch_idx = 0
        self.running_loss = []
        self.avg_loss = 0
        self.batch_idx = 0
        self.tqdm_metrics = {}
        self.callback_metrics = {}
        self.num_val_batches = 0
        self.num_training_batches = 0
        self.num_test_batches = 0
        self.get_train_dataloader = None
        self.get_test_dataloaders = None
        self.get_val_dataloaders = None
        self.is_iterable_train_dataloader = False

        # training state
        self.model = None
        self.testing = False
        self.disable_validation = False
        self.lr_schedulers = []
        self.optimizers = None
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0

        # configure early stop callback
        # creates a default one if none passed in
        self.early_stop_callback = None
        self.configure_early_stopping(early_stop_callback, logger)

        self.reduce_lr_on_plateau_scheduler = None

        # configure checkpoint callback
        self.checkpoint_callback = checkpoint_callback
        self.weights_save_path = weights_save_path

        # accumulated grads
        self.configure_accumulated_gradients(accumulate_grad_batches)

        # allow int, string and gpu list
        self.data_parallel_device_ids = parse_gpu_ids(gpus)
        self.root_gpu = determine_root_gpu_device(self.data_parallel_device_ids)

        # distributed backend choice
        self.use_ddp = False
        self.use_ddp2 = False
        self.use_dp = False
        self.single_gpu = False
        self.distributed_backend = distributed_backend
        self.set_distributed_mode(distributed_backend, num_nodes)

        # init flags for SLURM+ddp to work
        self.proc_rank = 0
        self.world_size = 1
        self.node_rank = 0
        self.configure_slurm_ddp(num_nodes)

        # nvidia setup
        self.set_nvidia_flags(self.is_slurm_managing_tasks, self.data_parallel_device_ids)

        # can't init progress bar here because starting a new process
        # means the progress_bar won't survive pickling
        self.show_progress_bar = show_progress_bar

        # logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval
        if add_row_log_interval is not None:
            # backward compatibility
            warnings.warn("`add_row_log_interval` has renamed to `row_log_interval` since v0.5.0"
                          " and will be removed in v0.8.0", DeprecationWarning)
            if not row_log_interval:  # in case you did not set the proper value
                row_log_interval = add_row_log_interval
        self.row_log_interval = row_log_interval

        # how much of the data to use
        self.determine_data_use_amount(train_percent_check, val_percent_check,
                                       test_percent_check, overfit_pct)

        # 16 bit mixed precision training using apex
        self.amp_level = amp_level
        self.init_amp(use_amp)

    @property
    def slurm_job_id(self):
        try:
            job_id = os.environ['SLURM_JOB_ID']
            job_id = int(job_id)
        except Exception:
            job_id = None
        return job_id

    def __parse_gpu_ids(self, gpus):
        """Parse GPUs id.

        :param list|str|int gpus: input GPU ids
        :return list(int):
        """
        # if gpus = -1 then use all available devices
        # otherwise, split the string using commas
        if gpus is not None:
            if isinstance(gpus, list):
                gpus = gpus
            elif isinstance(gpus, str):
                if gpus == '-1':
                    gpus = list(range(0, torch.cuda.device_count()))
                else:
                    gpus = [int(x.strip()) for x in gpus.split(',')]
            elif isinstance(gpus, int):
                gpus = gpus
            else:
                raise ValueError('`gpus` has to be a string, int or list of ints')

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
        else:
            return len(gpus)

    @property
    def data_parallel(self):
        return self.use_dp or self.use_ddp or self.use_ddp2

    @property
    def training_tqdm_dict(self):
        """Read-only for tqdm metrics.
        :return:
        """
        tqdm_dict = {
            'loss': '{0:.3f}'.format(self.avg_loss),
            'batch_idx': '{}'.format(self.batch_idx),
        }

        if self.truncated_bptt_steps is not None:
            tqdm_dict['split_idx'] = self.split_idx

        if self.logger is not None and self.logger.version is not None:
            tqdm_dict['v_num'] = self.logger.version

        tqdm_dict.update(self.tqdm_metrics)

        if self.on_gpu:
            tqdm_dict['gpu'] = '{}'.format(torch.cuda.current_device())

        return tqdm_dict

    @property
    def tng_tqdm_dic(self):
        """Read-only for tqdm metrics.

        .. warning:: Deprecated in v0.5.0. use training_tqdm_dict instead.
        :return:
        """
        warnings.warn("`tng_tqdm_dic` has renamed to `training_tqdm_dict` since v0.5.0"
                      " and will be removed in v0.8.0", DeprecationWarning)
        return self.training_tqdm_dict

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
                mp.spawn(self.ddp_train, nprocs=self.num_gpus, args=(model,))

        # 1 gpu or dp option triggers training using DP module
        # easier to avoid NCCL issues
        elif self.use_dp:
            self.dp_train(model)

        elif self.single_gpu:
            self.single_gpu_train(model)

        # ON CPU
        else:
            # run through amp wrapper
            if self.use_amp:
                raise MisconfigurationException('amp + cpu is not supported.  Please use a GPU option')

            # CHOOSE OPTIMIZER
            # allow for lr schedulers as well
            self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

            self.run_pretrain_routine(model)

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
            lr_schedulers, self.reduce_lr_on_plateau_scheduler = self.configure_schedulers(lr_schedulers)
            return optimizers, lr_schedulers

        # single list or tuple
        elif isinstance(optimizers, list) or isinstance(optimizers, tuple):
            return optimizers, []

    def configure_schedulers(self, schedulers):
        for i, scheduler in enumerate(schedulers):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                reduce_lr_on_plateau_scheduler = schedulers.pop(i)
                return schedulers, reduce_lr_on_plateau_scheduler
        return schedulers, None

    def run_pretrain_routine(self, model):
        """Sanity check a few things before starting actual training.

        :param model:
        """
        ref_model = model
        if self.data_parallel:
            ref_model = model.module

        # give model convenience properties
        ref_model.trainer = self

        # set local properties on the model
        self.copy_trainer_model_properties(ref_model)

        # link up experiment object
        if self.logger is not None:
            ref_model.logger = self.logger

            # save exp to get started
            if hasattr(ref_model, "hparams"):
                self.logger.log_hyperparams(ref_model.hparams)

            self.logger.save()

        if self.use_ddp or self.use_ddp2:
            dist.barrier()

        # set up checkpoint callback
        self.configure_checkpoint_callback()

        # register auto-resubmit when on SLURM
        self.register_slurm_signal_handlers()

        # transfer data loaders from model
        self.get_dataloaders(ref_model)

        # print model summary
        if self.proc_rank == 0 and self.weights_summary is not None:
            if self.weights_summary in ['full', 'top']:
                ref_model.summarize(mode=self.weights_summary)
            else:
                m = "weights_summary can be None, 'full' or 'top'"
                raise MisconfigurationException(m)

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # restore training and model before hpc call
        self.restore_weights(model)

        # when testing requested only run test and return
        if self.testing:
            self.run_evaluation(test=True)
            return

        # check if we should run validation during training
        self.disable_validation = ((self.num_val_batches == 0 or
                                   not self.is_overriden('validation_step')) and
                                   not self.fast_dev_run)

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        ref_model.on_sanity_check_start()
        ref_model.on_train_start()
        if not self.disable_validation and self.num_sanity_val_steps > 0:
            # init progress bars for validation sanity check
            pbar = tqdm.tqdm(desc='Validation sanity check',
                             total=self.num_sanity_val_steps * len(self.get_val_dataloaders()),
                             leave=False, position=2 * self.process_position,
                             disable=not self.show_progress_bar, dynamic_ncols=True, unit='batch')
            self.main_progress_bar = pbar
            # dummy validation progress bar
            self.val_progress_bar = tqdm.tqdm(disable=True)

            self.evaluate(model, self.get_val_dataloaders(), self.num_sanity_val_steps, self.testing)

            # close progress bars
            self.main_progress_bar.close()
            self.val_progress_bar.close()

        # init progress bar
        pbar = tqdm.tqdm(leave=True, position=2 * self.process_position,
                         disable=not self.show_progress_bar, dynamic_ncols=True, unit='batch',
                         file=sys.stdout)
        self.main_progress_bar = pbar

        # clear cache before training
        if self.on_gpu:
            torch.cuda.empty_cache()

        # CORE TRAINING LOOP
        self.train()

    def test(self, model=None):
        self.testing = True
        if model is not None:
            self.fit(model)
        else:
            self.run_evaluation(test=True)
