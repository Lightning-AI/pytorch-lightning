import os
import sys
import warnings
import logging as log
from typing import Union, Optional, List, Dict, Tuple, Iterable
from argparse import ArgumentParser

import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim.optimizer import Optimizer

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.profiler.profiler import BaseProfiler
from pytorch_lightning.trainer.auto_mix_precision import TrainerAMPMixin
from pytorch_lightning.trainer.callback_config import TrainerCallbackConfigMixin
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.distrib_data_parallel import TrainerDDPMixin
from pytorch_lightning.trainer.distrib_parts import (
    TrainerDPMixin,
    parse_gpu_ids,
    determine_root_gpu_device
)
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.callback_hook import TrainerCallbackHookMixin
from pytorch_lightning.trainer.deprecated_api import TrainerDeprecatedAPITillVer0_8
from pytorch_lightning.trainer.evaluation_loop import TrainerEvaluationLoopMixin
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from pytorch_lightning.trainer.training_io import TrainerIOMixin
from pytorch_lightning.trainer.training_loop import TrainerTrainLoopMixin
from pytorch_lightning.trainer.training_tricks import TrainerTrainingTricksMixin
from pytorch_lightning.utilities.debugging import MisconfigurationException
from pytorch_lightning.profiler import Profiler, PassThroughProfiler
from pytorch_lightning.callbacks import Callback


try:
    from apex import amp
except ImportError:
    APEX_AVAILABLE = False
else:
    APEX_AVAILABLE = True

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


class Trainer(
    TrainerIOMixin,
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
    TrainerCallbackHookMixin,
    TrainerDeprecatedAPITillVer0_8,
):

    def __init__(
            self,
            logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
            checkpoint_callback: Union[ModelCheckpoint, bool] = True,
            early_stop_callback: Optional[Union[EarlyStopping, bool]] = False,
            callbacks: List[Callback] = [],
            default_save_path: Optional[str] = None,
            gradient_clip_val: float = 0,
            gradient_clip=None,  # backward compatible, todo: remove in v0.8.0
            process_position: int = 0,
            nb_gpu_nodes=None,  # backward compatible, todo: remove in v0.8.0
            num_nodes: int = 1,
            gpus: Optional[Union[List[int], str, int]] = None,
            num_tpu_cores: Optional[int] = None,
            log_gpu_memory: Optional[str] = None,
            show_progress_bar: bool = True,
            progress_bar_refresh_rate: int = 50,
            overfit_pct: float = 0.0,
            track_grad_norm: int = -1,
            check_val_every_n_epoch: int = 1,
            fast_dev_run: bool = False,
            accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
            max_nb_epochs=None,  # backward compatible, todo: remove in v0.8.0
            min_nb_epochs=None,  # backward compatible, todo: remove in v0.8.0
            max_epochs: int = 1000,
            min_epochs: int = 1,
            max_steps: Optional[int] = None,
            min_steps: Optional[int] = None,
            train_percent_check: float = 1.0,
            val_percent_check: float = 1.0,
            test_percent_check: float = 1.0,
            val_check_interval: float = 1.0,
            log_save_interval: int = 100,
            row_log_interval: int = 10,
            add_row_log_interval=None,  # backward compatible, todo: remove in v0.8.0
            distributed_backend: Optional[str] = None,
            use_amp=False,  # backward compatible, todo: remove in v0.9.0
            precision: int = 32,
            print_nan_grads: bool = False,
            weights_summary: str = 'full',
            weights_save_path: Optional[str] = None,
            amp_level: str = 'O1',
            nb_sanity_val_steps=None,  # backward compatible, todo: remove in v0.8.0
            num_sanity_val_steps: int = 5,
            truncated_bptt_steps: Optional[int] = None,
            resume_from_checkpoint: Optional[str] = None,
            profiler: Optional[BaseProfiler] = None,
            benchmark: bool = False,
            reload_dataloaders_every_epoch: bool = False,
            **kwargs
    ):
        r"""

        Customize every aspect of training via flags

        Args:
            logger: Logger (or iterable collection of loggers) for experiment tracking.

            checkpoint_callback: Callback for checkpointing.

            early_stop_callback (:class:`pytorch_lightning.callbacks.EarlyStopping`):

            callbacks: Add a list of callbacks.

            default_save_path: Default path for logs and weights when no logger/ckpt_callback passed

            gradient_clip_val: 0 means don't clip.

            gradient_clip:
                .. warning:: deprecated 0.7.0 Use `gradient_clip_val` instead. Will remove 0.9.0.

            process_position: orders the tqdm bar when running multiple models on same machine.

            num_nodes: number of GPU nodes for distributed training.

            nb_gpu_nodes:
                .. warning:: .. deprecated:: 0.7.0
                    Use `num_nodes` instead. Will remove 0.9.0.

            gpus: Which GPUs to train on.

            num_tpu_cores: How many TPU cores to train on (1 or 8).

            log_gpu_memory: None, 'min_max', 'all'. Might slow performance

            show_progress_bar: If true shows tqdm progress bar

            progress_bar_refresh_rate: How often to refresh progress bar (in steps)

            track_grad_norm: -1 no tracking. Otherwise tracks that norm

            check_val_every_n_epoch: Check val every n train epochs.

            fast_dev_run: runs 1 batch of train, test  and val to find any bugs (ie: a sort of unit test).

            accumulate_grad_batches: Accumulates grads every k batches or as set up in the dict.

            max_epochs: Stop training once this number of epochs is reached.

            max_nb_epochs:
                .. warning:: .. deprecated:: 0.7.0
                    Use `max_epochs` instead. Will remove 0.9.0.

            min_epochs: Force training for at least these many epochs

            min_nb_epochs:
                .. warning:: .. deprecated:: 0.7.0
                    Use `min_epochs` instead. Will remove 0.9.0.

            max_steps: Stop training after this number of steps. Disabled by default (None).

            min_steps: Force training for at least these number of steps. Disabled by default (None).

            train_percent_check: How much of training dataset to check.

            val_percent_check: How much of validation dataset to check.

            test_percent_check: How much of test dataset to check.

            val_check_interval: How often within one training epoch to check the validation set

            log_save_interval: Writes logs to disk this often

            row_log_interval: How often to add logging rows (does not write to disk)

            add_row_log_interval:
                .. warning:: .. deprecated:: 0.7.0
                    Use `row_log_interval` instead. Will remove 0.9.0.

            distributed_backend: The distributed backend to use.

            use_amp:
                .. warning:: .. deprecated:: 0.7.0
                    Use `precision` instead. Will remove 0.9.0.

            precision: Full precision (32), half precision (16).

            print_nan_grads: Prints gradients with nan values

            weights_summary: Prints a summary of the weights when training begins.

            weights_save_path: Where to save weights if specified.

            amp_level: The optimization level to use (O1, O2, etc...).

            num_sanity_val_steps: Sanity check runs n batches of val before starting the training routine.

            nb_sanity_val_steps:
                .. warning:: .. deprecated:: 0.7.0
                    Use `num_sanity_val_steps` instead. Will remove 0.8.0.

            truncated_bptt_steps: Truncated back prop breaks performs backprop every k steps of

            resume_from_checkpoint: To resume training from a specific checkpoint pass in the path here.k

            profiler:  To profile individual steps during training and assist in

            reload_dataloaders_every_epoch: Set to True to reload dataloaders every epoch

            benchmark: If true enables cudnn.benchmark.
        """

        # Init callbacks
        self.callbacks = callbacks
        self.on_init_start()

        # benchmarking
        self.benchmark = benchmark
        if benchmark:
            torch.backends.cudnn.benchmark = True

        # Transfer params
        self.num_nodes = num_nodes
        # Backward compatibility, TODO: remove in v0.8.0
        if nb_gpu_nodes is not None:
            warnings.warn("Argument `nb_gpu_nodes` has renamed to `num_nodes` since v0.5.0"
                          " and this method will be removed in v0.8.0", DeprecationWarning)
            self.num_gpu_nodes = nb_gpu_nodes
        self.log_gpu_memory = log_gpu_memory

        self.gradient_clip_val = gradient_clip_val
        # Backward compatibility, TODO: remove in v0.8.0
        if gradient_clip is not None:
            warnings.warn("Argument `gradient_clip` has renamed to `gradient_clip_val` since v0.5.0"
                          " and this method will be removed in v0.8.0", DeprecationWarning)
            self.gradient_clip = gradient_clip

        self.reload_dataloaders_every_epoch = reload_dataloaders_every_epoch
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.track_grad_norm = track_grad_norm
        self.on_gpu = True if (gpus and torch.cuda.is_available()) else False

        # tpu config
        self.on_tpu = num_tpu_cores is not None
        self.num_tpu_cores = num_tpu_cores
        assert num_tpu_cores in [1, 8, None], 'num_tpu_cores can only be 1 or 8'

        self.process_position = process_position
        self.weights_summary = weights_summary

        self.max_epochs = max_epochs
        # Backward compatibility, TODO: remove in v0.8.0
        if max_nb_epochs is not None:
            warnings.warn("Argument `max_nb_epochs` has renamed to `max_epochs` since v0.5.0"
                          " and this method will be removed in v0.8.0", DeprecationWarning)
            self.max_nb_epochs = max_nb_epochs

        self.min_epochs = min_epochs
        # Backward compatibility, TODO: remove in v0.8.0
        if min_nb_epochs is not None:
            warnings.warn("Argument `min_nb_epochs` has renamed to `min_epochs` since v0.5.0"
                          " and this method will be removed in v0.8.0", DeprecationWarning)
            self.min_nb_epochs = min_nb_epochs

        self.max_steps = max_steps
        self.min_steps = min_steps

        self.num_sanity_val_steps = num_sanity_val_steps
        # Backward compatibility, TODO: remove in v0.8.0
        if nb_sanity_val_steps is not None:
            warnings.warn("Argument `nb_sanity_val_steps` has renamed to "
                          "`num_sanity_val_steps` since v0.5.0"
                          " and this method will be removed in v0.8.0", DeprecationWarning)
            self.nb_sanity_val_steps = nb_sanity_val_steps
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
            log.info(m)

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
        self.train_dataloader = None
        self.test_dataloaders = None
        self.val_dataloaders = None

        # training state
        self.model = None
        self.testing = False
        self.disable_validation = False
        self.lr_schedulers = []
        self.optimizers = None
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0

        # configure logger
        self.configure_logger(logger)

        # configure profiler
        if profiler is True:
            profiler = Profiler()
        self.profiler = profiler or PassThroughProfiler()

        # configure early stop callback
        # creates a default one if none passed in
        self.configure_early_stopping(early_stop_callback)

        # configure checkpoint callback
        self.checkpoint_callback = checkpoint_callback
        self.weights_save_path = weights_save_path

        # accumulated grads
        self.accumulate_grad_batches = accumulate_grad_batches
        self.configure_accumulated_gradients(accumulate_grad_batches)

        # allow int, string and gpu list
        self.gpus = gpus
        self.data_parallel_device_ids = parse_gpu_ids(self.gpus)
        self.root_gpu = determine_root_gpu_device(self.data_parallel_device_ids)

        # tpu state flags
        self.use_tpu = False
        self.tpu_local_core_rank = None
        self.tpu_global_core_rank = None

        # distributed backend choice
        self.use_ddp = False
        self.use_ddp2 = False
        self.use_dp = False
        self.single_gpu = False
        self.distributed_backend = distributed_backend
        self.set_distributed_mode(distributed_backend, self.num_nodes)

        # override dist backend when using tpus
        if self.on_tpu:
            self.init_tpu()
            self.current_tpu_idx = None

        # init flags for SLURM+ddp to work
        self.proc_rank = 0
        self.world_size = 1
        self.node_rank = 0
        self.configure_slurm_ddp(self.num_nodes)

        # nvidia setup
        self.set_nvidia_flags(self.is_slurm_managing_tasks, self.data_parallel_device_ids)

        # can't init progress bar here because starting a new process
        # means the progress_bar won't survive pickling
        self.show_progress_bar = show_progress_bar

        # logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval

        # backward compatibility
        if add_row_log_interval is not None:
            warnings.warn("`add_row_log_interval` has renamed to `row_log_interval` since v0.5.0"
                          " and this method will be removed in v0.8.0", DeprecationWarning)
            if not row_log_interval:  # in case you did not set the proper value
                row_log_interval = add_row_log_interval
        self.row_log_interval = row_log_interval

        # how much of the data to use
        self.overfit_pct = overfit_pct
        self.determine_data_use_amount(train_percent_check, val_percent_check,
                                       test_percent_check, overfit_pct)

        # 16 bit mixed precision training using apex
        self.amp_level = amp_level
        self.precision = precision

        assert self.precision in (16, 32), 'only 32 or 16 bit precision supported'

        if self.precision == 16 and self.num_tpu_cores is None:
            use_amp = True
        self.init_amp(use_amp)

        # Callback system
        self.on_init_end()

    @property
    def slurm_job_id(self) -> int:
        try:
            job_id = os.environ['SLURM_JOB_ID']
            job_id = int(job_id)
        except Exception:
            job_id = None
        return job_id

    @classmethod
    def default_attributes(cls):
        import inspect

        init_signature = inspect.signature(Trainer)

        args = {}
        for param_name in init_signature.parameters:
            value = init_signature.parameters[param_name].default
            args[param_name] = value

        return args

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        """Extend existing argparse by default `Trainer` attributes."""
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        trainer_default_params = Trainer.default_attributes()

        # TODO: get "help" from docstring :)
        for arg in trainer_default_params:
            parser.add_argument(
                f'--{arg}',
                default=trainer_default_params[arg],
                dest=arg,
                help='autogenerated by pl.Trainer'
            )

        return parser

    @classmethod
    def from_argparse_args(cls, args):

        params = vars(args)
        return cls(**params)

    @property
    def num_gpus(self) -> int:
        gpus = self.data_parallel_device_ids
        if gpus is None:
            return 0
        return len(gpus)

    @property
    def data_parallel(self) -> bool:
        return self.use_dp or self.use_ddp or self.use_ddp2

    @property
    def training_tqdm_dict(self) -> dict:
        """Read-only for tqdm metrics.
        :return:
        """
        ref_model = self.model if not self.data_parallel else self.model.module

        return dict(**ref_model.get_tqdm_dict(), **self.tqdm_metrics)

    @property
    def tng_tqdm_dic(self):
        """Read-only for tqdm metrics.

        :return: dictionary

        .. warning:: .. deprecated:: 0.5.0
                    Use `training_tqdm_dict` instead. Will remove 0.8.0.
        """
        warnings.warn("`tng_tqdm_dic` has renamed to `training_tqdm_dict` since v0.5.0"
                      " and this method will be removed in v0.8.0", DeprecationWarning)
        return self.training_tqdm_dict

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    def fit(
            self,
            model: LightningModule,
            train_dataloader: Optional[DataLoader] = None,
            val_dataloaders: Optional[DataLoader] = None,
            test_dataloaders: Optional[DataLoader] = None
    ):
        r"""
        Runs the full optimization routine.

        Args:
            model: Model to fit.

            train_dataloader: A Pytorch
                DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            val_dataloaders: Either a single
                Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped

            test_dataloaders: Either a single
                Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined test_dataloaders method this will be skipped

        Example::

            # Option 1,
            # Define the train_dataloader(), test_dataloader() and val_dataloader() fxs
            # in the lightningModule
            # RECOMMENDED FOR MOST RESEARCH AND APPLICATIONS TO MAINTAIN READABILITY
            trainer = Trainer()
            model = LightningModule()
            trainer.fit(model)

            # Option 2
            # in production cases we might want to pass different datasets to the same model
            # Recommended for PRODUCTION SYSTEMS
            train, val, test = DataLoader(...), DataLoader(...), DataLoader(...)
            trainer = Trainer()
            model = LightningModule()
            trainer.fit(model, train_dataloader=train,
                        val_dataloader=val, test_dataloader=test)

            # Option 1 & 2 can be mixed, for example the training set can be
            # defined as part of the model, and validation/test can then be
            # feed to .fit()

        """
        # bind logger and other properties
        model.logger = self.logger
        self.copy_trainer_model_properties(model)

        # set up the passed in dataloaders (if needed)
        self.__attach_dataloaders(model, train_dataloader, val_dataloaders, test_dataloaders)

        # download the data and do whatever transforms we need
        # do before any spawn calls so that the model can assign properties
        # only on proc 0 because no spawn has happened yet
        model.prepare_data()

        # route to appropriate start method
        # when using multi-node or DDP within a node start each module in a separate process
        if self.use_ddp2:
            task = int(os.environ['SLURM_LOCALID'])
            self.ddp_train(task, model)

        elif self.use_ddp:
            if self.is_slurm_managing_tasks:
                task = int(os.environ['SLURM_LOCALID'])
                self.ddp_train(task, model)
            else:
                self.__set_random_port()

                # track for predict
                self.model = model

                # train
                mp.spawn(self.ddp_train, nprocs=self.num_gpus, args=(model,))

                # load weights if not interrupted
                self.load_spawn_weights(model)
                self.model = model

        # 1 gpu or dp option triggers training using DP module
        # easier to avoid NCCL issues
        elif self.use_dp:
            self.dp_train(model)

        elif self.single_gpu:
            self.single_gpu_train(model)

        elif self.use_tpu:  # pragma: no cover
            log.info(f'training on {self.num_tpu_cores} TPU cores')

            #  COLAB_GPU is an env var available by default in Colab environments.
            start_method = 'fork' if os.getenv('COLAB_GPU') else 'spawn'

            # track for predict
            self.model = model

            # train
            xmp.spawn(self.tpu_train, args=(model,), nprocs=self.num_tpu_cores, start_method=start_method)

            # load weights if not interrupted
            self.load_spawn_weights(model)
            self.model = model

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

    def __set_random_port(self):
        """
        When running DDP NOT managed by SLURM, the ports might collide
        :return:
        """
        try:
            default_port = os.environ['MASTER_PORT']
        except Exception:
            import random
            default_port = random.randint(10000, 19000)
            os.environ['MASTER_PORT'] = str(default_port)

    def __attach_dataloaders(self, model, train_dataloader, val_dataloaders, test_dataloaders):
        # when dataloader is passed via fit, patch the train_dataloader
        # functions to overwrite with these implementations
        if train_dataloader is not None:
            if not self.is_overriden('training_step', model):
                m = 'You called .fit() with a train_dataloader but did not define training_step()'
                raise MisconfigurationException(m)

            model.train_dataloader = _PatchDataLoader(train_dataloader)

        if val_dataloaders is not None:
            if not self.is_overriden('validation_step', model):
                m = 'You called .fit() with a val_dataloaders but did not define validation_step()'
                raise MisconfigurationException(m)

            model.val_dataloader = _PatchDataLoader(val_dataloaders)

        if test_dataloaders is not None:
            if not self.is_overriden('test_step', model):
                m = 'You called .fit() with a test_dataloaders but did not define test_step()'
                raise MisconfigurationException(m)

            model.test_dataloader = _PatchDataLoader(test_dataloaders)

    def init_optimizers(
            self,
            optimizers: Union[Optimizer, Tuple[List, List], List[Optimizer], Tuple[Optimizer]]
    ) -> Tuple[List, List]:

        # single output, single optimizer
        if isinstance(optimizers, Optimizer):
            return [optimizers], []

        # two lists, optimizer + lr schedulers
        elif len(optimizers) == 2 and isinstance(optimizers[0], list):
            optimizers, lr_schedulers = optimizers
            lr_schedulers = self.configure_schedulers(lr_schedulers)
            return optimizers, lr_schedulers

        # single list or tuple, multiple optimizer
        elif isinstance(optimizers, (list, tuple)):
            return optimizers, []

        # unknown configuration
        else:
            raise ValueError('Unknown configuration for model optimizers. Output'
                             'from model.configure_optimizers() should either be:'
                             '* single output, single torch.optim.Optimizer'
                             '* single output, list of torch.optim.Optimizer'
                             '* two outputs, first being a list of torch.optim.Optimizer',
                             'second being a list of torch.optim.lr_scheduler')

    def configure_schedulers(self, schedulers: list):
        # Convert each scheduler into dict sturcture with relevant information
        lr_schedulers = []
        default_config = {'interval': 'epoch',  # default every epoch
                          'frequency': 1,  # default every epoch/batch
                          'reduce_on_plateau': False,  # most often not ReduceLROnPlateau scheduler
                          'monitor': 'val_loss'}  # default value to monitor for ReduceLROnPlateau
        for scheduler in schedulers:
            if isinstance(scheduler, dict):
                if 'scheduler' not in scheduler:
                    raise ValueError(f'Lr scheduler should have key `scheduler`',
                                     ' with item being a lr scheduler')
                scheduler['reduce_on_plateau'] = \
                    isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)

                lr_schedulers.append({**default_config, **scheduler})

            elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedulers.append({**default_config, 'scheduler': scheduler,
                                      'reduce_on_plateau': True})

            elif isinstance(scheduler, optim.lr_scheduler._LRScheduler):
                lr_schedulers.append({**default_config, 'scheduler': scheduler})
            else:
                raise ValueError(f'Input {scheduler} to lr schedulers '
                                 'is a invalid input.')
        return lr_schedulers

    def run_pretrain_routine(self, model: LightningModule):
        """Sanity check a few things before starting actual training.

        Args:
            model: The model to run sanity test on.
        """
        ref_model = model
        if self.data_parallel:
            ref_model = model.module

        # give model convenience properties
        ref_model.trainer = self

        # set local properties on the model
        self.copy_trainer_model_properties(ref_model)

        # log hyper-parameters
        if self.logger is not None:
            # save exp to get started
            if hasattr(ref_model, "hparams"):
                self.logger.log_hyperparams(ref_model.hparams)

            self.logger.save()

        if self.use_ddp or self.use_ddp2:
            dist.barrier()

        # wait for all models to restore weights
        if self.on_tpu and XLA_AVAILABLE:
            # wait for all processes to catch up
            torch_xla.core.xla_model.rendezvous("pl.Trainer.run_pretrain_routine")

        # register auto-resubmit when on SLURM
        self.register_slurm_signal_handlers()

        # print model summary
        # TODO: remove self.testing condition because model.summarize() is wiping out the weights
        if self.proc_rank == 0 and self.weights_summary is not None and not self.testing:
            if self.weights_summary in ['full', 'top']:
                ref_model.summarize(mode=self.weights_summary)
            else:
                m = "weights_summary can be None, 'full' or 'top'"
                raise MisconfigurationException(m)

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # set up checkpoint callback
        self.configure_checkpoint_callback()

        # restore training and model before hpc call
        self.restore_weights(model)

        # when testing requested only run test and return
        if self.testing:
            # only load test dataloader for testing
            # self.reset_test_dataloader(ref_model)
            self.run_evaluation(test_mode=True)
            return

        # check if we should run validation during training
        self.disable_validation = not self.is_overriden('validation_step') and not self.fast_dev_run

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        ref_model.on_sanity_check_start()
        if not self.disable_validation and self.num_sanity_val_steps > 0:
            self.reset_val_dataloader(ref_model)
            # init progress bars for validation sanity check
            pbar = tqdm(desc='Validation sanity check',
                        total=self.num_sanity_val_steps * len(self.val_dataloaders),
                        leave=False, position=2 * self.process_position,
                        disable=not self.show_progress_bar, dynamic_ncols=True)
            self.main_progress_bar = pbar
            # dummy validation progress bar
            self.val_progress_bar = tqdm(disable=True)

            eval_results = self.evaluate(model,
                                         self.val_dataloaders,
                                         self.num_sanity_val_steps,
                                         False)
            _, _, _, callback_metrics, _ = self.process_output(eval_results)

            # close progress bars
            self.main_progress_bar.close()
            self.val_progress_bar.close()

            if self.enable_early_stop:
                self.early_stop_callback.check_metrics(callback_metrics)

        # init progress bar
        pbar = tqdm(leave=True, position=2 * self.process_position,
                    disable=not self.show_progress_bar, dynamic_ncols=True,
                    file=sys.stdout)
        self.main_progress_bar = pbar

        # clear cache before training
        if self.on_gpu:
            torch.cuda.empty_cache()

        # CORE TRAINING LOOP
        self.train()

    def test(self, model: Optional[LightningModule] = None):
        r"""

        Separates from fit to make sure you never run on your test set until you want to.

        Args:
            model (:class:`.LightningModule`): The model to test.

        Example::

            # Option 1
            # run test after fitting
            trainer = Trainer()
            model = LightningModule()

            trainer.fit()
            trainer.test()

            # Option 2
            # run test from a loaded model
            model = LightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')
            trainer = Trainer()
            trainer.test(model)
        """

        self.testing = True
        if model is not None:
            self.model = model
            self.fit(model)
        elif self.use_ddp or self.use_tpu:  # pragma: no cover
            # attempt to load weights from a spawn
            path = os.path.join(self.default_save_path, '__temp_weight_ddp_end.ckpt')
            test_model = self.model
            if os.path.exists(path):
                test_model = self.load_spawn_weights(self.model)

            self.fit(test_model)
        else:
            self.run_evaluation(test_mode=True)

        self.testing = False


class _PatchDataLoader(object):
    r'''
    Callable object for patching dataloaders passed into trainer.fit().
    Use this class to override model.*_dataloader() and be pickle-compatible.

    Args:
        dataloader: Dataloader object to return when called.
    '''
    def __init__(self, dataloader: Union[List[DataLoader], DataLoader]):
        self.dataloader = dataloader

    def __call__(self) -> Union[List[DataLoader], DataLoader]:
        return self.dataloader
