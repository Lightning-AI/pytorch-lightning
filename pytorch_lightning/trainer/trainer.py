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

import inspect
import os
import warnings
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as torch_distrib
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.core.step_result import EvalResult
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.profiler import BaseProfiler, PassThroughProfiler, SimpleProfiler
from pytorch_lightning.trainer.callback_config import TrainerCallbackConfigMixin
from pytorch_lightning.trainer.callback_hook import TrainerCallbackHookMixin
from pytorch_lightning.trainer.configuration_validator import ConfigValidator
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.deprecated_api import TrainerDeprecatedAPITillVer0_10
from pytorch_lightning.trainer.distrib_data_parallel import TrainerDDPMixin
from pytorch_lightning.utilities import device_parser
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.trainer.lr_finder import TrainerLRFinderMixin
from pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin
from pytorch_lightning.trainer.states import TrainerState, trainer_state
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.trainer.training_io import TrainerIOMixin
from pytorch_lightning.trainer.training_tricks import TrainerTrainingTricksMixin
from pytorch_lightning.utilities import parsing, rank_zero_info, rank_zero_only, rank_zero_warn, AMPType
from pytorch_lightning.utilities.debugging import InternalDebugger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.trainer.evaluation_loop import EvaluationLoop
from pytorch_lightning.trainer.training_loop import TrainLoop
from pytorch_lightning.trainer.data_connector import DataConnector
from pytorch_lightning.accelerators.accelerator_connector import AcceleratorConnector
from pytorch_lightning.utilities import argparse_utils
from pytorch_lightning.trainer.logger_connector import LoggerConnector
from pytorch_lightning.trainer.lr_scheduler_connector import LRSchedulerConnector
from pytorch_lightning.trainer.model_connector import ModelConnector
from pytorch_lightning import _logger as log
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.trainer.initializer import Initializer
from pytorch_lightning.utilities.model_utils import is_overridden
from pytorch_lightning.trainer import docstrings

# warnings to ignore in trainer
warnings.filterwarnings(
    'ignore', message='torch.distributed.reduce_op is deprecated, ' 'please use torch.distributed.ReduceOp instead'
)

try:
    from apex import amp
except ImportError:
    amp = None

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True

try:
    import horovod.torch as hvd
except (ModuleNotFoundError, ImportError):
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True


class Trainer(
    TrainerIOMixin,
    TrainerCallbackHookMixin,
    TrainerModelHooksMixin,
    TrainerOptimizersMixin,
    TrainerDDPMixin,
    TrainerLoggingMixin,
    TrainerTrainingTricksMixin,
    TrainerDataLoadingMixin,
    TrainerCallbackConfigMixin,
    TrainerLRFinderMixin,
    TrainerDeprecatedAPITillVer0_10,
):
    def __init__(
        self,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        checkpoint_callback: Union[ModelCheckpoint, bool] = True,
        early_stop_callback: Optional[Union[EarlyStopping, bool]] = False,
        callbacks: Optional[List[Callback]] = None,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: float = 0,
        process_position: int = 0,
        num_nodes: int = 1,
        num_processes: int = 1,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        log_gpu_memory: Optional[str] = None,
        progress_bar_refresh_rate: int = 1,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: bool = False,
        accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
        max_epochs: int = 1000,
        min_epochs: int = 1,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        log_save_interval: int = 100,
        row_log_interval: int = 50,
        distributed_backend: Optional[str] = None,
        sync_batchnorm: bool = False,
        precision: int = 32,
        weights_summary: Optional[str] = ModelSummary.MODE_DEFAULT,
        weights_save_path: Optional[str] = None,
        num_sanity_val_steps: int = 2,
        truncated_bptt_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None,
        profiler: Optional[Union[BaseProfiler, bool]] = None,
        benchmark: bool = False,
        deterministic: bool = False,
        reload_dataloaders_every_epoch: bool = False,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        terminate_on_nan: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        amp_backend: str = 'native',
        amp_level: str = 'O2',  # backward compatible, todo: remove in v1.0.0
        val_percent_check: float = None,  # backward compatible, todo: remove in v0.10.0
        test_percent_check: float = None,  # backward compatible, todo: remove in v0.10.0
        train_percent_check: float = None,  # backward compatible, todo: remove in v0.10.0
        overfit_pct: float = None,  # backward compatible, todo: remove in v1.0.0
    ):
        super().__init__()

        self.deterministic = deterministic
        torch.backends.cudnn.deterministic = self.deterministic
        if self.deterministic:
            # fixing non-deterministic part of horovod
            # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
            os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)

        # init the default rank if exists
        # we need to call this here or NVIDIA flags and other messaging in init will show on all ranks
        # this way we only show it on rank 0
        if 'LOCAL_RANK' in os.environ:
            rank_zero_only.rank = int(os.environ['LOCAL_RANK'])

        # tracks internal state for debugging
        self.dev_debugger = InternalDebugger(self)
        self.config_validator = ConfigValidator(self)
        self.data_connector = DataConnector(self)
        self.lr_scheduler_connector = LRSchedulerConnector(self)
        self.accelerator_connector = AcceleratorConnector(self)
        self.logger_connector = LoggerConnector(self)
        self.model_connector = ModelConnector(self)
        self.initializer = Initializer(self)
        self.tuner = Tuner(self)
        self.accelerator_backend = None

        # loops
        self.evaluation_loop = EvaluationLoop(self)
        self.train_loop = TrainLoop(self)

        # training bookeeping
        self.total_batch_idx = 0
        self.running_loss = TensorRunningAccum(window_length=20)
        self.batch_idx = 0
        self.num_training_batches = 0
        self.num_val_batches = []
        self.num_sanity_val_batches = []
        self.num_test_batches = []
        self.train_dataloader = None
        self.test_dataloaders = None
        self.val_dataloaders = None

        # when true, prints test results
        self.verbose_test = True

        # when .test() is called, it sets this
        self.tested_ckpt_path = None

        # training state
        self.model = None
        self.datamodule = None
        self.testing = False
        self.prepare_data_per_node = prepare_data_per_node
        self.lr_schedulers = []
        self.optimizers = None
        self.optimizer_frequencies = []
        self.global_step = 0
        self.current_epoch = 0
        self.interrupted = False
        self.should_stop = False
        self.running_sanity_check = False
        self._state = TrainerState.INITIALIZING

        self._default_root_dir = default_root_dir or os.getcwd()
        self._weights_save_path = weights_save_path or self._default_root_dir

        # init callbacks
        self.callbacks = callbacks or []

        # configure early stop callback
        # creates a default one if none passed in
        early_stop_callback = self.configure_early_stopping(early_stop_callback)
        if early_stop_callback:
            self.callbacks.append(early_stop_callback)

        # configure checkpoint callback
        # it is important that this is the last callback to run
        # pass through the required args to figure out defaults
        checkpoint_callback = self.configure_checkpoint_callback(checkpoint_callback)
        if checkpoint_callback:
            self.callbacks.append(checkpoint_callback)

        # TODO refactor codebase (tests) to not directly reach into these callbacks
        self.checkpoint_callback = checkpoint_callback
        self.early_stop_callback = early_stop_callback

        self.on_init_start()

        # benchmarking
        self.benchmark = benchmark
        torch.backends.cudnn.benchmark = self.benchmark

        # Transfer params
        self.num_nodes = num_nodes
        self.log_gpu_memory = log_gpu_memory

        # sync-bn backend
        self.sync_batchnorm = sync_batchnorm

        self.gradient_clip_val = gradient_clip_val
        self.check_val_every_n_epoch = check_val_every_n_epoch

        if not isinstance(track_grad_norm, (int, float)) and track_grad_norm != 'inf':
            raise MisconfigurationException("track_grad_norm can be an int, a float or 'inf' (infinity norm).")
        self.track_grad_norm = float(track_grad_norm)

        self.tpu_cores = device_parser.parse_tpu_cores(tpu_cores)
        self.on_tpu = self.tpu_cores is not None

        self.tpu_id = self.tpu_cores[0] if isinstance(self.tpu_cores, list) else None

        if num_processes != 1 and distributed_backend != "ddp_cpu":
            rank_zero_warn("num_processes is only used for distributed_backend=\"ddp_cpu\". Ignoring it.")
        self.num_processes = num_processes

        self.weights_summary = weights_summary

        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.max_steps = max_steps
        self.min_steps = min_steps

        if num_sanity_val_steps == -1:
            self.num_sanity_val_steps = float('inf')
        else:
            self.num_sanity_val_steps = num_sanity_val_steps

        self.reload_dataloaders_every_epoch = reload_dataloaders_every_epoch

        self.auto_lr_find = auto_lr_find
        self.auto_scale_batch_size = auto_scale_batch_size
        self._is_data_prepared = False
        self.replace_sampler_ddp = replace_sampler_ddp

        self.truncated_bptt_steps = truncated_bptt_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.terminate_on_nan = terminate_on_nan
        self.shown_warnings = set()

        self.fast_dev_run = fast_dev_run
        if self.fast_dev_run:
            limit_train_batches = 1
            limit_val_batches = 1
            limit_test_batches = 1
            self.num_sanity_val_steps = 0
            self.max_epochs = 1
            rank_zero_info(
                'Running in fast_dev_run mode: will run a full train,' ' val and test loop using a single batch'
            )

        # configure profiler
        if profiler is True:
            profiler = SimpleProfiler()
        self.profiler = profiler or PassThroughProfiler()

        # accumulated grads
        self.accumulate_grad_batches = accumulate_grad_batches
        self.configure_accumulated_gradients(accumulate_grad_batches)

        # override with environment flag
        gpus = os.environ.get('PL_TRAINER_GPUS', gpus)

        # for gpus allow int, string and gpu list
        if auto_select_gpus and isinstance(gpus, int):
            self.gpus = self.tuner.pick_multiple_gpus(gpus)
        else:
            self.gpus = gpus

        self.data_parallel_device_ids = device_parser.parse_gpu_ids(self.gpus)
        self.root_gpu = device_parser.determine_root_gpu_device(self.data_parallel_device_ids)
        self.root_device = torch.device("cpu")

        self.on_gpu = True if (self.data_parallel_device_ids and torch.cuda.is_available()) else False

        # tpu state flags
        self.use_tpu = False
        self.tpu_local_core_rank = None
        self.tpu_global_core_rank = None

        # distributed backend choice
        self.distributed_backend = distributed_backend
        self.set_distributed_mode(distributed_backend)

        # override dist backend when using tpus
        if self.on_tpu:
            self.distributed_backend = 'tpu'
            self.init_tpu()

        # init flags for SLURM+DDP to work
        self.world_size = 1
        self.interactive_ddp_procs = []
        self.configure_slurm_ddp(self.num_nodes)
        self.node_rank = self.determine_ddp_node_rank()
        self.local_rank = self.determine_local_rank()
        self.global_rank = 0

        # NVIDIA setup
        self.set_nvidia_flags(self.is_slurm_managing_tasks, self.data_parallel_device_ids)

        self._progress_bar_callback = self.configure_progress_bar(progress_bar_refresh_rate, process_position)

        # logging
        self.configure_logger(logger)
        self.log_save_interval = log_save_interval
        self.row_log_interval = row_log_interval

        # how much of the data to use
        # TODO: remove in 0.10.0
        if overfit_pct is not None:
            rank_zero_warn(
                "Argument `overfit_pct` is now set by `overfit_batches` since v0.8.0"
                " and this argument will be removed in v0.10.0",
                DeprecationWarning,
            )
            overfit_batches = overfit_pct

        # TODO: remove in 0.10.0
        if val_percent_check is not None:
            rank_zero_warn(
                "Argument `val_percent_check` is now set by `limit_val_batches` since v0.8.0"
                " and this argument will be removed in v0.10.0",
                DeprecationWarning,
            )
            limit_val_batches = val_percent_check

        # TODO: remove in 0.10.0
        if test_percent_check is not None:
            rank_zero_warn(
                "Argument `test_percent_check` is now set by `limit_test_batches` since v0.8.0"
                " and this argument will be removed in v0.10.0",
                DeprecationWarning,
            )
            limit_test_batches = test_percent_check

        # TODO: remove in 0.10.0
        if train_percent_check is not None:
            rank_zero_warn(
                "Argument `train_percent_check` is now set by `limit_train_batches` since v0.8.0"
                " and this argument will be removed in v0.10.0",
                DeprecationWarning,
            )
            limit_train_batches = train_percent_check

        self.limit_train_batches = _determine_batch_limits(limit_train_batches, 'limit_train_batches')
        self.limit_val_batches = _determine_batch_limits(limit_val_batches, 'limit_val_batches')
        self.limit_test_batches = _determine_batch_limits(limit_test_batches, 'limit_test_batches')
        self.val_check_interval = _determine_batch_limits(val_check_interval, 'val_check_interval')
        self.overfit_batches = _determine_batch_limits(overfit_batches, 'overfit_batches')
        self.determine_data_use_amount(self.overfit_batches)

        # AMP init
        # These are the only lines needed after v0.8.0
        # we wrap the user's forward with autocast and give it back at the end of fit
        self.autocast_original_forward = None
        self.precision = precision
        self.scaler = None

        self.amp_level = amp_level
        self.initializer.init_amp(amp_backend)

        self.on_colab_kaggle = os.getenv('COLAB_GPU') or os.getenv('KAGGLE_URL_BASE')

        # Callback system
        self.on_init_end()

    @property
    def use_amp(self) -> bool:
        return self.precision == 16

    @property
    def callback_metrics(self):
        return self.logger_connector.callback_metrics

    @callback_metrics.setter
    def callback_metrics(self, x):
        self.logger_connector.callback_metrics = x

    @property
    def state(self) -> TrainerState:
        return self._state

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def slurm_job_id(self) -> Optional[int]:
        try:
            job_id = os.environ['SLURM_JOB_ID']
            job_id = int(job_id)

            # in interactive mode, don't make logs use the same job id
            in_slurm_interactive_mode = os.environ['SLURM_JOB_NAME'] == 'bash'
            if in_slurm_interactive_mode:
                job_id = None

        except Exception:
            job_id = None
        return job_id

    @classmethod
    def default_attributes(cls):
        init_signature = inspect.signature(Trainer)

        args = {}
        for param_name in init_signature.parameters:
            value = init_signature.parameters[param_name].default
            args[param_name] = value

        return args

    @classmethod
    def get_deprecated_arg_names(cls) -> List:
        """Returns a list with deprecated Trainer arguments."""
        depr_arg_names = []
        for name, val in cls.__dict__.items():
            if name.startswith('DEPRECATED') and isinstance(val, (tuple, list)):
                depr_arg_names.extend(val)
        return depr_arg_names

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs) -> 'Trainer':
        return argparse_utils.from_argparse_args(cls, args, **kwargs)

    @classmethod
    def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        return argparse_utils.parse_argparser(cls, arg_parser)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        r"""Extends existing argparse by default `Trainer` attributes.

        Args:
            parent_parser:
                The custom cli arguments parser, which will be extended by
                the Trainer default arguments.

        Only arguments of the allowed types (str, float, int, bool) will
        extend the `parent_parser`.

        Examples:
            >>> import argparse
            >>> import pprint
            >>> parser = argparse.ArgumentParser()
            >>> parser = Trainer.add_argparse_args(parser)
            >>> args = parser.parse_args([])
            >>> pprint.pprint(vars(args))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            {...
             'check_val_every_n_epoch': 1,
             'checkpoint_callback': True,
             'default_root_dir': None,
             'deterministic': False,
             'distributed_backend': None,
             'early_stop_callback': False,
             ...
             'logger': True,
             'max_epochs': 1000,
             'max_steps': None,
             'min_epochs': 1,
             'min_steps': None,
             ...
             'profiler': None,
             'progress_bar_refresh_rate': 1,
             ...}

        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False,)

        blacklist = ['kwargs']
        depr_arg_names = cls.get_deprecated_arg_names() + blacklist

        allowed_types = (str, int, float, bool)

        # TODO: get "help" from docstring :)
        for arg, arg_types, arg_default in (
            at for at in cls.get_init_arguments_and_types() if at[0] not in depr_arg_names
        ):
            arg_types = [at for at in allowed_types if at in arg_types]
            if not arg_types:
                # skip argument with not supported type
                continue
            arg_kwargs = {}
            if bool in arg_types:
                arg_kwargs.update(nargs="?", const=True)
                # if the only arg type is bool
                if len(arg_types) == 1:
                    use_type = parsing.str_to_bool
                # if only two args (str, bool)
                elif len(arg_types) == 2 and set(arg_types) == {str, bool}:
                    use_type = parsing.str_to_bool_or_str
                else:
                    # filter out the bool as we need to use more general
                    use_type = [at for at in arg_types if at is not bool][0]
            else:
                use_type = arg_types[0]

            if arg == 'gpus' or arg == 'tpu_cores':
                use_type = Trainer._gpus_allowed_type
                arg_default = Trainer._gpus_arg_default

            # hack for types in (int, float)
            if len(arg_types) == 2 and int in set(arg_types) and float in set(arg_types):
                use_type = Trainer._int_or_float_type

            # hack for track_grad_norm
            if arg == 'track_grad_norm':
                use_type = float

            parser.add_argument(
                f'--{arg}',
                dest=arg,
                default=arg_default,
                type=use_type,
                help='autogenerated by pl.Trainer',
                **arg_kwargs,
            )

        return parser

    def _gpus_allowed_type(x) -> Union[int, str]:
        if ',' in x:
            return str(x)
        else:
            return int(x)

    def _gpus_arg_default(x) -> Union[int, str]:
        if ',' in x:
            return str(x)
        else:
            return int(x)

    def _int_or_float_type(x) -> Union[int, float]:
        if '.' in str(x):
            return float(x)
        else:
            return int(x)

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
    def progress_bar_callback(self):
        return self._progress_bar_callback

    @property
    def progress_bar_dict(self) -> dict:
        """ Read-only for progress bar metrics. """
        ref_model = self.model if not self.data_parallel else self.model.module
        return dict(**ref_model.get_progress_bar_dict(), **self.logger_connector.progress_bar_metrics)

    @property
    def disable_validation(self) -> bool:
        """ Check if validation is disabled during training. """
        return not self.enable_validation

    @property
    def enable_validation(self) -> bool:
        """ Check if we should run validation during training. """
        val_loop_enabled = is_overridden('validation_step', self.get_model()) and self.limit_val_batches > 0
        return val_loop_enabled or self.fast_dev_run

    @property
    def default_root_dir(self) -> str:
        """
        The default location to save artifacts of loggers, checkpoints etc.
        It is used as a fallback if logger or checkpoint callback do not define specific save paths.
        """
        if get_filesystem(self._default_root_dir).protocol == "file":
            return os.path.normpath(self._default_root_dir)
        return self._default_root_dir

    @property
    def weights_save_path(self) -> str:
        """
        The default root location to save weights (checkpoints), e.g., when the
        :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` does not define a file path.
        """
        if get_filesystem(self._weights_save_path).protocol == "file":
            return os.path.normpath(self._weights_save_path)
        return self._weights_save_path

    def tune(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        # TODO: temporary, need to decide if tune or separate object

        # setup data, etc...
        self.setup_fit(model, train_dataloader, val_dataloaders, datamodule)

        # hook
        self.call_hook('on_fit_start', model)

        # hook
        self.data_connector.prepare_data(model)

        # Run auto batch size scaling
        if self.auto_scale_batch_size:
            if isinstance(self.auto_scale_batch_size, bool):
                self.auto_scale_batch_size = 'power'
            self.tuner.scale_batch_size(
                model,
                mode=self.auto_scale_batch_size,
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
            )
            model.logger = self.logger  # reset logger binding

        # Run learning rate finder:
        if self.auto_lr_find:
            self._run_lr_finder_internally(model)
            model.logger = self.logger  # reset logger binding

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    @trainer_state(entering=TrainerState.RUNNING, exiting=TrainerState.FINISHED)
    def fit(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        results = None

        # setup data, etc...
        self.setup_fit(model, train_dataloader, val_dataloaders, datamodule)

        # hook
        self.call_hook('on_fit_start', model)

        # hook
        self.data_connector.prepare_data(model)

        # set testing if set in environ
        self.testing = os.environ.get('PL_TESTING_MODE', self.testing)

        # -------------------------
        # TRAIN
        # -------------------------
        self.accelerator_backend = self.accelerator_connector.select_accelerator()
        self.accelerator_backend.setup(model)
        results = self.accelerator_backend.train()
        self.accelerator_backend.teardown()

        # -------------------------
        # POST-Training
        # -------------------------
        # hook
        self.call_hook('on_fit_end')

        # hook
        self.teardown('fit')
        if self.is_function_implemented('teardown'):
            model.teardown('fit')

        # return 1 when finished
        # used for testing or when we need to know that training succeeded
        return results or 1

    def setup_fit(self, model, train_dataloader, val_dataloaders, datamodule):
        # bind logger and other properties
        self.model_connector.copy_trainer_model_properties(model)

        # clean hparams
        if hasattr(model, 'hparams'):
            parsing.clean_namespace(model.hparams)

        # links data to the trainer
        self.data_connector.attach_data(model, train_dataloader, val_dataloaders, datamodule)

        # check that model is configured correctly
        self.config_validator.verify_loop_configurations(model)

    def setup_training(self, model: LightningModule):
        """Sanity check a few things before starting actual training.

        Args:
            model: The model to run sanity test on.
        """
        # --------------------------
        # Setup??
        # --------------------------
        ref_model = model
        if self.data_parallel:
            ref_model = model.module

        # give model convenience properties
        ref_model.trainer = self

        # set local properties on the model
        self.model_connector.copy_trainer_model_properties(ref_model)

        # init amp. Must be done here instead of __init__ to allow ddp to work
        if self.amp_backend == AMPType.NATIVE and self.precision == 16 and not self.use_tpu:
            self.scaler = torch.cuda.amp.GradScaler()

        # log hyper-parameters
        if self.logger is not None:
            # save exp to get started
            self.logger.log_hyperparams(ref_model.hparams)
            self.logger.log_graph(ref_model)
            self.logger.save()

        if self.use_ddp or self.use_ddp2:
            torch_distrib.barrier()

        # wait for all models to restore weights
        if self.on_tpu and XLA_AVAILABLE:
            # wait for all processes to catch up
            torch_xla.core.xla_model.rendezvous("pl.Trainer.setup_training")

        elif self.use_horovod:
            # wait for all processes to catch up
            hvd.join()

        # register auto-resubmit when on SLURM
        self.register_slurm_signal_handlers()

        # --------------------------
        # Pre-train
        # --------------------------
        # on pretrain routine start
        self.on_pretrain_routine_start(ref_model)
        if self.is_function_implemented('on_pretrain_routine_start'):
            ref_model.on_pretrain_routine_start()

        # print model summary
        if self.is_global_zero and self.weights_summary is not None and not self.testing:
            if self.weights_summary in ModelSummary.MODES:
                ref_model.summarize(mode=self.weights_summary)
            else:
                raise MisconfigurationException("weights_summary can be None, " + ", ".join(ModelSummary.MODES))

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # restore training and model before hpc is called
        self.restore_weights(model)

        # on pretrain routine end
        self.on_pretrain_routine_end(ref_model)
        if self.is_function_implemented('on_pretrain_routine_end'):
            ref_model.on_pretrain_routine_end()

    def train(self):
        self.run_sanity_check(self.get_model())

        # enable train mode
        model = self.get_model()
        model.train()
        torch.set_grad_enabled(True)

        # reload data when needed
        self.train_loop.reset_train_val_dataloaders(model)

        # hook
        self.train_loop.on_train_start()

        try:
            # run all epochs
            for epoch in range(self.current_epoch, self.max_epochs):

                # reset train dataloader
                if self.reload_dataloaders_every_epoch:
                    self.reset_train_dataloader(model)

                # hook
                self.train_loop.on_train_epoch_start(epoch)

                # run train epoch
                self.train_loop.run_training_epoch()

                if self.max_steps and self.max_steps <= self.global_step:

                    # hook
                    self.train_loop.on_train_end()
                    return

                # update LR schedulers
                self.lr_scheduler_connector.update_learning_rates(interval='epoch')

                # early stopping
                met_min_epochs = epoch >= self.min_epochs - 1
                met_min_steps = self.global_step >= self.min_steps if self.min_steps else True

                if self.should_stop:
                    if (met_min_epochs and met_min_steps):
                        self.train_loop.on_train_end()
                        return
                    else:
                        log.info('Trainer was signaled to stop but required minimum epochs'
                                 f' ({self.min_epochs}) or minimum steps ({self.min_steps}) has'
                                 ' not been met. Training will continue...')

            # hook
            self.train_loop.on_train_end()

        except KeyboardInterrupt:
            rank_zero_warn('Detected KeyboardInterrupt, attempting graceful shutdown...')

            # user could press ctrl+c many times... only shutdown once
            if not self.interrupted:
                self.interrupted = True
                self._state = TrainerState.INTERRUPTED
                self.on_keyboard_interrupt()

                # hook
                self.train_loop.on_train_end()

    def run_evaluation(self, test_mode: bool = False, max_batches=None):
        # bookkeeping
        self.evaluation_loop.testing = test_mode
        dataloaders, max_batches = self.evaluation_loop.get_evaluation_dataloaders(max_batches)
        if self.evaluation_loop.should_skip_evaluation(dataloaders, max_batches):
            return [], []

        # enable eval mode + no grads
        model = self.get_model()
        model.zero_grad()
        model.eval()
        torch.set_grad_enabled(False)

        # hook
        self.evaluation_loop.on_evaluation_start()

        # set up the eval loop
        self.evaluation_loop.setup(model, max_batches, dataloaders)

        # hook
        # TODO: should this be insider the dataloader loop?
        self.evaluation_loop.on_evaluation_epoch_start()

        # run validation/testing
        for dataloader_idx, dataloader in enumerate(dataloaders):
            # bookkeeping
            dl_outputs = []
            dataloader = self.accelerator_backend.process_dataloader(dataloader)
            dl_max_batches = self.evaluation_loop.max_batches[dataloader_idx]

            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    continue

                # stop short when running on limited batches
                if batch_idx >= dl_max_batches:
                    break

                # hook
                self.evaluation_loop.on_evaluation_batch_start(batch, batch_idx, dataloader_idx)

                # lightning module methods
                output = self.evaluation_loop.evaluation_step(test_mode, batch, batch_idx, dataloader_idx)
                output = self.evaluation_loop.evaluation_step_end(output)

                # hook
                self.evaluation_loop.on_evaluation_batch_end(batch, batch_idx, dataloader_idx)

                # clean up
                self.evaluation_loop.evaluation_batch_end_cleanup(output, batch_idx, dataloader_idx)
                self.evaluation_loop.log_step_metrics(output, batch_idx)

                # track epoch level metrics
                if output is not None:
                    dl_outputs.append(output)

            self.evaluation_loop.outputs.append(dl_outputs)

        # lightning module method
        eval_results = self.evaluation_loop.evaluation_epoch_end(num_dataloaders=len(dataloaders))

        # bookkeeping
        eval_loop_results = self.evaluation_loop.log_epoch_metrics(eval_results, test_mode)
        self.evaluation_loop.predictions.to_disk()

        # hook
        self.evaluation_loop.on_evaluation_epoch_end()

        # enable train mode again
        model.train()
        torch.set_grad_enabled(True)

        # hook
        self.evaluation_loop.on_evaluation_end()

        return eval_loop_results, eval_results

    def run_test(self):
        # only load test dataloader for testing
        # self.reset_test_dataloader(ref_model)
        eval_loop_results, _ = self.run_evaluation(test_mode=True)

        if len(eval_loop_results) == 0:
            return 1

        # remove the tensors from the eval results
        for i, result in enumerate(eval_loop_results):
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v.cpu().item()

        return eval_loop_results

    def train_or_test(self):
        if self.testing:
            results = self.run_test()
        else:
            results = self.train()
        return results

    def run_sanity_check(self, ref_model):
        using_val_step = ref_model.val_dataloader is not None and is_overridden('validation_step', ref_model)
        should_sanity_check = using_val_step and self.num_sanity_val_steps > 0 and self.limit_val_batches > 0

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        if should_sanity_check:
            self.reset_val_dataloader(ref_model)
            self.num_sanity_val_batches = [
                min(self.num_sanity_val_steps, val_batches) for val_batches in self.num_val_batches
            ]

            # hook and callback
            self.running_sanity_check = True
            self.on_sanity_check_start()

            # run eval step
            _, eval_results = self.run_evaluation(test_mode=False, max_batches=self.num_sanity_val_batches)

            # allow no returns from eval
            if eval_results is not None and len(eval_results) > 0:
                # when we get a list back, used only the last item
                if isinstance(eval_results, list):
                    eval_results = eval_results[-1]

                if isinstance(eval_results, EvalResult):
                    callback_metrics = eval_results.callback_metrics
                else:
                    _, _, _, callback_metrics, _ = self.process_output(eval_results)
                self.logger_connector.callback_metrics = callback_metrics

            self.on_sanity_check_end()
            self.running_sanity_check = False

    @trainer_state(entering=TrainerState.RUNNING, exiting=TrainerState.FINISHED)
    def test(
        self,
        model: Optional[LightningModule] = None,
        test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        ckpt_path: Optional[str] = 'best',
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ):
        # --------------------
        # SETUP HOOK
        # --------------------
        self.verbose_test = verbose

        if self.global_rank != 0:
            return

        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if test_dataloaders and datamodule:
            raise MisconfigurationException(
                'You cannot pass test_dataloaders to trainer.test if you supply a datamodule'
            )

        # Attach datamodule to get setup/prepare_data added to model before the call to it below
        self.data_connector.attach_datamodule(model or self.get_model(), datamodule, 'test')

        if model is not None:
            results = self.__test_given_model(model, test_dataloaders)
        else:
            results = self.__test_using_best_weights(ckpt_path, test_dataloaders)

        self.teardown('test')

        return results

    def __test_using_best_weights(self, ckpt_path, test_dataloaders):
        model = self.get_model()

        # if user requests the best checkpoint but we don't have it, error
        if ckpt_path == 'best' and self.checkpoint_callback.save_top_k <= 0:
            raise MisconfigurationException(
                'ckpt_path is "best", but ModelCheckpoint is not configured to save the best model.'
            )

        # load best weights
        if ckpt_path is not None:
            # ckpt_path is 'best' so load the best model
            if ckpt_path == 'best':
                ckpt_path = self.checkpoint_callback.best_model_path

            if len(ckpt_path) == 0:
                rank_zero_warn(
                    f'.test() found no path for the best weights, {ckpt_path}. Please '
                    f'specify a path for a checkpoint .test(ckpt_path=PATH)'
                )
                return {}

            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt['state_dict'])

        # attach dataloaders
        if test_dataloaders is not None:
            self.data_connector.attach_dataloaders(model, test_dataloaders=test_dataloaders)

        # run tests
        self.tested_ckpt_path = ckpt_path
        self.testing = True
        os.environ['PL_TESTING_MODE'] = '1'
        self.model = model
        results = self.fit(model)
        self.testing = False
        del os.environ['PL_TESTING_MODE']

        # teardown
        if self.is_function_implemented('teardown'):
            model_ref = self.get_model()
            model_ref.teardown('test')

        return results

    def __test_given_model(self, model, test_dataloaders):

        # attach data
        if test_dataloaders is not None:
            self.data_connector.attach_dataloaders(model, test_dataloaders=test_dataloaders)

        # run test
        # sets up testing so we short circuit to eval
        self.testing = True
        self.model = model
        results = self.fit(model)
        self.testing = False

        # teardown
        if self.is_function_implemented('teardown'):
            model.teardown('test')

        return results

    def call_setup_hook(self, model):
        # call setup after the ddp process has connected
        stage_name = 'test' if self.testing else 'fit'
        if self.datamodule is not None:
            called = self.datamodule.has_setup_test if self.testing else self.datamodule.has_setup_fit
            if not called:
                self.datamodule.setup(stage_name)
        self.setup(stage_name)
        model.setup(stage_name)

    def call_hook(self, hook_name, *args, **kwargs):
        # always profile hooks
        with self.profiler.profile(hook_name):

            # first call trainer hook
            if hasattr(self, hook_name):
                trainer_hook = getattr(self, hook_name)
                trainer_hook(*args, **kwargs)

            # next call hook in lightningModule
            output = None
            model_ref = self.get_model()
            if is_overridden(hook_name, model_ref):
                hook_fx = getattr(model_ref, hook_name)
                output = hook_fx(*args, **kwargs)

            # if the PL module doesn't have the hook then call the accelator
            # used to auto-reduce things for the user with Results obj
            elif hasattr(self.accelerator_backend, hook_name):
                accelerator_hook = getattr(self.accelerator_backend, hook_name)
                output = accelerator_hook(*args, **kwargs)

            return output


def _determine_batch_limits(batches: Union[int, float], name: str) -> Union[int, float]:
    if 0 <= batches <= 1:
        return batches
    elif batches > 1 and batches % 1.0 == 0:
        return int(batches)
    else:
        raise MisconfigurationException(
            f'You have passed invalid value {batches} for {name}, it has to be in [0.0, 1.0] or an int.'
        )


# add docstrings
Trainer.__init__.__doc__ = docstrings.trainer.init
Trainer.fit.__doc__ = docstrings.trainer.fit
Trainer.test.__doc__ = docstrings.trainer.test
