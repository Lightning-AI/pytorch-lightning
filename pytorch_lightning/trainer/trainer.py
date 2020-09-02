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
from pytorch_lightning.trainer.auto_mix_precision import TrainerAMPMixin
from pytorch_lightning.trainer.callback_config import TrainerCallbackConfigMixin
from pytorch_lightning.trainer.callback_hook import TrainerCallbackHookMixin
from pytorch_lightning.trainer.configuration_validator import ConfigValidator
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.deprecated_api import TrainerDeprecatedAPITillVer0_10
from pytorch_lightning.trainer.distrib_data_parallel import TrainerDDPMixin
from pytorch_lightning.trainer.distrib_parts import (TrainerDPMixin, _parse_gpu_ids, _parse_tpu_cores,
                                                     determine_root_gpu_device, pick_multiple_gpus)
from pytorch_lightning.trainer.evaluation_loop import TrainerEvaluationLoopMixin
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.trainer.lr_finder import TrainerLRFinderMixin
from pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin
from pytorch_lightning.trainer.states import TrainerState, trainer_state
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.trainer.training_io import TrainerIOMixin
from pytorch_lightning.trainer.training_loop import TrainerTrainLoopMixin
from pytorch_lightning.trainer.training_tricks import TrainerTrainingTricksMixin
from pytorch_lightning.utilities import parsing, rank_zero_info, rank_zero_only, rank_zero_warn, AMPType
from pytorch_lightning.utilities.debugging import InternalDebugger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.cloud_io import is_remote_path
from pytorch_lightning.trainer.evaluate_loop import EvaluationLoop
from pytorch_lightning.trainer.data_connector import DataConnector
from pytorch_lightning.accelerators.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.training_loop_temp import TrainLoop

from pytorch_lightning.utilities.model_utils import is_overridden

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
    TrainerAMPMixin,
    TrainerDPMixin,
    TrainerDDPMixin,
    TrainerLoggingMixin,
    TrainerTrainingTricksMixin,
    TrainerDataLoadingMixin,
    TrainerEvaluationLoopMixin,
    TrainerTrainLoopMixin,
    TrainerCallbackConfigMixin,
    TrainerLRFinderMixin,
    TrainerDeprecatedAPITillVer0_10,
):
    """
    Example:

        >>> import torch
        >>> from torch.nn import functional as F
        >>> from torch.utils.data import Dataset, DataLoader

        >>> # Define model
        >>> class SimpleModel(LightningModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.l1 = torch.nn.Linear(in_features=64, out_features=4)
        ...
        ...     def forward(self, x):
        ...         return torch.relu(self.l1(x.view(x.size(0), -1)))
        ...
        ...     def training_step(self, batch, batch_nb):
        ...         x, y = batch
        ...         loss = F.cross_entropy(self(x), y)
        ...         return {'loss': loss, 'log': {'train_loss': loss}}
        ...
        ...     def test_step(self, batch, batch_nb):
        ...         x, y = batch
        ...         loss = F.cross_entropy(self(x), y)
        ...         return {'loss': loss, 'log': {'test_loss': loss}}
        ...
        ...     def configure_optimizers(self):
        ...         return torch.optim.Adam(self.parameters(), lr=0.02)
        ...
        >>> # Define dataset
        >>> class SimpleDataset(Dataset):
        ...     def __init__(self, num_samples=200):
        ...         self.input_seq = torch.randn(num_samples, 64)
        ...         self.output_seq = torch.randint(0, 4, (num_samples,))
        ...
        ...     def __len__(self):
        ...         return len(self.input_seq)
        ...
        ...     def __getitem__(self, item):
        ...         return self.input_seq[item], self.output_seq[item]
        ...
        >>> train_loader = DataLoader(SimpleDataset(), batch_size=8)
        >>> model = SimpleModel()
        >>> # Define Trainer and fit model
        >>> trainer = Trainer(max_epochs=1, progress_bar_refresh_rate=0)
        >>> trainer.fit(model, train_loader)
        1
        >>> test_outputs = trainer.test(model, train_loader, verbose=False)
        >>> len(test_outputs)
        25
    """

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
        r"""

        Customize every aspect of training via flags

        Args:
            logger: Logger (or iterable collection of loggers) for experiment tracking.

            checkpoint_callback: Callback for checkpointing.

            early_stop_callback (:class:`pytorch_lightning.callbacks.EarlyStopping`):

            callbacks: Add a list of callbacks.

            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Default: ``os.getcwd()``.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

            gradient_clip_val: 0 means don't clip.

            process_position: orders the progress bar when running multiple models on same machine.

            num_nodes: number of GPU nodes for distributed training.

            gpus: number of gpus to train on (int) or which GPUs to train on (list or str) applied per node

            auto_select_gpus:

                If enabled and `gpus` is an integer, pick available
                gpus automatically. This is especially useful when
                GPUs are configured to be in "exclusive mode", such
                that only one process at a time can access them.

            tpu_cores: How many TPU cores to train on (1 or 8) / Single TPU to train on [1]

            log_gpu_memory: None, 'min_max', 'all'. Might slow performance

            progress_bar_refresh_rate: How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
                Ignored when a custom callback is passed to :paramref:`~Trainer.callbacks`.

            overfit_batches: Overfit a percent of training data (float) or a set number of batches (int). Default: 0.0

            overfit_pct:
                .. warning:: .. deprecated:: 0.8.0

                    Use `overfit_batches` instead. Will be removed in 0.10.0.

            track_grad_norm: -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm.

            check_val_every_n_epoch: Check val every n train epochs.

            fast_dev_run: runs 1 batch of train, test and val to find any bugs (ie: a sort of unit test).

            accumulate_grad_batches: Accumulates grads every k batches or as set up in the dict.

            max_epochs: Stop training once this number of epochs is reached.

            min_epochs: Force training for at least these many epochs

            max_steps: Stop training after this number of steps. Disabled by default (None).

            min_steps: Force training for at least these number of steps. Disabled by default (None).

            limit_train_batches: How much of training dataset to check (floats = percent, int = num_batches)

            limit_val_batches: How much of validation dataset to check (floats = percent, int = num_batches)

            limit_test_batches: How much of test dataset to check (floats = percent, int = num_batches)

            train_percent_check:
                .. warning:: .. deprecated:: 0.8.0

                    Use `limit_train_batches` instead. Will remove v0.10.0.

            val_percent_check:
                .. warning:: .. deprecated:: 0.8.0

                    Use `limit_val_batches` instead. Will remove v0.10.0.

            test_percent_check:
                .. warning:: .. deprecated:: 0.8.0

                    Use `limit_test_batches` instead. Will remove v0.10.0.

            val_check_interval: How often to check the validation set. Use float to check within a training epoch,
                use int to check every n steps (batches).

            log_save_interval: Writes logs to disk this often

            row_log_interval: How often to add logging rows (does not write to disk)

            distributed_backend: The distributed backend to use (dp, ddp, ddp2, ddp_spawn, ddp_cpu)

            sync_batchnorm: Synchronize batch norm layers between process groups/whole world.

            precision: Full precision (32), half precision (16). Can be used on CPU, GPU or TPUs.

            weights_summary: Prints a summary of the weights when training begins.

            weights_save_path: Where to save weights if specified. Will override default_root_dir
                    for checkpoints only. Use this if for whatever reason you need the checkpoints
                    stored in a different place than the logs written in `default_root_dir`.
                    Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
                    Defaults to `default_root_dir`.

            amp_backend: The mixed precision backend to use ("native" or "apex")

            amp_level: The optimization level to use (O1, O2, etc...).

            num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
                Set it to `-1` to run all batches in all validation dataloaders. Default: 2

            truncated_bptt_steps: Truncated back prop breaks performs backprop every k steps of much longer
                sequence.

            resume_from_checkpoint: To resume training from a specific checkpoint pass in the path here.
                This can be a URL.

            profiler:  To profile individual steps during training and assist in identifying bottlenecks.

            reload_dataloaders_every_epoch: Set to True to reload dataloaders every epoch.

            auto_lr_find: If set to True, will `initially` run a learning rate finder,
                trying to optimize initial learning for faster convergence. Sets learning
                rate in self.lr or self.learning_rate in the LightningModule.
                To use a different key, set a string instead of True with the key name.

            replace_sampler_ddp: Explicitly enables or disables sampler replacement. If not specified this
                will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for
                train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it,
                you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.

            benchmark: If true enables cudnn.benchmark.

            deterministic: If true enables cudnn.deterministic.

            terminate_on_nan: If set to True, will terminate training (by raising a `ValueError`) at the
                end of each training batch, if any of the parameters or the loss are NaN or +/-inf.

            auto_scale_batch_size: If set to True, will `initially` run a batch size
                finder trying to find the largest batch size that fits into memory.
                The result will be stored in self.batch_size in the LightningModule.
                Additionally, can be set to either `power` that estimates the batch size through
                a power search or `binsearch` that estimates the batch size through a binary search.

            prepare_data_per_node: If True, each LOCAL_RANK=0 will call prepare data.
                Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data
        """
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

        # training bookeeping
        self.total_batch_idx = 0
        self.running_loss = TensorRunningAccum(window_length=20)
        self.batch_idx = 0
        self.progress_bar_metrics = {}
        self.callback_metrics = {}
        self.logged_metrics = {}
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

        self.tpu_cores = _parse_tpu_cores(tpu_cores)
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
            self.gpus = pick_multiple_gpus(gpus)
        else:
            self.gpus = gpus

        self.data_parallel_device_ids = _parse_gpu_ids(self.gpus)
        self.root_gpu = determine_root_gpu_device(self.data_parallel_device_ids)
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
        self.init_amp(amp_backend)

        self.on_colab_kaggle = os.getenv('COLAB_GPU') or os.getenv('KAGGLE_URL_BASE')

        # tracks internal state for debugging
        self.dev_debugger = InternalDebugger(self)
        self.config_validator = ConfigValidator(self)
        self.data_connector = DataConnector(self)
        self.accelerator_connector = AcceleratorConnector(self)
        self.accelerator_backend = None

        # loops
        self.evaluation_loop = EvaluationLoop(self)
        self.train_loop = TrainLoop(self)

        # Callback system
        self.on_init_end()

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
    def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
        r"""Scans the Trainer signature and returns argument names, types and default values.

        Returns:
            List with tuples of 3 values:
            (argument name, set with argument types, argument default value).

        Examples:
            >>> args = Trainer.get_init_arguments_and_types()
            >>> import pprint
            >>> pprint.pprint(sorted(args))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            [('accumulate_grad_batches',
              (<class 'int'>, typing.Dict[int, int], typing.List[list]),
              1),
             ...
             ('callbacks',
              (typing.List[pytorch_lightning.callbacks.base.Callback],
               <class 'NoneType'>),
               None),
             ('check_val_every_n_epoch', (<class 'int'>,), 1),
             ...
             ('max_epochs', (<class 'int'>,), 1000),
             ...
             ('precision', (<class 'int'>,), 32),
             ('prepare_data_per_node', (<class 'bool'>,), True),
             ('process_position', (<class 'int'>,), 0),
             ('profiler',
              (<class 'pytorch_lightning.profiler.profilers.BaseProfiler'>,
               <class 'bool'>,
               <class 'NoneType'>),
              None),
             ...
        """
        trainer_default_params = inspect.signature(cls).parameters
        name_type_default = []
        for arg in trainer_default_params:
            arg_type = trainer_default_params[arg].annotation
            arg_default = trainer_default_params[arg].default
            try:
                arg_types = tuple(arg_type.__args__)
            except AttributeError:
                arg_types = (arg_type,)

            name_type_default.append((arg, arg_types, arg_default))

        return name_type_default

    @classmethod
    def get_deprecated_arg_names(cls) -> List:
        """Returns a list with deprecated Trainer arguments."""
        depr_arg_names = []
        for name, val in cls.__dict__.items():
            if name.startswith('DEPRECATED') and isinstance(val, (tuple, list)):
                depr_arg_names.extend(val)
        return depr_arg_names

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

    @classmethod
    def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        """Parse CLI arguments, required for custom bool types."""
        args = arg_parser.parse_args() if isinstance(arg_parser, ArgumentParser) else arg_parser

        types_default = {
            arg: (arg_types, arg_default) for arg, arg_types, arg_default in cls.get_init_arguments_and_types()
        }

        modified_args = {}
        for k, v in vars(args).items():
            if k in types_default and v is None:
                # We need to figure out if the None is due to using nargs="?" or if it comes from the default value
                arg_types, arg_default = types_default[k]
                if bool in arg_types and isinstance(arg_default, bool):
                    # Value has been passed as a flag => It is currently None, so we need to set it to True
                    # We always set to True, regardless of the default value.
                    # Users must pass False directly, but when passing nothing True is assumed.
                    # i.e. the only way to disable somthing that defaults to True is to use the long form:
                    # "--a_default_true_arg False" becomes False, while "--a_default_false_arg" becomes None,
                    # which then becomes True here.

                    v = True

            modified_args[k] = v
        return Namespace(**modified_args)

    @classmethod
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs) -> 'Trainer':
        """
        Create an instance from CLI arguments.

        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
                parsed and passed to the :class:`Trainer`.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
                These must be valid Trainer arguments.

        Example:
            >>> parser = ArgumentParser(add_help=False)
            >>> parser = Trainer.add_argparse_args(parser)
            >>> parser.add_argument('--my_custom_arg', default='something')  # doctest: +SKIP
            >>> args = Trainer.parse_argparser(parser.parse_args(""))
            >>> trainer = Trainer.from_argparse_args(args, logger=False)
        """
        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid Trainer args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        trainer_kwargs = dict((name, params[name]) for name in valid_kwargs if name in params)
        trainer_kwargs.update(**kwargs)

        return cls(**trainer_kwargs)

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
        return dict(**ref_model.get_progress_bar_dict(), **self.progress_bar_metrics)

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
        if is_remote_path(self._default_root_dir):
            # it is a remote uri, use as is
            return self._default_root_dir
        return os.path.normpath(self._default_root_dir)

    @property
    def weights_save_path(self) -> str:
        """
        The default root location to save weights (checkpoints), e.g., when the
        :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` does not define a file path.
        """
        if is_remote_path(self._weights_save_path):
            # it is a remote uri, use as is
            return self._weights_save_path
        return os.path.normpath(self._weights_save_path)

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
            self.scale_batch_size(model, mode=self.auto_scale_batch_size)
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

        Example::

            # Option 1,
            # Define the train_dataloader() and val_dataloader() fxs
            # in the lightningModule
            # RECOMMENDED FOR MOST RESEARCH AND APPLICATIONS TO MAINTAIN READABILITY
            trainer = Trainer()
            model = LightningModule()
            trainer.fit(model)

            # Option 2
            # in production cases we might want to pass different datasets to the same model
            # Recommended for PRODUCTION SYSTEMS
            train, val = DataLoader(...), DataLoader(...)
            trainer = Trainer()
            model = LightningModule()
            trainer.fit(model, train_dataloader=train, val_dataloaders=val)

            # Option 1 & 2 can be mixed, for example the training set can be
            # defined as part of the model, and validation can then be feed to .fit()

        """
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
        self.copy_trainer_model_properties(model)

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
        self.copy_trainer_model_properties(ref_model)

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
                self.callback_metrics = callback_metrics

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
        r"""

        Separates from fit to make sure you never run on your test set until you want to.

        Args:
            model: The model to test.

            test_dataloaders: Either a single
                Pytorch Dataloader or a list of them, specifying validation samples.

            ckpt_path: Either ``best`` or path to the checkpoint you wish to test.
                If ``None``, use the weights from the last epoch to test. Default to ``best``.

            verbose: If True, prints the test results

        Returns:
            The final test result dictionary. If no test_epoch_end is defined returns a list of dictionaries

        Example::

            # Option 1
            # run test with the best checkpoint from ``ModelCheckpoint`` after fitting.
            test = DataLoader(...)
            trainer = Trainer()
            model = LightningModule()

            trainer.fit(model)
            trainer.test(test_dataloaders=test)

            # Option 2
            # run test with the specified checkpoint after fitting
            test = DataLoader(...)
            trainer = Trainer()
            model = LightningModule()

            trainer.fit(model)
            trainer.test(test_dataloaders=test, ckpt_path='path/to/checkpoint.ckpt')

            # Option 3
            # run test with the weights from the end of training after fitting
            test = DataLoader(...)
            trainer = Trainer()
            model = LightningModule()

            trainer.fit(model)
            trainer.test(test_dataloaders=test, ckpt_path=None)

            # Option 4
            # run test from a loaded model. ``ckpt_path`` is ignored in this case.
            test = DataLoader(...)
            model = LightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')
            trainer = Trainer()
            trainer.test(model, test_dataloaders=test)
        """
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

    def barrier(self, name):
        if self.use_ddp or self.use_ddp2:
            pass
            # torch_distrib.barrier()

        if self.on_tpu and XLA_AVAILABLE:
            # wait for all processes to catch up
            torch_xla.core.xla_model.rendezvous(f'pl.Trainer.{name}')

    def call_setup_hook(self, model):
        # call setup after the ddp process has connected
        stage_name = 'test' if self.testing else 'fit'
        if self.datamodule is not None:
            called = self.datamodule.has_setup_test if self.testing else self.datamodule.has_setup_fit
            if not called:
                self.datamodule.setup(stage_name)
        self.setup(stage_name)
        model.setup(stage_name)

    def init_amp(self, amp_type: str):
        assert self.precision in (16, 32), 'only 32 or 16 bit precision supported'
        self.amp_backend = None
        self._setup_amp_backend(amp_type)

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
