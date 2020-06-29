import inspect
import os
from argparse import ArgumentParser, Namespace
from typing import Union, Optional, List, Dict, Tuple, Iterable, Any

import torch
import torch.distributed as torch_distrib
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.profiler import SimpleProfiler, PassThroughProfiler, BaseProfiler
from pytorch_lightning.trainer.auto_mix_precision import TrainerAMPMixin, NATIVE_AMP_AVALAIBLE
from pytorch_lightning.trainer.callback_config import TrainerCallbackConfigMixin
from pytorch_lightning.trainer.callback_hook import TrainerCallbackHookMixin
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.deprecated_api import (
    TrainerDeprecatedAPITillVer0_9, TrainerDeprecatedAPITillVer0_10)
from pytorch_lightning.trainer.distrib_data_parallel import TrainerDDPMixin
from pytorch_lightning.trainer.distrib_parts import (
    TrainerDPMixin, _parse_gpu_ids, determine_root_gpu_device, pick_multiple_gpus, _parse_tpu_cores)
from pytorch_lightning.trainer.evaluation_loop import TrainerEvaluationLoopMixin
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.trainer.training_io import TrainerIOMixin
from pytorch_lightning.trainer.training_loop import TrainerTrainLoopMixin
from pytorch_lightning.trainer.training_tricks import TrainerTrainingTricksMixin
from pytorch_lightning.trainer.lr_finder import TrainerLRFinderMixin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import rank_zero_warn, parsing, rank_zero_info, rank_zero_only

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
    TrainerDeprecatedAPITillVer0_9,
    TrainerDeprecatedAPITillVer0_10,
):
    DEPRECATED_IN_0_9 = ('use_amp', 'show_progress_bar', 'training_tqdm_dict', 'num_tpu_cores')

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
        precision: int = 32,
        print_nan_grads: bool = False,  # backward compatible, todo: remove in v0.9.0
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
        amp_level: str = 'O2',  # backward compatible, todo: remove in v1.0.0
        num_tpu_cores: Optional[int] = None,  # backward compatible, todo: remove in v0.9.0
        use_amp=None,  # backward compatible, todo: remove in v0.9.0
        show_progress_bar=None,  # backward compatible, todo: remove in v0.9.0
        val_percent_check: float = None,  # backward compatible, todo: remove in v0.10.0
        test_percent_check: float = None,  # backward compatible, todo: remove in v0.10.0
        train_percent_check: float = None,  # backward compatible, todo: remove in v0.10.0
        overfit_pct: float = None  # backward compatible, todo: remove in v1.0.0
    ):
        r"""

        Customize every aspect of training via flags

        Args:
            logger: Logger (or iterable collection of loggers) for experiment tracking.

            checkpoint_callback: Callback for checkpointing.

            early_stop_callback (:class:`pytorch_lightning.callbacks.EarlyStopping`):

            callbacks: Add a list of callbacks.

            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed

            gradient_clip_val: 0 means don't clip.

            gradient_clip:
                .. warning:: .. deprecated:: 0.7.0

                    Use `gradient_clip_val` instead. Will remove 0.9.0.

            process_position: orders the progress bar when running multiple models on same machine.

            num_nodes: number of GPU nodes for distributed training.

            nb_gpu_nodes:
                .. warning:: .. deprecated:: 0.7.0

                    Use `num_nodes` instead. Will remove 0.9.0.

            gpus: Which GPUs to train on.

            auto_select_gpus:

                If enabled and `gpus` is an integer, pick available
                gpus automatically. This is especially useful when
                GPUs are configured to be in "exclusive mode", such
                that only one process at a time can access them.

            tpu_cores: How many TPU cores to train on (1 or 8) / Single TPU to train on [1]

            num_tpu_cores: How many TPU cores to train on (1 or 8)
                .. warning:: .. deprecated:: 0.7.6. Will remove 0.9.0.

            log_gpu_memory: None, 'min_max', 'all'. Might slow performance

            show_progress_bar:
                .. warning:: .. deprecated:: 0.7.2

                        Set `progress_bar_refresh_rate` to positive integer to enable. Will remove 0.9.0.

            progress_bar_refresh_rate: How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
                Ignored when a custom callback is passed to :paramref:`~Trainer.callbacks`.

            overfit_batches: Overfit a percent of training data (float) or a set number of batches (int). Default: 0.0

            overfit_pct:
                .. warning:: .. deprecated:: 0.8.0

                    Use `overfit_batches` instead. Will be removed in 0.10.0.

            track_grad_norm: -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm.

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

            val_check_interval: How often within one training epoch to check the validation set

            log_save_interval: Writes logs to disk this often

            row_log_interval: How often to add logging rows (does not write to disk)

            add_row_log_interval:
                .. warning:: .. deprecated:: 0.7.0

                    Use `row_log_interval` instead. Will remove 0.9.0.

            distributed_backend: The distributed backend to use (dp, ddp, ddp2, ddp_spawn, ddp_cpu)

            use_amp:
                .. warning:: .. deprecated:: 0.7.0

                    Use `precision` instead. Will remove 0.9.0.

            precision: Full precision (32), half precision (16).

            print_nan_grads:
                .. warning:: .. deprecated:: 0.7.2

                    Has no effect. When detected, NaN grads will be printed automatically.
                    Will remove 0.9.0.

            weights_summary: Prints a summary of the weights when training begins.

            weights_save_path: Where to save weights if specified. Will override default_root_dir
                    for checkpoints only. Use this if for whatever reason you need the checkpoints
                    stored in a different place than the logs written in `default_root_dir`.

            amp_level: The optimization level to use (O1, O2, etc...).

            num_sanity_val_steps: Sanity check runs n batches of val before starting the training routine.

            truncated_bptt_steps: Truncated back prop breaks performs backprop every k steps of

            resume_from_checkpoint: To resume training from a specific checkpoint pass in the path here.
                This can be a URL.

            profiler:  To profile individual steps during training and assist in

            reload_dataloaders_every_epoch: Set to True to reload dataloaders every epoch

            auto_lr_find: If set to True, will `initially` run a learning rate finder,
                trying to optimize initial learning for faster convergence. Sets learning
                rate in self.lr or self.learning_rate in the LightningModule.
                To use a different key, set a string instead of True with the key name.

            replace_sampler_ddp: Explicitly enables or disables sampler replacement.
                If not specified this will toggled automatically ddp is used

            benchmark: If true enables cudnn.benchmark.

            deterministic: If true enables cudnn.deterministic

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
            rank_zero_only.rank = os.environ['LOCAL_RANK']

        # training bookeeping
        self.total_batch_idx = 0
        self.running_loss = TensorRunningAccum(window_length=20)
        self.batch_idx = 0
        self.progress_bar_metrics = {}
        self.callback_metrics = {}
        self.num_training_batches = 0
        self.num_val_batches = []
        self.num_test_batches = []
        self.train_dataloader = None
        self.test_dataloaders = None
        self.val_dataloaders = None

        # training state
        self.model = None
        self.testing = False
        self.disable_validation = False
        self.prepare_data_per_node = prepare_data_per_node
        self.lr_schedulers = []
        self.optimizers = None
        self.optimizer_frequencies = []
        self.global_step = 0
        self.current_epoch = 0
        self.interrupted = False
        self.should_stop = False

        # set default save path if user didn't provide one
        if default_root_dir is None:
            default_root_dir = os.getcwd()
        self.default_root_dir = default_root_dir

        self.configure_logger(logger)

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
        self.weights_save_path = weights_save_path
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

        self.gradient_clip_val = gradient_clip_val
        self.check_val_every_n_epoch = check_val_every_n_epoch

        if not isinstance(track_grad_norm, (int, float)) and track_grad_norm != 'inf':
            raise MisconfigurationException(
                "track_grad_norm can be an int, a float or 'inf' (infinity norm).")
        self.track_grad_norm = float(track_grad_norm)

        self.on_gpu = True if (gpus and torch.cuda.is_available()) else False

        # tpu config
        if num_tpu_cores is not None:
            rank_zero_warn("Argument `num_tpu_cores` is now set by `tpu_cores` since v0.7.6"
                           " and this argument will be removed in v0.9.0", DeprecationWarning)

        if tpu_cores is None:
            tpu_cores = num_tpu_cores
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

        self.num_sanity_val_steps = num_sanity_val_steps
        # Backward compatibility, TODO: remove in v0.9.0
        if print_nan_grads:
            rank_zero_warn("Argument `print_nan_grads` has no effect and will be removed in v0.9.0."
                           " NaN grads will be printed automatically when detected.", DeprecationWarning)

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
            self.num_sanity_val_steps = 0
            self.max_epochs = 1
            rank_zero_info('Running in fast_dev_run mode: will run a full train,'
                           ' val and test loop using a single batch')

        # configure profiler
        if profiler is True:
            profiler = SimpleProfiler()
        self.profiler = profiler or PassThroughProfiler()

        # accumulated grads
        self.accumulate_grad_batches = accumulate_grad_batches
        self.configure_accumulated_gradients(accumulate_grad_batches)

        # for gpus allow int, string and gpu list
        if auto_select_gpus and isinstance(gpus, int):
            self.gpus = pick_multiple_gpus(gpus)
        else:
            self.gpus = gpus

        self.data_parallel_device_ids = _parse_gpu_ids(self.gpus)
        self.root_gpu = determine_root_gpu_device(self.data_parallel_device_ids)
        self.root_device = torch.device("cpu")

        # tpu state flags
        self.use_tpu = False
        self.tpu_local_core_rank = None
        self.tpu_global_core_rank = None

        # distributed backend choice
        self.distributed_backend = distributed_backend
        self.set_distributed_mode(distributed_backend)

        # override dist backend when using tpus
        if self.on_tpu:
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

        # backward compatibility
        if show_progress_bar is not None:
            self.show_progress_bar = show_progress_bar

        self._progress_bar_callback = self.configure_progress_bar(progress_bar_refresh_rate, process_position)

        # logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval

        self.row_log_interval = row_log_interval

        # how much of the data to use
        # TODO: remove in 0.10.0
        if overfit_pct is not None:
            rank_zero_warn("Argument `overfit_pct` is now set by `overfit_batches` since v0.8.0"
                           " and this argument will be removed in v0.10.0", DeprecationWarning)
            overfit_batches = overfit_pct

        # convert floats to ints
        self.overfit_batches = _determine_limit_batches(overfit_batches)

        # TODO: remove in 0.10.0
        if val_percent_check is not None:
            rank_zero_warn("Argument `val_percent_check` is now set by `limit_val_batches` since v0.8.0"
                           " and this argument will be removed in v0.10.0", DeprecationWarning)
            limit_val_batches = val_percent_check

        # TODO: remove in 0.10.0
        if test_percent_check is not None:
            rank_zero_warn("Argument `test_percent_check` is now set by `limit_test_batches` since v0.8.0"
                           " and this argument will be removed in v0.10.0", DeprecationWarning)
            limit_test_batches = test_percent_check

        # TODO: remove in 0.10.0
        if train_percent_check is not None:
            rank_zero_warn("Argument `train_percent_check` is now set by `limit_train_batches` since v0.8.0"
                           " and this argument will be removed in v0.10.0", DeprecationWarning)
            limit_train_batches = train_percent_check

        self.limit_test_batches = _determine_limit_batches(limit_test_batches)
        self.limit_val_batches = _determine_limit_batches(limit_val_batches)
        self.limit_train_batches = _determine_limit_batches(limit_train_batches)
        self.determine_data_use_amount(self.overfit_batches)

        # AMP init
        # These are the only lines needed after v0.8.0
        # we wrap the user's forward with autocast and give it back at the end of fit
        self.autocast_original_forward = None
        self.precision = precision
        self.scaler = None

        # Backward compatibility, TODO: remove in v0.9.0
        if use_amp is not None:
            rank_zero_warn("Argument `use_amp` is now set by `precision` since v0.7.0"
                           " and this method will be removed in v0.9.0", DeprecationWarning)
            self.precision = 16 if use_amp else 32

        self.amp_level = amp_level
        self.init_amp()

        self.on_colab_kaggle = os.getenv('COLAB_GPU') or os.getenv('KAGGLE_URL_BASE')

        # Callback system
        self.on_init_end()

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
             ('print_nan_grads', (<class 'bool'>,), False),
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
        parser = ArgumentParser(parents=[parent_parser], add_help=False, )

        blacklist = ['kwargs']
        depr_arg_names = cls.get_deprecated_arg_names() + blacklist

        allowed_types = (str, float, int, bool)

        # TODO: get "help" from docstring :)
        for arg, arg_types, arg_default in (at for at in cls.get_init_arguments_and_types()
                                            if at[0] not in depr_arg_names):
            arg_types = [at for at in allowed_types if at in arg_types]
            if not arg_types:
                # skip argument with not supported type
                continue
            arg_kwargs = {}
            if bool in arg_types:
                arg_kwargs.update(nargs="?")
                # if the only arg type is bool
                if len(arg_types) == 1:
                    # redefine the type for ArgParser needed
                    def use_type(x):
                        return bool(parsing.str_to_bool(x))
                else:
                    # filter out the bool as we need to use more general
                    use_type = [at for at in arg_types if at is not bool][0]
            else:
                use_type = arg_types[0]

            if arg == 'gpus' or arg == 'tpu_cores':
                use_type = Trainer._allowed_type
                arg_default = Trainer._arg_default

            parser.add_argument(
                f'--{arg}',
                dest=arg,
                default=arg_default,
                type=use_type,
                help='autogenerated by pl.Trainer',
                **arg_kwargs,
            )

        return parser

    def _allowed_type(x) -> Union[int, str]:
        if ',' in x:
            return str(x)
        else:
            return int(x)

    def _arg_default(x) -> Union[int, str]:
        if ',' in x:
            return str(x)
        else:
            return int(x)

    @staticmethod
    def parse_argparser(arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        """Parse CLI arguments, required for custom bool types."""
        args = arg_parser.parse_args() if isinstance(arg_parser, ArgumentParser) else arg_parser
        args = {k: True if v is None else v for k, v in vars(args).items()}
        return Namespace(**args)

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

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    def fit(
            self,
            model: LightningModule,
            train_dataloader: Optional[DataLoader] = None,
            val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None
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
        # bind logger and other properties
        model.logger = self.logger
        self.copy_trainer_model_properties(model)

        # clean hparams
        if hasattr(model, 'hparams'):
            parsing.clean_namespace(model.hparams)

        # set up the passed in dataloaders (if needed)
        self.__attach_dataloaders(model, train_dataloader, val_dataloaders)

        # check that model is configured correctly
        self.check_model_configuration(model)

        # callbacks
        self.on_fit_start()
        if self.is_function_implemented('on_fit_start', model):
            model.on_fit_start()

        # on multi-gpu jobs we only want to manipulate (download, etc) on node_rank=0, local_rank=0
        # or in the case where each node needs to do its own manipulation in which case just local_rank=0
        if self.can_prepare_data():
            model.prepare_data()
            self._is_data_prepared = True

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

        # route to appropriate start method
        # when using multi-node or DDP within a node start each module in a separate process
        if self.use_ddp2:
            if self.is_slurm_managing_tasks:
                task = int(os.environ['SLURM_LOCALID'])

            # torchelastic or general non_slurm ddp2
            elif 'WORLD_SIZE' in os.environ and ('GROUP_RANK' in os.environ or 'NODE_RANK' in os.environ):
                task = int(os.environ['LOCAL_RANK'])

            self.ddp_train(task, model)
        elif self.use_ddp:
            if self.is_slurm_managing_tasks:
                task = int(os.environ['SLURM_LOCALID'])
                self.ddp_train(task, model)

            # torchelastic or general non_slurm ddp
            elif 'WORLD_SIZE' in os.environ and ('GROUP_RANK' in os.environ or 'NODE_RANK' in os.environ):
                task = int(os.environ['LOCAL_RANK'])
                self.ddp_train(task, model)

            elif self.distributed_backend == 'ddp_cpu':
                self.set_random_port()
                self.model = model
                mp.spawn(self.ddp_train, nprocs=self.num_processes, args=(model,))

            elif self.distributed_backend == 'ddp_spawn':
                self.set_random_port()
                model.share_memory()

                # spin up peers
                mp.spawn(self.ddp_train, nprocs=self.num_processes, args=(model, ))

            elif self.distributed_backend == 'ddp':
                self.set_random_port()
                self.spawn_ddp_children(model)

        # 1 gpu or dp option triggers training using DP module
        # easier to avoid NCCL issues
        elif self.use_dp:
            self.dp_train(model)

        elif self.use_horovod:
            self.horovod_train(model)

        elif self.single_gpu:
            self.single_gpu_train(model)

        elif self.use_tpu:  # pragma: no-cover
            rank_zero_info(f'training on {self.tpu_cores} TPU cores')

            if not XLA_AVAILABLE:
                raise MisconfigurationException('No TPU devices found.')

            #  COLAB_GPU is an env var available by default in Colab environments.
            start_method = 'fork' if self.on_colab_kaggle else 'spawn'

            # track for predict
            self.model = model

            # train
            if self.tpu_id is not None:
                self.tpu_train(self.tpu_id, model)
            else:
                xmp.spawn(self.tpu_train, args=(model,), nprocs=self.tpu_cores, start_method=start_method)

            # load weights if not interrupted
            self.load_spawn_weights(model)
            self.model = model

        # ON CPU
        else:
            # run through amp wrapper
            if self.use_amp:
                raise MisconfigurationException('amp + cpu is not supported.  Please use a GPU option')

            # call setup after the ddp process has connected
            self.setup('fit')
            if self.is_function_implemented('setup', model):
                model.setup('fit')

            # CHOOSE OPTIMIZER
            # allow for lr schedulers as well
            self.optimizers, self.lr_schedulers, self.optimizer_frequencies = self.init_optimizers(model)

            self.run_pretrain_routine(model)

        # callbacks
        self.on_fit_end()

        # model hooks
        if self.is_function_implemented('on_fit_end'):
            model.on_fit_end()

        self.teardown('fit')
        if self.is_function_implemented('teardown'):
            model.teardown('fit')

        # return 1 when finished
        # used for testing or when we need to know that training succeeded
        return 1

    def can_prepare_data(self):
        if self.prepare_data_per_node:
            return self.local_rank == 0

        else:
            return self.node_rank == 0 and self.local_rank == 0

    def __attach_dataloaders(self, model, train_dataloader=None, val_dataloaders=None, test_dataloaders=None):
        # when dataloader is passed via fit, patch the train_dataloader
        # functions to overwrite with these implementations
        if train_dataloader is not None:
            model.train_dataloader = _PatchDataLoader(train_dataloader)

        if val_dataloaders is not None:
            model.val_dataloader = _PatchDataLoader(val_dataloaders)

        if test_dataloaders is not None:
            model.test_dataloader = _PatchDataLoader(test_dataloaders)

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

        # init amp. Must be done here instead of __init__ to allow ddp to work
        if NATIVE_AMP_AVALAIBLE and self.precision == 16:
            self.scaler = torch.cuda.amp.GradScaler()

        # log hyper-parameters
        if self.logger is not None:
            # save exp to get started
            self.logger.log_hyperparams(ref_model.hparams)

            self.logger.save()

        if self.use_ddp or self.use_ddp2:
            torch_distrib.barrier()

        # wait for all models to restore weights
        if self.on_tpu and XLA_AVAILABLE:
            # wait for all processes to catch up
            torch_xla.core.xla_model.rendezvous("pl.Trainer.run_pretrain_routine")

        elif self.use_horovod:
            # wait for all processes to catch up
            hvd.join()

        # register auto-resubmit when on SLURM
        self.register_slurm_signal_handlers()

        # print model summary
        if self.is_global_zero and self.weights_summary is not None and not self.testing:
            if self.weights_summary in ModelSummary.MODES:
                ref_model.summarize(mode=self.weights_summary)
            else:
                raise MisconfigurationException(
                    "weights_summary can be None, " + ", ".join(ModelSummary.MODES)
                )

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # restore training and model before hpc call
        self.restore_weights(model)

        # when testing requested only run test and return
        if self.testing:
            # only load test dataloader for testing
            # self.reset_test_dataloader(ref_model)
            self.run_evaluation(test_mode=True)
            return

        # check if we should run validation during training
        self.disable_validation = not (self.is_overridden('validation_step') and self.limit_val_batches > 0) \
            and not self.fast_dev_run

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        if not self.disable_validation and self.num_sanity_val_steps > 0:
            self.reset_val_dataloader(ref_model)

            # hook and callback
            ref_model.on_sanity_check_start()
            self.on_sanity_check_start()

            num_loaders = len(self.val_dataloaders)
            max_batches = [self.num_sanity_val_steps] * num_loaders
            eval_results = self._evaluate(model,
                                          self.val_dataloaders,
                                          max_batches,
                                          False)
            _, _, _, callback_metrics, _ = self.process_output(eval_results)
            self.callback_metrics = callback_metrics

            self.on_sanity_check_end()

        # clear cache before training
        if self.on_gpu and self.root_gpu is not None:
            # use context because of:
            # https://discuss.pytorch.org/t/out-of-memory-when-i-use-torch-cuda-empty-cache/57898
            with torch.cuda.device(f'cuda:{self.root_gpu}'):
                torch.cuda.empty_cache()

        # CORE TRAINING LOOP
        self.train()

    def test(
            self,
            model: Optional[LightningModule] = None,
            test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
            ckpt_path: Optional[str] = 'best'
    ):
        r"""

        Separates from fit to make sure you never run on your test set until you want to.

        Args:
            model: The model to test.

            test_dataloaders: Either a single
                Pytorch Dataloader or a list of them, specifying validation samples.

            ckpt_path: Either ``best`` or path to the checkpoint you wish to test.
                If ``None``, use the weights from the last epoch to test. Default to ``best``.

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
        self.setup('test')
        model_ref = self.model if model is None else model
        if self.is_function_implemented('setup', model_ref):
            model_ref.setup('test')

        self.barrier('test_setup')

        if model is None and ckpt_path == 'best' and self.checkpoint_callback.save_top_k <= 0:
            raise MisconfigurationException(
                'ckpt_path is "best", but ModelCheckpoint is not configured to save the best model.')

        # if model is not given (None), ckpt_path is given,
        # load the given checkpoint for testing
        if model is None and ckpt_path is not None:
            # ckpt_path is 'best' so load the best model
            if ckpt_path == 'best':
                ckpt_path = self.checkpoint_callback.best_model_path
            model = self.get_model().load_from_checkpoint(ckpt_path)

        self.testing = True

        if test_dataloaders is not None:
            if model:
                self.__attach_dataloaders(model, test_dataloaders=test_dataloaders)
            else:
                self.__attach_dataloaders(self.model, test_dataloaders=test_dataloaders)

        if model is not None:
            self.model = model
            self.fit(model)

        # on tpu, .spawn means we don't have a trained model
        # TODO: remove TPU spawn
        elif self.use_tpu:  # pragma: no-cover
            # attempt to load weights from a spawn
            path = os.path.join(self.default_root_dir, '__temp_weight_ddp_end.ckpt')
            test_model = self.model
            if os.path.exists(path):
                test_model = self.load_spawn_weights(self.model)

            self.fit(test_model)
        else:
            self.run_evaluation(test_mode=True)

        self.testing = False

        self.teardown('test')
        if self.is_function_implemented('teardown'):
            model_ref = self.get_model()
            model_ref.teardown('test')

    def check_model_configuration(self, model: LightningModule):
        r"""
        Checks that the model is configured correctly before training or testing is started.

        Args:
            model: The model to check the configuration.

        """
        # Check training_step, train_dataloader, configure_optimizer methods
        if not self.testing:
            if not self.is_overridden('training_step', model):
                raise MisconfigurationException(
                    'No `training_step()` method defined. Lightning `Trainer` expects as minimum a'
                    ' `training_step()`, `training_dataloader()` and `configure_optimizers()` to be defined.')

            if not self.is_overridden('train_dataloader', model):
                raise MisconfigurationException(
                    'No `train_dataloader()` method defined. Lightning `Trainer` expects as minimum a'
                    ' `training_step()`, `training_dataloader()` and `configure_optimizers()` to be defined.')

            if not self.is_overridden('configure_optimizers', model):
                raise MisconfigurationException(
                    'No `configure_optimizers()` method defined. Lightning `Trainer` expects as minimum a'
                    ' `training_step()`, `training_dataloader()` and `configure_optimizers()` to be defined.')

            # Check val_dataloader, validation_step and validation_epoch_end
            if self.is_overridden('val_dataloader', model):
                if not self.is_overridden('validation_step', model):
                    raise MisconfigurationException('You have passed in a `val_dataloader()`'
                                                    ' but have not defined `validation_step()`.')
                else:
                    if not self.is_overridden('validation_epoch_end', model):
                        rank_zero_warn(
                            'You have defined a `val_dataloader()` and have defined a `validation_step()`,'
                            ' you may also want to define `validation_epoch_end()` for accumulating stats.',
                            RuntimeWarning
                        )
            else:
                if self.is_overridden('validation_step', model):
                    raise MisconfigurationException('You have defined `validation_step()`,'
                                                    ' but have not passed in a `val_dataloader()`.')

        # Check test_dataloader, test_step and test_epoch_end
        if self.is_overridden('test_dataloader', model):
            if not self.is_overridden('test_step', model):
                raise MisconfigurationException('You have passed in a `test_dataloader()`'
                                                ' but have not defined `test_step()`.')
            else:
                if not self.is_overridden('test_epoch_end', model):
                    rank_zero_warn(
                        'You have defined a `test_dataloader()` and have defined a `test_step()`, you may also want to'
                        ' define `test_epoch_end()` for accumulating stats.', RuntimeWarning
                    )
        else:
            if self.testing and self.is_overridden('test_step', model):
                raise MisconfigurationException('You have defined `test_step()` but did not'
                                                ' implement `test_dataloader` nor passed in `.test(test_dataloader)`.')

    def barrier(self, name):
        if self.use_ddp or self.use_ddp2:
            torch_distrib.barrier()

        if self.on_tpu and XLA_AVAILABLE:
            # wait for all processes to catch up
            torch_xla.core.xla_model.rendezvous(f'pl.Trainer.{name}')


class _PatchDataLoader(object):
    r"""
    Callable object for patching dataloaders passed into trainer.fit().
    Use this class to override model.*_dataloader() and be pickle-compatible.

    Args:
        dataloader: Dataloader object to return when called.

    """
    def __init__(self, dataloader: Union[List[DataLoader], DataLoader]):
        self.dataloader = dataloader

        # cannot pickle __code__ so cannot verify if PatchDataloader
        # exists which shows dataloader methods have been overwritten.
        # so, we hack it by using the string representation
        self.patch_loader_code = str(self.__call__.__code__)

    def __call__(self) -> Union[List[DataLoader], DataLoader]:
        return self.dataloader


def _determine_limit_batches(batches: Union[int, float]) -> Union[int, float]:
    if 0 <= batches <= 1:
        return batches
    elif batches > 1 and batches % 1.0 == 0:
        return int(batches)
    else:
        raise MisconfigurationException(
            f'You have passed invalid value {batches}, it has to be in (0, 1) or nature number.'
        )
