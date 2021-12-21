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
"""Trainer to automate the training."""
import inspect
import logging
import os
import traceback
import warnings
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Tuple, Union
from weakref import proxy

import torch
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.accelerators import Accelerator, IPUAccelerator
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint, ProgressBarBase
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import DummyLogger, LoggerCollection
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loops import PredictionLoop, TrainingBatchLoop, TrainingEpochLoop
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.plugins import DDPSpawnPlugin, ParallelPlugin, PLUGIN_INPUT, PrecisionPlugin, TrainingTypePlugin
from pytorch_lightning.profiler import (
    AdvancedProfiler,
    BaseProfiler,
    PassThroughProfiler,
    PyTorchProfiler,
    SimpleProfiler,
    XLAProfiler,
)
from pytorch_lightning.trainer.callback_hook import TrainerCallbackHookMixin
from pytorch_lightning.trainer.configuration_validator import verify_loop_configurations
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.trainer.connectors.env_vars_connector import _defaults_from_env_vars
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.trainer.connectors.signal_connector import SignalConnector
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from pytorch_lightning.tuner.auto_gpu_select import pick_multiple_gpus
from pytorch_lightning.tuner.lr_finder import _LRFinder
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities import (
    _IPU_AVAILABLE,
    _TPU_AVAILABLE,
    device_parser,
    DeviceType,
    DistributedType,
    GradClipAlgorithmType,
    parsing,
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_warn,
)
from pytorch_lightning.utilities.argparse import (
    add_argparse_args,
    from_argparse_args,
    parse_argparser,
    parse_env_variables,
)
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.distributed import distributed_available
from pytorch_lightning.utilities.exceptions import ExitGracefullyException, MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from pytorch_lightning.utilities.meta import is_on_meta_device, materialize_module
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.utilities.types import (
    _EVALUATE_OUTPUT,
    _PATH,
    _PREDICT_OUTPUT,
    EVAL_DATALOADERS,
    LRSchedulerTypeUnion,
    TRAIN_DATALOADERS,
)

log = logging.getLogger(__name__)
# warnings to ignore in trainer
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead"
)


class Trainer(
    TrainerCallbackHookMixin,
    TrainerModelHooksMixin,
    TrainerOptimizersMixin,
    TrainerDataLoadingMixin,
):
    # Needed because of LightningOptimizer
    _lightning_optimizers = None

    @_defaults_from_env_vars
    def __init__(
        self,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        checkpoint_callback: Optional[bool] = None,
        enable_checkpointing: bool = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        process_position: int = 0,
        num_nodes: int = 1,
        num_processes: int = 1,
        devices: Optional[Union[List[int], str, int]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        ipus: Optional[int] = None,
        log_gpu_memory: Optional[str] = None,  # TODO: Remove in 1.7
        progress_bar_refresh_rate: Optional[int] = None,  # TODO: remove in v1.7
        enable_progress_bar: bool = True,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        limit_predict_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        flush_logs_every_n_steps: Optional[int] = None,
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = None,
        strategy: Optional[Union[str, TrainingTypePlugin]] = None,
        sync_batchnorm: bool = False,
        precision: Union[int, str] = 32,
        enable_model_summary: bool = True,
        weights_summary: Optional[str] = "top",
        weights_save_path: Optional[str] = None,
        num_sanity_val_steps: int = 2,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[BaseProfiler, str]] = None,
        benchmark: bool = False,
        deterministic: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        reload_dataloaders_every_epoch: bool = False,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        detect_anomaly: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        prepare_data_per_node: Optional[bool] = None,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        amp_backend: str = "native",
        amp_level: Optional[str] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = "max_size_cycle",
        stochastic_weight_avg: bool = False,
        terminate_on_nan: Optional[bool] = None,
    ):
        r"""
        Customize every aspect of training via flags.

        Args:

            accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto")
                as well as custom accelerator instances.

                .. deprecated:: v1.5
                    Passing training strategies (e.g., 'ddp') to ``accelerator`` has been deprecated in v1.5.0
                    and will be removed in v1.7.0. Please use the ``strategy`` argument instead.

            accumulate_grad_batches: Accumulates grads every k batches or as set up in the dict.

            amp_backend: The mixed precision backend to use ("native" or "apex").

            amp_level: The optimization level to use (O1, O2, etc...). By default it will be set to "O2"
                if ``amp_backend`` is set to "apex".

            auto_lr_find: If set to True, will make trainer.tune() run a learning rate finder,
                trying to optimize initial learning for faster convergence. trainer.tune() method will
                set the suggested learning rate in self.lr or self.learning_rate in the LightningModule.
                To use a different key set a string instead of True with the key name.

            auto_scale_batch_size: If set to True, will `initially` run a batch size
                finder trying to find the largest batch size that fits into memory.
                The result will be stored in self.batch_size in the LightningModule.
                Additionally, can be set to either `power` that estimates the batch size through
                a power search or `binsearch` that estimates the batch size through a binary search.

            auto_select_gpus: If enabled and ``gpus`` is an integer, pick available
                gpus automatically. This is especially useful when
                GPUs are configured to be in "exclusive mode", such
                that only one process at a time can access them.

            benchmark: If true enables cudnn.benchmark.

            callbacks: Add a callback or list of callbacks.

            checkpoint_callback: If ``True``, enable checkpointing.

                .. deprecated:: v1.5
                    ``checkpoint_callback`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please consider using ``enable_checkpointing`` instead.

            enable_checkpointing: If ``True``, enable checkpointing.
                It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`.

            check_val_every_n_epoch: Check val every n train epochs.

            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Default: ``os.getcwd()``.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

            detect_anomaly: Enable anomaly detection for the autograd engine.

            deterministic: If ``True``, sets whether PyTorch operations must use deterministic algorithms.
                Default: ``False``.

            devices: Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`,
                based on the accelerator type.

            fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
                of train, val and test to find any bugs (ie: a sort of unit test).

            flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).

                .. deprecated:: v1.5
                    ``flush_logs_every_n_steps`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please configure flushing directly in the logger instead.

            gpus: Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node

            gradient_clip_val: The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
                gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.

            gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
                to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
                be set to ``"norm"``.

            limit_train_batches: How much of training dataset to check (float = fraction, int = num_batches).

            limit_val_batches: How much of validation dataset to check (float = fraction, int = num_batches).

            limit_test_batches: How much of test dataset to check (float = fraction, int = num_batches).

            limit_predict_batches: How much of prediction dataset to check (float = fraction, int = num_batches).

            logger: Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
                the default ``TensorBoardLogger``. ``False`` will disable logging. If multiple loggers are
                provided and the `save_dir` property of that logger is not set, local files (checkpoints,
                profiler traces, etc.) are saved in ``default_root_dir`` rather than in the ``log_dir`` of any
                of the individual loggers.

            log_gpu_memory: None, 'min_max', 'all'. Might slow performance.

                .. deprecated:: v1.5
                    Deprecated in v1.5.0 and will be removed in v1.7.0
                    Please use the ``DeviceStatsMonitor`` callback directly instead.

            log_every_n_steps: How often to log within steps (defaults to every 50 steps).

            prepare_data_per_node: If True, each LOCAL_RANK=0 will call prepare data.
                Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data

                .. deprecated:: v1.5
                    Deprecated in v1.5.0 and will be removed in v1.7.0
                    Please set ``prepare_data_per_node`` in LightningDataModule or LightningModule directly instead.

            process_position: Orders the progress bar when running multiple models on same machine.

                .. deprecated:: v1.5
                    ``process_position`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please pass :class:`~pytorch_lightning.callbacks.progress.TQDMProgressBar` with ``process_position``
                    directly to the Trainer's ``callbacks`` argument instead.

            progress_bar_refresh_rate: How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
                Ignored when a custom progress bar is passed to :paramref:`~Trainer.callbacks`. Default: None, means
                a suitable value will be chosen based on the environment (terminal, Google COLAB, etc.).

                .. deprecated:: v1.5
                    ``progress_bar_refresh_rate`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please pass :class:`~pytorch_lightning.callbacks.progress.TQDMProgressBar` with ``refresh_rate``
                    directly to the Trainer's ``callbacks`` argument instead. To disable the progress bar,
                    pass ``enable_progress_bar = False`` to the Trainer.

            enable_progress_bar: Whether to enable to progress bar by default.

            profiler: To profile individual steps during training and assist in identifying bottlenecks.

            overfit_batches: Overfit a fraction of training data (float) or a set number of batches (int).

            plugins: Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.

            precision: Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16).
                Can be used on CPU, GPU or TPUs.

            max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
                If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
                To enable infinite training, set ``max_epochs = -1``.

            min_epochs: Force training for at least these many epochs. Disabled by default (None).
                If both min_epochs and min_steps are not specified, defaults to ``min_epochs = 1``.

            max_steps: Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
                and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
                ``max_epochs`` to ``-1``.

            min_steps: Force training for at least these number of steps. Disabled by default (None).

            max_time: Stop training after this amount of time has passed. Disabled by default (None).
                The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
                :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
                :class:`datetime.timedelta`.

            num_nodes: Number of GPU nodes for distributed training.

            num_processes: Number of processes for distributed training with ``accelerator="cpu"``.

            num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
                Set it to `-1` to run all batches in all validation dataloaders.

            reload_dataloaders_every_n_epochs: Set to a non-negative integer to reload dataloaders every n epochs.

            reload_dataloaders_every_epoch: Set to True to reload dataloaders every epoch.

                .. deprecated:: v1.4
                    ``reload_dataloaders_every_epoch`` has been deprecated in v1.4 and will be removed in v1.6.
                    Please use ``reload_dataloaders_every_n_epochs``.

            replace_sampler_ddp: Explicitly enables or disables sampler replacement. If not specified this
                will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for
                train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it,
                you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.

            resume_from_checkpoint: Path/URL of the checkpoint from which training is resumed. If there is
                no checkpoint file at the path, an exception is raised. If resuming from mid-epoch checkpoint,
                training will start from the beginning of the next epoch.

                .. deprecated:: v1.5
                    ``resume_from_checkpoint`` is deprecated in v1.5 and will be removed in v1.7.
                    Please pass the path to ``Trainer.fit(..., ckpt_path=...)`` instead.

            strategy: Supports different training strategies with aliases
                as well custom training type plugins.

            sync_batchnorm: Synchronize batch norm layers between process groups/whole world.

            terminate_on_nan: If set to True, will terminate training (by raising a `ValueError`) at the
                end of each training batch, if any of the parameters or the loss are NaN or +/-inf.

                .. deprecated:: v1.5
                    Trainer argument ``terminate_on_nan`` was deprecated in v1.5 and will be removed in 1.7.
                    Please use ``detect_anomaly`` instead.

            detect_anomaly: Enable anomaly detection for the autograd engine.

            tpu_cores: How many TPU cores to train on (1 or 8) / Single TPU to train on [1]

            ipus: How many IPUs to train on.

            track_grad_norm: -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm. If using
                Automatic Mixed Precision (AMP), the gradients will be unscaled before logging them.

            val_check_interval: How often to check the validation set. Use float to check within a training epoch,
                use int to check every n steps (batches).

            enable_model_summary: Whether to enable model summarization by default.

            weights_summary: Prints a summary of the weights when training begins.

                .. deprecated:: v1.5
                    ``weights_summary`` has been deprecated in v1.5 and will be removed in v1.7.
                    To disable the summary, pass ``enable_model_summary = False`` to the Trainer.
                    To customize the summary, pass :class:`~pytorch_lightning.callbacks.model_summary.ModelSummary`
                    directly to the Trainer's ``callbacks`` argument.

            weights_save_path: Where to save weights if specified. Will override default_root_dir
                for checkpoints only. Use this if for whatever reason you need the checkpoints
                stored in a different place than the logs written in `default_root_dir`.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
                Defaults to `default_root_dir`.

            move_metrics_to_cpu: Whether to force internal logged metrics to be moved to cpu.
                This can save some gpu memory, but can make training slower. Use with attention.

            multiple_trainloader_mode: How to loop over the datasets when there are multiple train loaders.
                In 'max_size_cycle' mode, the trainer ends one epoch when the largest dataset is traversed,
                and smaller datasets reload when running out of their data. In 'min_size' mode, all the datasets
                reload when reaching the minimum length of datasets.

            stochastic_weight_avg: Whether to use `Stochastic Weight Averaging (SWA)
                <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>`_.

                .. deprecated:: v1.5
                    ``stochastic_weight_avg`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please pass :class:`~pytorch_lightning.callbacks.stochastic_weight_avg.StochasticWeightAveraging`
                    directly to the Trainer's ``callbacks`` argument instead.
        """
        super().__init__()
        Trainer._log_api_event("init")
        self.state = TrainerState()

        gpu_ids, tpu_cores = self._parse_devices(gpus, auto_select_gpus, tpu_cores)

        # init connectors
        self._data_connector = DataConnector(self, multiple_trainloader_mode)

        self._accelerator_connector = AcceleratorConnector(
            num_processes,
            devices,
            tpu_cores,
            ipus,
            accelerator,
            strategy,
            gpus,
            gpu_ids,
            num_nodes,
            sync_batchnorm,
            benchmark,
            replace_sampler_ddp,
            deterministic,
            precision,
            amp_backend,
            amp_level,
            plugins,
        )
        self.logger_connector = LoggerConnector(self, log_gpu_memory)
        self._callback_connector = CallbackConnector(self)
        self.checkpoint_connector = CheckpointConnector(self, resume_from_checkpoint)
        self.signal_connector = SignalConnector(self)
        self.tuner = Tuner(self)

        fit_loop = FitLoop(
            min_epochs=(1 if (min_epochs is None and min_steps is None and max_time is None) else min_epochs),
            max_epochs=(
                max_epochs if max_epochs is not None else (1000 if (max_steps == -1 and max_time is None) else -1)
            ),
        )
        training_epoch_loop = TrainingEpochLoop(min_steps, max_steps)
        training_batch_loop = TrainingBatchLoop()
        training_validation_loop = EvaluationLoop()
        training_epoch_loop.connect(batch_loop=training_batch_loop, val_loop=training_validation_loop)
        fit_loop.connect(epoch_loop=training_epoch_loop)

        # default .fit() loop
        self.fit_loop = fit_loop

        # default .validate() loop
        self.validate_loop = EvaluationLoop()

        # default .test() loop
        self.test_loop = EvaluationLoop()

        # default .predict() loop
        self.predict_loop = PredictionLoop()

        # Needed because of LightningOptimizer
        self._lightning_optimizers = None

        # .validate() and .test() set this when they load a checkpoint
        self.validated_ckpt_path: Optional[str] = None
        self.tested_ckpt_path: Optional[str] = None
        self.predicted_ckpt_path: Optional[str] = None

        # todo: remove in v1.7
        self._weights_summary: Optional[str] = None

        # init callbacks
        # Declare attributes to be set in _callback_connector on_trainer_init
        self._callback_connector.on_trainer_init(
            callbacks,
            checkpoint_callback,
            enable_checkpointing,
            enable_progress_bar,
            progress_bar_refresh_rate,
            process_position,
            default_root_dir,
            weights_save_path,
            enable_model_summary,
            weights_summary,
            stochastic_weight_avg,
            max_time,
            accumulate_grad_batches,
        )

        # hook
        self.on_init_start()

        # init optimizer + lr scheduler related flags
        self.lr_schedulers = []
        self.optimizers = []
        self.optimizer_frequencies = []

        # init data flags
        self._data_connector.on_trainer_init(
            check_val_every_n_epoch,
            reload_dataloaders_every_n_epochs,
            reload_dataloaders_every_epoch,
            prepare_data_per_node,
        )

        if terminate_on_nan is not None:
            rank_zero_deprecation(
                "Trainer argument `terminate_on_nan` was deprecated in v1.5 and will be removed in 1.7."
                " Please use `Trainer(detect_anomaly=True)` instead."
            )
            if not isinstance(terminate_on_nan, bool):
                raise TypeError(f"`terminate_on_nan` should be a bool, got {terminate_on_nan}.")

        # gradient clipping
        if gradient_clip_val is not None and not isinstance(gradient_clip_val, (int, float)):
            raise TypeError(f"`gradient_clip_val` should be an int or a float. Got {gradient_clip_val}.")

        if gradient_clip_algorithm is not None and not GradClipAlgorithmType.supported_type(
            gradient_clip_algorithm.lower()
        ):
            raise MisconfigurationException(
                f"`gradient_clip_algorithm` {gradient_clip_algorithm} is invalid. "
                f"Allowed algorithms: {GradClipAlgorithmType.supported_types()}."
            )

        # gradient norm tracking
        if track_grad_norm != -1 and not (
            (isinstance(track_grad_norm, (int, float)) or track_grad_norm == "inf") and float(track_grad_norm) > 0
        ):
            raise MisconfigurationException(
                f"`track_grad_norm` must be a positive number or 'inf' (infinity norm). Got {track_grad_norm}."
            )

        self._terminate_on_nan = terminate_on_nan
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = (
            GradClipAlgorithmType(gradient_clip_algorithm.lower())
            if gradient_clip_algorithm is not None
            else gradient_clip_algorithm
        )
        self.track_grad_norm: float = float(track_grad_norm)

        self._detect_anomaly: bool = detect_anomaly
        self._setup_on_init(num_sanity_val_steps)

        # configure tuner
        self.tuner.on_trainer_init(auto_lr_find, auto_scale_batch_size)

        # configure profiler
        self.__init_profiler(profiler)

        # init logger flags
        self.logger: Optional[LightningLoggerBase]
        self.logger_connector.on_trainer_init(logger, flush_logs_every_n_steps, log_every_n_steps, move_metrics_to_cpu)

        # init debugging flags
        self._init_debugging_flags(
            limit_train_batches,
            limit_val_batches,
            limit_test_batches,
            limit_predict_batches,
            val_check_interval,
            overfit_batches,
            fast_dev_run,
        )

        # Callback system
        self.on_init_end()

    def _init_debugging_flags(
        self,
        limit_train_batches,
        limit_val_batches,
        limit_test_batches,
        limit_predict_batches,
        val_check_interval,
        overfit_batches,
        fast_dev_run,
    ):
        if not isinstance(fast_dev_run, (bool, int)):
            raise MisconfigurationException(
                f"fast_dev_run={fast_dev_run} is not a valid configuration. It should be either a bool or an int >= 0"
            )

        if isinstance(fast_dev_run, int) and (fast_dev_run < 0):
            raise MisconfigurationException(
                f"fast_dev_run={fast_dev_run} is not a valid configuration. It should be >= 0."
            )

        self.fast_dev_run = fast_dev_run
        fast_dev_run = int(fast_dev_run)

        # set fast_dev_run=True when it is 1, used while logging
        if fast_dev_run == 1:
            self.fast_dev_run = True

        if fast_dev_run:
            limit_train_batches = fast_dev_run
            limit_val_batches = fast_dev_run
            limit_test_batches = fast_dev_run
            limit_predict_batches = fast_dev_run
            self.fit_loop.max_steps = fast_dev_run
            self.num_sanity_val_steps = 0
            self.fit_loop.max_epochs = 1
            val_check_interval = 1.0
            self.check_val_every_n_epoch = 1
            self.logger = DummyLogger() if self.logger is not None else None

            rank_zero_info(
                "Running in fast_dev_run mode: will run a full train,"
                f" val, test and prediction loop using {fast_dev_run} batch(es)."
            )

        self.limit_train_batches = _determine_batch_limits(limit_train_batches, "limit_train_batches")
        self.limit_val_batches = _determine_batch_limits(limit_val_batches, "limit_val_batches")
        self.limit_test_batches = _determine_batch_limits(limit_test_batches, "limit_test_batches")
        self.limit_predict_batches = _determine_batch_limits(limit_predict_batches, "limit_predict_batches")
        self.val_check_interval = _determine_batch_limits(val_check_interval, "val_check_interval")
        self.overfit_batches = _determine_batch_limits(overfit_batches, "overfit_batches")
        self.determine_data_use_amount(self.overfit_batches)

    def determine_data_use_amount(self, overfit_batches: float) -> None:
        """Use less data for debugging purposes."""
        if overfit_batches > 0:
            self.limit_train_batches = overfit_batches
            self.limit_val_batches = overfit_batches
            self.limit_test_batches = overfit_batches

    def _setup_on_init(self, num_sanity_val_steps: int) -> None:
        self._log_device_info()

        self.should_stop = False
        self.state = TrainerState()
        self.num_training_batches = float("inf")
        self.train_dataloader = None

        if num_sanity_val_steps == -1:
            self.num_sanity_val_steps = float("inf")
        else:
            self.num_sanity_val_steps = num_sanity_val_steps

        self.num_sanity_val_batches = []
        self.num_test_batches = []
        self.num_val_batches = []
        self.test_dataloaders = None
        self.val_dataloaders = None

        # when true, print evaluation results in .validate() and .test()
        self.verbose_evaluate = True

        self.num_predict_batches = []

    def _call_and_handle_interrupt(self, trainer_fn: Callable, *args: Any, **kwargs: Any) -> Any:
        r"""
        Error handling, intended to be used only for main trainer function entry points (fit, validate, test, predict)
        as all errors should funnel through them

        Args:
            trainer_fn: one of (fit, validate, test, predict)
            *args: positional arguments to be passed to the `trainer_fn`
            **kwargs: keyword arguments to be passed to `trainer_fn`
        """
        try:
            return trainer_fn(*args, **kwargs)
        # TODO: treat KeyboardInterrupt as BaseException (delete the code below) in v1.7
        except KeyboardInterrupt as exception:
            rank_zero_warn("Detected KeyboardInterrupt, attempting graceful shutdown...")
            # user could press Ctrl+c many times... only shutdown once
            if not self.interrupted:
                self.state.status = TrainerStatus.INTERRUPTED
                self.on_keyboard_interrupt()
                self.on_exception(exception)
        except BaseException as exception:
            self.state.status = TrainerStatus.INTERRUPTED
            if distributed_available() and self.world_size > 1:
                # try syncing remaing processes, kill otherwise
                self.training_type_plugin.reconciliate_processes(traceback.format_exc())
            self._on_exception()
            # reset bookkeeping
            self.state.stage = None
            self.on_exception(exception)
            # shutdown workers
            self._data_connector.teardown()
            raise

    def fit(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        train_dataloader=None,  # TODO: remove with 1.6
        ckpt_path: Optional[str] = None,
    ) -> None:
        r"""
        Runs the full optimization routine.

        Args:
            model: Model to fit.

            train_dataloaders: A collection of :class:`torch.utils.data.DataLoader` or a
                :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying training samples.
                In the case of multiple dataloaders, please see this :ref:`page <multiple-training-dataloaders>`.

            val_dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.

            ckpt_path: Path/URL of the checkpoint from which training is resumed. If there is
                no checkpoint file at the path, an exception is raised. If resuming from mid-epoch checkpoint,
                training will start from the beginning of the next epoch.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
        """
        if train_dataloader is not None:
            rank_zero_deprecation(
                "`trainer.fit(train_dataloader)` is deprecated in v1.4 and will be removed in v1.6."
                " Use `trainer.fit(train_dataloaders)` instead. HINT: added 's'"
            )
            train_dataloaders = train_dataloader
        self._call_and_handle_interrupt(
            self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
        )

    def _fit_impl(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        Trainer._log_api_event("fit")

        self.state.fn = TrainerFn.FITTING
        self.state.status = TrainerStatus.RUNNING
        self.training = True

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloaders, LightningDataModule):
            datamodule = train_dataloaders
            train_dataloaders = None
        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if (train_dataloaders is not None or val_dataloaders is not None) and datamodule is not None:
            raise MisconfigurationException(
                "You cannot pass `train_dataloader` or `val_dataloaders` to `trainer.fit(datamodule=...)`"
            )

        # links data to the trainer
        self._data_connector.attach_data(
            model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, datamodule=datamodule
        )

        # TODO: ckpt_path only in v1.7
        ckpt_path = ckpt_path or self.resume_from_checkpoint
        self._run(model, ckpt_path=ckpt_path)

        assert self.state.stopped
        self.training = False

    def validate(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
        val_dataloaders=None,  # TODO: remove with 1.6
    ) -> _EVALUATE_OUTPUT:
        r"""
        Perform one evaluation epoch over the validation set.

        Args:
            model: The model to validate.

            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them,
                or a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying validation samples.

            ckpt_path: Either ``best`` or path to the checkpoint you wish to validate.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.

            verbose: If True, prints the validation results.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

        Returns:
            List of dictionaries with metrics logged during the validation phase, e.g., in model- or callback hooks
            like :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step`,
            :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_epoch_end`, etc.
            The length of the list corresponds to the number of validation dataloaders used.
        """
        if val_dataloaders is not None:
            rank_zero_deprecation(
                "`trainer.validate(val_dataloaders)` is deprecated in v1.4 and will be removed in v1.6."
                " Use `trainer.validate(dataloaders)` instead."
            )
            dataloaders = val_dataloaders
        return self._call_and_handle_interrupt(self._validate_impl, model, dataloaders, ckpt_path, verbose, datamodule)

    def _validate_impl(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ) -> _EVALUATE_OUTPUT:
        # --------------------
        # SETUP HOOK
        # --------------------
        Trainer._log_api_event("validate")
        self.verbose_evaluate = verbose

        self.state.fn = TrainerFn.VALIDATING
        self.state.status = TrainerStatus.RUNNING
        self.validating = True

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(dataloaders, LightningDataModule):
            datamodule = dataloaders
            dataloaders = None
        # If you supply a datamodule you can't supply val_dataloaders
        if dataloaders is not None and datamodule:
            raise MisconfigurationException("You cannot pass both `trainer.validate(dataloaders=..., datamodule=...)`")

        model_provided = model is not None
        model = model or self.lightning_module
        if model is None:
            raise MisconfigurationException(
                "`model` must be provided to `trainer.validate()` when it hasn't been passed in a previous run"
            )

        # links data to the trainer
        self._data_connector.attach_data(model, val_dataloaders=dataloaders, datamodule=datamodule)

        self.validated_ckpt_path = self.__set_ckpt_path(
            ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        )

        # run validate
        results = self._run(model, ckpt_path=self.validated_ckpt_path)

        assert self.state.stopped
        self.validating = False

        return results

    def test(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
        test_dataloaders=None,  # TODO: remove with 1.6
    ) -> _EVALUATE_OUTPUT:
        r"""
        Perform one evaluation epoch over the test set.
        It's separated from fit to make sure you never run on your test set until you want to.

        Args:
            model: The model to test.

            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them,
                or a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying test samples.

            ckpt_path: Either ``best`` or path to the checkpoint you wish to test.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.

            verbose: If True, prints the test results.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

        Returns:
            List of dictionaries with metrics logged during the test phase, e.g., in model- or callback hooks
            like :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step`,
            :meth:`~pytorch_lightning.core.lightning.LightningModule.test_epoch_end`, etc.
            The length of the list corresponds to the number of test dataloaders used.
        """
        if test_dataloaders is not None:
            rank_zero_deprecation(
                "`trainer.test(test_dataloaders)` is deprecated in v1.4 and will be removed in v1.6."
                " Use `trainer.test(dataloaders)` instead."
            )
            dataloaders = test_dataloaders
        return self._call_and_handle_interrupt(self._test_impl, model, dataloaders, ckpt_path, verbose, datamodule)

    def _test_impl(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ) -> _EVALUATE_OUTPUT:
        # --------------------
        # SETUP HOOK
        # --------------------
        Trainer._log_api_event("test")
        self.verbose_evaluate = verbose

        self.state.fn = TrainerFn.TESTING
        self.state.status = TrainerStatus.RUNNING
        self.testing = True

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(dataloaders, LightningDataModule):
            datamodule = dataloaders
            dataloaders = None
        # If you supply a datamodule you can't supply test_dataloaders
        if dataloaders is not None and datamodule:
            raise MisconfigurationException("You cannot pass both `trainer.test(dataloaders=..., datamodule=...)`")

        model_provided = model is not None
        model = model or self.lightning_module
        if model is None:
            raise MisconfigurationException(
                "`model` must be provided to `trainer.test()` when it hasn't been passed in a previous run"
            )

        # links data to the trainer
        self._data_connector.attach_data(model, test_dataloaders=dataloaders, datamodule=datamodule)

        self.tested_ckpt_path = self.__set_ckpt_path(
            ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        )

        # run test
        results = self._run(model, ckpt_path=self.tested_ckpt_path)

        assert self.state.stopped
        self.testing = False

        return results

    def predict(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        datamodule: Optional[LightningDataModule] = None,
        return_predictions: Optional[bool] = None,
        ckpt_path: Optional[str] = None,
    ) -> Optional[_PREDICT_OUTPUT]:
        r"""
        Run inference on your data.
        This will call the model forward function to compute predictions. Useful to perform distributed
        and batched predictions. Logging is disabled in the predict hooks.

        Args:
            model: The model to predict with.

            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them,
                or a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying prediction samples.

            datamodule: The datamodule with a predict_dataloader method that returns one or more dataloaders.

            return_predictions: Whether to return predictions.
                ``True`` by default except when an accelerator that spawns processes is used (not supported).

            ckpt_path: Either ``best`` or path to the checkpoint you wish to predict.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.

        Returns:
            Returns a list of dictionaries, one for each provided dataloader containing their respective predictions.
        """
        return self._call_and_handle_interrupt(
            self._predict_impl, model, dataloaders, datamodule, return_predictions, ckpt_path
        )

    def _predict_impl(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        datamodule: Optional[LightningDataModule] = None,
        return_predictions: Optional[bool] = None,
        ckpt_path: Optional[str] = None,
    ) -> Optional[_PREDICT_OUTPUT]:
        # --------------------
        # SETUP HOOK
        # --------------------
        Trainer._log_api_event("predict")

        self.state.fn = TrainerFn.PREDICTING
        self.state.status = TrainerStatus.RUNNING
        self.predicting = True

        self.predict_loop.return_predictions = return_predictions

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(dataloaders, LightningDataModule):
            datamodule = dataloaders
            dataloaders = None
        if dataloaders is not None and datamodule:
            raise MisconfigurationException("You cannot pass both `trainer.predict(dataloaders=..., datamodule=...)`")

        model_provided = model is not None
        model = model or self.lightning_module
        if model is None:
            raise MisconfigurationException(
                "`model` must be provided to `trainer.predict()` when it hasn't been passed in a previous run"
            )

        # links data to the trainer
        self._data_connector.attach_data(model, predict_dataloaders=dataloaders, datamodule=datamodule)

        self.predicted_ckpt_path = self.__set_ckpt_path(
            ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        )

        results = self._run(model, ckpt_path=self.predicted_ckpt_path)

        assert self.state.stopped
        self.predicting = False

        return results

    def tune(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        scale_batch_size_kwargs: Optional[Dict[str, Any]] = None,
        lr_find_kwargs: Optional[Dict[str, Any]] = None,
        train_dataloader=None,  # TODO: remove with 1.6
    ) -> Dict[str, Optional[Union[int, _LRFinder]]]:
        r"""
        Runs routines to tune hyperparameters before training.

        Args:
            model: Model to tune.

            train_dataloaders: A collection of :class:`torch.utils.data.DataLoader` or a
                :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying training samples.
                In the case of multiple dataloaders, please see this :ref:`page <multiple-training-dataloaders>`.

            val_dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

            scale_batch_size_kwargs: Arguments for :func:`~pytorch_lightning.tuner.batch_size_scaling.scale_batch_size`

            lr_find_kwargs: Arguments for :func:`~pytorch_lightning.tuner.lr_finder.lr_find`
        """
        Trainer._log_api_event("tune")

        self.state.fn = TrainerFn.TUNING
        self.state.status = TrainerStatus.RUNNING
        self.tuning = True

        if train_dataloader is not None:
            rank_zero_deprecation(
                "`trainer.tune(train_dataloader)` is deprecated in v1.4 and will be removed in v1.6."
                " Use `trainer.tune(train_dataloaders)` instead. HINT: added 's'"
            )
            train_dataloaders = train_dataloader
        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloaders, LightningDataModule):
            datamodule = train_dataloaders
            train_dataloaders = None
        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if (train_dataloaders is not None or val_dataloaders is not None) and datamodule is not None:
            raise MisconfigurationException(
                "You cannot pass `train_dataloader` or `val_dataloaders` to `trainer.tune(datamodule=...)`"
            )

        # links data to the trainer
        self._data_connector.attach_data(
            model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, datamodule=datamodule
        )

        result = self.tuner._tune(model, scale_batch_size_kwargs=scale_batch_size_kwargs, lr_find_kwargs=lr_find_kwargs)

        assert self.state.stopped
        self.tuning = False

        return result

    def _restore_modules_and_callbacks(self, checkpoint_path: Optional[_PATH] = None) -> None:
        # restore modules after setup
        self.checkpoint_connector.resume_start(checkpoint_path)
        self.checkpoint_connector.restore_model()
        self.checkpoint_connector.restore_datamodule()
        if self.state.fn == TrainerFn.FITTING:
            # restore callback states
            self.checkpoint_connector.restore_callbacks()

    def _run(
        self, model: "pl.LightningModule", ckpt_path: Optional[str] = None
    ) -> Optional[Union[_EVALUATE_OUTPUT, _PREDICT_OUTPUT]]:
        # clean hparams
        if hasattr(model, "hparams"):
            parsing.clean_namespace(model.hparams)

        # attach model to the training type plugin
        self.training_type_plugin.connect(model)

        self._callback_connector._attach_model_callbacks()
        self._callback_connector._attach_model_logging_functions()

        verify_loop_configurations(self)

        # hook
        self._data_connector.prepare_data()

        # ----------------------------
        # SET UP TRAINING
        # ----------------------------
        self.call_hook("on_before_accelerator_backend_setup")
        self.accelerator.setup_environment()
        self._call_setup_hook()  # allow user to setup lightning_module in accelerator environment

        # check if we should delay restoring checkpoint till later
        if not self.training_type_plugin.restore_checkpoint_after_pre_dispatch:
            self._restore_modules_and_callbacks(ckpt_path)

        self._call_configure_sharded_model()  # allow user to setup in model sharded environment
        self.accelerator.setup(self)

        # ----------------------------
        # INSPECT THE CORE LOOPS
        # ----------------------------
        fr"""
             Lightning internal flow looks like this:
        {Trainer.fit} or {Trainer.test} or {Trainer.predict}  ||
                                |                             ||
                        create accelerator                    ||
                                |                             ||
                         {self._dispatch}                     ||
                                |                             ||  LIGHTNING
         {self.training_type_plugin.start_training}           ||
       or {self.training_type_plugin.start_evaluating}        ||
       or {self.training_type_plugin.start_predicting}        ||  FLOW
                                |                             ||
                         {self.run_stage}                     ||
                                |                             ||  DIRECTION
                        {self._run_train}                     ||
                     or {self._run_evaluate}                  ||
                     or {self._run_predict}                   ||
                                |                             ||
                             results                          \/
        This is used to guide readers to the core loops: train, test, predict.
        {self._run_predict} is the simplest to understand, use `Go to Definition` to read it :)
        Search for `start_training` or `start_evaluating` or `start_predicting` in
        `pytorch_lightning/plugins/training_type_plugin` to find accelerator dispatch functions.
        """

        # ----------------------------
        # TRAIN
        # ----------------------------

        # reset logger connector
        self.logger_connector.reset_results()
        self.logger_connector.reset_metrics()

        # hook
        if self.state.fn == TrainerFn.FITTING:
            self.call_hook("on_fit_start")

        # plugin will setup fitting (e.g. ddp will launch child processes)
        self._pre_dispatch()

        if self.training_type_plugin.restore_checkpoint_after_pre_dispatch:
            self._restore_modules_and_callbacks(ckpt_path)

        # restore optimizers, etc.
        self.checkpoint_connector.restore_training_state()

        self.checkpoint_connector.resume_end()

        # dispatch `start_training` or `start_evaluating` or `start_predicting`
        self._dispatch()

        # plugin will finalized fitting (e.g. ddp_spawn will load trained model)
        self._post_dispatch()

        # ----------------------------
        # POST-Training CLEAN UP
        # ----------------------------
        # hook
        if self.state.fn == TrainerFn.FITTING:
            self.call_hook("on_fit_end")

        # teardown if necessary (similar calls for spawn plugins are excluded as they have
        # been included at the end of `new_process` functions)
        if not isinstance(self.training_type_plugin, DDPSpawnPlugin):
            self._call_teardown_hook()

        if self.state.status != TrainerStatus.INTERRUPTED:
            self.state.status = TrainerStatus.FINISHED
        self.state.stage = None

        return self.training_type_plugin.results

    def _pre_dispatch(self):
        self.accelerator.pre_dispatch(self)
        self._log_hyperparams()

    def _log_hyperparams(self) -> None:
        # log hyper-parameters
        hparams_initial = None

        if self.logger is not None:
            # save exp to get started (this is where the first experiment logs are written)
            datamodule_log_hyperparams = self.datamodule._log_hyperparams if self.datamodule is not None else False

            if self.lightning_module._log_hyperparams and datamodule_log_hyperparams:
                datamodule_hparams = self.datamodule.hparams_initial
                lightning_hparams = self.lightning_module.hparams_initial
                inconsistent_keys = []
                for key in lightning_hparams.keys() & datamodule_hparams.keys():
                    lm_val, dm_val = lightning_hparams[key], datamodule_hparams[key]
                    if type(lm_val) != type(dm_val):
                        inconsistent_keys.append(key)
                    elif isinstance(lm_val, torch.Tensor) and id(lm_val) != id(dm_val):
                        inconsistent_keys.append(key)
                    elif lm_val != dm_val:
                        inconsistent_keys.append(key)
                if inconsistent_keys:
                    raise MisconfigurationException(
                        f"Error while merging hparams: the keys {inconsistent_keys} are present "
                        "in both the LightningModule's and LightningDataModule's hparams "
                        "but have different values."
                    )
                hparams_initial = {**lightning_hparams, **datamodule_hparams}
            elif self.lightning_module._log_hyperparams:
                hparams_initial = self.lightning_module.hparams_initial
            elif datamodule_log_hyperparams:
                hparams_initial = self.datamodule.hparams_initial

            if hparams_initial is not None:
                self.logger.log_hyperparams(hparams_initial)
            self.logger.log_graph(self.lightning_module)
            self.logger.save()

    def _post_dispatch(self):
        self.accelerator.post_dispatch(self)
        # these `teardown` calls are here instead of in `_call_teardown_hook` since they are internal teardowns
        # which need to happen before.
        self.accelerator.teardown()
        self._data_connector.teardown()
        self._active_loop.teardown()
        self.logger_connector.teardown()
        self.signal_connector.teardown()

    def _dispatch(self):
        if self.evaluating:
            self.training_type_plugin.start_evaluating(self)
        elif self.predicting:
            self.training_type_plugin.start_predicting(self)
        else:
            self.training_type_plugin.start_training(self)

    def run_stage(self):
        self.accelerator.dispatch(self)
        self.__setup_profiler()

        if self.evaluating:
            return self._run_evaluate()
        if self.predicting:
            return self._run_predict()
        return self._run_train()

    def _pre_training_routine(self):
        # wait for all to join if on distributed
        self.training_type_plugin.barrier("setup_training")

        # register signals
        self.signal_connector.register_signal_handlers()

        # --------------------------
        # Pre-train
        # --------------------------
        self.call_hook("on_pretrain_routine_start")

        self.call_hook("on_pretrain_routine_end")

    def _run_train(self) -> None:
        self._pre_training_routine()

        if not self.is_global_zero and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        self._run_sanity_check(self.lightning_module)

        # enable train mode
        self.model.train()
        torch.set_grad_enabled(True)

        self.fit_loop.trainer = self
        with torch.autograd.set_detect_anomaly(self._detect_anomaly):
            self.fit_loop.run()

    def _run_evaluate(self) -> _EVALUATE_OUTPUT:
        if not self.is_global_zero and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        assert self.evaluating

        # reload dataloaders
        self._evaluation_loop._reload_evaluation_dataloaders()

        # reset trainer on this loop and all child loops in case user connected a custom loop
        self._evaluation_loop.trainer = self

        with self.profiler.profile(f"run_{self.state.stage}_evaluation"), torch.no_grad():
            eval_loop_results = self._evaluation_loop.run()

        # remove the tensors from the eval results
        for result in eval_loop_results:
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v.cpu().item()

        return eval_loop_results

    def _run_predict(self) -> Optional[_PREDICT_OUTPUT]:
        self.reset_predict_dataloader(self.lightning_module)
        # reset trainer on this loop and all child loops in case user connected a custom loop
        self.predict_loop.trainer = self
        with torch.no_grad():
            return self.predict_loop.run()

    def _run_sanity_check(self, ref_model):
        using_val_step = self._data_connector._val_dataloader_source.is_defined() and is_overridden(
            "validation_step", ref_model
        )
        should_sanity_check = using_val_step and self.num_sanity_val_steps > 0 and self.limit_val_batches > 0

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        if should_sanity_check:
            stage = self.state.stage
            self.sanity_checking = True

            # reset logger connector
            self.logger_connector.reset_results()
            self.logger_connector.reset_metrics()

            self.call_hook("on_sanity_check_start")

            # reload dataloaders
            self._evaluation_loop._reload_evaluation_dataloaders()

            # run eval step
            with torch.no_grad():
                self._evaluation_loop.run()

            self.call_hook("on_sanity_check_end")

            # reset logger connector
            self.logger_connector.reset_results()
            self.logger_connector.reset_metrics()

            # reset the seed to what it was before sanity check
            # prevents sanity check to affect random sampling in training
            reset_seed()

            # restore the previous stage when the sanity check if finished
            self.state.stage = stage

    def __set_ckpt_path(self, ckpt_path: Optional[str], model_provided: bool, model_connected: bool) -> Optional[str]:
        if model_provided and ckpt_path is None:
            # use passed model to function without loading weights
            return

        fn = self.state.fn.value

        if model_connected and ckpt_path is None:
            rank_zero_warn(
                f"`.{fn}(ckpt_path=None)` was called without a model."
                " The best model of the previous `fit` call will be used."
                f" You can pass `{fn}(ckpt_path='best')` to use and best model"
                " checkpoint and avoid this warning or"
                " `ckpt_path=trainer.checkpoint_callback.last_model_path` to use the last model."
            )
            ckpt_path = "best"

        if ckpt_path == "best":
            # if user requests the best checkpoint but we don't have it, error
            if not self.checkpoint_callback:
                raise MisconfigurationException(
                    f'`.{fn}(ckpt_path="best")` is set but `ModelCheckpoint` is not configured.'
                )
            if not self.checkpoint_callback.best_model_path:
                if self.fast_dev_run:
                    raise MisconfigurationException(
                        f"You cannot execute `.{fn}()` with `fast_dev_run=True` unless you do"
                        f" `.{fn}(ckpt_path=PATH)` as no checkpoint path was generated during fitting."
                    )
                raise MisconfigurationException(
                    f'`.{fn}(ckpt_path="best")` is set but `ModelCheckpoint` is not configured to save the best model.'
                )
            # load best weights
            ckpt_path = self.checkpoint_callback.best_model_path

        if not ckpt_path:
            raise MisconfigurationException(
                f"`.{fn}()` found no path for the best weights: {ckpt_path!r}. Please"
                f" specify a path for a checkpoint `.{fn}(ckpt_path=PATH)`"
            )
        return ckpt_path

    def _call_setup_hook(self) -> None:
        fn = self.state.fn._setup_fn

        self.training_type_plugin.barrier("pre_setup")

        if self.datamodule is not None:
            self.datamodule.setup(stage=fn)
        self.call_hook("setup", stage=fn)

        self.training_type_plugin.barrier("post_setup")

    def _call_configure_sharded_model(self) -> None:
        with self.accelerator.model_sharded_context():
            self._handle_meta_model()
            self.call_hook("configure_sharded_model")
            self.call_hook("on_configure_sharded_model")

    def _handle_meta_model(self) -> None:
        if not is_on_meta_device(self.lightning_module):
            return

        if isinstance(self.training_type_plugin, DDPSpawnPlugin):
            raise MisconfigurationException("LightningModule on meta device isn't supported with spawn.")

        materialize_module(self.lightning_module)
        # the trainer reference is lost during materialization
        self.lightning_module.trainer = proxy(self)

    def _call_teardown_hook(self) -> None:
        fn = self.state.fn._setup_fn

        if self.datamodule is not None:
            self.datamodule.teardown(stage=fn)

        self.call_hook("teardown", stage=fn)

        self.lightning_module._current_fx_name = None
        self.lightning_module._current_dataloader_idx = None
        # these could have become stale if metrics are defined in `setup`
        self.lightning_module._metric_attributes = None

        # todo: TPU 8 cores hangs in flush with TensorBoard. Might do for all loggers.
        # It might be related to xla tensors blocked when moving the cpu kill loggers.
        if self.logger is not None:
            self.logger.finalize("success")

        # summarize profile results
        self.profiler.describe()

    def call_hook(
        self, hook_name: str, *args: Any, pl_module: Optional["pl.LightningModule"] = None, **kwargs: Any
    ) -> Any:
        pl_module = self.lightning_module or pl_module
        if pl_module:
            prev_fx_name = pl_module._current_fx_name
            pl_module._current_fx_name = hook_name

        # always profile hooks
        with self.profiler.profile(hook_name):

            # first call trainer hook
            callback_fx = getattr(self, hook_name, None)
            if callable(callback_fx):
                callback_fx(*args, **kwargs)

            # next call hook in lightningModule
            output = None
            model_fx = getattr(pl_module, hook_name, None)
            if callable(model_fx):
                output = model_fx(*args, **kwargs)

            # *Bad code alert*
            # The `Accelerator` mostly calls the `TrainingTypePlugin` but some of those calls are deprecated.
            # The following logic selectively chooses which hooks are called on each object.
            # In the case of `setup` and `teardown`, the hooks on the `LightningModule` should not call the hooks of the
            # same name in these objects as they are meant to be managed outside of the `LightningModule` lifecycle.
            # All of this should be fixed by #8506

            # call the accelerator hook
            if hook_name in ("on_train_start",) and hasattr(self.accelerator, hook_name):
                accelerator_hook = getattr(self.accelerator, hook_name)
                accelerator_output = accelerator_hook(*args, **kwargs)
                # Rely on the accelerator output if lightningModule hook returns nothing
                # Required for cases such as DataParallel where we reduce the output for the user
                # todo: move this data parallel logic into the data parallel plugin
                output = accelerator_output if output is None else output

            # call the ttp hook
            if hook_name not in ("setup", "teardown", "on_train_start") and hasattr(
                self.training_type_plugin, hook_name
            ):
                ttp_hook = getattr(self.training_type_plugin, hook_name)
                ttp_output = ttp_hook(*args, **kwargs)
                output = ttp_output if output is None else output

        if pl_module:
            # restore current_fx when nested context
            pl_module._current_fx_name = prev_fx_name

        return output

    @staticmethod
    def _parse_devices(
        gpus: Optional[Union[List[int], str, int]],
        auto_select_gpus: bool,
        tpu_cores: Optional[Union[List[int], str, int]],
    ) -> Tuple[Optional[List[int]], Optional[Union[List[int], int]]]:
        if auto_select_gpus and isinstance(gpus, int):
            gpus = pick_multiple_gpus(gpus)

        # TODO (@seannaren, @kaushikb11): Include IPU parsing logic here
        gpu_ids = device_parser.parse_gpu_ids(gpus)
        tpu_cores = device_parser.parse_tpu_cores(tpu_cores)
        return gpu_ids, tpu_cores

    @staticmethod
    def _log_api_event(event: str) -> None:
        torch._C._log_api_usage_once("lightning.trainer." + event)

    def __init_profiler(self, profiler: Optional[Union[BaseProfiler, str]]) -> None:
        if isinstance(profiler, str):
            PROFILERS = {
                "simple": SimpleProfiler,
                "advanced": AdvancedProfiler,
                "pytorch": PyTorchProfiler,
                "xla": XLAProfiler,
            }
            profiler = profiler.lower()
            if profiler not in PROFILERS:
                raise MisconfigurationException(
                    "When passing string value for the `profiler` parameter of `Trainer`,"
                    f" it can only be one of {list(PROFILERS.keys())}"
                )
            profiler_class = PROFILERS[profiler]
            profiler = profiler_class()
        self.profiler: BaseProfiler = profiler or PassThroughProfiler()

    def __setup_profiler(self) -> None:
        local_rank = self.local_rank if self.world_size > 1 else None
        self.profiler._lightning_module = proxy(self.lightning_module)
        self.profiler.setup(stage=self.state.fn._setup_fn, local_rank=local_rank, log_dir=self.log_dir)

    def _log_device_info(self) -> None:
        rank_zero_info(f"GPU available: {torch.cuda.is_available()}, used: {self._device_type == DeviceType.GPU}")

        num_tpu_cores = self.tpu_cores if self.tpu_cores is not None and self._device_type == DeviceType.TPU else 0
        rank_zero_info(f"TPU available: {_TPU_AVAILABLE}, using: {num_tpu_cores} TPU cores")

        num_ipus = self.ipus if self.ipus is not None else 0
        rank_zero_info(f"IPU available: {_IPU_AVAILABLE}, using: {num_ipus} IPUs")

        if torch.cuda.is_available() and self._device_type != DeviceType.GPU:
            rank_zero_warn(
                "GPU available but not used. Set the gpus flag in your trainer `Trainer(gpus=1)` or script `--gpus=1`."
            )

        if _TPU_AVAILABLE and self._device_type != DeviceType.TPU:
            rank_zero_warn(
                "TPU available but not used. Set the `tpu_cores` flag in your trainer"
                " `Trainer(tpu_cores=8)` or script `--tpu_cores=8`."
            )

        if _IPU_AVAILABLE and self._device_type != DeviceType.IPU and not isinstance(self.accelerator, IPUAccelerator):
            rank_zero_warn(
                "IPU available but not used. Set the `ipus` flag in your trainer"
                " `Trainer(ipus=8)` or script `--ipus=8`."
            )

    def _on_exception(self):
        if not _fault_tolerant_training():
            return
        # save a checkpoint for fault tolerant training. we don't use `log_dir` to minimize the chances of failure.
        file_path = os.path.join(self.default_root_dir, ".pl_auto_save.ckpt")
        self.save_checkpoint(file_path)

    """
    Accelerator properties
    """

    @property
    def accelerator(self) -> Accelerator:
        return self._accelerator_connector.accelerator

    @property
    def training_type_plugin(self) -> TrainingTypePlugin:
        return self.accelerator.training_type_plugin

    @property
    def precision_plugin(self) -> PrecisionPlugin:
        return self.accelerator.precision_plugin

    @property
    def global_rank(self) -> int:
        return self.training_type_plugin.global_rank

    @property
    def local_rank(self) -> int:
        # some training types define a local rank
        return getattr(self.training_type_plugin, "local_rank", 0)

    @property
    def node_rank(self) -> int:
        # some training types define a node rank
        return getattr(self.training_type_plugin, "node_rank", 0)

    @property
    def world_size(self) -> int:
        # some training types define a world size
        return getattr(self.training_type_plugin, "world_size", 1)

    @property
    def should_rank_save_checkpoint(self) -> bool:
        return self.training_type_plugin.should_rank_save_checkpoint

    @property
    def _distrib_type(self) -> DistributedType:
        return self._accelerator_connector._distrib_type

    @property
    def _device_type(self) -> DeviceType:
        return self._accelerator_connector._device_type

    @property
    def num_nodes(self) -> int:
        return self._accelerator_connector.num_nodes

    @property
    def num_processes(self) -> int:
        return self._accelerator_connector.num_processes

    @property
    def root_gpu(self) -> Optional[int]:
        return self._accelerator_connector.root_gpu

    @property
    def tpu_cores(self) -> int:
        return self._accelerator_connector.tpu_cores

    @property
    def ipus(self) -> int:
        return self._accelerator_connector.num_ipus

    @property
    def num_gpus(self) -> int:
        return self._accelerator_connector.num_gpus

    @property
    def devices(self) -> Optional[Union[List[int], str, int]]:
        return self._accelerator_connector.devices

    @property
    def data_parallel_device_ids(self) -> Optional[List[int]]:
        return self._accelerator_connector.parallel_device_ids

    @property
    def lightning_module(self) -> "pl.LightningModule":
        return self.accelerator.lightning_module

    @property
    def optimizers(self) -> List[Optimizer]:
        return self.accelerator.optimizers

    @optimizers.setter
    def optimizers(self, new_optims: Optional[List[Optimizer]]) -> None:
        # Necessary to rewrap optimizers to lightning
        # They will be re-created when accessing
        # the `lightning_optimizers` trainer property
        self._lightning_optimizers = None

        self.accelerator.optimizers = new_optims

    @property
    def lr_schedulers(self) -> List[LRSchedulerTypeUnion]:
        return self.accelerator.lr_schedulers

    @lr_schedulers.setter
    def lr_schedulers(self, new_schedulers: List[LRSchedulerTypeUnion]) -> None:
        self.accelerator.lr_schedulers = new_schedulers

    @property
    def optimizer_frequencies(self) -> list:
        return self.accelerator.optimizer_frequencies

    @optimizer_frequencies.setter
    def optimizer_frequencies(self, new_freqs: list) -> None:
        self.accelerator.optimizer_frequencies = new_freqs

    @property
    def amp_backend(self) -> Optional[str]:
        return self.accelerator.amp_backend

    @property
    def precision(self) -> Union[str, int]:
        return self.accelerator.precision

    @property
    def scaler(self):
        return self.accelerator.scaler

    @property
    def gpus(self) -> Optional[Union[List[int], str, int]]:
        return self._accelerator_connector.gpus

    @property
    def model(self) -> torch.nn.Module:
        """The LightningModule, but possibly wrapped into DataParallel or DistributedDataParallel.

        To access the pure LightningModule, use
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.lightning_module` instead.
        """
        return self.accelerator.model

    @model.setter
    def model(self, model: torch.nn.Module) -> None:
        """Setter for the model, pass-through to accelerator and plugin where the model reference is stored. Used
        by the Tuner to reset the state of Trainer and Accelerator.

        Args:
            model: The LightningModule, possibly wrapped into DataParallel or DistributedDataParallel, depending
                on the backend.
        """
        self.accelerator.model = model

    """
    General properties
    """

    @property
    def log_dir(self) -> Optional[str]:
        if self.logger is None:
            dirpath = self.default_root_dir
        elif isinstance(self.logger, TensorBoardLogger):
            dirpath = self.logger.log_dir
        elif isinstance(self.logger, LoggerCollection):
            dirpath = self.default_root_dir
        else:
            dirpath = self.logger.save_dir

        dirpath = self.training_type_plugin.broadcast(dirpath)
        return dirpath

    @property
    def use_amp(self) -> bool:
        return self.precision == 16

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def slurm_job_id(self) -> Optional[int]:
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            try:
                job_id = int(job_id)
            except ValueError:
                job_id = None

        # in interactive mode, don't make logs use the same job id
        in_slurm_interactive_mode = os.environ.get("SLURM_JOB_NAME") == "bash"
        if in_slurm_interactive_mode:
            job_id = None
        return job_id

    @property
    def lightning_optimizers(self) -> List[LightningOptimizer]:
        if self._lightning_optimizers is None:
            self.convert_to_lightning_optimizers()
        return self._lightning_optimizers

    @property
    def distributed_sampler_kwargs(self) -> Optional[dict]:
        if isinstance(self.training_type_plugin, ParallelPlugin):
            return self.training_type_plugin.distributed_sampler_kwargs

    @property
    def data_parallel(self) -> bool:
        return self._distrib_type in (
            DistributedType.DP,
            DistributedType.DDP,
            DistributedType.DDP_SPAWN,
            DistributedType.DDP2,
        )

    @property
    def progress_bar_callback(self) -> Optional[ProgressBarBase]:
        return self._progress_bar_callback

    @property
    def progress_bar_dict(self) -> dict:
        """Read-only for progress bar metrics."""
        rank_zero_deprecation(
            "`trainer.progress_bar_dict` is deprecated in v1.5 and will be removed in v1.7."
            " Use `ProgressBarBase.get_metrics` instead."
        )
        ref_model = self.lightning_module
        ref_model = cast(pl.LightningModule, ref_model)
        if self.progress_bar_callback:
            return self.progress_bar_callback.get_metrics(self, ref_model)
        return self.progress_bar_metrics

    @property
    def _should_reload_dl_epoch(self) -> bool:
        """Check if dataloader should be reloaded in the current epoch."""
        n_epochs = self.reload_dataloaders_every_n_epochs
        return n_epochs and (not self.current_epoch % n_epochs)

    @property
    def disable_validation(self) -> bool:
        """Check if validation is disabled during training."""
        rank_zero_deprecation(
            "`trainer.disable_validation` is deprecated in v1.4 and will be removed in v1.6."
            " Use `not trainer.enable_validation` instead."
        )
        return not self.enable_validation

    @property
    def enable_validation(self) -> bool:
        """Check if we should run validation during training."""
        model_ref = self.lightning_module
        val_loop_enabled = is_overridden("validation_step", model_ref) and self.limit_val_batches > 0
        return val_loop_enabled

    @property
    def default_root_dir(self) -> str:
        """The default location to save artifacts of loggers, checkpoints etc.

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

    @property
    def early_stopping_callback(self) -> Optional[EarlyStopping]:
        """The first :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` callback in the
        Trainer.callbacks list, or ``None`` if it doesn't exist."""
        callbacks = self.early_stopping_callbacks
        return callbacks[0] if len(callbacks) > 0 else None

    @property
    def early_stopping_callbacks(self) -> List[EarlyStopping]:
        """A list of all instances of :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` found in
        the Trainer.callbacks list."""
        return [c for c in self.callbacks if isinstance(c, EarlyStopping)]

    @property
    def prediction_writer_callbacks(self) -> List[BasePredictionWriter]:
        """A list of all instances of :class:`~pytorch_lightning.callbacks.prediction_writer.BasePredictionWriter`
        found in the Trainer.callbacks list."""
        return [cb for cb in self.callbacks if isinstance(cb, BasePredictionWriter)]

    @property
    def checkpoint_callback(self) -> Optional[ModelCheckpoint]:
        """The first :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` callback in the
        Trainer.callbacks list, or ``None`` if it doesn't exist."""
        callbacks = self.checkpoint_callbacks
        return callbacks[0] if len(callbacks) > 0 else None

    @property
    def checkpoint_callbacks(self) -> List[ModelCheckpoint]:
        """A list of all instances of :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` found
        in the Trainer.callbacks list."""
        return [c for c in self.callbacks if isinstance(c, ModelCheckpoint)]

    @property
    def resume_from_checkpoint(self) -> Optional[Union[str, Path]]:
        resume_from_checkpoint = self.checkpoint_connector.resume_from_checkpoint_fit_path
        if resume_from_checkpoint is not None:
            rank_zero_deprecation(
                "`trainer.resume_from_checkpoint` is deprecated in v1.5 and will be removed in v1.7."
                " Specify the fit checkpoint path with `trainer.fit(ckpt_path=)` instead."
            )

        return resume_from_checkpoint

    def save_checkpoint(self, filepath: _PATH, weights_only: bool = False) -> None:
        self.checkpoint_connector.save_checkpoint(filepath, weights_only)

    """
    Parsing properties
    """

    @classmethod
    def default_attributes(cls) -> dict:
        init_signature = inspect.signature(cls)
        return {k: v.default for k, v in init_signature.parameters.items()}

    @classmethod
    def get_deprecated_arg_names(cls) -> List:
        """Returns a list with deprecated Trainer arguments."""
        depr_arg_names = []
        for name, val in cls.__dict__.items():
            if name.startswith("DEPRECATED") and isinstance(val, (tuple, list)):
                depr_arg_names.extend(val)
        return depr_arg_names

    @classmethod
    def from_argparse_args(cls: Any, args: Union[Namespace, ArgumentParser], **kwargs) -> Any:
        return from_argparse_args(cls, args, **kwargs)

    @classmethod
    def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        return parse_argparser(cls, arg_parser)

    @classmethod
    def match_env_arguments(cls) -> Namespace:
        return parse_env_variables(cls)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        return add_argparse_args(cls, parent_parser, **kwargs)

    """
    State properties
    """

    @property
    def interrupted(self) -> bool:
        return self.state.status == TrainerStatus.INTERRUPTED

    @property
    def training(self) -> bool:
        return self.state.stage == RunningStage.TRAINING

    @training.setter
    def training(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.TRAINING
        elif self.training:
            self.state.stage = None

    @property
    def testing(self) -> bool:
        return self.state.stage == RunningStage.TESTING

    @testing.setter
    def testing(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.TESTING
        elif self.testing:
            self.state.stage = None

    @property
    def predicting(self) -> bool:
        return self.state.stage == RunningStage.PREDICTING

    @predicting.setter
    def predicting(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.PREDICTING
        elif self.predicting:
            self.state.stage = None

    @property
    def tuning(self) -> bool:
        return self.state.stage == RunningStage.TUNING

    @tuning.setter
    def tuning(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.TUNING
        elif self.tuning:
            self.state.stage = None

    @property
    def validating(self) -> bool:
        return self.state.stage == RunningStage.VALIDATING

    @validating.setter
    def validating(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.VALIDATING
        elif self.validating:
            self.state.stage = None

    @property
    def evaluating(self) -> bool:
        return self.state.stage and self.state.stage.evaluating

    @property
    def sanity_checking(self) -> bool:
        return self.state.stage == RunningStage.SANITY_CHECKING

    @sanity_checking.setter
    def sanity_checking(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.SANITY_CHECKING
        elif self.sanity_checking:
            self.state.stage = None

    """
    Loop properties
    """

    @property
    def global_step(self) -> int:
        return self.fit_loop.global_step

    @property
    def current_epoch(self) -> int:
        return self.fit_loop.current_epoch

    @property
    def max_epochs(self) -> int:
        return self.fit_loop.max_epochs

    @property
    def min_epochs(self) -> Optional[int]:
        return self.fit_loop.min_epochs

    @property
    def max_steps(self) -> int:
        return self.fit_loop.max_steps

    @property
    def min_steps(self) -> Optional[int]:
        return self.fit_loop.min_steps

    @property
    def is_last_batch(self) -> bool:
        return self.fit_loop.epoch_loop.batch_progress.is_last_batch

    @property
    def fit_loop(self) -> FitLoop:
        return self._fit_loop

    @fit_loop.setter
    def fit_loop(self, loop: FitLoop):
        """Attach a custom fit loop to this Trainer.

        It will run with
        :meth:`~pytorch_lighting.trainer.trainer.Trainer.fit`.
        """
        loop.trainer = self
        self._fit_loop = loop

    @property
    def validate_loop(self) -> EvaluationLoop:
        return self._validate_loop

    @validate_loop.setter
    def validate_loop(self, loop: EvaluationLoop):
        """Attach a custom validation loop to this Trainer.

        It will run with
        :meth:`~pytorch_lighting.trainer.trainer.Trainer.validate`. Note that this loop is different from the one
        running during training inside the :meth:`pytorch_lightning.trainer.trainer.Trainer.fit` call.
        """
        loop.trainer = self
        self._validate_loop = loop

    @property
    def test_loop(self) -> EvaluationLoop:
        return self._test_loop

    @test_loop.setter
    def test_loop(self, loop: EvaluationLoop):
        """Attach a custom test loop to this Trainer.

        It will run with
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.test`.
        """
        loop.trainer = self
        self._test_loop = loop

    @property
    def predict_loop(self) -> PredictionLoop:
        return self._predict_loop

    @predict_loop.setter
    def predict_loop(self, loop: PredictionLoop):
        """Attach a custom prediction loop to this Trainer.

        It will run with
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`.
        """
        loop.trainer = self
        self._predict_loop = loop

    @property
    def _evaluation_loop(self) -> EvaluationLoop:
        if self.state.fn in (TrainerFn.FITTING, TrainerFn.TUNING):
            return self.fit_loop.epoch_loop.val_loop
        if self.state.fn == TrainerFn.VALIDATING:
            return self.validate_loop
        if self.state.fn == TrainerFn.TESTING:
            return self.test_loop
        raise RuntimeError("The `Trainer._evaluation_loop` property isn't defined. Accessed outside of scope")

    @property
    def _active_loop(self) -> Optional[Union[FitLoop, EvaluationLoop, PredictionLoop]]:
        if self.training:
            return self.fit_loop
        if self.sanity_checking or self.evaluating:
            return self._evaluation_loop
        if self.predicting:
            return self.predict_loop

    """
    Logging properties
    """

    @property
    def callback_metrics(self) -> dict:
        return self.logger_connector.callback_metrics

    @property
    def logged_metrics(self) -> dict:
        return self.logger_connector.logged_metrics

    @property
    def progress_bar_metrics(self) -> dict:
        return self.logger_connector.progress_bar_metrics

    @property
    def _results(self) -> Optional[ResultCollection]:
        active_loop = self._active_loop
        if active_loop is not None:
            return active_loop._results

    def _exit_gracefully_on_signal(self) -> None:
        if _fault_tolerant_training() and self._terminate_gracefully:
            caller = inspect.stack()[1]
            class_name = caller[0].f_locals["self"].__class__.__name__
            raise ExitGracefullyException(f"Exiting gracefully on {class_name}:{caller.function}")

    @property
    def weights_summary(self) -> Optional[str]:
        rank_zero_deprecation("`Trainer.weights_summary` is deprecated in v1.5 and will be removed in v1.7.")
        return self._weights_summary

    @weights_summary.setter
    def weights_summary(self, val: Optional[str]) -> None:
        rank_zero_deprecation("Setting `Trainer.weights_summary` is deprecated in v1.5 and will be removed in v1.7.")
        self._weights_summary = val

    """
    Other
    """

    # TODO: refactor this so that it can be done in LightningOptimizer
    def __getstate__(self):
        # remove lightning_optimizers
        self._lightning_optimizers = None
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    @property
    def train_loop(self) -> FitLoop:
        rank_zero_deprecation(
            "`Trainer.train_loop` has been renamed to `Trainer.fit_loop` and will be removed in v1.6."
        )
        return self.fit_loop

    @property
    def terminate_on_nan(self) -> bool:
        rank_zero_deprecation("`Trainer.terminate_on_nan` is deprecated in v1.5 and will be removed in 1.7.")
        return self._terminate_on_nan

    @terminate_on_nan.setter
    def terminate_on_nan(self, val: bool) -> None:
        rank_zero_deprecation(
            f"Setting `Trainer.terminate_on_nan = {val}` is deprecated in v1.5 and will be removed in 1.7."
            f" Please set `Trainer(detect_anomaly={val})` instead."
        )
        self._terminate_on_nan = val  # : 212


def _determine_batch_limits(batches: Union[int, float], name: str) -> Union[int, float]:
    if 0 <= batches <= 1:
        return batches
    if batches > 1 and batches % 1.0 == 0:
        return int(batches)
    raise MisconfigurationException(
        f"You have passed invalid value {batches} for {name}, it has to be in [0.0, 1.0] or an int."
    )
