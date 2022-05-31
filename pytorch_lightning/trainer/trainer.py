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
import math
import os
import traceback
import warnings
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Type, Union
from weakref import proxy

import torch
from packaging.version import Version
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.accelerators import Accelerator, GPUAccelerator, HPUAccelerator, IPUAccelerator, TPUAccelerator
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint, ProgressBarBase
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import DummyLogger, LoggerCollection
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loops import PredictionLoop, TrainingEpochLoop
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.utilities import _parse_loop_limits, _reset_progress
from pytorch_lightning.plugins import (
    ApexMixedPrecisionPlugin,
    NativeMixedPrecisionPlugin,
    PLUGIN_INPUT,
    PrecisionPlugin,
)
from pytorch_lightning.plugins.environments.slurm_environment import SLURMEnvironment
from pytorch_lightning.profiler import (
    AdvancedProfiler,
    PassThroughProfiler,
    Profiler,
    PyTorchProfiler,
    SimpleProfiler,
    XLAProfiler,
)
from pytorch_lightning.strategies import ParallelStrategy, Strategy
from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy
from pytorch_lightning.trainer.callback_hook import TrainerCallbackHookMixin
from pytorch_lightning.trainer.configuration_validator import verify_loop_configurations
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from pytorch_lightning.trainer.connectors.signal_connector import SignalConnector
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.tuner.lr_finder import _LRFinder
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities import (
    _HPU_AVAILABLE,
    _IPU_AVAILABLE,
    _TPU_AVAILABLE,
    AMPType,
    GradClipAlgorithmType,
    parsing,
)
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.argparse import (
    _defaults_from_env_vars,
    add_argparse_args,
    from_argparse_args,
    parse_argparser,
    parse_env_variables,
)
from pytorch_lightning.utilities.auto_restart import _add_capture_metadata_collate
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.data import _auto_add_worker_init_fn, has_len_all_ranks
from pytorch_lightning.utilities.distributed import distributed_available
from pytorch_lightning.utilities.exceptions import ExitGracefullyException, MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from pytorch_lightning.utilities.meta import is_on_meta_device, materialize_module
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.seed import isolate_rng
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import (
    _EVALUATE_OUTPUT,
    _PATH,
    _PREDICT_OUTPUT,
    EVAL_DATALOADERS,
    LRSchedulerConfig,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
from pytorch_lightning.utilities.warnings import PossibleUserWarning

log = logging.getLogger(__name__)
# warnings to ignore in trainer
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead"
)


class Trainer(
    TrainerCallbackHookMixin,  # TODO: Remove in v1.8
    TrainerOptimizersMixin,  # TODO: Remove in v1.8
    TrainerDataLoadingMixin,  # TODO: Remove in v1.8
):
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
        num_processes: Optional[int] = None,
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
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        val_check_interval: Optional[Union[int, float]] = None,
        flush_logs_every_n_steps: Optional[int] = None,
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = None,
        strategy: Optional[Union[str, Strategy]] = None,
        sync_batchnorm: bool = False,
        precision: Union[int, str] = 32,
        enable_model_summary: bool = True,
        weights_summary: Optional[str] = "top",
        weights_save_path: Optional[str] = None,  # TODO: Remove in 1.8
        num_sanity_val_steps: int = 2,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[Profiler, str]] = None,
        benchmark: Optional[bool] = None,
        deterministic: Optional[bool] = None,
        reload_dataloaders_every_n_epochs: int = 0,
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
    ) -> None:
        r"""
        Customize every aspect of training via flags.

        Args:

            accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "auto")
                as well as custom accelerator instances.

                .. deprecated:: v1.5
                    Passing training strategies (e.g., 'ddp') to ``accelerator`` has been deprecated in v1.5.0
                    and will be removed in v1.7.0. Please use the ``strategy`` argument instead.

            accumulate_grad_batches: Accumulates grads every k batches or as set up in the dict.
                Default: ``None``.

            amp_backend: The mixed precision backend to use ("native" or "apex").
                Default: ``'native''``.

            amp_level: The optimization level to use (O1, O2, etc...). By default it will be set to "O2"
                if ``amp_backend`` is set to "apex".

            auto_lr_find: If set to True, will make trainer.tune() run a learning rate finder,
                trying to optimize initial learning for faster convergence. trainer.tune() method will
                set the suggested learning rate in self.lr or self.learning_rate in the LightningModule.
                To use a different key set a string instead of True with the key name.
                Default: ``False``.

            auto_scale_batch_size: If set to True, will `initially` run a batch size
                finder trying to find the largest batch size that fits into memory.
                The result will be stored in self.batch_size in the LightningModule.
                Additionally, can be set to either `power` that estimates the batch size through
                a power search or `binsearch` that estimates the batch size through a binary search.
                Default: ``False``.

            auto_select_gpus: If enabled and ``gpus`` or ``devices`` is an integer, pick available
                gpus automatically. This is especially useful when
                GPUs are configured to be in "exclusive mode", such
                that only one process at a time can access them.
                Default: ``False``.

            benchmark: The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to.
                The value for ``torch.backends.cudnn.benchmark`` set in the current session will be used
                (``False`` if not manually set). If :paramref:`~pytorch_lightning.trainer.Trainer.deterministic` is set
                to ``True``, this will default to ``False``. Override to manually set a different value.
                Default: ``None``.

            callbacks: Add a callback or list of callbacks.
                Default: ``None``.

            checkpoint_callback: If ``True``, enable checkpointing.
                Default: ``None``.

                .. deprecated:: v1.5
                    ``checkpoint_callback`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please consider using ``enable_checkpointing`` instead.

            enable_checkpointing: If ``True``, enable checkpointing.
                It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`.
                Default: ``True``.

            check_val_every_n_epoch: Check val every n train epochs.
                Default: ``1``.


            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Default: ``os.getcwd()``.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

            detect_anomaly: Enable anomaly detection for the autograd engine.
                Default: ``False``.

            deterministic: If ``True``, sets whether PyTorch operations must use deterministic algorithms.
                If not set, defaults to ``False``.
                Default: ``None``.

            devices: Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`,
                based on the accelerator type.

            fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
                of train, val and test to find any bugs (ie: a sort of unit test).
                Default: ``False``.

            flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).

                .. deprecated:: v1.5
                    ``flush_logs_every_n_steps`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please configure flushing directly in the logger instead.

            gpus: Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node
                Default: ``None``.

            gradient_clip_val: The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
                gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.
                Default: ``None``.

            gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
                to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
                be set to ``"norm"``.

            limit_train_batches: How much of training dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            limit_val_batches: How much of validation dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            limit_test_batches: How much of test dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            limit_predict_batches: How much of prediction dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            logger: Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
                the default ``TensorBoardLogger``. ``False`` will disable logging. If multiple loggers are
                provided and the `save_dir` property of that logger is not set, local files (checkpoints,
                profiler traces, etc.) are saved in ``default_root_dir`` rather than in the ``log_dir`` of any
                of the individual loggers.
                Default: ``True``.

            log_gpu_memory: None, 'min_max', 'all'. Might slow performance.

                .. deprecated:: v1.5
                    Deprecated in v1.5.0 and will be removed in v1.7.0
                    Please use the ``DeviceStatsMonitor`` callback directly instead.

            log_every_n_steps: How often to log within steps.
                Default: ``50``.

            prepare_data_per_node: If True, each LOCAL_RANK=0 will call prepare data.
                Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data

                .. deprecated:: v1.5
                    Deprecated in v1.5.0 and will be removed in v1.7.0
                    Please set ``prepare_data_per_node`` in ``LightningDataModule`` and/or
                    ``LightningModule`` directly instead.

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
                Default: ``False``.

            profiler: To profile individual steps during training and assist in identifying bottlenecks.
                Default: ``None``.

            overfit_batches: Overfit a fraction of training data (float) or a set number of batches (int).
                Default: ``0.0``.

            plugins: Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
                Default: ``None``.

            precision: Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16).
                Can be used on CPU, GPU, TPUs, HPUs or IPUs.
                Default: ``32``.

            max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
                If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
                To enable infinite training, set ``max_epochs = -1``.

            min_epochs: Force training for at least these many epochs. Disabled by default (None).

            max_steps: Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
                and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
                ``max_epochs`` to ``-1``.

            min_steps: Force training for at least these number of steps. Disabled by default (``None``).

            max_time: Stop training after this amount of time has passed. Disabled by default (``None``).
                The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
                :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
                :class:`datetime.timedelta`.

            num_nodes: Number of GPU nodes for distributed training.
                Default: ``1``.

            num_processes: Number of processes for distributed training with ``accelerator="cpu"``.
                Default: ``1``.

            num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
                Set it to `-1` to run all batches in all validation dataloaders.
                Default: ``2``.

            reload_dataloaders_every_n_epochs: Set to a non-negative integer to reload dataloaders every n epochs.
                Default: ``0``.

            replace_sampler_ddp: Explicitly enables or disables sampler replacement. If not specified this
                will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for
                train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it,
                you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.

            resume_from_checkpoint: Path/URL of the checkpoint from which training is resumed. If there is
                no checkpoint file at the path, an exception is raised. If resuming from mid-epoch checkpoint,
                training will start from the beginning of the next epoch.

                .. deprecated:: v1.5
                    ``resume_from_checkpoint`` is deprecated in v1.5 and will be removed in v2.0.
                    Please pass the path to ``Trainer.fit(..., ckpt_path=...)`` instead.

            strategy: Supports different training strategies with aliases
                as well custom strategies.
                Default: ``None``.

            sync_batchnorm: Synchronize batch norm layers between process groups/whole world.
                Default: ``False``.

            terminate_on_nan: If set to True, will terminate training (by raising a `ValueError`) at the
                end of each training batch, if any of the parameters or the loss are NaN or +/-inf.

                .. deprecated:: v1.5
                    Trainer argument ``terminate_on_nan`` was deprecated in v1.5 and will be removed in 1.7.
                    Please use ``detect_anomaly`` instead.

            detect_anomaly: Enable anomaly detection for the autograd engine.
                Default: ``False``.

            tpu_cores: How many TPU cores to train on (1 or 8) / Single TPU to train on (1)
                Default: ``None``.

            ipus: How many IPUs to train on.
                Default: ``None``.

            track_grad_norm: -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm. If using
                Automatic Mixed Precision (AMP), the gradients will be unscaled before logging them.
                Default: ``-1``.

            val_check_interval: How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
                after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
                batches.
                Default: ``1.0``.

            enable_model_summary: Whether to enable model summarization by default.
                Default: ``True``.

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

                .. deprecated:: v1.6
                    ``weights_save_path`` has been deprecated in v1.6 and will be removed in v1.8. Please pass
                    ``dirpath`` directly to the :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`
                    callback.

            move_metrics_to_cpu: Whether to force internal logged metrics to be moved to cpu.
                This can save some gpu memory, but can make training slower. Use with attention.
                Default: ``False``.

            multiple_trainloader_mode: How to loop over the datasets when there are multiple train loaders.
                In 'max_size_cycle' mode, the trainer ends one epoch when the largest dataset is traversed,
                and smaller datasets reload when running out of their data. In 'min_size' mode, all the datasets
                reload when reaching the minimum length of datasets.
                Default: ``"max_size_cycle"``.

            stochastic_weight_avg: Whether to use `Stochastic Weight Averaging (SWA)
                <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>`_.
                Default: ``False``.

                .. deprecated:: v1.5
                    ``stochastic_weight_avg`` has been deprecated in v1.5 and will be removed in v1.7.
                    Please pass :class:`~pytorch_lightning.callbacks.stochastic_weight_avg.StochasticWeightAveraging`
                    directly to the Trainer's ``callbacks`` argument instead.
        """
        super().__init__()
        Trainer._log_api_event("init")
        log.detail(f"{self.__class__.__name__}: Initializing trainer with parameters: {locals()}")
        self.state = TrainerState()

        # init connectors
        self._data_connector = DataConnector(self, multiple_trainloader_mode)

        self._accelerator_connector = AcceleratorConnector(
            num_processes=num_processes,
            devices=devices,
            tpu_cores=tpu_cores,
            ipus=ipus,
            accelerator=accelerator,
            strategy=strategy,
            gpus=gpus,
            num_nodes=num_nodes,
            sync_batchnorm=sync_batchnorm,
            benchmark=benchmark,
            replace_sampler_ddp=replace_sampler_ddp,
            deterministic=deterministic,
            auto_select_gpus=auto_select_gpus,
            precision=precision,
            amp_type=amp_backend,
            amp_level=amp_level,
            plugins=plugins,
        )
        self._logger_connector = LoggerConnector(self, log_gpu_memory)
        self._callback_connector = CallbackConnector(self)
        self._checkpoint_connector = CheckpointConnector(self, resume_from_checkpoint)
        self._signal_connector = SignalConnector(self)
        self.tuner = Tuner(self)

        min_steps, max_steps, min_epochs, max_epochs, max_time = _parse_loop_limits(
            min_steps, max_steps, min_epochs, max_epochs, max_time
        )
        fit_loop = FitLoop(min_epochs=min_epochs, max_epochs=max_epochs)
        training_epoch_loop = TrainingEpochLoop(min_steps=min_steps, max_steps=max_steps)
        fit_loop.connect(epoch_loop=training_epoch_loop)

        # default .fit() loop
        self.fit_loop = fit_loop

        # default .validate() loop
        self.validate_loop = EvaluationLoop()

        # default .test() loop
        self.test_loop = EvaluationLoop()

        # default .predict() loop
        self.predict_loop = PredictionLoop()

        # set when a checkpoint is loaded via `Trainer.{fit,validate,test,predict}`.
        self._ckpt_path: Optional[str] = None

        # .validate(), predict() and .test() set these when they load a checkpoint. They will be removed in favor of
        #  the unified read-only `Trainer.ckpt_path` attribute in v1.8
        self._validated_ckpt_path: Optional[str] = None  # TODO: remove in v1.8
        self._tested_ckpt_path: Optional[str] = None  # TODO: remove in v1.8
        self._predicted_ckpt_path: Optional[str] = None  # TODO: remove in v1.8

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
        self._call_callback_hooks("on_init_start")

        # init data flags
        self.check_val_every_n_epoch: int
        self._data_connector.on_trainer_init(
            check_val_every_n_epoch,
            reload_dataloaders_every_n_epochs,
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
        self.gradient_clip_val: Union[int, float] = gradient_clip_val
        self.gradient_clip_algorithm: Optional[GradClipAlgorithmType] = (
            GradClipAlgorithmType(gradient_clip_algorithm.lower()) if gradient_clip_algorithm is not None else None
        )
        self.track_grad_norm: float = float(track_grad_norm)

        self._detect_anomaly: bool = detect_anomaly
        self._setup_on_init(num_sanity_val_steps)

        # configure tuner
        self.tuner.on_trainer_init(auto_lr_find, auto_scale_batch_size)

        # configure profiler
        self.__init_profiler(profiler)

        # init logger flags
        self._loggers: List[LightningLoggerBase]
        self._logger_connector.on_trainer_init(logger, flush_logs_every_n_steps, log_every_n_steps, move_metrics_to_cpu)

        # init debugging flags
        self.val_check_interval: Union[int, float]
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
        self._call_callback_hooks("on_init_end")

    def _init_debugging_flags(
        self,
        limit_train_batches: Optional[Union[int, float]],
        limit_val_batches: Optional[Union[int, float]],
        limit_test_batches: Optional[Union[int, float]],
        limit_predict_batches: Optional[Union[int, float]],
        val_check_interval: Optional[Union[int, float]],
        overfit_batches: Union[int, float],
        fast_dev_run: Union[int, bool],
    ) -> None:
        if isinstance(fast_dev_run, int) and (fast_dev_run < 0):
            raise MisconfigurationException(
                f"fast_dev_run={fast_dev_run} is not a valid configuration. It should be >= 0."
            )

        self.fast_dev_run = fast_dev_run

        # set fast_dev_run=True when it is 1, used while logging
        if fast_dev_run == 1:
            self.fast_dev_run = True

        if fast_dev_run:
            num_batches = int(fast_dev_run)
            limit_train_batches = num_batches
            limit_val_batches = num_batches
            limit_test_batches = num_batches
            limit_predict_batches = num_batches
            self.fit_loop.max_steps = num_batches
            self.num_sanity_val_steps = 0
            self.fit_loop.max_epochs = 1
            val_check_interval = 1.0
            self.check_val_every_n_epoch = 1
            self.loggers = [DummyLogger()] if self.loggers else []

            rank_zero_info(
                "Running in fast_dev_run mode: will run a full train,"
                f" val, test and prediction loop using {num_batches} batch(es)."
            )

        self.limit_train_batches = _determine_batch_limits(limit_train_batches, "limit_train_batches")
        self.limit_val_batches = _determine_batch_limits(limit_val_batches, "limit_val_batches")
        self.limit_test_batches = _determine_batch_limits(limit_test_batches, "limit_test_batches")
        self.limit_predict_batches = _determine_batch_limits(limit_predict_batches, "limit_predict_batches")
        self.val_check_interval = _determine_batch_limits(val_check_interval, "val_check_interval")
        self.overfit_batches = _determine_batch_limits(overfit_batches, "overfit_batches")
        self._determine_data_use_amount(self.overfit_batches)

    def _determine_data_use_amount(self, overfit_batches: float) -> None:
        """Use less data for debugging purposes."""
        if overfit_batches > 0:
            self.limit_train_batches = overfit_batches
            self.limit_val_batches = 0

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
        self._last_train_dl_reload_epoch = float("-inf")
        self._last_val_dl_reload_epoch = float("-inf")

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
            if self.strategy.launcher is not None:
                return self.strategy.launcher.launch(trainer_fn, *args, trainer=self, **kwargs)
            else:
                return trainer_fn(*args, **kwargs)
        # TODO: treat KeyboardInterrupt as BaseException (delete the code below) in v1.7
        except KeyboardInterrupt as exception:
            rank_zero_warn("Detected KeyboardInterrupt, attempting graceful shutdown...")
            # user could press Ctrl+c many times... only shutdown once
            if not self.interrupted:
                self.state.status = TrainerStatus.INTERRUPTED
                self._call_callback_hooks("on_keyboard_interrupt")
                self._call_callback_hooks("on_exception", exception)
        except BaseException as exception:
            self.state.status = TrainerStatus.INTERRUPTED
            if distributed_available() and self.world_size > 1:
                # try syncing remaining processes, kill otherwise
                self.strategy.reconciliate_processes(traceback.format_exc())
            self._call_callback_hooks("on_exception", exception)
            self._teardown()
            # teardown might access the stage so we reset it after
            self.state.stage = None
            raise

    def fit(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        r"""
        Runs the full optimization routine.

        Args:
            model: Model to fit.

            train_dataloaders: A collection of :class:`torch.utils.data.DataLoader` or a
                :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying training samples.
                In the case of multiple dataloaders, please see this :ref:`section <multiple-dataloaders>`.

            val_dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.

            ckpt_path: Path/URL of the checkpoint from which training is resumed. If there is
                no checkpoint file at the path, an exception is raised. If resuming from mid-epoch checkpoint,
                training will start from the beginning of the next epoch.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
        """
        self.strategy.model = model
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
        log.detail(f"{self.__class__.__name__}: trainer fit stage")

        self.state.fn = TrainerFn.FITTING
        self.state.status = TrainerStatus.RUNNING
        self.training = True
        self._last_train_dl_reload_epoch = float("-inf")
        self._last_val_dl_reload_epoch = float("-inf")

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

        # TODO: ckpt_path only in v2.0
        ckpt_path = ckpt_path or self.resume_from_checkpoint
        self._ckpt_path = self.__set_ckpt_path(
            ckpt_path, model_provided=True, model_connected=self.lightning_module is not None
        )
        results = self._run(model, ckpt_path=self.ckpt_path)

        assert self.state.stopped
        self.training = False
        return results

    def validate(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
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
        self.strategy.model = model or self.lightning_module
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
        log.detail(f"{self.__class__.__name__}: trainer validate stage")

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

        self.validate_loop.verbose = verbose

        # links data to the trainer
        self._data_connector.attach_data(model, val_dataloaders=dataloaders, datamodule=datamodule)

        self._ckpt_path = self.__set_ckpt_path(
            ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        )

        self._validated_ckpt_path = self.ckpt_path  # TODO: remove in v1.8

        # run validate
        results = self._run(model, ckpt_path=self.ckpt_path)

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
        self.strategy.model = model or self.lightning_module
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
        log.detail(f"{self.__class__.__name__}: trainer test stage")

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

        self.test_loop.verbose = verbose

        # links data to the trainer
        self._data_connector.attach_data(model, test_dataloaders=dataloaders, datamodule=datamodule)

        self._ckpt_path = self.__set_ckpt_path(
            ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        )

        self._tested_ckpt_path = self.ckpt_path  # TODO: remove in v1.8

        # run test
        results = self._run(model, ckpt_path=self.ckpt_path)

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
        self.strategy.model = model or self.lightning_module
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
        log.detail(f"{self.__class__.__name__}: trainer predict stage")

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

        self._ckpt_path = self.__set_ckpt_path(
            ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        )

        self._predicted_ckpt_path = self.ckpt_path  # TODO: remove in v1.8

        results = self._run(model, ckpt_path=self.ckpt_path)

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
    ) -> Dict[str, Optional[Union[int, _LRFinder]]]:
        r"""
        Runs routines to tune hyperparameters before training.

        Args:
            model: Model to tune.

            train_dataloaders: A collection of :class:`torch.utils.data.DataLoader` or a
                :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying training samples.
                In the case of multiple dataloaders, please see this :ref:`section <multiple-dataloaders>`.

            val_dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

            scale_batch_size_kwargs: Arguments for :func:`~pytorch_lightning.tuner.batch_size_scaling.scale_batch_size`

            lr_find_kwargs: Arguments for :func:`~pytorch_lightning.tuner.lr_finder.lr_find`
        """
        Trainer._log_api_event("tune")

        self.state.fn = TrainerFn.TUNING
        self.state.status = TrainerStatus.RUNNING
        self.tuning = True

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

        with isolate_rng():
            result = self.tuner._tune(
                model, scale_batch_size_kwargs=scale_batch_size_kwargs, lr_find_kwargs=lr_find_kwargs
            )

        assert self.state.stopped
        self.tuning = False

        return result

    def _restore_modules_and_callbacks(self, checkpoint_path: Optional[_PATH] = None) -> None:
        # restore modules after setup
        self._checkpoint_connector.resume_start(checkpoint_path)
        self._checkpoint_connector._restore_quantization_callbacks()
        self._checkpoint_connector.restore_model()
        self._checkpoint_connector.restore_datamodule()
        if self.state.fn == TrainerFn.FITTING:
            # restore callback states
            self._checkpoint_connector.restore_callbacks()

    def _run(
        self, model: "pl.LightningModule", ckpt_path: Optional[str] = None
    ) -> Optional[Union[_EVALUATE_OUTPUT, _PREDICT_OUTPUT]]:
        # clean hparams
        if hasattr(model, "hparams"):
            parsing.clean_namespace(model.hparams)

        # attach model to the strategy
        self.strategy.connect(model)

        self._callback_connector._attach_model_callbacks()
        self._callback_connector._attach_model_logging_functions()

        verify_loop_configurations(self)

        # hook
        log.detail(f"{self.__class__.__name__}: preparing data")
        self._data_connector.prepare_data()

        # ----------------------------
        # SET UP TRAINING
        # ----------------------------
        self._call_callback_hooks("on_before_accelerator_backend_setup")
        log.detail(f"{self.__class__.__name__}: setting up strategy environment")
        self.strategy.setup_environment()
        self.__setup_profiler()

        self._call_setup_hook()  # allow user to setup lightning_module in accelerator environment

        # check if we should delay restoring checkpoint till later
        if not self.strategy.restore_checkpoint_after_setup:
            log.detail(f"{self.__class__.__name__}: restoring module and callbacks from checkpoint path: {ckpt_path}")
            self._restore_modules_and_callbacks(ckpt_path)

        log.detail(f"{self.__class__.__name__}: configuring sharded model")
        self._call_configure_sharded_model()  # allow user to setup in model sharded environment

        # ----------------------------
        # INSPECT THE CORE LOOPS
        # ----------------------------
        rf"""
             Lightning internal flow looks like this:
        {Trainer.fit} or {Trainer.test} or {Trainer.predict}  ||
                                |                             ||
                         spawn processes                      ||
                 {self.strategy.setup_environment}            ||
                                |                             ||
                        setup accelerator                     ||
                           and strategy                       ||  LIGHTNING
                                |                             ||
                        {self._run_stage}                     ||  FLOW
                                |                             ||
                        {self._run_train}                     ||  DIRECTION
                     or {self._run_evaluate}                  ||
                     or {self._run_predict}                   ||
                                |                             ||
                             results                          \/
        This is used to guide readers to the core loops: train, test, predict.
        {self._run_predict} is the simplest to understand, use `Go to Definition` to read it :)
        """

        # ----------------------------
        # TRAIN
        # ----------------------------

        # reset logger connector
        self._logger_connector.reset_results()
        self._logger_connector.reset_metrics()

        # strategy will configure model and move it to the device
        self.strategy.setup(self)

        # hook
        if self.state.fn == TrainerFn.FITTING:
            self._call_callback_hooks("on_fit_start")
            self._call_lightning_module_hook("on_fit_start")

        self._log_hyperparams()

        if self.strategy.restore_checkpoint_after_setup:
            log.detail(f"{self.__class__.__name__}: restoring module and callbacks from checkpoint path: {ckpt_path}")
            self._restore_modules_and_callbacks(ckpt_path)

        # restore optimizers, etc.
        log.detail(f"{self.__class__.__name__}: restoring training state")
        self._checkpoint_connector.restore_training_state()

        self._checkpoint_connector.resume_end()

        results = self._run_stage()

        log.detail(f"{self.__class__.__name__}: trainer tearing down")
        self._teardown()

        # ----------------------------
        # POST-Training CLEAN UP
        # ----------------------------
        # hook
        if self.state.fn == TrainerFn.FITTING:
            self._call_callback_hooks("on_fit_end")
            self._call_lightning_module_hook("on_fit_end")

        log.detail(f"{self.__class__.__name__}: calling teardown hooks")
        self._call_teardown_hook()

        self.state.status = TrainerStatus.FINISHED
        self.state.stage = None

        return results

    def _log_hyperparams(self) -> None:
        if not self.loggers:
            return
        # log hyper-parameters
        hparams_initial = None

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

        for logger in self.loggers:
            if hparams_initial is not None:
                logger.log_hyperparams(hparams_initial)
            logger.log_graph(self.lightning_module)
            logger.save()

    def _teardown(self):
        """This is the Trainer's internal teardown, unrelated to the `teardown` hooks in LightningModule and
        Callback; those are handled by :meth:`_call_teardown_hook`."""
        self.strategy.post_dispatch(self)
        self.strategy.teardown()
        loop = self._active_loop
        # loop should never be `None` here but it can because we don't know the trainer stage with `ddp_spawn`
        if loop is not None:
            loop.teardown()
        self._logger_connector.teardown()
        self._signal_connector.teardown()

    def run_stage(self) -> None:
        rank_zero_deprecation(
            "`Trainer.run_stage` is deprecated in v1.6 and will be removed in v1.8. Use"
            " `Trainer.{fit,validate,test,predict}` instead."
        )
        return self._run_stage()

    def _run_stage(self):
        self.strategy.barrier("run-stage")
        self.strategy.dispatch(self)

        if self.evaluating:
            return self._run_evaluate()
        if self.predicting:
            return self._run_predict()
        return self._run_train()

    def _pre_training_routine(self):
        # wait for all to join if on distributed
        self.strategy.barrier("setup_training")

        # register signals
        self._signal_connector.register_signal_handlers()

        # --------------------------
        # Pre-train
        # --------------------------
        self._call_callback_hooks("on_pretrain_routine_start")
        self._call_lightning_module_hook("on_pretrain_routine_start")

        self._call_callback_hooks("on_pretrain_routine_end")
        self._call_lightning_module_hook("on_pretrain_routine_end")

    def _run_train(self) -> None:
        self._pre_training_routine()

        with isolate_rng():
            self._run_sanity_check()

        # enable train mode
        self.model.train()
        torch.set_grad_enabled(True)

        self.fit_loop.trainer = self
        with torch.autograd.set_detect_anomaly(self._detect_anomaly):
            self.fit_loop.run()

    def _run_evaluate(self) -> _EVALUATE_OUTPUT:
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

    def _run_sanity_check(self) -> None:
        val_loop = self.fit_loop.epoch_loop.val_loop

        should_sanity_check = (
            self.enable_validation
            and self.num_sanity_val_steps > 0
            # do not sanity check if restarting because it would mess up the loaded state
            and not val_loop.restarting
        )

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        if should_sanity_check:
            stage = self.state.stage
            self.sanity_checking = True

            # reset logger connector
            self._logger_connector.reset_results()
            self._logger_connector.reset_metrics()

            self._call_callback_hooks("on_sanity_check_start")

            # reload dataloaders
            val_loop._reload_evaluation_dataloaders()
            self.num_sanity_val_batches = [
                min(self.num_sanity_val_steps, val_batches) for val_batches in self.num_val_batches
            ]

            # run eval step
            with torch.no_grad():
                val_loop.run()

            self._call_callback_hooks("on_sanity_check_end")

            # reset logger connector
            self._logger_connector.reset_results()
            self._logger_connector.reset_metrics()

            # reset the progress tracking state after sanity checking. we don't need to set the state before
            # because sanity check only runs when we are not restarting
            _reset_progress(val_loop)

            # restore the previous stage when the sanity check if finished
            self.state.stage = stage

    def __set_ckpt_path(self, ckpt_path: Optional[str], model_provided: bool, model_connected: bool) -> Optional[str]:
        # fault-tolerance takes precedence
        from pytorch_lightning.callbacks.fault_tolerance import _FaultToleranceCheckpoint

        ft_checkpoints = [cb for cb in self.callbacks if isinstance(cb, _FaultToleranceCheckpoint)]
        if ft_checkpoints:
            ft_ckpt_path = ft_checkpoints[0].ckpt_path
            fs = get_filesystem(ft_ckpt_path)
            if fs.exists(ft_ckpt_path):
                return ft_ckpt_path

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
            if len(self.checkpoint_callbacks) > 1:
                rank_zero_warn(
                    f'`.{fn}(ckpt_path="best")` is called with Trainer configured with multiple `ModelCheckpoint`'
                    " callbacks. It will use the best checkpoint path from first checkpoint callback."
                )

            if not self.checkpoint_callback:
                raise MisconfigurationException(
                    f'`.{fn}(ckpt_path="best")` is set but `ModelCheckpoint` is not configured.'
                )

            if not self.checkpoint_callback.best_model_path:
                if self.fast_dev_run:
                    raise MisconfigurationException(
                        f'You cannot execute `.{fn}(ckpt_path="best")` with `fast_dev_run=True`.'
                        f" Please pass an exact checkpoint path to `.{fn}(ckpt_path=...)`"
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

        self.strategy.barrier("pre_setup")

        if self.datamodule is not None:
            self.datamodule.setup(stage=fn)
        self._call_callback_hooks("setup", stage=fn)
        self._call_lightning_module_hook("setup", stage=fn)

        self.strategy.barrier("post_setup")

    def _call_configure_sharded_model(self) -> None:
        with self.strategy.model_sharded_context():
            self._handle_meta_model()
            self._call_lightning_module_hook("configure_sharded_model")
            self._call_callback_hooks("on_configure_sharded_model")

    def _handle_meta_model(self) -> None:
        if not is_on_meta_device(self.lightning_module):
            return

        if isinstance(self.strategy, DDPSpawnStrategy):
            raise MisconfigurationException("LightningModule on meta device isn't supported with spawn.")

        materialize_module(self.lightning_module)
        # the trainer reference is lost during materialization
        self.lightning_module.trainer = proxy(self)

    def _call_teardown_hook(self) -> None:
        fn = self.state.fn._setup_fn

        if self.datamodule is not None:
            self.datamodule.teardown(stage=fn)

        self._call_callback_hooks("teardown", stage=fn)
        self._call_lightning_module_hook("teardown", stage=fn)

        self.lightning_module._current_fx_name = None
        # these could have become stale if metrics are defined in `setup`
        self.lightning_module._metric_attributes = None

        # todo: TPU 8 cores hangs in flush with TensorBoard. Might do for all loggers.
        # It might be related to xla tensors blocked when moving the cpu kill loggers.
        for logger in self.loggers:
            logger.finalize("success")

        # summarize profile results
        self.profiler.describe()

    def call_hook(
        self, hook_name: str, *args: Any, pl_module: Optional["pl.LightningModule"] = None, **kwargs: Any
    ) -> Any:
        r"""
        .. deprecated:: v1.6
            The Trainer's `call_hook` method was deprecated in v1.6 and will be removed in v1.8.
        """
        rank_zero_deprecation("The Trainer's `call_hook` method was deprecated in v1.6 and will be removed in v1.8.")
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

            # call the strategy hook
            if hook_name not in ("setup", "teardown", "on_train_start") and hasattr(self.strategy, hook_name):
                strategy_hook = getattr(self.strategy, hook_name)
                strategy_output = strategy_hook(*args, **kwargs)
                output = strategy_output if output is None else output

        if pl_module:
            # restore current_fx when nested context
            pl_module._current_fx_name = prev_fx_name

        return output

    def _call_lightning_module_hook(
        self,
        hook_name: str,
        *args: Any,
        pl_module: Optional["pl.LightningModule"] = None,
        **kwargs: Any,
    ) -> Any:
        pl_module = pl_module or self.lightning_module

        if pl_module is None:
            raise TypeError("No Lightning Module is available to call hooks on")

        fn = getattr(pl_module, hook_name)
        if not callable(fn):
            return

        prev_fx_name = pl_module._current_fx_name
        pl_module._current_fx_name = hook_name

        with self.profiler.profile(f"[LightningModule]{pl_module.__class__.__name__}.{hook_name}"):
            output = fn(*args, **kwargs)

        # restore current_fx when nested context
        pl_module._current_fx_name = prev_fx_name

        return output

    def _call_callback_hooks(
        self,
        hook_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        log.debug(f"{self.__class__.__name__}: calling callback hook: {hook_name}")
        # TODO: remove if block in v1.8
        if hook_name in ("on_init_start", "on_init_end"):
            # these `Callback` hooks are the only ones that do not take a lightning module.
            # we also don't profile bc profiler hasn't been set yet
            for callback in self.callbacks:
                fn = getattr(callback, hook_name)
                if callable(fn):
                    fn(self, *args, **kwargs)
            return

        pl_module = self.lightning_module
        if pl_module:
            prev_fx_name = pl_module._current_fx_name
            pl_module._current_fx_name = hook_name

        # TODO: remove if block in v1.7
        if hook_name == "on_train_batch_start":
            with self.profiler.profile(hook_name):
                self._on_train_batch_start(*args, **kwargs)
        elif hook_name == "on_train_batch_end":
            with self.profiler.profile(hook_name):
                self._on_train_batch_end(*args, **kwargs)
        else:
            for callback in self.callbacks:
                fn = getattr(callback, hook_name)
                if callable(fn):
                    with self.profiler.profile(f"[Callback]{callback.state_key}.{hook_name}"):
                        fn(self, self.lightning_module, *args, **kwargs)

        if pl_module:
            # restore current_fx when nested context
            pl_module._current_fx_name = prev_fx_name

    # TODO: Delete this in v1.7 (deprecations: #9816 and #11148)
    def _on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        r"""Called when the training batch begins. This function is needed because of two different deprecations affecting
        the original function in TrainerCallbackHookMixin: #9816 and #11148.
        """
        for callback in self.callbacks:
            if is_param_in_hook_signature(callback.on_train_batch_start, "dataloader_idx", explicit=True):
                callback.on_train_batch_start(self, self.lightning_module, batch, batch_idx, 0)
            else:
                callback.on_train_batch_start(self, self.lightning_module, batch, batch_idx)

    # TODO: Delete this in v1.7 (deprecations: #9816 and #11148)
    def _on_train_batch_end(self, outputs: STEP_OUTPUT, batch, batch_idx, dataloader_idx=0):
        r"""Called when the training batch ends. This function is needed because of two different deprecations affecting
        the original function in TrainerCallbackHookMixin: #9816 and #11148.
        """
        for callback in self.callbacks:
            if is_param_in_hook_signature(callback.on_train_batch_end, "dataloader_idx", explicit=True):
                callback.on_train_batch_end(self, self.lightning_module, outputs, batch, batch_idx, 0)
            else:
                callback.on_train_batch_end(self, self.lightning_module, outputs, batch, batch_idx)

    def _call_callbacks_state_dict(self) -> Dict[str, dict]:
        """Called when saving a model checkpoint, calls and returns every callback's `state_dict`, keyed by
        `Callback.state_key`."""
        callback_state_dicts = {}
        for callback in self.callbacks:
            state_dict = callback.state_dict()
            if state_dict:
                callback_state_dicts[callback.state_key] = state_dict
        return callback_state_dicts

    def _call_callbacks_on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving a model checkpoint, calls every callback's `on_save_checkpoint` hook.

        Will be removed in v1.8: If state is returned, we insert the callback state into
        ``checkpoint["callbacks"][Callback.state_key]``. It overrides ``state_dict`` if already present.
        """
        pl_module = self.lightning_module
        if pl_module:
            prev_fx_name = pl_module._current_fx_name
            pl_module._current_fx_name = "on_save_checkpoint"

        for callback in self.callbacks:
            with self.profiler.profile(f"[Callback]{callback.state_key}.on_save_checkpoint"):
                state = callback.on_save_checkpoint(self, self.lightning_module, checkpoint)
            if state:
                rank_zero_deprecation(
                    f"Returning a value from `{callback.__class__.__name__}.on_save_checkpoint` is deprecated in v1.6"
                    " and will be removed in v1.8. Please override `Callback.state_dict`"
                    " to return state to be saved."
                )
                checkpoint["callbacks"][callback.state_key] = state

        if pl_module:
            # restore current_fx when nested context
            pl_module._current_fx_name = prev_fx_name

    def _call_callbacks_on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading a model checkpoint.

        Calls every callback's `on_load_checkpoint` hook. We have a dedicated function for this rather than using
        `_call_callback_hooks` because we have special logic for getting callback_states.
        """
        pl_module = self.lightning_module
        if pl_module:
            prev_fx_name = pl_module._current_fx_name
            pl_module._current_fx_name = "on_load_checkpoint"

        callback_states: Dict[Union[Type, str], Dict] = checkpoint.get("callbacks")

        if callback_states is None:
            return

        is_legacy_ckpt = Version(checkpoint["pytorch-lightning_version"]) < Version("1.5.0dev")
        current_callbacks_keys = {cb._legacy_state_key if is_legacy_ckpt else cb.state_key for cb in self.callbacks}
        difference = callback_states.keys() - current_callbacks_keys
        if difference:
            rank_zero_warn(
                "Be aware that when using `ckpt_path`,"
                " callbacks used to create the checkpoint need to be provided during `Trainer` instantiation."
                f" Please add the following callbacks: {list(difference)}.",
            )

        for callback in self.callbacks:
            state = callback_states.get(callback.state_key, callback_states.get(callback._legacy_state_key))
            if state:
                state = deepcopy(state)
                with self.profiler.profile(f"[Callback]{callback.state_key}.on_load_checkpoint"):
                    callback.on_load_checkpoint(self, self.lightning_module, state)

        if pl_module:
            # restore current_fx when nested context
            pl_module._current_fx_name = prev_fx_name

    def _call_callbacks_load_state_dict(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading a model checkpoint, calls every callback's `load_state_dict`."""
        callback_states: Dict[Union[Type, str], Dict] = checkpoint.get("callbacks")

        if callback_states is None:
            return

        for callback in self.callbacks:
            state = callback_states.get(callback.state_key, callback_states.get(callback._legacy_state_key))
            if state:
                state = deepcopy(state)
                callback.load_state_dict(state)

    def _call_strategy_hook(
        self,
        hook_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        pl_module = self.lightning_module
        prev_fx_name = pl_module._current_fx_name
        pl_module._current_fx_name = hook_name

        fn = getattr(self.strategy, hook_name)
        if not callable(fn):
            return

        with self.profiler.profile(f"[Strategy]{self.strategy.__class__.__name__}.{hook_name}"):
            output = fn(*args, **kwargs)

        # restore current_fx when nested context
        pl_module._current_fx_name = prev_fx_name

        return output

    @staticmethod
    def _log_api_event(event: str) -> None:
        torch._C._log_api_usage_once("lightning.trainer." + event)

    def __init_profiler(self, profiler: Optional[Union[Profiler, str]]) -> None:
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
        self.profiler: Profiler = profiler or PassThroughProfiler()

    def __setup_profiler(self) -> None:
        local_rank = self.local_rank if self.world_size > 1 else None
        self.profiler._lightning_module = proxy(self.lightning_module)
        self.profiler.setup(stage=self.state.fn._setup_fn, local_rank=local_rank, log_dir=self.log_dir)

    def _log_device_info(self) -> None:
        rank_zero_info(
            f"GPU available: {torch.cuda.is_available()}, used: {isinstance(self.accelerator, GPUAccelerator)}"
        )

        num_tpu_cores = self.num_devices if isinstance(self.accelerator, TPUAccelerator) else 0
        rank_zero_info(f"TPU available: {_TPU_AVAILABLE}, using: {num_tpu_cores} TPU cores")

        num_ipus = self.num_devices if isinstance(self.accelerator, IPUAccelerator) else 0
        rank_zero_info(f"IPU available: {_IPU_AVAILABLE}, using: {num_ipus} IPUs")

        num_hpus = self.num_devices if isinstance(self.accelerator, HPUAccelerator) else 0
        rank_zero_info(f"HPU available: {_HPU_AVAILABLE}, using: {num_hpus} HPUs")

        if torch.cuda.is_available() and not isinstance(self.accelerator, GPUAccelerator):
            rank_zero_warn(
                "GPU available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='gpu', devices={GPUAccelerator.auto_device_count()})`.",
                category=PossibleUserWarning,
            )

        if _TPU_AVAILABLE and not isinstance(self.accelerator, TPUAccelerator):
            rank_zero_warn(
                "TPU available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='tpu', devices={TPUAccelerator.auto_device_count()})`."
            )

        if _IPU_AVAILABLE and not isinstance(self.accelerator, IPUAccelerator):
            rank_zero_warn(
                "IPU available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='ipu', devices={IPUAccelerator.auto_device_count()})`."
            )

        if _HPU_AVAILABLE and not isinstance(self.accelerator, HPUAccelerator):
            rank_zero_warn(
                "HPU available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='hpu', devices={HPUAccelerator.auto_device_count()})`."
            )

    """
    Data loading methods
    """

    def reset_train_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the train dataloader and initialises required variables (number of batches, when to validate,
        etc.).

        Args:
            model: The ``LightningModule`` if calling this outside of the trainer scope.
        """
        source = self._data_connector._train_dataloader_source
        pl_module = self.lightning_module or model
        has_step = is_overridden("training_step", pl_module)
        enable_training = self.limit_train_batches > 0
        if not (source.is_defined() and has_step and enable_training):
            return

        self.train_dataloader = self._data_connector._request_dataloader(RunningStage.TRAINING, model=model)

        if self.overfit_batches > 0:
            self.train_dataloader = self._data_connector._resolve_overfit_batches(self.train_dataloader)

        # automatically add samplers
        self.train_dataloader = apply_to_collection(
            self.train_dataloader,
            (DataLoader, CombinedLoader),
            self._data_connector._prepare_dataloader,
            mode=RunningStage.TRAINING,
        )
        loaders = (
            self.train_dataloader.loaders
            if isinstance(self.train_dataloader, CombinedLoader)
            else self.train_dataloader
        )

        # check the workers recursively
        apply_to_collection(loaders, DataLoader, self._data_connector._worker_check, "train_dataloader")

        # add worker_init_fn for correct seeding in worker processes
        apply_to_collection(loaders, DataLoader, _auto_add_worker_init_fn, rank=self.global_rank)

        # add collate_fn to collect metadata for fault tolerant training
        if _fault_tolerant_training():
            apply_to_collection(loaders, DataLoader, _add_capture_metadata_collate)

        # wrap the sequence of train loaders to a CombinedLoader object for computing the num_training_batches
        if not isinstance(self.train_dataloader, CombinedLoader):
            self.train_dataloader = CombinedLoader(loaders, self._data_connector.multiple_trainloader_mode)

        module = model or self.lightning_module or self.datamodule
        orig_train_batches = self.num_training_batches = (
            len(self.train_dataloader)
            if has_len_all_ranks(self.train_dataloader, self.strategy, module)
            else float("inf")
        )
        if orig_train_batches == 0:
            return

        # store epoch of dataloader reset for reload_dataloaders_every_n_epochs
        self._last_train_dl_reload_epoch = self.current_epoch

        if isinstance(self.limit_train_batches, int):
            self.num_training_batches = min(orig_train_batches, self.limit_train_batches)
        elif self.num_training_batches != float("inf"):
            self.num_training_batches = int(orig_train_batches * self.limit_train_batches)
        elif self.limit_train_batches != 1.0:
            raise MisconfigurationException(
                "When using an `IterableDataset`, `Trainer(limit_train_batches)` must be `1.0` or an int."
                "An int specifies `num_training_batches` to use."
            )

        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
            if self.val_check_batch > self.num_training_batches:
                raise ValueError(
                    f"`val_check_interval` ({self.val_check_interval}) must be less than or equal "
                    f"to the number of the training batches ({self.num_training_batches}). "
                    "If you want to disable validation set `limit_val_batches` to 0.0 instead."
                )
        else:
            if not has_len_all_ranks(self.train_dataloader, self.strategy, module):
                if self.val_check_interval == 1.0:
                    self.val_check_batch = float("inf")
                else:
                    raise MisconfigurationException(
                        "When using an IterableDataset for `train_dataloader`,"
                        " `Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies"
                        " checking validation every k training batches."
                    )
            else:
                self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
                self.val_check_batch = max(1, self.val_check_batch)

        if self.loggers and self.num_training_batches < self.log_every_n_steps:
            rank_zero_warn(
                f"The number of training batches ({self.num_training_batches}) is smaller than the logging interval"
                f" Trainer(log_every_n_steps={self.log_every_n_steps}). Set a lower value for log_every_n_steps if"
                " you want to see logs for the training epoch.",
                category=PossibleUserWarning,
            )

        if (
            self.num_training_batches == 0
            and self.limit_train_batches > 0.0
            and isinstance(self.limit_train_batches, float)
            and orig_train_batches != float("inf")
        ):
            min_percentage = 1.0 / orig_train_batches
            raise MisconfigurationException(
                f"You requested to check {self.limit_train_batches} of the `train_dataloader` but"
                f" {self.limit_train_batches} * {orig_train_batches} < 1. Please increase the"
                f" `limit_train_batches` argument. Try at least"
                f" `limit_train_batches={min_percentage}`"
            )

    def reset_val_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the validation dataloader and determines the number of batches.

        Args:
            model: The ``LightningModule`` if called outside of the trainer scope.
        """
        source = self._data_connector._val_dataloader_source
        pl_module = self.lightning_module or model
        has_step = is_overridden("validation_step", pl_module)
        enable_validation = self.limit_val_batches > 0
        if source.is_defined() and has_step and enable_validation:
            self.num_val_batches, self.val_dataloaders = self._data_connector._reset_eval_dataloader(
                RunningStage.VALIDATING, model=pl_module
            )

            # store epoch of dataloader reset for reload_dataloaders_every_n_epochs
            self._last_val_dl_reload_epoch = self.current_epoch

    def reset_test_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the test dataloader and determines the number of batches.

        Args:
            model: The ``LightningModule`` if called outside of the trainer scope.
        """
        source = self._data_connector._test_dataloader_source
        pl_module = self.lightning_module or model
        has_step = is_overridden("test_step", pl_module)
        enable_testing = self.limit_test_batches > 0
        if source.is_defined() and has_step and enable_testing:
            self.num_test_batches, self.test_dataloaders = self._data_connector._reset_eval_dataloader(
                RunningStage.TESTING, model=pl_module
            )

    def reset_predict_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the predict dataloader and determines the number of batches.

        Args:
            model: The ``LightningModule`` if called outside of the trainer scope.
        """
        source = self._data_connector._predict_dataloader_source
        pl_module = self.lightning_module or model
        enable_prediction = self.limit_predict_batches > 0
        if source.is_defined() and enable_prediction:
            self.num_predict_batches, self.predict_dataloaders = self._data_connector._reset_eval_dataloader(
                RunningStage.PREDICTING, model=pl_module
            )

    def reset_train_val_dataloaders(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets train and val dataloaders if none are attached to the trainer.

        The val dataloader must be initialized before training loop starts, as the training loop
        inspects the val dataloader to determine whether to run the evaluation loop.
        Args:
            model: The ``LightningModule`` if called outside of the trainer scope.
        """
        if self.train_dataloader is None:
            self.reset_train_dataloader(model=model)
        if self.val_dataloaders is None:
            self.reset_val_dataloader(model=model)

    """
    Accelerator properties
    """

    @property
    def accelerator(self) -> Accelerator:
        return self.strategy.accelerator

    @property
    def strategy(self) -> Strategy:
        return self._accelerator_connector.strategy

    @property
    def training_type_plugin(self) -> Strategy:
        rank_zero_deprecation(
            "`Trainer.training_type_plugin` is deprecated in v1.6 and will be removed in v1.8. Use"
            " `Trainer.strategy` instead."
        )
        return self.strategy

    @property
    def precision_plugin(self) -> PrecisionPlugin:
        return self.strategy.precision_plugin

    @property
    def global_rank(self) -> int:
        return self.strategy.global_rank

    @property
    def local_rank(self) -> int:
        # some strategies define a local rank
        return getattr(self.strategy, "local_rank", 0)

    @property
    def node_rank(self) -> int:
        # some strategies define a node rank
        return getattr(self.strategy, "node_rank", 0)

    @property
    def world_size(self) -> int:
        # some strategies define a world size
        return getattr(self.strategy, "world_size", 1)

    @property
    def should_rank_save_checkpoint(self) -> bool:
        rank_zero_deprecation(
            "`Trainer.should_rank_save_checkpoint` is deprecated in v1.6 and will be removed in v1.8.", stacklevel=5
        )
        strategy = self.strategy
        return (
            isinstance(strategy, pl.strategies.TPUSpawnStrategy) and strategy.local_rank == 0 or strategy.is_global_zero
        )

    @property
    def num_nodes(self) -> int:
        return getattr(self.strategy, "num_nodes", 1)

    @property
    def device_ids(self) -> List[int]:
        """List of device indexes per node."""
        devices = (
            self.strategy.parallel_devices
            if isinstance(self.strategy, ParallelStrategy)
            else [self.strategy.root_device]
        )
        device_ids = []
        for idx, device in enumerate(devices):
            if isinstance(device, torch.device):
                device_ids.append(device.index or idx)
            elif isinstance(device, int):
                device_ids.append(device)
        return device_ids

    @property
    def num_devices(self) -> int:
        """Number of devices the trainer uses per node."""
        return len(self.device_ids)

    @property
    def num_processes(self) -> int:
        rank_zero_deprecation(
            "`Trainer.num_processes` is deprecated in v1.6 and will be removed in v1.8. "
            "Please use `Trainer.num_devices` instead."
        )
        return self.num_devices

    @property
    def root_gpu(self) -> Optional[int]:
        rank_zero_deprecation(
            "`Trainer.root_gpu` is deprecated in v1.6 and will be removed in v1.8. "
            "Please use `Trainer.strategy.root_device.index` instead."
        )
        return self.strategy.root_device.index if isinstance(self.accelerator, GPUAccelerator) else None

    @property
    def tpu_cores(self) -> int:
        rank_zero_deprecation(
            "`Trainer.tpu_cores` is deprecated in v1.6 and will be removed in v1.8. "
            "Please use `Trainer.num_devices` instead."
        )
        return self.num_devices if isinstance(self.accelerator, TPUAccelerator) else 0

    @property
    def ipus(self) -> int:
        rank_zero_deprecation(
            "`Trainer.ipus` was deprecated in v1.6 and will be removed in v1.8."
            " Please use `Trainer.num_devices` instead."
        )
        return self.num_devices if isinstance(self.accelerator, IPUAccelerator) else 0

    @property
    def num_gpus(self) -> int:
        rank_zero_deprecation(
            "`Trainer.num_gpus` was deprecated in v1.6 and will be removed in v1.8."
            " Please use `Trainer.num_devices` instead."
        )
        return self.num_devices if isinstance(self.accelerator, GPUAccelerator) else 0

    @property
    def devices(self) -> int:
        rank_zero_deprecation(
            "`Trainer.devices` was deprecated in v1.6 and will be removed in v1.8."
            " Please use `Trainer.num_devices` or `Trainer.device_ids` to get device information instead."
        )
        return self.num_devices

    @property
    def data_parallel_device_ids(self) -> Optional[List[int]]:
        rank_zero_deprecation(
            "`Trainer.data_parallel_device_ids` was deprecated in v1.6 and will be removed in v1.8."
            " Please use `Trainer.device_ids` instead."
        )
        return self.device_ids if isinstance(self.accelerator, GPUAccelerator) else None

    @property
    def lightning_module(self) -> "pl.LightningModule":
        # TODO: this is actually an optional return
        return self.strategy.lightning_module

    @property
    def optimizers(self) -> List[Optimizer]:
        return self.strategy.optimizers

    @optimizers.setter
    def optimizers(self, new_optims: Optional[List[Optimizer]]) -> None:
        self.strategy.optimizers = new_optims

    @property
    def lightning_optimizers(self) -> Dict[int, LightningOptimizer]:
        rank_zero_deprecation(
            "`Trainer.lightning_optimizers` is deprecated in v1.6 and will be removed in v1.8", stacklevel=5
        )
        return self.strategy._lightning_optimizers

    @property
    def lr_scheduler_configs(self) -> List[LRSchedulerConfig]:
        return self.strategy.lr_scheduler_configs

    @property
    def lr_schedulers(self) -> List[Dict[str, Any]]:
        rank_zero_deprecation(
            "`Trainer.lr_schedulers` is deprecated in v1.6 and will be removed in v1.8."
            " You can use `trainer.lr_scheduler_configs` instead which contains dataclasses instead of dictionaries.",
            stacklevel=5,
        )
        from dataclasses import asdict

        return [asdict(config) for config in self.strategy.lr_scheduler_configs]

    @property
    def optimizer_frequencies(self) -> List[int]:
        return self.strategy.optimizer_frequencies

    @optimizer_frequencies.setter
    def optimizer_frequencies(self, new_freqs: List[int]) -> None:
        self.strategy.optimizer_frequencies = new_freqs

    @property
    def amp_backend(self) -> Optional[AMPType]:
        if isinstance(self.precision_plugin, ApexMixedPrecisionPlugin):
            return AMPType.APEX
        if isinstance(self.precision_plugin, NativeMixedPrecisionPlugin):
            return AMPType.NATIVE
        return None

    @property
    def precision(self) -> Union[str, int]:
        return self.strategy.precision_plugin.precision

    @property
    def scaler(self) -> Optional[Any]:
        return getattr(self.precision_plugin, "scaler", None)

    @property
    def gpus(self) -> Optional[Union[List[int], str, int]]:
        rank_zero_deprecation(
            "`Trainer.gpus` was deprecated in v1.6 and will be removed in v1.8."
            " Please use `Trainer.num_devices` or `Trainer.device_ids` to get device information instead."
        )
        return self._accelerator_connector.gpus

    @property
    def model(self) -> torch.nn.Module:
        """The LightningModule, but possibly wrapped into DataParallel or DistributedDataParallel.

        To access the pure LightningModule, use
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.lightning_module` instead.
        """
        return self.strategy.model

    @model.setter
    def model(self, model: torch.nn.Module) -> None:
        """Setter for the model, pass-through to accelerator and plugin where the model reference is stored. Used
        by the Tuner to reset the state of Trainer and Accelerator.

        Args:
            model: The LightningModule, possibly wrapped into DataParallel or DistributedDataParallel, depending
                on the backend.
        """
        self.strategy.model = model

    """
    General properties
    """

    @property
    def log_dir(self) -> Optional[str]:
        if len(self.loggers) == 1:
            if isinstance(self.logger, TensorBoardLogger):
                dirpath = self.logger.log_dir
            else:
                dirpath = self.logger.save_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath

    @property
    def use_amp(self) -> bool:
        rank_zero_deprecation(
            "`Trainer.use_amp` is deprecated in v1.6.0 and will be removed in v1.8.0."
            " Please use `Trainer.amp_backend` instead."
        )
        return self.precision == 16

    @property
    def is_global_zero(self) -> bool:
        return self.strategy.is_global_zero

    @property
    def slurm_job_id(self) -> Optional[int]:
        rank_zero_deprecation("Method `slurm_job_id` is deprecated in v1.6.0 and will be removed in v1.7.0.")
        return SLURMEnvironment.job_id()

    @property
    def distributed_sampler_kwargs(self) -> Optional[dict]:
        if isinstance(self.strategy, ParallelStrategy):
            return self.strategy.distributed_sampler_kwargs

    @property
    def data_parallel(self) -> bool:
        return isinstance(self.strategy, ParallelStrategy)

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
    def enable_validation(self) -> bool:
        """Check if we should run validation during training."""
        return (
            self._data_connector._val_dataloader_source.is_defined()
            and is_overridden("validation_step", self.lightning_module)
            and self.limit_val_batches > 0
        )

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

        .. deprecated:: v1.6
            `Trainer.weights_save_path` has been deprecated in v1.6 and will be removed in v1.8.
        """
        rank_zero_deprecation("`Trainer.weights_save_path` has been deprecated in v1.6 and will be removed in v1.8.")
        return self._weights_save_path_internal

    # TODO: Remove _weights_save_path_internal in v1.8
    @property
    def _weights_save_path_internal(self) -> str:
        """This is an internal implementation of weights_save_path which allows weights_save_path to be used
        internally by the framework without emitting a deprecation warning.

        To be removed in v1.8.
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
    def progress_bar_callback(self) -> Optional[ProgressBarBase]:
        """An instance of :class:`~pytorch_lightning.callbacks.progress.base.ProgressBarBase` found in the
        Trainer.callbacks list, or ``None`` if one doesn't exist."""
        for c in self.callbacks:
            if isinstance(c, ProgressBarBase):
                return c
        return None

    @property
    def resume_from_checkpoint(self) -> Optional[Union[str, Path]]:
        resume_from_checkpoint = self._checkpoint_connector.resume_from_checkpoint_fit_path
        if resume_from_checkpoint is not None:
            rank_zero_deprecation(
                "`trainer.resume_from_checkpoint` is deprecated in v1.5 and will be removed in v2.0."
                " Specify the fit checkpoint path with `trainer.fit(ckpt_path=)` instead.",
                stacklevel=5,
            )

        return resume_from_checkpoint

    @property
    def ckpt_path(self) -> Optional[str]:
        """Set to the path/URL of a checkpoint loaded via :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`,
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate`,
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.test`, or
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`. ``None`` otherwise."""
        return self._ckpt_path

    @property
    def validated_ckpt_path(self) -> Optional[str]:
        rank_zero_deprecation(
            "The `Trainer.validated_ckpt_path` attribute was deprecated in v1.6 and will be removed in v1.8. The"
            " path of a checkpoint loaded via `Trainer.{fit,validate,test,predict}` should be accessed via"
            " `Trainer.ckpt_path` instead.",
            stacklevel=5,
        )
        return self._validated_ckpt_path

    @validated_ckpt_path.setter
    def validated_ckpt_path(self, ckpt_path: Optional[str]) -> None:
        rank_zero_deprecation(
            "The `Trainer.validated_ckpt_path` attribute was deprecated in v1.6 and will be removed in v1.8. The"
            " path of a checkpoint loaded via `Trainer.{fit,validate,test,predict}` should be accessed via the"
            " read-only `Trainer.ckpt_path`.",
            stacklevel=5,
        )
        self._validated_ckpt_path = ckpt_path

    @property
    def tested_ckpt_path(self) -> Optional[str]:
        rank_zero_deprecation(
            "The `Trainer.tested_ckpt_path` attribute was deprecated in v1.6 and will be removed in v1.8. The"
            " path of a checkpoint loaded via `Trainer.{fit,validate,test,predict}` should be accessed via"
            " `Trainer.ckpt_path` instead.",
            stacklevel=5,
        )
        return self._tested_ckpt_path

    @tested_ckpt_path.setter
    def tested_ckpt_path(self, ckpt_path: Optional[str]) -> None:
        rank_zero_deprecation(
            "The `Trainer.tested_ckpt_path` attribute was deprecated in v1.6 and will be removed in v1.8. The"
            " path of a checkpoint loaded via `Trainer.{fit,validate,test,predict}` should be accessed via the"
            " read-only `Trainer.ckpt_path` instead.",
            stacklevel=5,
        )
        self._tested_ckpt_path = ckpt_path

    @property
    def predicted_ckpt_path(self) -> Optional[str]:
        rank_zero_deprecation(
            "The `Trainer.predicted_ckpt_path` attribute was deprecated in v1.6 and will be removed in v1.8. The"
            " path of a checkpoint loaded via `Trainer.{fit,validate,test,predict}` should be accessed via"
            " `Trainer.ckpt_path` instead.",
            stacklevel=5,
        )
        return self._predicted_ckpt_path

    @predicted_ckpt_path.setter
    def predicted_ckpt_path(self, ckpt_path: Optional[str]) -> None:
        rank_zero_deprecation(
            "The `Trainer.predicted_ckpt_path` attribute was deprecated in v1.6 and will be removed in v1.8. The"
            " path of a checkpoint loaded via `Trainer.{fit,validate,test,predict}` should be accessed via the"
            " read-only `Trainer.ckpt_path` instead.",
            stacklevel=5,
        )
        self._predicted_ckpt_path = ckpt_path

    def save_checkpoint(
        self, filepath: _PATH, weights_only: bool = False, storage_options: Optional[Any] = None
    ) -> None:
        r"""
        Runs routine to create a checkpoint.

        Args:
            filepath: Path where checkpoint is saved.
            weights_only: If ``True``, will only save the model weights.
            storage_options: parameter for how to save to storage, passed to ``CheckpointIO`` plugin

        """
        self._checkpoint_connector.save_checkpoint(filepath, weights_only=weights_only, storage_options=storage_options)

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
        """The number of optimizer steps taken (does not reset each epoch).

        This includes multiple optimizers and TBPTT steps (if enabled).
        """
        return self.fit_loop.epoch_loop.global_step

    @property
    def current_epoch(self) -> int:
        """The current epoch, updated after the epoch end hooks are run."""
        return self.fit_loop.epoch_progress.current.completed

    @property
    def max_epochs(self) -> int:
        return self.fit_loop.max_epochs

    @property
    def min_epochs(self) -> int:
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
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`.
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
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate`. Note that this loop is different from the one
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
    def verbose_evaluate(self) -> bool:
        rank_zero_deprecation(
            "The `Trainer.verbose_evaluate` property has been deprecated and will be removed in v1.8. The current value"
            " returned is the union of the validate and test loop values. You can choose which one to access with"
            " `trainer.{validate,test}_loop.verbose`.",
            stacklevel=5,
        )
        return self.validate_loop.verbose or self.test_loop.verbose

    @verbose_evaluate.setter
    def verbose_evaluate(self, verbose: bool) -> None:
        rank_zero_deprecation(
            "The `Trainer.verbose_evaluate` property has been deprecated and will be removed in v1.8. This will set"
            " the value for both trainer.{validate,test}_loop.verbose`.",
            stacklevel=5,
        )
        self.validate_loop.verbose = verbose
        self.test_loop.verbose = verbose

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
    def logger(self) -> Optional[LightningLoggerBase]:
        loggers = self.loggers
        if len(loggers) == 0:
            return None
        if len(loggers) == 1:
            return loggers[0]
        else:
            rank_zero_deprecation(
                "Using `trainer.logger` when multiple loggers are configured."
                " This behavior will change in v1.8 when `LoggerCollection` is removed, and"
                " `trainer.logger` will return the first logger available.",
                stacklevel=5,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return LoggerCollection(loggers)

    @logger.setter
    def logger(self, logger: Optional[LightningLoggerBase]) -> None:
        if not logger:
            self.loggers = []
        elif isinstance(logger, LoggerCollection):
            self.loggers = list(logger)
        else:
            self.loggers = [logger]

    @property
    def loggers(self) -> List[LightningLoggerBase]:
        return self._loggers

    @loggers.setter
    def loggers(self, loggers: Optional[List[LightningLoggerBase]]) -> None:
        self._loggers = loggers if loggers else []

    @property
    def callback_metrics(self) -> dict:
        return self._logger_connector.callback_metrics

    @property
    def logged_metrics(self) -> dict:
        return self._logger_connector.logged_metrics

    @property
    def progress_bar_metrics(self) -> dict:
        return self._logger_connector.progress_bar_metrics

    @property
    def _results(self) -> Optional[_ResultCollection]:
        active_loop = self._active_loop
        if active_loop is not None:
            return active_loop._results

    def _exit_gracefully_on_signal(self) -> None:
        if not _fault_tolerant_training() or not self._should_terminate_gracefully():
            return
        raise ExitGracefullyException(0)

    def _should_terminate_gracefully(self) -> bool:
        value = torch.tensor(int(self._terminate_gracefully), device=self.strategy.root_device)
        return self.strategy.reduce(value, reduce_op="sum") > 0

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

    @property
    def estimated_stepping_batches(self) -> Union[int, float]:
        r"""
        Estimated stepping batches for the complete training inferred from DataLoaders, gradient
        accumulation factor and distributed setup.

        Examples::

            def configure_optimizers(self):
                optimizer = ...
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
                )
                return [optimizer], [scheduler]

        """
        accumulation_scheduler = self.accumulation_scheduler

        if accumulation_scheduler.epochs != [0]:
            raise MisconfigurationException(
                "Estimated stepping batches cannot be computed with different"
                " `accumulate_grad_batches` at different epochs."
            )

        # infinite training
        if self.max_epochs == -1 and self.max_steps == -1:
            return float("inf")

        if self.train_dataloader is None:
            rank_zero_info("Loading `train_dataloader` to estimate number of stepping batches.")
            self.reset_train_dataloader()

        total_batches = self.num_training_batches

        # iterable dataset
        if total_batches == float("inf"):
            return self.max_steps

        self.accumulate_grad_batches = accumulation_scheduler.get_accumulate_grad_batches(self.current_epoch)
        effective_batch_size = self.accumulate_grad_batches
        max_estimated_steps = math.ceil(total_batches / effective_batch_size) * max(self.max_epochs, 1)

        max_estimated_steps = min(max_estimated_steps, self.max_steps) if self.max_steps != -1 else max_estimated_steps
        return max_estimated_steps

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


def _determine_batch_limits(batches: Optional[Union[int, float]], name: str) -> Union[int, float]:
    if batches is None:
        # batches is optional to know if the user passed a value so that we can show the above info messages only to the
        # users that set a value explicitly
        return 1.0

    # differentiating based on the type can be error-prone for users. show a message describing the chosen behaviour
    if isinstance(batches, int) and batches == 1:
        if name == "limit_train_batches":
            message = "1 batch per epoch will be used."
        elif name == "val_check_interval":
            message = "validation will run after every batch."
        else:
            message = "1 batch will be used."
        rank_zero_info(f"`Trainer({name}=1)` was configured so {message}")
    elif isinstance(batches, float) and batches == 1.0:
        if name == "limit_train_batches":
            message = "100% of the batches per epoch will be used."
        elif name == "val_check_interval":
            message = "validation will run at the end of the training epoch."
        else:
            message = "100% of the batches will be used."
        rank_zero_info(f"`Trainer({name}=1.0)` was configured so {message}.")

    if 0 <= batches <= 1:
        return batches
    if batches > 1 and batches % 1.0 == 0:
        return int(batches)
    raise MisconfigurationException(
        f"You have passed invalid value {batches} for {name}, it has to be in [0.0, 1.0] or an int."
    )
