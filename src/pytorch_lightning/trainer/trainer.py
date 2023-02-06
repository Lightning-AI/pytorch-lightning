# Copyright The Lightning AI team.
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

# THIS FILE MUST READ EASILY, FOR UNDERSTANDING AND DEBUGGING PURPOSES.
# DO NOT OBSCURE THE TRAINING LOOP
# THIS IS A HARD REQUIREMENT TO CONTRIBUTING TO LIGHTNING
# WE FAVOR READABILITY OVER ENGINEERING-CONSTRUCTS BY DESIGN
# DO NOT REMOVE THIS NOTICE
# - WILLIAM FALCON

"""Trainer to automate the training."""
import inspect
import logging
import math
import os
import warnings
from argparse import _ArgumentGroup, ArgumentParser, Namespace
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union
from weakref import proxy

import torch
import torch.distributed as dist
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.imports import module_available
from packaging.version import Version
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import Literal

import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.data import _auto_add_worker_init_fn
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.types import _PATH
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.accelerators import Accelerator, TPUAccelerator
from pytorch_lightning.callbacks import Callback, Checkpoint, EarlyStopping, ProgressBarBase
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loops import PredictionLoop, TrainingEpochLoop
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.utilities import _parse_loop_limits, _reset_progress
from pytorch_lightning.plugins import ApexMixedPrecisionPlugin, MixedPrecisionPlugin, PLUGIN_INPUT, PrecisionPlugin
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.strategies import (
    DDPFullyShardedNativeStrategy,
    DDPStrategy,
    ParallelStrategy,
    SingleDeviceStrategy,
    Strategy,
)
from pytorch_lightning.trainer import call, setup
from pytorch_lightning.trainer.configuration_validator import verify_loop_configurations
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT,
    _PRECISION_INPUT_STR,
    AcceleratorConnector,
)
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.connectors.logger_connector.result import _OUT_DICT, _PBAR_DICT, _ResultCollection
from pytorch_lightning.trainer.connectors.signal_connector import SignalConnector
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.tuner.tuning import _TunerResult, Tuner
from pytorch_lightning.utilities import GradClipAlgorithmType, parsing
from pytorch_lightning.utilities.argparse import (
    _defaults_from_env_vars,
    add_argparse_args,
    from_argparse_args,
    parse_argparser,
    parse_env_variables,
)
from pytorch_lightning.utilities.auto_restart import _add_capture_metadata_collate
from pytorch_lightning.utilities.data import has_len_all_ranks
from pytorch_lightning.utilities.exceptions import ExitGracefullyException, MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.seed import isolate_rng
from pytorch_lightning.utilities.types import (
    _EVALUATE_OUTPUT,
    _PREDICT_OUTPUT,
    EVAL_DATALOADERS,
    LRSchedulerConfig,
    TRAIN_DATALOADERS,
)

log = logging.getLogger(__name__)
# warnings to ignore in trainer
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead"
)


class Trainer:
    @_defaults_from_env_vars
    def __init__(
        self,
        logger: Union[Logger, Iterable[Logger], bool] = True,
        enable_checkpointing: bool = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[_PATH] = None,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        num_nodes: int = 1,
        num_processes: Optional[int] = None,  # TODO: Remove in 2.0
        devices: Optional[Union[List[int], str, int]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in 2.0
        auto_select_gpus: Optional[bool] = None,  # TODO: Remove in 2.0
        tpu_cores: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in 2.0
        ipus: Optional[int] = None,  # TODO: Remove in 2.0
        enable_progress_bar: bool = True,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: Optional[int] = 1,
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
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = None,
        strategy: Optional[Union[str, Strategy]] = None,
        sync_batchnorm: bool = False,
        precision: _PRECISION_INPUT = 32,
        enable_model_summary: bool = True,
        num_sanity_val_steps: int = 2,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[Profiler, str]] = None,
        benchmark: Optional[bool] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        reload_dataloaders_every_n_epochs: int = 0,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        detect_anomaly: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        amp_backend: Optional[str] = None,  # TODO: Remove in v2.0.0
        amp_level: Optional[str] = None,  # TODO: Remove in v2.0.0
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = "max_size_cycle",
        inference_mode: bool = True,
    ) -> None:
        r"""
        Customize every aspect of training via flags.

        Args:

            accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
                as well as custom accelerator instances.

            accumulate_grad_batches: Accumulates grads every k batches or as set up in the dict.
                Default: ``None``.

            amp_backend: The mixed precision backend to use ("native" or "apex").
                Default: ``'native''``.

                .. deprecated:: v1.9
                    Setting ``amp_backend`` inside the ``Trainer`` is deprecated in v1.8.0 and will be removed
                    in v2.0.0. This argument was only relevant for apex which is being removed.

            amp_level: The optimization level to use (O1, O2, etc...). By default it will be set to "O2"
                if ``amp_backend`` is set to "apex".

                .. deprecated:: v1.8
                    Setting ``amp_level`` inside the ``Trainer`` is deprecated in v1.8.0 and will be removed
                    in v2.0.0.

            auto_lr_find: If set to True, will make trainer.tune() run a learning rate finder,
                trying to optimize initial learning for faster convergence. trainer.tune() method will
                set the suggested learning rate in self.lr or self.learning_rate in the LightningModule.
                To use a different key set a string instead of True with the key name.
                Default: ``False``.

            auto_scale_batch_size: If set to True, will `initially` run a batch size
                finder trying to find the largest batch size that fits into memory.
                The result will be stored in self.batch_size in the LightningModule
                or LightningDataModule depending on your setup.
                Additionally, can be set to either `power` that estimates the batch size through
                a power search or `binsearch` that estimates the batch size through a binary search.
                Default: ``False``.

            auto_select_gpus: If enabled and ``gpus`` or ``devices`` is an integer, pick available
                gpus automatically. This is especially useful when
                GPUs are configured to be in "exclusive mode", such
                that only one process at a time can access them.
                Default: ``False``.

                .. deprecated:: v1.9
                    ``auto_select_gpus`` has been deprecated in v1.9.0 and will be removed in v2.0.0.
                    Please use the function :func:`~lightning_fabric.accelerators.cuda.find_usable_cuda_devices`
                    instead.

            benchmark: The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to.
                The value for ``torch.backends.cudnn.benchmark`` set in the current session will be used
                (``False`` if not manually set). If :paramref:`~pytorch_lightning.trainer.Trainer.deterministic` is set
                to ``True``, this will default to ``False``. Override to manually set a different value.
                Default: ``None``.

            callbacks: Add a callback or list of callbacks.
                Default: ``None``.

            enable_checkpointing: If ``True``, enable checkpointing.
                It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`.
                Default: ``True``.

            check_val_every_n_epoch: Perform a validation loop every after every `N` training epochs. If ``None``,
                validation will be done solely based on the number of training batches, requiring ``val_check_interval``
                to be an integer value.
                Default: ``1``.

            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Default: ``os.getcwd()``.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

            detect_anomaly: Enable anomaly detection for the autograd engine.
                Default: ``False``.

            deterministic: If ``True``, sets whether PyTorch operations must use deterministic algorithms.
                Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations
                that don't support deterministic mode (requires PyTorch 1.11+). If not set, defaults to ``False``.
                Default: ``None``.

            devices: Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`,
                based on the accelerator type.

            fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
                of train, val and test to find any bugs (ie: a sort of unit test).
                Default: ``False``.

            gpus: Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node
                Default: ``None``.

                .. deprecated:: v1.7
                    ``gpus`` has been deprecated in v1.7 and will be removed in v2.0.
                    Please use ``accelerator='gpu'`` and ``devices=x`` instead.

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
                the default ``TensorBoardLogger`` if it is installed, otherwise ``CSVLogger``.
                ``False`` will disable logging. If multiple loggers are provided, local files
                (checkpoints, profiler traces, etc.) are saved in the ``log_dir`` of he first logger.
                Default: ``True``.

            log_every_n_steps: How often to log within steps.
                Default: ``50``.

            enable_progress_bar: Whether to enable to progress bar by default.
                Default: ``True``.

            profiler: To profile individual steps during training and assist in identifying bottlenecks.
                Default: ``None``.

            overfit_batches: Overfit a fraction of training/validation data (float) or a set number of batches (int).
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

                .. deprecated:: v1.7
                    ``num_processes`` has been deprecated in v1.7 and will be removed in v2.0.
                    Please use ``accelerator='cpu'`` and ``devices=x`` instead.

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

            tpu_cores: How many TPU cores to train on (1 or 8) / Single TPU to train on (1)
                Default: ``None``.

                .. deprecated:: v1.7
                    ``tpu_cores`` has been deprecated in v1.7 and will be removed in v2.0.
                    Please use ``accelerator='tpu'`` and ``devices=x`` instead.

            ipus: How many IPUs to train on.
                Default: ``None``.

                .. deprecated:: v1.7
                    ``ipus`` has been deprecated in v1.7 and will be removed in v2.0.
                    Please use ``accelerator='ipu'`` and ``devices=x`` instead.

            track_grad_norm: -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm. If using
                Automatic Mixed Precision (AMP), the gradients will be unscaled before logging them.
                Default: ``-1``.

            val_check_interval: How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
                after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
                batches. An ``int`` value can only be higher than the number of training batches when
                ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
                across epochs or during iteration-based training.
                Default: ``1.0``.

            enable_model_summary: Whether to enable model summarization by default.
                Default: ``True``.

            move_metrics_to_cpu: Whether to force internal logged metrics to be moved to cpu.
                This can save some gpu memory, but can make training slower. Use with attention.
                Default: ``False``.

            multiple_trainloader_mode: How to loop over the datasets when there are multiple train loaders.
                In 'max_size_cycle' mode, the trainer ends one epoch when the largest dataset is traversed,
                and smaller datasets reload when running out of their data. In 'min_size' mode, all the datasets
                reload when reaching the minimum length of datasets.
                Default: ``"max_size_cycle"``.

            inference_mode: Whether to use :func:`torch.inference_mode` or :func:`torch.no_grad` during
                evaluation (``validate``/``test``/``predict``).
        """
        super().__init__()
        Trainer._log_api_event("init")
        log.detail(f"{self.__class__.__name__}: Initializing trainer with parameters: {locals()}")
        self.state = TrainerState()

        if default_root_dir is not None:
            default_root_dir = os.fspath(default_root_dir)

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
        self._logger_connector = LoggerConnector(self)
        self._callback_connector = CallbackConnector(self)
        self._checkpoint_connector = CheckpointConnector(self, resume_from_checkpoint)
        self._signal_connector = SignalConnector(self)
        self.tuner = Tuner(self)

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

        # init callbacks
        # Declare attributes to be set in _callback_connector on_trainer_init
        self._callback_connector.on_trainer_init(
            callbacks,
            enable_checkpointing,
            enable_progress_bar,
            default_root_dir,
            enable_model_summary,
            max_time,
            accumulate_grad_batches,
        )

        # init data flags
        self.check_val_every_n_epoch: Optional[int]
        self._data_connector.on_trainer_init(
            val_check_interval,
            reload_dataloaders_every_n_epochs,
            check_val_every_n_epoch,
        )

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

        self.gradient_clip_val: Optional[Union[int, float]] = gradient_clip_val
        self.gradient_clip_algorithm: Optional[GradClipAlgorithmType] = (
            GradClipAlgorithmType(gradient_clip_algorithm.lower()) if gradient_clip_algorithm is not None else None
        )
        self.track_grad_norm: float = float(track_grad_norm)

        self._inference_mode: bool = inference_mode

        self._detect_anomaly: bool = detect_anomaly
        self._setup_on_init()

        # configure tuner
        self.tuner.on_trainer_init(auto_lr_find, auto_scale_batch_size)

        # configure profiler
        setup._init_profiler(self, profiler)

        # init logger flags
        self._loggers: List[Logger]
        self._logger_connector.on_trainer_init(logger, log_every_n_steps, move_metrics_to_cpu)

        # init debugging flags
        self.val_check_batch: Union[int, float]
        self.val_check_interval: Union[int, float]
        self.num_sanity_val_steps: Union[int, float]
        self.limit_train_batches: Union[int, float]
        self.limit_val_batches: Union[int, float]
        self.limit_test_batches: Union[int, float]
        self.limit_predict_batches: Union[int, float]
        setup._init_debugging_flags(
            self,
            limit_train_batches,
            limit_val_batches,
            limit_test_batches,
            limit_predict_batches,
            fast_dev_run,
            overfit_batches,
            val_check_interval,
            num_sanity_val_steps,
        )

    def _setup_on_init(self) -> None:
        setup._log_device_info(self)

        self.should_stop = False
        self.state = TrainerState()
        self.num_training_batches = float("inf")

        self.train_dataloader: Optional[Union[CombinedLoader, TRAIN_DATALOADERS]] = None

        self.num_sanity_val_batches: List[Union[int, float]] = []
        self.num_test_batches: List[Union[int, float]] = []
        self.num_val_batches: List[Union[int, float]] = []
        self.num_predict_batches: List[Union[int, float]] = []

        self.test_dataloaders: Optional[List[DataLoader]] = None
        self.val_dataloaders: Optional[List[DataLoader]] = None
        self.predict_dataloaders: Optional[List[DataLoader]] = None
        self._last_train_dl_reload_epoch = float("-inf")
        self._last_val_dl_reload_epoch = float("-inf")

    def _maybe_unwrap_optimized(self, model: object) -> "pl.LightningModule":
        if not _TORCH_GREATER_EQUAL_2_0:
            if not isinstance(model, pl.LightningModule):
                raise TypeError(f"`model` must be a `LightningModule`, got `{type(model).__qualname__}`")
            return model
        from torch._dynamo import OptimizedModule

        if isinstance(model, OptimizedModule):
            return model.from_compiled(model)
        if isinstance(model, pl.LightningModule):
            return model
        raise TypeError(
            f"`model` must be a `LightningModule` or `torch._dynamo.OptimizedModule`, got `{type(model).__qualname__}`"
        )

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

            ckpt_path: Path/URL of the checkpoint from which training is resumed. Could also be one of two special
                keywords ``"last"`` and ``"hpc"``. If there is no checkpoint file at the path, an exception is raised.
                If resuming from mid-epoch checkpoint, training will start from the beginning of the next epoch.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
        """
        model = self._maybe_unwrap_optimized(model)
        self.strategy._lightning_module = model
        call._call_and_handle_interrupt(
            self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
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
        self._ckpt_path = self._checkpoint_connector._set_ckpt_path(
            self.state.fn,
            ckpt_path,  # type: ignore[arg-type]
            model_provided=True,
            model_connected=self.lightning_module is not None,
        )
        self._run(model, ckpt_path=self.ckpt_path)

        assert self.state.stopped
        self.training = False
        return

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

            ckpt_path: Either ``"best"``, ``"last"``, ``"hpc"`` or path to the checkpoint you wish to validate.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.

            verbose: If True, prints the validation results.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

        Returns:
            List of dictionaries with metrics logged during the validation phase, e.g., in model- or callback hooks
            like :meth:`~pytorch_lightning.core.module.LightningModule.validation_step`,
            :meth:`~pytorch_lightning.core.module.LightningModule.validation_epoch_end`, etc.
            The length of the list corresponds to the number of validation dataloaders used.
        """
        if model is None:
            # do we still have a reference from a previous call?
            if self.lightning_module is None:
                raise TypeError(
                    "`Trainer.validate()` requires a `LightningModule` when it hasn't been passed in a previous run"
                )
        else:
            model = self._maybe_unwrap_optimized(model)
            self.strategy._lightning_module = model
        return call._call_and_handle_interrupt(
            self, self._validate_impl, model, dataloaders, ckpt_path, verbose, datamodule
        )

    def _validate_impl(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ) -> Optional[Union[_PREDICT_OUTPUT, _EVALUATE_OUTPUT]]:
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

        if model is None:
            model = self.lightning_module
            model_provided = False
        else:
            model_provided = True

        self.validate_loop.verbose = verbose

        # links data to the trainer
        self._data_connector.attach_data(model, val_dataloaders=dataloaders, datamodule=datamodule)

        self._ckpt_path = self._checkpoint_connector._set_ckpt_path(
            self.state.fn, ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
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

            ckpt_path: Either ``"best"``, ``"last"``, ``"hpc"`` or path to the checkpoint you wish to test.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.

            verbose: If True, prints the test results.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

        Returns:
            List of dictionaries with metrics logged during the test phase, e.g., in model- or callback hooks
            like :meth:`~pytorch_lightning.core.module.LightningModule.test_step`,
            :meth:`~pytorch_lightning.core.module.LightningModule.test_epoch_end`, etc.
            The length of the list corresponds to the number of test dataloaders used.
        """
        if model is None:
            # do we still have a reference from a previous call?
            if self.lightning_module is None:
                raise TypeError(
                    "`Trainer.test()` requires a `LightningModule` when it hasn't been passed in a previous run"
                )
        else:
            model = self._maybe_unwrap_optimized(model)
            self.strategy._lightning_module = model
        return call._call_and_handle_interrupt(
            self, self._test_impl, model, dataloaders, ckpt_path, verbose, datamodule
        )

    def _test_impl(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[str] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ) -> Optional[Union[_PREDICT_OUTPUT, _EVALUATE_OUTPUT]]:
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

        if model is None:
            model = self.lightning_module
            model_provided = False
        else:
            model_provided = True

        self.test_loop.verbose = verbose

        # links data to the trainer
        self._data_connector.attach_data(model, test_dataloaders=dataloaders, datamodule=datamodule)

        self._ckpt_path = self._checkpoint_connector._set_ckpt_path(
            self.state.fn, ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
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

            ckpt_path: Either ``"best"``, ``"last"``, ``"hpc"`` or path to the checkpoint you wish to predict.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.

        Returns:
            Returns a list of dictionaries, one for each provided dataloader containing their respective predictions.

        See :ref:`Lightning inference section<deploy/production_basic:Predict step with your LightningModule>` for more.
        """
        if model is None:
            # do we still have a reference from a previous call?
            if self.lightning_module is None:
                raise TypeError(
                    "`Trainer.predict()` requires a `LightningModule` when it hasn't been passed in a previous run"
                )
        else:
            model = self._maybe_unwrap_optimized(model)
            self.strategy._lightning_module = model
        return call._call_and_handle_interrupt(
            self, self._predict_impl, model, dataloaders, datamodule, return_predictions, ckpt_path
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

        self.predict_loop.return_predictions = return_predictions  # type: ignore[assignment]

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(dataloaders, LightningDataModule):
            datamodule = dataloaders
            dataloaders = None
        if dataloaders is not None and datamodule:
            raise MisconfigurationException("You cannot pass both `trainer.predict(dataloaders=..., datamodule=...)`")

        if model is None:
            model = self.lightning_module
            model_provided = False
        else:
            model_provided = True

        # links data to the trainer
        self._data_connector.attach_data(model, predict_dataloaders=dataloaders, datamodule=datamodule)

        self._ckpt_path = self._checkpoint_connector._set_ckpt_path(
            self.state.fn, ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
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
        dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        scale_batch_size_kwargs: Optional[Dict[str, Any]] = None,
        lr_find_kwargs: Optional[Dict[str, Any]] = None,
        method: Literal["fit", "validate", "test", "predict"] = "fit",
    ) -> _TunerResult:
        r"""
        Runs routines to tune hyperparameters before training.

        Args:
            model: Model to tune.

            train_dataloaders: A collection of :class:`torch.utils.data.DataLoader` or a
                :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying training samples.
                In the case of multiple dataloaders, please see this :ref:`section <multiple-dataloaders>`.

            val_dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.

            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying val/test/predict
                samples used for running tuner on validation/testing/prediction.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

            scale_batch_size_kwargs: Arguments for :func:`~pytorch_lightning.tuner.batch_size_scaling.scale_batch_size`

            lr_find_kwargs: Arguments for :func:`~pytorch_lightning.tuner.lr_finder.lr_find`

            method: Method to run tuner on. It can be any of ``("fit", "validate", "test", "predict")``.
        """
        model = self._maybe_unwrap_optimized(model)
        Trainer._log_api_event("tune")

        with isolate_rng():
            result = self.tuner._tune(
                model,
                train_dataloaders,
                val_dataloaders,
                dataloaders,
                datamodule,
                scale_batch_size_kwargs=scale_batch_size_kwargs,
                lr_find_kwargs=lr_find_kwargs,
                method=method,
            )

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
        if model._compiler_ctx is not None:
            supported_strategies = [SingleDeviceStrategy, DDPStrategy, DDPFullyShardedNativeStrategy]
            if self.strategy is not None and not any(isinstance(self.strategy, s) for s in supported_strategies):
                supported_strategy_names = ", ".join(s.__name__ for s in supported_strategies)
                raise RuntimeError(
                    "Using a compiled model is incompatible with the current strategy: "
                    f"{self.strategy.__class__.__name__}. "
                    f"Only {supported_strategy_names} support compilation. "
                    "Either switch to one of the supported strategies or avoid passing in "
                    "a compiled model."
                )

        if self.state.fn == TrainerFn.FITTING:
            min_epochs, max_epochs = _parse_loop_limits(
                self.min_steps, self.max_steps, self.min_epochs, self.max_epochs, self
            )
            self.fit_loop.min_epochs = min_epochs
            self.fit_loop.max_epochs = max_epochs

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
                elif isinstance(lm_val, Tensor) and id(lm_val) != id(dm_val):
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

    def _teardown(self) -> None:
        """This is the Trainer's internal teardown, unrelated to the `teardown` hooks in LightningModule and
        Callback; those are handled by :meth:`_call_teardown_hook`."""
        self.strategy.teardown()
        loop = self._active_loop
        # loop should never be `None` here but it can because we don't know the trainer stage with `ddp_spawn`
        if loop is not None:
            loop.teardown()
        self._logger_connector.teardown()
        self._signal_connector.teardown()

    def _run_stage(self) -> Optional[Union[_PREDICT_OUTPUT, _EVALUATE_OUTPUT]]:
        self.strategy.barrier("run-stage")
        self.strategy.dispatch(self)

        if self.evaluating:
            return self._run_evaluate()
        if self.predicting:
            return self._run_predict()
        self._run_train()

    def _pre_training_routine(self) -> None:
        # wait for all to join if on distributed
        self.strategy.barrier("setup_training")

        # register signals
        self._signal_connector.register_signal_handlers()

    def _run_train(self) -> None:
        self._pre_training_routine()

        with isolate_rng():
            self._run_sanity_check()

        # enable train mode
        assert self.model is not None
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

        with self.profiler.profile(f"run_{self.state.stage}_evaluation"), _evaluation_context(
            self.accelerator, self._inference_mode
        ):
            eval_loop_results = self._evaluation_loop.run()

        # remove the tensors from the eval results
        for result in eval_loop_results:
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, Tensor):
                        result[k] = v.cpu().item()

        return eval_loop_results

    def _run_predict(self) -> Optional[_PREDICT_OUTPUT]:
        self.reset_predict_dataloader(self.lightning_module)
        # reset trainer on this loop and all child loops in case user connected a custom loop
        self.predict_loop.trainer = self
        with _evaluation_context(self.accelerator, self._inference_mode):
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

    def _call_setup_hook(self) -> None:
        assert self.state.fn is not None
        fn = self.state.fn

        self.strategy.barrier("pre_setup")

        if self.datamodule is not None:
            self._call_lightning_datamodule_hook("setup", stage=fn)
        self._call_callback_hooks("setup", stage=fn)
        self._call_lightning_module_hook("setup", stage=fn)

        self.strategy.barrier("post_setup")

    def _call_configure_sharded_model(self) -> None:
        with self.strategy.model_sharded_context():
            # experimental support for torchdistx
            if module_available("torchdistx.deferred_init"):
                from torchdistx.deferred_init import materialize_module

                materialize_module(self.lightning_module)

            self._call_lightning_module_hook("configure_sharded_model")

    def _call_teardown_hook(self) -> None:
        assert self.state.fn is not None
        fn = self.state.fn

        if self.datamodule is not None:
            self._call_lightning_datamodule_hook("teardown", stage=fn)

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

    def _call_lightning_module_hook(
        self,
        hook_name: str,
        *args: Any,
        pl_module: Optional["pl.LightningModule"] = None,
        **kwargs: Any,
    ) -> Any:
        pl_module = pl_module or self.lightning_module

        if pl_module is None:
            raise TypeError("No `LightningModule` is available to call hooks on.")

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

    def _call_lightning_datamodule_hook(
        self,
        hook_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self.datamodule is None:
            raise TypeError("No `LightningDataModule` is available to call hooks on.")

        fn = getattr(self.datamodule, hook_name)
        if callable(fn):
            with self.profiler.profile(f"[LightningDataModule]{self.datamodule.__class__.__name__}.{hook_name}"):
                return fn(*args, **kwargs)

    def _call_callback_hooks(
        self,
        hook_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        log.debug(f"{self.__class__.__name__}: calling callback hook: {hook_name}")

        pl_module = self.lightning_module
        if pl_module:
            prev_fx_name = pl_module._current_fx_name
            pl_module._current_fx_name = hook_name

        for callback in self.callbacks:
            fn = getattr(callback, hook_name)
            if callable(fn):
                with self.profiler.profile(f"[Callback]{callback.state_key}.{hook_name}"):
                    fn(self, self.lightning_module, *args, **kwargs)

        if pl_module:
            # restore current_fx when nested context
            pl_module._current_fx_name = prev_fx_name

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
        """Called when saving a model checkpoint, calls every callback's `on_save_checkpoint` hook."""
        pl_module = self.lightning_module
        if pl_module:
            prev_fx_name = pl_module._current_fx_name
            pl_module._current_fx_name = "on_save_checkpoint"

        for callback in self.callbacks:
            with self.profiler.profile(f"[Callback]{callback.state_key}.on_save_checkpoint"):
                state = callback.on_save_checkpoint(self, self.lightning_module, checkpoint)
            if state is not None:
                # TODO: Remove this error message in v2.0
                raise ValueError(
                    f"Returning a value from `{callback.__class__.__name__}.on_save_checkpoint` was deprecated in v1.6"
                    f" and is no longer supported as of v1.8. Please override `Callback.state_dict` to return state"
                    f" to be saved."
                )

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

        callback_states: Optional[Dict[Union[Type, str], Dict]] = checkpoint.get("callbacks")

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
            with self.profiler.profile(f"[Callback]{callback.state_key}.on_load_checkpoint"):
                callback.on_load_checkpoint(self, self.lightning_module, checkpoint)

        if pl_module:
            # restore current_fx when nested context
            pl_module._current_fx_name = prev_fx_name

    def _call_callbacks_load_state_dict(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading a model checkpoint, calls every callback's `load_state_dict`."""
        callback_states: Optional[Dict[Union[Type, str], Dict]] = checkpoint.get("callbacks")

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

    def __setup_profiler(self) -> None:
        assert self.state.fn is not None
        local_rank = self.local_rank if self.world_size > 1 else None
        self.profiler._lightning_module = proxy(self.lightning_module)
        self.profiler.setup(stage=self.state.fn, local_rank=local_rank, log_dir=self.log_dir)

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
        pl_module = model or self.lightning_module
        has_step = is_overridden("training_step", pl_module)
        enable_training = self.limit_train_batches > 0
        if not (source.is_defined() and has_step and enable_training):
            return

        self.train_dataloader = self._data_connector._request_dataloader(RunningStage.TRAINING)

        if self.overfit_batches > 0:
            self.train_dataloader = self._data_connector._resolve_overfit_batches(
                self.train_dataloader, mode=RunningStage.TRAINING
            )

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
            len(self.train_dataloader)  # type: ignore[arg-type]
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
            if self.val_check_batch > self.num_training_batches and self.check_val_every_n_epoch is not None:
                raise ValueError(
                    f"`val_check_interval` ({self.val_check_interval}) must be less than or equal "
                    f"to the number of the training batches ({self.num_training_batches}). "
                    "If you want to disable validation set `limit_val_batches` to 0.0 instead."
                    "If you want to validate based on the total training batches, set `check_val_every_n_epoch=None`."
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
            # store epoch of dataloader reset for reload_dataloaders_every_n_epochs
            # it should not reload again if it has already reloaded during sanity_check
            if self.state.fn == TrainerFn.FITTING and (
                (self.sanity_checking and self.fit_loop.epoch_loop._should_check_val_epoch())
                or not self.sanity_checking
            ):
                self._last_val_dl_reload_epoch = self.current_epoch

            self.num_val_batches, self.val_dataloaders = self._data_connector._reset_eval_dataloader(
                RunningStage.VALIDATING, model=pl_module
            )

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

    """
    Accelerator properties
    """

    @property
    def accelerator(self) -> Accelerator:
        assert self.strategy.accelerator
        return self.strategy.accelerator

    @property
    def strategy(self) -> Strategy:
        # TODO(fabric): remove ignore after merging Fabric and PL strategies
        return self._accelerator_connector.strategy  # type: ignore[return-value]

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
        assert devices is not None
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
    def lightning_module(self) -> "pl.LightningModule":
        # TODO: this is actually an optional return
        return self.strategy.lightning_module  # type: ignore[return-value]

    @property
    def optimizers(self) -> List[Optimizer]:
        return self.strategy.optimizers

    @optimizers.setter
    def optimizers(self, new_optims: List[Optimizer]) -> None:
        self.strategy.optimizers = new_optims

    @property
    def lr_scheduler_configs(self) -> List[LRSchedulerConfig]:
        return self.strategy.lr_scheduler_configs

    @property
    def optimizer_frequencies(self) -> List[int]:
        return self.strategy.optimizer_frequencies

    @optimizer_frequencies.setter
    def optimizer_frequencies(self, new_freqs: List[int]) -> None:
        self.strategy.optimizer_frequencies = new_freqs

    @property
    def amp_backend(self) -> Optional[str]:
        rank_zero_deprecation(
            "The NVIDIA/apex AMP implementation has been deprecated upstream. Consequently, its integration inside"
            " PyTorch Lightning has been deprecated in v1.9.0 and will be removed in v2.0.0."
            " Accessing `Trainer.amp_backend` will not be supported. You can assume it will be `'native'`",
            stacklevel=6,
        )
        if isinstance(self.precision_plugin, ApexMixedPrecisionPlugin):
            return "apex"
        if isinstance(self.precision_plugin, MixedPrecisionPlugin):
            return "native"
        return None

    @property
    def precision(self) -> _PRECISION_INPUT_STR:
        return self.strategy.precision_plugin.precision

    @property
    def scaler(self) -> Optional[Any]:
        return getattr(self.precision_plugin, "scaler", None)

    @property
    def model(self) -> Optional[torch.nn.Module]:
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
        if len(self.loggers) > 0:
            if not isinstance(self.loggers[0], TensorBoardLogger):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath

    @property
    def is_global_zero(self) -> bool:
        return self.strategy.is_global_zero

    @property
    def distributed_sampler_kwargs(self) -> Optional[Dict[str, Any]]:
        if isinstance(self.strategy, ParallelStrategy):
            return self.strategy.distributed_sampler_kwargs

    @property
    def data_parallel(self) -> bool:
        return isinstance(self.strategy, ParallelStrategy)

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
    def checkpoint_callback(self) -> Optional[Checkpoint]:
        """The first :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` callback in the
        Trainer.callbacks list, or ``None`` if it doesn't exist."""
        callbacks = self.checkpoint_callbacks
        return callbacks[0] if len(callbacks) > 0 else None

    @property
    def checkpoint_callbacks(self) -> List[Checkpoint]:
        """A list of all instances of :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` found
        in the Trainer.callbacks list."""
        return [c for c in self.callbacks if isinstance(c, Checkpoint)]

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
        if self.model is None:
            raise AttributeError(
                "Saving a checkpoint is only possible if a model is attached to the Trainer. Did you call"
                " `Trainer.save_checkpoint()` before calling `Trainer.{fit,validate,test,predict}`?"
            )
        self._checkpoint_connector.save_checkpoint(filepath, weights_only=weights_only, storage_options=storage_options)

    """
    Parsing properties
    """

    @classmethod
    def default_attributes(cls) -> dict:
        init_signature = inspect.signature(cls)
        return {k: v.default for k, v in init_signature.parameters.items()}

    @classmethod
    def from_argparse_args(cls: Any, args: Union[Namespace, ArgumentParser], **kwargs: Any) -> Any:
        return from_argparse_args(cls, args, **kwargs)

    @classmethod
    def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        return parse_argparser(cls, arg_parser)

    @classmethod
    def match_env_arguments(cls) -> Namespace:
        return parse_env_variables(cls)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs: Any) -> Union[_ArgumentGroup, ArgumentParser]:
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
        rank_zero_deprecation("`Trainer.tuning` has been deprecated in v1.8.0 and will be removed in v2.0.0.")
        return self.state.stage == RunningStage.TUNING

    @tuning.setter
    def tuning(self, val: bool) -> None:
        rank_zero_deprecation("Setting `Trainer.tuning` has been deprecated in v1.8.0 and will be removed in v2.0.0.")

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
        return self.state.stage is not None and self.state.stage.evaluating

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
    def max_epochs(self) -> Optional[int]:
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
        """Whether trainer is executing the last batch."""
        return self.fit_loop.epoch_loop.batch_progress.is_last_batch

    @property
    def fit_loop(self) -> FitLoop:
        return self._fit_loop

    @fit_loop.setter
    def fit_loop(self, loop: FitLoop) -> None:
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
    def validate_loop(self, loop: EvaluationLoop) -> None:
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
    def test_loop(self, loop: EvaluationLoop) -> None:
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
    def predict_loop(self, loop: PredictionLoop) -> None:
        """Attach a custom prediction loop to this Trainer.

        It will run with
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`.
        """
        loop.trainer = self
        self._predict_loop = loop

    @property
    def _evaluation_loop(self) -> EvaluationLoop:
        if self.state.fn == TrainerFn.FITTING:
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
    def logger(self) -> Optional[Logger]:
        return self.loggers[0] if len(self.loggers) > 0 else None

    @logger.setter
    def logger(self, logger: Optional[Logger]) -> None:
        if not logger:
            self.loggers = []
        else:
            self.loggers = [logger]

    @property
    def loggers(self) -> List[Logger]:
        return self._loggers

    @loggers.setter
    def loggers(self, loggers: Optional[List[Logger]]) -> None:
        self._loggers = loggers if loggers else []

    @property
    def callback_metrics(self) -> Dict:
        # TODO: the true typing return can include dictionaries as defined in
        # `pytorch_lightning.trainer.connectors.logger_connector.result._OUT_DICT`
        return self._logger_connector.callback_metrics

    @property
    def logged_metrics(self) -> _OUT_DICT:
        return self._logger_connector.logged_metrics

    @property
    def progress_bar_metrics(self) -> _PBAR_DICT:
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
        return bool(self.strategy.reduce(value, reduce_op="sum") > 0)

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
        if self.max_epochs == -1:
            return float("inf") if self.max_steps == -1 else self.max_steps

        if self.train_dataloader is None:
            rank_zero_info("Loading `train_dataloader` to estimate number of stepping batches.")
            self.reset_train_dataloader()

        total_batches = self.num_training_batches

        # iterable dataset
        if total_batches == float("inf"):
            return self.max_steps

        assert self.max_epochs is not None
        self.accumulate_grad_batches = accumulation_scheduler.get_accumulate_grad_batches(self.current_epoch)
        effective_batch_size = self.accumulate_grad_batches
        max_estimated_steps = math.ceil(total_batches / effective_batch_size) * max(self.max_epochs, 1)

        max_estimated_steps = min(max_estimated_steps, self.max_steps) if self.max_steps != -1 else max_estimated_steps
        return max_estimated_steps


@contextmanager
def _evaluation_context(accelerator: Accelerator, inference_mode: bool = True) -> Generator:
    # inference mode is not supported with gloo backend (#9431) and TPU accelerators.
    context_manager_class = (
        torch.inference_mode
        if inference_mode
        and not (dist.is_available() and dist.is_initialized() and dist.get_backend() == "gloo")
        and not isinstance(accelerator, TPUAccelerator)
        else torch.no_grad
    )
    with context_manager_class():
        yield
