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
import logging
import os
import traceback
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from weakref import proxy

import torch

import pytorch_lightning as pl
from pytorch_lightning.accelerators import Accelerator, IPUAccelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loops import TrainingBatchLoop, TrainingEpochLoop
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.dataloader.prediction_loop import PredictionLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.plugins import Plugin
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.profiler import (
    AdvancedProfiler,
    BaseProfiler,
    PassThroughProfiler,
    PyTorchProfiler,
    SimpleProfiler,
    XLAProfiler,
)
from pytorch_lightning.trainer.callback_hook import TrainerCallbackHookMixin
from pytorch_lightning.trainer.configuration_validator import ConfigValidator
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.trainer.connectors.debugging_connector import DebuggingConnector
from pytorch_lightning.trainer.connectors.env_vars_connector import _defaults_from_env_vars
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.connectors.model_connector import ModelConnector
from pytorch_lightning.trainer.connectors.optimizer_connector import OptimizerConnector
from pytorch_lightning.trainer.connectors.slurm_connector import SLURMConnector
from pytorch_lightning.trainer.connectors.training_trick_connector import TrainingTricksConnector
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.deprecated_api import DeprecatedTrainerAttributes
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin
from pytorch_lightning.trainer.properties import TrainerProperties
from pytorch_lightning.trainer.states import TrainerFn, TrainerState, TrainerStatus
from pytorch_lightning.trainer.training_tricks import TrainerTrainingTricksMixin
from pytorch_lightning.tuner.auto_gpu_select import pick_multiple_gpus
from pytorch_lightning.tuner.lr_finder import _LRFinder
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities import (
    _IPU_AVAILABLE,
    _TPU_AVAILABLE,
    device_parser,
    DeviceType,
    parsing,
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_warn,
)
from pytorch_lightning.utilities.debugging import InternalDebugger
from pytorch_lightning.utilities.distributed import distributed_available
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_enabled
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS

log = logging.getLogger(__name__)
# warnings to ignore in trainer
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead"
)


class Trainer(
    TrainerProperties,
    TrainerCallbackHookMixin,
    TrainerModelHooksMixin,
    TrainerOptimizersMixin,
    TrainerLoggingMixin,
    TrainerTrainingTricksMixin,
    TrainerDataLoadingMixin,
    DeprecatedTrainerAttributes,
):
    @_defaults_from_env_vars
    def __init__(
        self,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        checkpoint_callback: bool = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = "norm",
        process_position: int = 0,
        num_nodes: int = 1,
        num_processes: int = 1,
        devices: Optional[Union[List[int], str, int]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        ipus: Optional[int] = None,
        log_gpu_memory: Optional[str] = None,
        progress_bar_refresh_rate: Optional[int] = None,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        limit_predict_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        flush_logs_every_n_steps: int = 100,
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = None,
        sync_batchnorm: bool = False,
        precision: int = 32,
        weights_summary: Optional[str] = "top",
        weights_save_path: Optional[str] = None,
        num_sanity_val_steps: int = 2,
        truncated_bptt_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[BaseProfiler, str]] = None,
        benchmark: bool = False,
        deterministic: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        reload_dataloaders_every_epoch: bool = False,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        terminate_on_nan: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        plugins: Optional[Union[List[Union[Plugin, ClusterEnvironment, str]], Plugin, ClusterEnvironment, str]] = None,
        amp_backend: str = "native",
        amp_level: str = "O2",
        distributed_backend: Optional[str] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = "max_size_cycle",
        stochastic_weight_avg: bool = False,
    ):
        r"""
        Customize every aspect of training via flags

        Args:

            accelerator: Previously known as distributed_backend (dp, ddp, ddp2, etc...).
                Can also take in an accelerator object for custom hardware.

            accumulate_grad_batches: Accumulates grads every k batches or as set up in the dict.

            amp_backend: The mixed precision backend to use ("native" or "apex")

            amp_level: The optimization level to use (O1, O2, etc...).

            auto_lr_find: If set to True, will make trainer.tune() run a learning rate finder,
                trying to optimize initial learning for faster convergence. trainer.tune() method will
                set the suggested learning rate in self.lr or self.learning_rate in the LightningModule.
                To use a different key set a string instead of True with the key name.

            auto_scale_batch_size: If set to True, will `initially` run a batch size
                finder trying to find the largest batch size that fits into memory.
                The result will be stored in self.batch_size in the LightningModule.
                Additionally, can be set to either `power` that estimates the batch size through
                a power search or `binsearch` that estimates the batch size through a binary search.

            auto_select_gpus: If enabled and `gpus` is an integer, pick available
                gpus automatically. This is especially useful when
                GPUs are configured to be in "exclusive mode", such
                that only one process at a time can access them.

            benchmark: If true enables cudnn.benchmark.

            callbacks: Add a callback or list of callbacks.

            checkpoint_callback: If ``True``, enable checkpointing.
                It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`.

            check_val_every_n_epoch: Check val every n train epochs.

            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Default: ``os.getcwd()``.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

            deterministic: If true enables cudnn.deterministic.

            devices: Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`,
                based on the accelerator type.

            distributed_backend: deprecated. Please use 'accelerator'

            fast_dev_run: runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
                of train, val and test to find any bugs (ie: a sort of unit test).

            flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).

            gpus: number of gpus to train on (int) or which GPUs to train on (list or str) applied per node

            gradient_clip_val: 0 means don't clip.

            gradient_clip_algorithm: 'value' means clip_by_value, 'norm' means clip_by_norm. Default: 'norm'

            limit_train_batches: How much of training dataset to check (float = fraction, int = num_batches)

            limit_val_batches: How much of validation dataset to check (float = fraction, int = num_batches)

            limit_test_batches: How much of test dataset to check (float = fraction, int = num_batches)

            limit_predict_batches: How much of prediction dataset to check (float = fraction, int = num_batches)

            logger: Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
                the default ``TensorBoardLogger``. ``False`` will disable logging. If multiple loggers are
                provided and the `save_dir` property of that logger is not set, local files (checkpoints,
                profiler traces, etc.) are saved in ``default_root_dir`` rather than in the ``log_dir`` of any
                of the individual loggers.

            log_gpu_memory: None, 'min_max', 'all'. Might slow performance

            log_every_n_steps: How often to log within steps (defaults to every 50 steps).

            prepare_data_per_node: If True, each LOCAL_RANK=0 will call prepare data.
                Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data

            process_position: orders the progress bar when running multiple models on same machine.

            progress_bar_refresh_rate: How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
                Ignored when a custom progress bar is passed to :paramref:`~Trainer.callbacks`. Default: None, means
                a suitable value will be chosen based on the environment (terminal, Google COLAB, etc.).

            profiler: To profile individual steps during training and assist in identifying bottlenecks.

            overfit_batches: Overfit a fraction of training data (float) or a set number of batches (int).

            plugins: Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.

            precision: Double precision (64), full precision (32) or half precision (16). Can be used on CPU, GPU or
                TPUs.

            max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
                If both max_epochs and max_steps are not specified, defaults to ``max_epochs`` = 1000.

            min_epochs: Force training for at least these many epochs. Disabled by default (None).
                If both min_epochs and min_steps are not specified, defaults to ``min_epochs`` = 1.

            max_steps: Stop training after this number of steps. Disabled by default (None).

            min_steps: Force training for at least these number of steps. Disabled by default (None).

            max_time: Stop training after this amount of time has passed. Disabled by default (None).
                The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
                :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
                :class:`datetime.timedelta`.

            num_nodes: number of GPU nodes for distributed training.

            num_processes: number of processes for distributed training with distributed_backend="ddp_cpu"

            num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
                Set it to `-1` to run all batches in all validation dataloaders.

            reload_dataloaders_every_n_epochs: Set to a non-negative integer to reload dataloaders every n epochs.
                Default: 0

            reload_dataloaders_every_epoch: Set to True to reload dataloaders every epoch.

                .. deprecated:: v1.4
                    ``reload_dataloaders_every_epoch`` has been deprecated in v1.4 and will be removed in v1.6.
                    Please use ``reload_dataloaders_every_n_epochs``.

            replace_sampler_ddp: Explicitly enables or disables sampler replacement. If not specified this
                will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for
                train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it,
                you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.

            resume_from_checkpoint: Path/URL of the checkpoint from which training is resumed. If there is
                no checkpoint file at the path, start from scratch. If resuming from mid-epoch checkpoint,
                training will start from the beginning of the next epoch.

            sync_batchnorm: Synchronize batch norm layers between process groups/whole world.

            terminate_on_nan: If set to True, will terminate training (by raising a `ValueError`) at the
                end of each training batch, if any of the parameters or the loss are NaN or +/-inf.

            tpu_cores: How many TPU cores to train on (1 or 8) / Single TPU to train on [1]

            ipus: How many IPUs to train on.

            track_grad_norm: -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm.

            truncated_bptt_steps: Deprecated in v1.3 to be removed in 1.5.
                Please use :paramref:`~pytorch_lightning.core.lightning.LightningModule.truncated_bptt_steps` instead.

            val_check_interval: How often to check the validation set. Use float to check within a training epoch,
                use int to check every n steps (batches).

            weights_summary: Prints a summary of the weights when training begins.

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
                <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>_`

        """
        super().__init__()
        Trainer._log_api_event("init")
        self.state = TrainerState()

        gpu_ids, tpu_cores = self._parse_devices(gpus, auto_select_gpus, tpu_cores)

        # init connectors
        self.dev_debugger = InternalDebugger(self)
        self.config_validator = ConfigValidator(self)
        self.data_connector = DataConnector(self, multiple_trainloader_mode)
        self.optimizer_connector = OptimizerConnector(self)

        self.accelerator_connector = AcceleratorConnector(
            num_processes,
            devices,
            tpu_cores,
            ipus,
            distributed_backend,
            accelerator,
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
        self.model_connector = ModelConnector(self)
        self.callback_connector = CallbackConnector(self)
        self.debugging_connector = DebuggingConnector(self)
        self.training_tricks_connector = TrainingTricksConnector(self)
        self.checkpoint_connector = CheckpointConnector(self, resume_from_checkpoint)
        self.slurm_connector = SLURMConnector(self)
        self.tuner = Tuner(self)

        fit_loop = FitLoop(
            min_epochs=(1 if (min_epochs is None and min_steps is None and max_time is None) else min_epochs),
            max_epochs=(1000 if (max_epochs is None and max_steps is None and max_time is None) else max_epochs),
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

        # training state
        if weights_summary is not None and weights_summary not in ModelSummary.MODES:
            raise MisconfigurationException(
                f"`weights_summary` can be None, {', '.join(ModelSummary.MODES)}, but got {weights_summary}"
            )
        self.weights_summary = weights_summary

        # init callbacks
        # Declare attributes to be set in callback_connector on_trainer_init
        self.callback_connector.on_trainer_init(
            callbacks,
            checkpoint_callback,
            progress_bar_refresh_rate,
            process_position,
            default_root_dir,
            weights_save_path,
            stochastic_weight_avg,
            max_time,
        )

        # hook
        self.on_init_start()

        # init optimizer + lr scheduler related flags
        self.optimizer_connector.on_trainer_init()

        # init data flags
        self.data_connector.on_trainer_init(
            check_val_every_n_epoch,
            reload_dataloaders_every_n_epochs,
            reload_dataloaders_every_epoch,
            prepare_data_per_node,
        )

        # init training tricks
        self.training_tricks_connector.on_trainer_init(
            gradient_clip_val,
            gradient_clip_algorithm,
            track_grad_norm,
            accumulate_grad_batches,
            truncated_bptt_steps,
            terminate_on_nan,
        )
        self._setup_on_init(num_sanity_val_steps)

        # configure tuner
        self.tuner.on_trainer_init(auto_lr_find, auto_scale_batch_size)

        # configure profiler
        self.__init_profiler(profiler)

        # init logger flags
        self.logger_connector.on_trainer_init(logger, flush_logs_every_n_steps, log_every_n_steps, move_metrics_to_cpu)

        # init debugging flags
        self.debugging_connector.on_init_start(
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

    def _setup_on_init(self, num_sanity_val_steps: int) -> None:
        self._log_device_info()

        self.should_stop = False
        self.state = TrainerState()
        self.num_training_batches = 0
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

        # .validate() and .test() set this when they load a checkpoint
        self.validated_ckpt_path = None
        self.tested_ckpt_path = None

        # when true, print evaluation results in .validate() and .test()
        self.verbose_evaluate = True

        self.num_predict_batches = []
        self.predicted_ckpt_path = None

    def fit(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        train_dataloader=None,  # noqa TODO: remove with 1.6
    ) -> None:
        r"""
        Runs the full optimization routine.

        Args:
            model: Model to fit.

            train_dataloaders: A collection of :class:`torch.utils.data.DataLoader` or a
                :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying training samples.
                In the case of multiple dataloaders, please see this :ref:`page <multiple-training-dataloaders>`.

            val_dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
        """
        Trainer._log_api_event("fit")

        self.state.fn = TrainerFn.FITTING
        self.state.status = TrainerStatus.RUNNING
        self.training = True

        if train_dataloader is not None:
            rank_zero_deprecation(
                "`trainer.fit(train_dataloader)` is deprecated in v1.4 and will be removed in v1.6."
                " Use `trainer.fit(train_dataloaders)` instead. HINT: added 's'"
            )
            train_dataloaders = train_dataloader
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
        self.data_connector.attach_data(
            model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, datamodule=datamodule
        )

        self.checkpoint_connector.resume_start()

        self._run(model)

        assert self.state.stopped
        self.training = False

    def validate(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[str] = "best",
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
        val_dataloaders=None,  # noqa TODO: remove with 1.6
    ) -> _EVALUATE_OUTPUT:
        r"""
        Perform one evaluation epoch over the validation set.

        Args:
            model: The model to validate.

            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them,
                or a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying validation samples.

            ckpt_path: Either ``best`` or path to the checkpoint you wish to validate.
                If ``None``, use the current weights of the model.
                When the model is given as argument, this parameter will not apply.

            verbose: If True, prints the validation results.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

        Returns:
            List of dictionaries with metrics logged during the validation phase, e.g., in model- or callback hooks
            like :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step`,
            :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_epoch_end`, etc.
            The length of the list corresponds to the number of validation dataloaders used.
        """
        # --------------------
        # SETUP HOOK
        # --------------------
        Trainer._log_api_event("validate")
        self.verbose_evaluate = verbose

        self.state.fn = TrainerFn.VALIDATING
        self.state.status = TrainerStatus.RUNNING
        self.validating = True

        if val_dataloaders is not None:
            rank_zero_deprecation(
                "`trainer.validate(val_dataloaders)` is deprecated in v1.4 and will be removed in v1.6."
                " Use `trainer.validate(dataloaders)` instead."
            )
            dataloaders = val_dataloaders
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
        self.data_connector.attach_data(model, val_dataloaders=dataloaders, datamodule=datamodule)

        if not model_provided:
            self.validated_ckpt_path = self.__load_ckpt_weights(ckpt_path)

        # run validate
        results = self._run(model)

        assert self.state.stopped
        self.validating = False

        return results

    def test(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[str] = "best",
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
        test_dataloaders=None,  # noqa TODO: remove with 1.6
    ) -> _EVALUATE_OUTPUT:
        r"""
        Perform one evaluation epoch over the test set. It's separated from
        fit to make sure you never run on your test set until you want to.

        Args:
            model: The model to test.

            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them,
                or a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying test samples.

            ckpt_path: Either ``best`` or path to the checkpoint you wish to test.
                If ``None``, use the current weights of the model.
                When the model is given as argument, this parameter will not apply.

            verbose: If True, prints the test results.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

        Returns:
            List of dictionaries with metrics logged during the test phase, e.g., in model- or callback hooks
            like :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step`,
            :meth:`~pytorch_lightning.core.lightning.LightningModule.test_epoch_end`, etc.
            The length of the list corresponds to the number of test dataloaders used.
        """
        # --------------------
        # SETUP HOOK
        # --------------------
        Trainer._log_api_event("test")
        self.verbose_evaluate = verbose

        self.state.fn = TrainerFn.TESTING
        self.state.status = TrainerStatus.RUNNING
        self.testing = True

        if test_dataloaders is not None:
            rank_zero_deprecation(
                "`trainer.test(test_dataloaders)` is deprecated in v1.4 and will be removed in v1.6."
                " Use `trainer.test(dataloaders)` instead."
            )
            dataloaders = test_dataloaders
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
        self.data_connector.attach_data(model, test_dataloaders=dataloaders, datamodule=datamodule)

        if not model_provided:
            self.tested_ckpt_path = self.__load_ckpt_weights(ckpt_path)

        # run test
        results = self._run(model)

        assert self.state.stopped
        self.testing = False

        return results

    def predict(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        datamodule: Optional[LightningDataModule] = None,
        return_predictions: Optional[bool] = None,
        ckpt_path: Optional[str] = "best",
    ) -> Optional[_PREDICT_OUTPUT]:
        r"""

        Separates from fit to make sure you never run on your predictions set until you want to.
        This will call the model forward function to compute predictions.

        Args:
            model: The model to predict with.

            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them,
                or a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying prediction samples.

            datamodule: The datamodule with a predict_dataloader method that returns one or more dataloaders.

            return_predictions: Whether to return predictions.
                ``True`` by default except when an accelerator that spawns processes is used (not supported).

            ckpt_path: Either ``best`` or path to the checkpoint you wish to use to predict.
                If ``None``, use the current weights of the model.
                When the model is given as argument, this parameter will not apply.

        Returns:
            Returns a list of dictionaries, one for each provided dataloader containing their respective predictions.
        """

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
        self.data_connector.attach_data(model, predict_dataloaders=dataloaders, datamodule=datamodule)

        if not model_provided:
            self.predicted_ckpt_path = self.__load_ckpt_weights(ckpt_path)

        results = self._run(model)

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
        train_dataloader=None,  # noqa TODO: remove with 1.6
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
        self.data_connector.attach_data(
            model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, datamodule=datamodule
        )

        result = self.tuner._tune(model, scale_batch_size_kwargs=scale_batch_size_kwargs, lr_find_kwargs=lr_find_kwargs)

        assert self.state.stopped
        self.tuning = False

        return result

    def _run(self, model: "pl.LightningModule") -> Optional[Union[_EVALUATE_OUTPUT, _PREDICT_OUTPUT]]:
        # clean hparams
        if hasattr(model, "hparams"):
            parsing.clean_namespace(model.hparams)

        self.config_validator.verify_loop_configurations(model)

        # attach model log function to callback
        self.callback_connector.attach_model_logging_functions(model)

        # hook
        self.data_connector.prepare_data(model)
        self.callback_connector._attach_model_callbacks(model, self)

        # ----------------------------
        # SET UP TRAINING
        # ----------------------------
        self.call_hook("on_before_accelerator_backend_setup", model)
        self.accelerator.connect(model)
        self.accelerator.setup_environment()
        self._call_setup_hook(model)  # allow user to setup lightning_module in accelerator environment

        # restore modules after setup
        self.checkpoint_connector.restore_datamodule()
        self.checkpoint_connector.restore_model()
        # restore callback states
        self.checkpoint_connector.restore_callbacks()

        self._call_configure_sharded_model(model)  # allow user to setup in model sharded environment
        self.accelerator.setup(self, model)  # note: this sets up self.lightning_module

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
                  {self.accelerator.start_training}           ||
                or {self.accelerator.start_evaluating}        ||
                or {self.accelerator.start_predicting}        ||  FLOW
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
        """  # noqa: W605

        # ----------------------------
        # TRAIN
        # ----------------------------
        # hook
        if self.state.fn == TrainerFn.FITTING:
            self.call_hook("on_fit_start")

        # plugin will setup fitting (e.g. ddp will launch child processes)
        self._pre_dispatch()

        # restore optimizers, etc.
        self.checkpoint_connector.restore_training_state()

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

        # teardown
        self._call_teardown_hook(model)

        if self.state.status != TrainerStatus.INTERRUPTED:
            self.state.status = TrainerStatus.FINISHED
        self.state.stage = None

        return self.accelerator.results

    def _pre_dispatch(self):
        self.accelerator.pre_dispatch(self)
        self._log_hyperparams()

    def _log_hyperparams(self):
        # log hyper-parameters
        hparams_initial = None

        if self.logger is not None:
            # save exp to get started (this is where the first experiment logs are written)
            datamodule_log_hyperparams = self.datamodule._log_hyperparams if self.datamodule is not None else False

            if self.lightning_module._log_hyperparams and datamodule_log_hyperparams:
                datamodule_hparams = self.datamodule.hparams_initial
                lightning_hparams = self.lightning_module.hparams_initial

                colliding_keys = lightning_hparams.keys() & datamodule_hparams.keys()
                if colliding_keys:
                    raise MisconfigurationException(
                        f"Error while merging hparams: the keys {colliding_keys} are present "
                        "in both the LightningModule's and LightningDataModule's hparams."
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
        self._active_loop.teardown()
        self.logger_connector.teardown()

    def _dispatch(self):
        if self.evaluating:
            self.accelerator.start_evaluating(self)
        elif self.predicting:
            self.accelerator.start_predicting(self)
        else:
            self.accelerator.start_training(self)

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
        self.accelerator.barrier("setup_training")

        # register auto-resubmit when on SLURM
        self.slurm_connector.register_slurm_signal_handlers()

        self.checkpoint_connector.resume_end()

        # --------------------------
        # Pre-train
        # --------------------------
        # on pretrain routine start
        ref_model = self.lightning_module

        self.on_pretrain_routine_start()
        ref_model.on_pretrain_routine_start()

        # print model summary
        if self.is_global_zero and self.weights_summary is not None and not self.testing:
            max_depth = ModelSummary.MODES[self.weights_summary]
            ref_model.summarize(max_depth=max_depth)

        # on pretrain routine end
        self.on_pretrain_routine_end()
        ref_model.on_pretrain_routine_end()

    def _run_train(self) -> None:
        self._pre_training_routine()

        if not self.is_global_zero and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        self._run_sanity_check(self.lightning_module)

        # enable train mode
        self.model.train()
        torch.set_grad_enabled(True)

        # reload data when needed
        model = self.lightning_module

        self.reset_train_val_dataloaders(model)

        try:
            # reset trainer on this loop and all child loops in case user connected a custom loop
            self.fit_loop.trainer = self
            self.fit_loop.run()
        except KeyboardInterrupt:
            rank_zero_warn("Detected KeyboardInterrupt, attempting graceful shutdown...")
            # user could press Ctrl+c many times... only shutdown once
            if not self.interrupted:
                self.state.status = TrainerStatus.INTERRUPTED
                self.on_keyboard_interrupt()
                # same treatment as below
                self.accelerator.on_train_end()
        except BaseException:
            self.state.status = TrainerStatus.INTERRUPTED
            if distributed_available() and self.world_size > 1:
                # try syncing remaing processes, kill otherwise
                self.training_type_plugin.reconciliate_processes(traceback.format_exc())
            # give accelerators a chance to finish
            self.accelerator.on_train_end()
            self._on_expection()
            # reset bookkeeping
            self.state.stage = None
            raise

    def _run_evaluate(self) -> _EVALUATE_OUTPUT:
        if not self.is_global_zero and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        assert self.evaluating

        # reload dataloaders
        self._evaluation_loop.reload_evaluation_dataloaders()

        # reset trainer on this loop and all child loops in case user connected a custom loop
        self._evaluation_loop.trainer = self

        with self.profiler.profile(f"run_{self.state.stage}_evaluation"), torch.no_grad():
            eval_loop_results = self._evaluation_loop.run()

        # remove the tensors from the eval results
        for i, result in enumerate(eval_loop_results):
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
        using_val_step = ref_model.val_dataloader is not None and is_overridden("validation_step", ref_model)
        should_sanity_check = using_val_step and self.num_sanity_val_steps > 0 and self.limit_val_batches > 0

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        if should_sanity_check:
            stage = self.state.stage
            self.sanity_checking = True

            # hook and callback
            self.on_sanity_check_start()

            # reload dataloaders
            self._evaluation_loop.reload_evaluation_dataloaders()

            # run eval step
            with torch.no_grad():
                self._evaluation_loop.run()

            self.on_sanity_check_end()

            # reset validation metrics
            self.logger_connector.reset()

            # reset the seed to what it was before sanity check
            # prevents sanity check to affect random sampling in training
            reset_seed()

            # restore the previous stage when the sanity check if finished
            self.state.stage = stage

    def __load_ckpt_weights(self, ckpt_path: Optional[str]) -> Optional[str]:
        if ckpt_path is None:
            return

        fn = self.state.fn.value

        if ckpt_path == "best":
            # if user requests the best checkpoint but we don't have it, error
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
                f'`.{fn}()` found no path for the best weights: "{ckpt_path}". Please'
                f" specify a path for a checkpoint `.{fn}(ckpt_path=PATH)`"
            )

        # only one process running at this point for TPUs, as spawn isn't triggered yet
        # todo: move this logic internally within the barrier.
        if not self._device_type == DeviceType.TPU:
            self.training_type_plugin.barrier()

        self.checkpoint_connector.restore_model_weights(ckpt_path)
        return ckpt_path

    def _call_setup_hook(self, model: "pl.LightningModule") -> None:
        fn = self.state.fn._setup_fn

        self.accelerator.barrier("pre_setup")

        if self.datamodule is not None:
            self.datamodule.setup(stage=fn)
        self.setup(model, stage=fn)
        model.setup(stage=fn)

        self.accelerator.barrier("post_setup")

    def _call_configure_sharded_model(self, model: "pl.LightningModule") -> None:
        # Call configure sharded model hook if accelerator requests. In some cases
        # we will not call the hook; the hook has initialized the sharded model for example.

        # used on the model if the user re-create a trainer with resume_from_checkpoint
        model_call_configure_sharded_model_hook = getattr(model, "call_configure_sharded_model_hook", False)
        if self.accelerator.call_configure_sharded_model_hook and not model_call_configure_sharded_model_hook:
            with self.accelerator.model_sharded_context():
                model.configure_sharded_model()
                self.configure_sharded_model(model)
            model.call_configure_sharded_model_hook = True
            self.accelerator.call_configure_sharded_model_hook = False

    def _call_teardown_hook(self, model: "pl.LightningModule") -> None:
        fn = self.state.fn._setup_fn

        if self.datamodule is not None:
            self.datamodule.teardown(stage=fn)
        self.profiler.teardown(stage=fn)

        self.data_connector.detach_data(self.lightning_module)

        self.teardown(stage=fn)
        model.teardown(stage=fn)

        model._current_fx_name = None
        model._current_dataloader_idx = None
        # these could have become stale if metrics are defined in `setup`
        model._metric_attributes = None

    def call_hook(self, hook_name: str, *args, **kwargs) -> Any:
        # Note this implementation is copy/pasted into the TrainLoop class in TrainingEpochLoop._on_train_epoch_end_hook
        # This was done to manage the deprecation of the `outputs` argument to on_train_epoch_end
        # If making changes to this function, ensure that those changes are also made to
        # TrainingEpochLoop._on_train_epoch_end_hook
        if self.lightning_module:
            prev_fx_name = self.lightning_module._current_fx_name
            self.lightning_module._current_fx_name = hook_name

        # always profile hooks
        with self.profiler.profile(hook_name):

            # first call trainer hook
            if hasattr(self, hook_name):
                trainer_hook = getattr(self, hook_name)
                trainer_hook(*args, **kwargs)

            # next call hook in lightningModule
            output = None
            model_ref = self.lightning_module
            if is_overridden(hook_name, model_ref):
                hook_fx = getattr(model_ref, hook_name)
                output = hook_fx(*args, **kwargs)

            # call the accelerator hook
            if hasattr(self.accelerator, hook_name):
                accelerator_hook = getattr(self.accelerator, hook_name)
                accelerator_output = accelerator_hook(*args, **kwargs)
                # Rely on the accelerator output if lightningModule hook returns nothing
                # Required for cases such as DataParallel where we reduce the output for the user
                # todo: move this data parallel logic into the data parallel plugin
                output = accelerator_output if output is None else output

        if self.lightning_module:
            # restore current_fx when nested context
            self.lightning_module._current_fx_name = prev_fx_name

        return output

    def _parse_devices(
        self,
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
                "GPU available but not used. Set the gpus flag in your trainer"
                " `Trainer(gpus=1)` or script `--gpus=1`."
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

    def _on_expection(self):
        if not self.is_global_zero or not _fault_tolerant_enabled():
            return
        # save a checkpoint for fault tolerant training. we don't use `log_dir` to minimize the chances of failure.
        file_path = os.path.join(self.default_root_dir, ".pl_auto_save.ckpt")
        self.save_checkpoint(file_path)
