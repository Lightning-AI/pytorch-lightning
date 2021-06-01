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
import warnings
from datetime import timedelta
from itertools import count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from weakref import proxy

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.plugins import Plugin
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.profiler import (
    AdvancedProfiler,
    BaseProfiler,
    PassThroughProfiler,
    PyTorchProfiler,
    SimpleProfiler,
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
from pytorch_lightning.trainer.connectors.logger_connector.result import Result
from pytorch_lightning.trainer.connectors.model_connector import ModelConnector
from pytorch_lightning.trainer.connectors.optimizer_connector import OptimizerConnector
from pytorch_lightning.trainer.connectors.slurm_connector import SLURMConnector
from pytorch_lightning.trainer.connectors.training_trick_connector import TrainingTricksConnector
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.deprecated_api import DeprecatedTrainerAttributes
from pytorch_lightning.trainer.evaluation_loop import EvaluationLoop
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin
from pytorch_lightning.trainer.predict_loop import PredictLoop
from pytorch_lightning.trainer.properties import TrainerProperties
from pytorch_lightning.trainer.states import TrainerFn, TrainerState, TrainerStatus
from pytorch_lightning.trainer.training_loop import TrainLoop
from pytorch_lightning.trainer.training_tricks import TrainerTrainingTricksMixin
from pytorch_lightning.tuner.lr_finder import _LRFinder
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities import DeviceType, parsing, rank_zero_warn
from pytorch_lightning.utilities.debugging import InternalDebugger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT

log = logging.getLogger(__name__)
# warnings to ignore in trainer
warnings.filterwarnings(
    'ignore', message='torch.distributed.reduce_op is deprecated, '
    'please use torch.distributed.ReduceOp instead'
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
        gradient_clip_algorithm: str = 'norm',
        process_position: int = 0,
        num_nodes: int = 1,
        num_processes: int = 1,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
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
        weights_summary: Optional[str] = 'top',
        weights_save_path: Optional[str] = None,
        num_sanity_val_steps: int = 2,
        truncated_bptt_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[BaseProfiler, str]] = None,
        benchmark: bool = False,
        deterministic: bool = False,
        reload_dataloaders_every_epoch: bool = False,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        terminate_on_nan: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        plugins: Optional[Union[List[Union[Plugin, ClusterEnvironment, str]], Plugin, ClusterEnvironment, str]] = None,
        amp_backend: str = 'native',
        amp_level: str = 'O2',
        distributed_backend: Optional[str] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = 'max_size_cycle',
        stochastic_weight_avg: bool = False
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
                the default ``TensorBoardLogger``. ``False`` will disable logging.

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

            reload_dataloaders_every_epoch: Set to True to reload dataloaders every epoch.

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
        distributed_backend = distributed_backend or accelerator

        # init connectors
        self.dev_debugger = InternalDebugger(self)
        self.config_validator = ConfigValidator(self)
        self.data_connector = DataConnector(self, multiple_trainloader_mode)
        self.optimizer_connector = OptimizerConnector(self)

        self.accelerator_connector = AcceleratorConnector(
            num_processes, tpu_cores, distributed_backend, auto_select_gpus, gpus, num_nodes, sync_batchnorm, benchmark,
            replace_sampler_ddp, deterministic, precision, amp_backend, amp_level, plugins
        )
        self.logger_connector = LoggerConnector(self, log_gpu_memory)
        self.model_connector = ModelConnector(self)
        self.callback_connector = CallbackConnector(self)
        self.debugging_connector = DebuggingConnector(self)
        self.training_tricks_connector = TrainingTricksConnector(self)
        self.checkpoint_connector = CheckpointConnector(self)
        self.slurm_connector = SLURMConnector(self)
        self.tuner = Tuner(self)
        self.train_loop = TrainLoop(self, max_epochs, min_epochs, max_steps, min_steps, num_sanity_val_steps)
        self.evaluation_loop = EvaluationLoop(self)
        self.predict_loop = PredictLoop(self)

        # training state
        if weights_summary is not None and weights_summary not in ModelSummary.MODES:
            raise MisconfigurationException(
                f"`weights_summary` can be None, {', '.join(ModelSummary.MODES)}, but got {weights_summary}"
            )
        self.weights_summary = weights_summary
        self.shown_warnings = set()

        # init callbacks
        # Declare attributes to be set in callback_connector on_trainer_init
        self.callback_connector.on_trainer_init(
            callbacks,
            checkpoint_callback,
            progress_bar_refresh_rate,
            process_position,
            default_root_dir,
            weights_save_path,
            resume_from_checkpoint,
            stochastic_weight_avg,
            max_time,
        )

        # hook
        self.on_init_start()

        # init optimizer + lr scheduler related flags
        self.optimizer_connector.on_trainer_init()

        # init data flags
        self.data_connector.on_trainer_init(
            check_val_every_n_epoch, reload_dataloaders_every_epoch, prepare_data_per_node
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
        self.evaluation_loop.on_trainer_init()
        self.predict_loop.on_trainer_init()

        # configure tuner
        self.tuner.on_trainer_init(auto_lr_find, auto_scale_batch_size)

        # configure profiler
        self.__init_profiler(profiler)

        # init logger flags
        self.logger_connector.on_trainer_init(
            logger,
            flush_logs_every_n_steps,
            log_every_n_steps,
            move_metrics_to_cpu,
        )

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

    def fit(
        self,
        model: LightningModule,
        train_dataloader: Any = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ) -> None:
        r"""
        Runs the full optimization routine.

        Args:
            model: Model to fit.

            train_dataloader: Either a single PyTorch DataLoader or a collection of these
                (list, dict, nested lists and dicts). In the case of multiple dataloaders, please
                see this :ref:`page <multiple-training-dataloaders>`

            val_dataloaders: Either a single Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
        """
        Trainer._log_api_event("fit")

        self.state.fn = TrainerFn.FITTING
        self.state.status = TrainerStatus.RUNNING
        self.training = True

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloader, LightningDataModule):
            datamodule = train_dataloader
            train_dataloader = None
        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if (train_dataloader is not None or val_dataloaders is not None) and datamodule is not None:
            raise MisconfigurationException(
                'You cannot pass `train_dataloader` or `val_dataloaders` to `trainer.fit(datamodule=...)`'
            )

        # links data to the trainer
        self.data_connector.attach_data(
            model, train_dataloader=train_dataloader, val_dataloaders=val_dataloaders, datamodule=datamodule
        )

        self._run(model)

        assert self.state.stopped
        self.training = False

    def validate(
        self,
        model: Optional[LightningModule] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        ckpt_path: Optional[str] = 'best',
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ) -> _EVALUATE_OUTPUT:
        r"""
        Perform one evaluation epoch over the validation set.

        Args:
            model: The model to validate.

            val_dataloaders: Either a single PyTorch DataLoader or a list of them,
                specifying validation samples.

            ckpt_path: Either ``best`` or path to the checkpoint you wish to validate.
                If ``None``, use the current weights of the model.
                When the model is given as argument, this parameter will not apply.

            verbose: If True, prints the validation results.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

        Returns:
            The dictionary with final validation results returned by validation_epoch_end.
            If validation_epoch_end is not defined, the output is a list of the dictionaries
            returned by validation_step.
        """
        # --------------------
        # SETUP HOOK
        # --------------------
        Trainer._log_api_event("validate")
        self.verbose_evaluate = verbose

        self.state.fn = TrainerFn.VALIDATING
        self.state.status = TrainerStatus.RUNNING
        self.validating = True

        # If you supply a datamodule you can't supply val_dataloaders
        if val_dataloaders is not None and datamodule:
            raise MisconfigurationException(
                'You cannot pass both `trainer.validate(val_dataloaders=..., datamodule=...)`'
            )

        model_provided = model is not None
        model = model or self.lightning_module
        if model is None:
            raise MisconfigurationException(
                "`model` must be provided to `trainer.validate()` when it hasn't been passed in a previous run"
            )

        # links data to the trainer
        self.data_connector.attach_data(model, val_dataloaders=val_dataloaders, datamodule=datamodule)

        if not model_provided:
            self.validated_ckpt_path = self.__load_ckpt_weights(ckpt_path)

        # run validate
        results = self._run(model)

        assert self.state.stopped
        self.validating = False

        return results

    def test(
        self,
        model: Optional[LightningModule] = None,
        test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        ckpt_path: Optional[str] = 'best',
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ) -> _EVALUATE_OUTPUT:
        r"""
        Perform one evaluation epoch over the test set. It's separated from
        fit to make sure you never run on your test set until you want to.

        Args:
            model: The model to test.

            test_dataloaders: Either a single PyTorch DataLoader or a list of them,
                specifying test samples.

            ckpt_path: Either ``best`` or path to the checkpoint you wish to test.
                If ``None``, use the current weights of the model.
                When the model is given as argument, this parameter will not apply.

            verbose: If True, prints the test results.

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

        Returns:
            Returns a list of dictionaries, one for each test dataloader containing their respective metrics.
        """
        # --------------------
        # SETUP HOOK
        # --------------------
        Trainer._log_api_event("test")
        self.verbose_evaluate = verbose

        self.state.fn = TrainerFn.TESTING
        self.state.status = TrainerStatus.RUNNING
        self.testing = True

        # If you supply a datamodule you can't supply test_dataloaders
        if test_dataloaders is not None and datamodule:
            raise MisconfigurationException('You cannot pass both `trainer.test(test_dataloaders=..., datamodule=...)`')

        model_provided = model is not None
        model = model or self.lightning_module
        if model is None:
            raise MisconfigurationException(
                "`model` must be provided to `trainer.test()` when it hasn't been passed in a previous run"
            )

        # links data to the trainer
        self.data_connector.attach_data(model, test_dataloaders=test_dataloaders, datamodule=datamodule)

        if not model_provided:
            self.tested_ckpt_path = self.__load_ckpt_weights(ckpt_path)

        # run test
        results = self._run(model)

        assert self.state.stopped
        self.testing = False

        return results

    def predict(
        self,
        model: Optional[LightningModule] = None,
        dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
        return_predictions: Optional[bool] = None,
        ckpt_path: Optional[str] = 'best',
    ) -> Optional[_PREDICT_OUTPUT]:
        r"""

        Separates from fit to make sure you never run on your predictions set until you want to.
        This will call the model forward function to compute predictions.

        Args:
            model: The model to predict with.

            dataloaders: Either a single PyTorch DataLoader or a list of them, specifying inference samples.

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

        if dataloaders is not None and datamodule:
            raise MisconfigurationException('You cannot pass both `trainer.predict(dataloaders=..., datamodule=...)`')

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
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
        scale_batch_size_kwargs: Optional[Dict[str, Any]] = None,
        lr_find_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Optional[Union[int, _LRFinder]]]:
        r"""
        Runs routines to tune hyperparameters before training.

        Args:
            model: Model to tune.

            train_dataloader: A Pytorch DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            val_dataloaders: Either a single Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped

            datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

            scale_batch_size_kwargs: Arguments for :func:`~pytorch_lightning.tuner.batch_size_scaling.scale_batch_size`

            lr_find_kwargs: Arguments for :func:`~pytorch_lightning.tuner.lr_finder.lr_find`
        """
        Trainer._log_api_event("tune")

        self.state.fn = TrainerFn.TUNING
        self.state.status = TrainerStatus.RUNNING
        self.tuning = True

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloader, LightningDataModule):
            datamodule = train_dataloader
            train_dataloader = None
        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if (train_dataloader is not None or val_dataloaders is not None) and datamodule is not None:
            raise MisconfigurationException(
                'You cannot pass `train_dataloader` or `val_dataloaders` to `trainer.tune(datamodule=...)`'
            )

        # links data to the trainer
        self.data_connector.attach_data(
            model, train_dataloader=train_dataloader, val_dataloaders=val_dataloaders, datamodule=datamodule
        )

        result = self.tuner._tune(model, scale_batch_size_kwargs=scale_batch_size_kwargs, lr_find_kwargs=lr_find_kwargs)

        assert self.state.stopped
        self.tuning = False

        return result

    def _run(self, model: LightningModule) -> Optional[Union[_EVALUATE_OUTPUT, _PREDICT_OUTPUT]]:
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
        self._call_configure_sharded_model(model)  # allow user to setup in model sharded environment
        self.accelerator.setup(self, model)  # note: this sets up self.lightning_module

        # ----------------------------
        # INSPECT THE CORE LOOPS
        # ----------------------------
        f"""
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
                     or {self._run_evaluation}                ||
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

        # dispatch `start_training` or `start_evaluating` or `start_predicting`
        self._dispatch()

        # plugin will finalized fitting (e.g. ddp_spawn will load trained model)
        self._post_dispatch()

        # ----------------------------
        # POST-Training CLEAN UP
        # ----------------------------
        # hook
        if self.state.fn == TrainerFn.FITTING:
            self.call_hook('on_fit_end')

        # teardown
        self._call_teardown_hook(model)

        if self.state.status != TrainerStatus.INTERRUPTED:
            self.state.status = TrainerStatus.FINISHED
        self.state.stage = None

        return self.accelerator.results

    def _pre_dispatch(self):
        self.accelerator.pre_dispatch(self)

        # log hyper-parameters
        if self.logger is not None:
            # save exp to get started (this is where the first experiment logs are written)
            self.logger.log_hyperparams(self.lightning_module.hparams_initial)
            self.logger.log_graph(self.lightning_module)
            self.logger.save()

    def _post_dispatch(self):
        self.accelerator.post_dispatch(self)
        self.accelerator.teardown()

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

        # --------------------------
        # Pre-train
        # --------------------------
        # on pretrain routine start
        ref_model = self.lightning_module

        self.on_pretrain_routine_start()
        ref_model.on_pretrain_routine_start()

        # print model summary
        if self.is_global_zero and self.weights_summary is not None and not self.testing:
            ref_model.summarize(mode=self.weights_summary)

        # restore training and model before hpc is called
        self.checkpoint_connector.restore_weights()

        # on pretrain routine end
        self.on_pretrain_routine_end()
        ref_model.on_pretrain_routine_end()

    def _run_train(self) -> None:
        self._pre_training_routine()

        if not self.is_global_zero and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        self._run_sanity_check(self.lightning_module)

        self.checkpoint_connector.has_trained = False

        # enable train mode
        self.model.train()
        torch.set_grad_enabled(True)

        # reload data when needed
        model = self.lightning_module
        self.train_loop.reset_train_val_dataloaders(model)

        # hook
        self.train_loop.on_train_start()

        try:
            if self.train_loop.should_skip_training():
                return
            # run all epochs
            epochs = range(self.current_epoch, self.max_epochs) if self.max_epochs else count(self.current_epoch)
            for epoch in epochs:

                # hook
                self.train_loop.on_train_epoch_start(epoch)

                with self.profiler.profile("run_training_epoch"):
                    # run train epoch
                    self.train_loop.run_training_epoch()

                if self.max_steps and self.max_steps <= self.global_step:
                    self.train_loop.on_train_end()
                    return

                # early stopping
                met_min_epochs = (epoch >= self.min_epochs - 1) if self.min_epochs else True
                met_min_steps = self.global_step >= self.min_steps if self.min_steps else True

                if self.should_stop:
                    if met_min_epochs and met_min_steps:
                        self.train_loop.on_train_end()
                        return
                    else:
                        log.info(
                            'Trainer was signaled to stop but required minimum epochs'
                            f' ({self.min_epochs}) or minimum steps ({self.min_steps}) has'
                            ' not been met. Training will continue...'
                        )
                        self.should_stop = False

            # hook
            self.train_loop.on_train_end()

        except KeyboardInterrupt:
            rank_zero_warn('Detected KeyboardInterrupt, attempting graceful shutdown...')
            # user could press Ctrl+c many times... only shutdown once
            if not self.interrupted:
                self.state.status = TrainerStatus.INTERRUPTED
                self.on_keyboard_interrupt()
                # same treatment as below
                self.accelerator.on_train_end()
                self.state.stage = None
        except BaseException:
            self.state.status = TrainerStatus.INTERRUPTED
            # give accelerators a chance to finish
            self.accelerator.on_train_end()
            # reset bookkeeping
            self.state.stage = None
            raise

    def _run_evaluation(self) -> _EVALUATE_OUTPUT:
        if not (self.evaluating or self.sanity_checking):
            rank_zero_warn(
                f"`trainer._run_evaluation()` was called but the running stage is set to {self.state.stage}."
                " This should not happen normally. Setting it to `RunningStage.VALIDATING`", RuntimeWarning
            )
            self.validating = True

        # prepare dataloaders
        dataloaders, max_batches = self.evaluation_loop.get_evaluation_dataloaders()

        # check if we want to skip this evaluation
        if self.evaluation_loop.should_skip_evaluation(max_batches):
            return [], []

        # enable eval mode + no grads
        self.evaluation_loop.on_evaluation_model_eval()
        # ref model
        model = self.lightning_module
        model.zero_grad()
        torch.set_grad_enabled(False)

        # hook
        self.evaluation_loop.on_evaluation_start()

        # set up the eval loop
        self.evaluation_loop.setup(max_batches, dataloaders)

        # hook
        self.evaluation_loop.on_evaluation_epoch_start()

        # run validation/testing
        for dataloader_idx, dataloader in enumerate(dataloaders):
            # bookkeeping
            dl_outputs = []
            dataloader = self.accelerator.process_dataloader(dataloader)
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
                with self.profiler.profile("evaluation_step_and_end"):
                    output = self.evaluation_loop.evaluation_step(batch, batch_idx, dataloader_idx)
                    output = self.evaluation_loop.evaluation_step_end(output)

                # hook + store predictions
                self.evaluation_loop.on_evaluation_batch_end(output, batch, batch_idx, dataloader_idx)

                # log batch metrics
                self.logger_connector.log_evaluation_step_metrics()

                # track epoch level outputs
                dl_outputs = self._track_output_for_epoch_end(dl_outputs, output)

            # store batch level output per dataloader
            if self.evaluation_loop.should_track_batch_outputs_for_epoch_end:
                self.evaluation_loop.outputs.append(dl_outputs)

        outputs = self.evaluation_loop.outputs

        # reset outputs
        self.evaluation_loop.outputs = []

        # with a single dataloader don't pass a 2D list
        if len(outputs) > 0 and self.evaluation_loop.num_dataloaders == 1:
            outputs = outputs[0]

        # lightning module method
        self.evaluation_loop.evaluation_epoch_end(outputs)

        # hook
        self.evaluation_loop.on_evaluation_epoch_end()

        # log epoch metrics
        eval_loop_results = self.logger_connector.get_evaluate_epoch_results()

        # hook
        self.evaluation_loop.on_evaluation_end()

        # save predictions to disk
        self.evaluation_loop.predictions.to_disk()

        # enable train mode again
        self.evaluation_loop.on_evaluation_model_train()

        # reset cached results
        self.logger_connector.reset()

        torch.set_grad_enabled(True)

        return eval_loop_results

    def _track_output_for_epoch_end(self, outputs, output):
        if output is not None:
            if isinstance(output, Result):
                output = output.detach()
                if self.move_metrics_to_cpu:
                    output = output.cpu()
            elif isinstance(output, dict):
                output = recursive_detach(output, to_cpu=self.move_metrics_to_cpu)
            elif isinstance(output, torch.Tensor) and output.is_cuda and self.move_metrics_to_cpu:
                output = output.cpu()
            outputs.append(output)
        return outputs

    def _run_evaluate(self) -> _EVALUATE_OUTPUT:
        if not self.is_global_zero and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        assert self.evaluating

        with self.profiler.profile(f"run_{self.state.stage}_evaluation"):
            eval_loop_results = self._run_evaluation()

        # remove the tensors from the eval results
        for i, result in enumerate(eval_loop_results):
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v.cpu().item()

        return eval_loop_results

    def _run_predict(self) -> Optional[_PREDICT_OUTPUT]:
        # prepare dataloaders
        dataloaders, max_batches = self.predict_loop.get_predict_dataloaders()

        # check if we want to skip this evaluation
        if self.predict_loop.should_skip_predict(max_batches):
            return []

        # set up the eval loop
        self.predict_loop.setup(max_batches, dataloaders)

        # call hook
        self.predict_loop.on_predict_start()

        # run validation/testing
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dataloader = self.accelerator.process_dataloader(dataloader)
            dl_max_batches = self.predict_loop.max_batches[dataloader_idx]
            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    continue

                # stop short when running on limited batches
                if batch_idx >= dl_max_batches:
                    break

                # lightning module methods
                with self.profiler.profile("predict_step"):
                    self.predict_loop.predict_step(batch, batch_idx, dataloader_idx)

        # call hook
        results = self.predict_loop.on_predict_epoch_end()

        # call hook
        self.predict_loop.on_predict_end()

        return results

    def _run_sanity_check(self, ref_model):
        using_val_step = ref_model.val_dataloader is not None and is_overridden('validation_step', ref_model)
        should_sanity_check = using_val_step and self.num_sanity_val_steps > 0 and self.limit_val_batches > 0

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        if should_sanity_check:
            stage = self.state.stage
            self.sanity_checking = True

            # hook and callback
            self.on_sanity_check_start()

            # run eval step
            self._run_evaluation()

            self.on_sanity_check_end()

            self.state.stage = stage

            # reset the seed to what it was before sanity check
            # prevents sanity check to affect random sampling in training
            reset_seed()

    def __load_ckpt_weights(self, ckpt_path: Optional[str]) -> Optional[str]:
        if ckpt_path is None:
            return

        fn = self.state.fn.value

        if ckpt_path == 'best':
            # if user requests the best checkpoint but we don't have it, error
            if not self.checkpoint_callback.best_model_path:
                if self.fast_dev_run:
                    raise MisconfigurationException(
                        f'You cannot execute `.{fn}()` with `fast_dev_run=True` unless you do'
                        f' `.{fn}(ckpt_path=PATH)` as no checkpoint path was generated during fitting.'
                    )
                raise MisconfigurationException(
                    f'`.{fn}(ckpt_path="best")` is set but `ModelCheckpoint` is not configured to save the best model.'
                )
            # load best weights
            ckpt_path = self.checkpoint_callback.best_model_path

        if not ckpt_path:
            raise MisconfigurationException(
                f'`.{fn}()` found no path for the best weights: "{ckpt_path}". Please'
                f' specify a path for a checkpoint `.{fn}(ckpt_path=PATH)`'
            )

        # only one process running at this point for TPUs, as spawn isn't triggered yet
        # todo: move this logic internally within the barrier.
        if not self._device_type == DeviceType.TPU:
            self.training_type_plugin.barrier()

        self.training_type_plugin.restore_model_state_from_ckpt_path(
            ckpt_path, map_location=lambda storage, loc: storage
        )
        return ckpt_path

    def _call_setup_hook(self, model: LightningModule) -> None:
        fn = self.state.fn._setup_fn

        self.accelerator.barrier("pre_setup")

        if self.datamodule is not None:
            self.datamodule.setup(stage=fn)
        self.setup(model, stage=fn)
        model.setup(stage=fn)

        self.accelerator.barrier("post_setup")

    def _call_configure_sharded_model(self, model: LightningModule) -> None:
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

    def _call_teardown_hook(self, model: LightningModule) -> None:
        fn = self.state.fn._setup_fn

        if self.datamodule is not None:
            self.datamodule.teardown(stage=fn)
        self.profiler.teardown(stage=fn)
        self.teardown(stage=fn)
        model.teardown(stage=fn)

        model._current_fx_name = None
        model._current_dataloader_idx = None

    def _reset_result_and_set_fx_name(self, hook_name: str) -> bool:
        # on_before_zero_grad is called within training_step
        # TODO(@carmocca): Result should handle this logic
        if "batch_start" in hook_name or hook_name in ("on_before_zero_grad", "on_after_backward"):
            return True
        model_ref = self.lightning_module
        if model_ref is not None:
            # used to track current hook name called
            model_ref._results = Result()
            model_ref._current_fx_name = hook_name
        return False

    def _cache_logged_metrics(self):
        model_ref = self.lightning_module
        if model_ref is not None:
            # capture logging for this hook
            self.logger_connector.cache_logged_metrics()

    def call_hook(self, hook_name: str, *args, **kwargs) -> Any:
        # Note this implementation is copy/pasted into the TrainLoop class in TrainLoop._on_train_epoch_end_hook
        # This was done to manage the deprecation of the `outputs` argument to on_train_epoch_end
        # If making changes to this function, ensure that those changes are also made to
        # TrainLoop._on_train_epoch_end_hook

        # set hook_name to model + reset Result obj
        skip = self._reset_result_and_set_fx_name(hook_name)

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

            # if the PL module doesn't have the hook then call the accelerator
            # used to auto-reduce things for the user with Results obj
            elif hasattr(self.accelerator, hook_name):
                accelerator_hook = getattr(self.accelerator, hook_name)
                output = accelerator_hook(*args, **kwargs)

        if not skip:
            self._cache_logged_metrics()
        return output

    @staticmethod
    def _log_api_event(event: str) -> None:
        torch._C._log_api_usage_once("lightning.trainer." + event)

    def __init_profiler(self, profiler: Optional[Union[BaseProfiler, str]]) -> None:
        if isinstance(profiler, str):
            PROFILERS = {
                "simple": SimpleProfiler,
                "advanced": AdvancedProfiler,
                "pytorch": PyTorchProfiler,
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
