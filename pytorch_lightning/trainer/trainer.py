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
from itertools import count
from pathlib import Path
from traceback import print_exc
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.plugins import Plugin
from pytorch_lightning.profiler import BaseProfiler
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
from pytorch_lightning.trainer.connectors.profiler_connector import ProfilerConnector
from pytorch_lightning.trainer.connectors.slurm_connector import SLURMConnector
from pytorch_lightning.trainer.connectors.training_trick_connector import TrainingTricksConnector
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.deprecated_api import DeprecatedDistDeviceAttributes, DeprecatedTrainerAttributes
from pytorch_lightning.trainer.evaluation_loop import EvaluationLoop
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin
from pytorch_lightning.trainer.predict_loop import PredictLoop
from pytorch_lightning.trainer.properties import TrainerProperties
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.trainer.training_loop import TrainLoop
from pytorch_lightning.trainer.training_tricks import TrainerTrainingTricksMixin
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.debugging import InternalDebugger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.model_helpers import is_overridden

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
    DeprecatedDistDeviceAttributes,
    DeprecatedTrainerAttributes,
):

    @_defaults_from_env_vars
    def __init__(
        self,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        checkpoint_callback: bool = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: float = 0,
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
        plugins: Optional[Union[Plugin, str, list]] = None,
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

            limit_train_batches: How much of training dataset to check (float = fraction, int = num_batches)

            limit_val_batches: How much of validation dataset to check (float = fraction, int = num_batches)

            limit_test_batches: How much of test dataset to check (float = fraction, int = num_batches)

            limit_predict_batches: How much of prediction dataset to check (float = fraction, int = num_batches)

            logger: Logger (or iterable collection of loggers) for experiment tracking.

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

            precision: Full precision (32), half precision (16). Can be used on CPU, GPU or TPUs.

            max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
                If both max_epochs and max_steps are not specified, defaults to ``max_epochs`` = 1000.

            min_epochs: Force training for at least these many epochs. Disabled by default (None).
                If both min_epochs and min_steps are not specified, defaults to ``min_epochs`` = 1.

            max_steps: Stop training after this number of steps. Disabled by default (None).

            min_steps: Force training for at least these number of steps. Disabled by default (None).

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

            truncated_bptt_steps: Truncated back prop breaks performs backprop every k steps of much longer
                sequence.

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

        distributed_backend = distributed_backend or accelerator

        # init connectors
        self.dev_debugger = InternalDebugger(self)
        self.config_validator = ConfigValidator(self)
        self.data_connector = DataConnector(self)
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
        self.profile_connector = ProfilerConnector(self)
        self.checkpoint_connector = CheckpointConnector(self)
        self.slurm_connector = SLURMConnector(self)
        self.tuner = Tuner(self)
        self.train_loop = TrainLoop(self, multiple_trainloader_mode)
        self.evaluation_loop = EvaluationLoop(self)
        self.predict_loop = PredictLoop(self)

        # training state
        self.weights_summary = weights_summary
        self.shown_warnings = set()

        # init callbacks
        # Declare attributes to be set in callback_connector on_trainer_init
        self.callback_connector.on_trainer_init(
            callbacks, checkpoint_callback, progress_bar_refresh_rate, process_position, default_root_dir,
            weights_save_path, resume_from_checkpoint, stochastic_weight_avg
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
            gradient_clip_val, track_grad_norm, accumulate_grad_batches, truncated_bptt_steps, terminate_on_nan
        )
        self.train_loop.on_trainer_init(
            max_epochs,
            min_epochs,
            max_steps,
            min_steps,
            num_sanity_val_steps,
            weights_summary,
        )
        self.evaluation_loop.on_trainer_init()

        # configure tuner
        self.tuner.on_trainer_init(auto_lr_find, auto_scale_batch_size)

        # configure profiler
        self.profile_connector.on_trainer_init(profiler)

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
    ):
        r"""
        Runs the full optimization routine.

        Args:
            datamodule: A instance of :class:`LightningDataModule`.

            model: Model to fit.

            train_dataloader: Either a single PyTorch DataLoader or a collection of these
                (list, dict, nested lists and dicts). In the case of multiple dataloaders, please
                see this :ref:`page <multiple-training-dataloaders>`

            val_dataloaders: Either a single Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped

        """
        # we reuse fit for other functions. When already set, it shouldn't be modified.
        if not self.state.running:
            self.state = TrainerState.FITTING
        if self._running_stage is None:
            self.training = True

        # set local properties on the model
        self.model_connector.copy_trainer_model_properties(model)

        # ----------------------------
        # LINK DATA
        # ----------------------------
        # setup data, etc...
        self.train_loop.setup_fit(model, train_dataloader, val_dataloaders, datamodule)

        # hook
        self.data_connector.prepare_data(model)
        self.callback_connector._attach_model_callbacks(model, self)

        # ----------------------------
        # SET UP TRAINING
        # ----------------------------
        self.call_hook("on_before_accelerator_backend_setup", model)
        self.accelerator.connect(model)
        self.accelerator.setup_environment()
        self.call_setup_hook(model)  # allow user to setup lightning_module in accelerator environment
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
                         {self.dispatch}                      ||
                                |                             ||  LIGHTNING
                {self.accelerator.start_training} or          ||
                {self.accelerator.start_evaluating} or        ||  FLOW
                {self.accelerator.start_predicting}           ||
                                |                             ||  DIRECTION
                        {self.run_train} or                   ||
                     {self.run_evaluation} or                 ||
                       {self.run_predict}                     ||
                                |                             ||
                             results                          \/
        This is used to guide readers to the core loops: train, test, predict.
        {self.run_predict} is the simplest to understand, use `Go to Definition` to read it :)
        Search for `start_training` or `start_evaluating` or `start_predicting` in
        `pytorch_lightning/plugins/training_type_plugin` to find accelerator dispatch functions.
        """  # noqa: W605

        # ----------------------------
        # TRAIN
        # ----------------------------
        # hook
        if self.state == TrainerState.FITTING:
            self.call_hook("on_fit_start")

        # plugin will setup fitting (e.g. ddp will launch child processes)
        self.pre_dispatch()

        # dispatch `start_training` or `start_evaluating` or `start_predicting`
        self.dispatch()

        # plugin will finalized fitting (e.g. ddp_spawn will load trained model)
        self.post_dispatch()

        # ----------------------------
        # POST-Training CLEAN UP
        # ----------------------------
        # hook
        if self.state == TrainerState.FITTING:
            self.call_hook('on_fit_end')

        # teardown
        self.call_teardown_hook(model)

        if self.state != TrainerState.INTERRUPTED:
            self.state = TrainerState.FINISHED
        self._running_stage = None

        # return 1 when finished
        # used for testing or when we need to know that training succeeded
        return self.accelerator.results or 1

    def pre_dispatch(self):
        self.accelerator.pre_dispatch()

        # log hyper-parameters
        if self.logger is not None:
            # save exp to get started (this is where the first experiment logs are written)
            self.logger.log_hyperparams(self.lightning_module.hparams_initial)
            self.logger.log_graph(self.lightning_module)
            self.logger.save()

    def post_dispatch(self):
        self.accelerator.post_dispatch()
        self.accelerator.teardown()

    def dispatch(self):
        if self.evaluating:
            self.accelerator.start_evaluating(self)
        elif self.predicting:
            self.accelerator.start_predicting(self)
        else:
            self.accelerator.start_training(self)

    def run_stage(self):
        results = None
        if self.evaluating:
            results = self.run_evaluate()
        elif self.predicting:
            results = self.run_predict()
        else:
            self.run_train()
        return results

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
            if self.weights_summary in ModelSummary.MODES:
                ref_model.summarize(mode=self.weights_summary)
            else:
                raise MisconfigurationException("weights_summary can be None, " + ", ".join(ModelSummary.MODES))

        # restore training and model before hpc is called
        self.checkpoint_connector.restore_weights()

        # on pretrain routine end
        self.on_pretrain_routine_end()
        ref_model.on_pretrain_routine_end()

    def run_train(self) -> None:

        self._pre_training_routine()

        if not self.is_global_zero and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        self.run_sanity_check(self.lightning_module)

        self.checkpoint_connector.has_trained = False

        # enable train mode
        model = self.lightning_module
        model.train()
        torch.set_grad_enabled(True)

        # reload data when needed
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
                    return

                # early stopping
                met_min_epochs = (epoch >= self.min_epochs - 1) if self.min_epochs else True
                met_min_steps = self.global_step >= self.min_steps if self.min_steps else True

                if self.should_stop:
                    if met_min_epochs and met_min_steps:
                        return
                    else:
                        log.info(
                            'Trainer was signaled to stop but required minimum epochs'
                            f' ({self.min_epochs}) or minimum steps ({self.min_steps}) has'
                            ' not been met. Training will continue...'
                        )

            # hook
            self.train_loop.on_train_end()

        except KeyboardInterrupt:
            rank_zero_warn('Detected KeyboardInterrupt, attempting graceful shutdown...')
            # user could press Ctrl+c many times... only shutdown once
            if not self.interrupted:
                self.state = TrainerState.INTERRUPTED
                self.on_keyboard_interrupt()
        except (RuntimeError, AssertionError):
            # if an exception is raised, the finally block is executed and can hide the actual exception
            # that was initially raised if `on_train_end` also raises an exception. we want to avoid that
            # for assertions and other runtime errors so we aren't misled while debugging
            print_exc()
        finally:
            # hook
            self.train_loop.on_train_end()

    def run_evaluation(self, on_epoch=False):
        if not (self.evaluating or self.sanity_checking):
            rank_zero_warn(
                f"`trainer.run_evaluation()` was called but the running stage is set to {self._running_stage}."
                " This should not happen normally. Setting it to `RunningStage.VALIDATING`", RuntimeWarning
            )
            self.validating = True

        # reset cached results
        self.logger_connector.reset()

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
        self.evaluation_loop.setup(model, max_batches, dataloaders)

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
                self.evaluation_loop.log_evaluation_step_metrics(output, batch_idx)

                # track epoch level outputs
                dl_outputs = self.track_output_for_epoch_end(dl_outputs, output)

            # store batch level output per dataloader
            self.evaluation_loop.outputs.append(dl_outputs)

        # lightning module method
        deprecated_eval_results = self.evaluation_loop.evaluation_epoch_end()

        # hook
        self.evaluation_loop.on_evaluation_epoch_end()

        # update epoch-level lr_schedulers
        if on_epoch:
            self.optimizer_connector.update_learning_rates(interval='epoch')

        # hook
        self.evaluation_loop.on_evaluation_end()

        # log epoch metrics
        eval_loop_results = self.evaluation_loop.log_epoch_metrics_on_evaluation_end()

        # save predictions to disk
        self.evaluation_loop.predictions.to_disk()

        # enable train mode again
        self.evaluation_loop.on_evaluation_model_train()

        torch.set_grad_enabled(True)

        return eval_loop_results, deprecated_eval_results

    def track_output_for_epoch_end(self, outputs, output):
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

    def run_evaluate(self):
        if not self.is_global_zero and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        assert self.evaluating

        with self.profiler.profile(f"run_{self._running_stage}_evaluation"):
            eval_loop_results, _ = self.run_evaluation()

        if len(eval_loop_results) == 0:
            return 1

        # remove the tensors from the eval results
        for i, result in enumerate(eval_loop_results):
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v.cpu().item()

        return eval_loop_results

    def run_predict(self):
        # prepare dataloaders
        dataloaders, max_batches = self.predict_loop.get_predict_dataloaders()

        # check if we want to skip this evaluation
        if self.predict_loop.should_skip_predict(max_batches):
            return []

        # ref model
        model = self.lightning_module

        # enable eval mode + no grads
        self.predict_loop.on_predict_model_eval()
        model.zero_grad()
        torch.set_grad_enabled(False)

        # set up the eval loop
        self.predict_loop.setup(model, max_batches, dataloaders)

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
                with self.profiler.profile("predict"):
                    self.predict_loop.predict(batch, batch_idx, dataloader_idx)

        results = self.predict_loop.on_predict_epoch_end()
        return results

    def run_sanity_check(self, ref_model):
        using_val_step = ref_model.val_dataloader is not None and is_overridden('validation_step', ref_model)
        should_sanity_check = using_val_step and self.num_sanity_val_steps > 0 and self.limit_val_batches > 0

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        if should_sanity_check:
            stage = self._running_stage
            self.sanity_checking = True

            # hook and callback
            self.on_sanity_check_start()

            # run eval step
            _, eval_results = self.run_evaluation()

            # allow no returns from eval
            if eval_results is not None and len(eval_results) > 0:
                # when we get a list back, used only the last item
                if isinstance(eval_results, list):
                    eval_results = eval_results[-1]

                _, _, _, callback_metrics, _ = self.process_dict_result(eval_results)
                self.logger_connector.callback_metrics = callback_metrics

            self.on_sanity_check_end()

            self._running_stage = stage

    def validate(
        self,
        model: Optional[LightningModule] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        ckpt_path: Optional[str] = 'best',
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ):
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

            datamodule: A instance of :class:`LightningDataModule`.

        Returns:
            The dictionary with final validation results returned by validation_epoch_end.
            If validation_epoch_end is not defined, the output is a list of the dictionaries
            returned by validation_step.
        """
        # --------------------
        # SETUP HOOK
        # --------------------
        self.verbose_evaluate = verbose

        self.state = TrainerState.VALIDATING
        self.validating = True

        # If you supply a datamodule you can't supply val_dataloaders
        if val_dataloaders and datamodule:
            raise MisconfigurationException(
                'You cannot pass both `trainer.validate(val_dataloaders=..., datamodule=...)`'
            )

        model_provided = model is not None
        model = model or self.lightning_module

        # Attach datamodule to get setup/prepare_data added to model before the call to it below
        self.data_connector.attach_datamodule(model, datamodule)
        #  Attach dataloaders (if given)
        self.data_connector.attach_dataloaders(model, val_dataloaders=val_dataloaders)

        if not model_provided:
            self.validated_ckpt_path = self.__load_ckpt_weights(model, ckpt_path=ckpt_path)

        # run validate
        results = self.fit(model)

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
    ):
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

            datamodule: A instance of :class:`LightningDataModule`.

        Returns:
            Returns a list of dictionaries, one for each test dataloader containing their respective metrics.
        """
        # --------------------
        # SETUP HOOK
        # --------------------
        self.verbose_evaluate = verbose

        self.state = TrainerState.TESTING
        self.testing = True

        # If you supply a datamodule you can't supply test_dataloaders
        if test_dataloaders and datamodule:
            raise MisconfigurationException('You cannot pass both `trainer.test(test_dataloaders=..., datamodule=...)`')

        model_provided = model is not None
        model = model or self.lightning_module

        # Attach datamodule to get setup/prepare_data added to model before the call to it below
        self.data_connector.attach_datamodule(model, datamodule)
        #  Attach dataloaders (if given)
        self.data_connector.attach_dataloaders(model, test_dataloaders=test_dataloaders)

        if not model_provided:
            self.tested_ckpt_path = self.__load_ckpt_weights(model, ckpt_path=ckpt_path)

        # run test
        results = self.fit(model)

        assert self.state.stopped
        self.testing = False

        return results

    def __load_ckpt_weights(
        self,
        model,
        ckpt_path: Optional[str] = None,
    ) -> Optional[str]:
        # if user requests the best checkpoint but we don't have it, error
        if ckpt_path == 'best' and not self.checkpoint_callback.best_model_path:
            raise MisconfigurationException(
                'ckpt_path is "best", but `ModelCheckpoint` is not configured to save the best model.'
            )

        # load best weights
        if ckpt_path is not None:
            # ckpt_path is 'best' so load the best model
            if ckpt_path == 'best':
                ckpt_path = self.checkpoint_callback.best_model_path

            if not ckpt_path:
                fn = self.state.value
                raise MisconfigurationException(
                    f'`.{fn}()` found no path for the best weights: "{ckpt_path}". Please'
                    ' specify a path for a checkpoint `.{fn}(ckpt_path=PATH)`'
                )

            self.training_type_plugin.barrier()

            ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(ckpt['state_dict'])
        return ckpt_path

    def predict(
        self,
        model: Optional[LightningModule] = None,
        dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        r"""

        Separates from fit to make sure you never run on your predictions set until you want to.

        This will call the model forward function to compute predictions.

        Args:
            model: The model to predict on.

            dataloaders: Either a single
                Pytorch Dataloader or a list of them, specifying inference samples.

            datamodule: A instance of :class:`LightningDataModule`.

        Returns:
            Returns a list of dictionaries, one for each provided dataloader containing their respective predictions.
        """

        # --------------------
        # SETUP HOOK
        # --------------------
        # If you supply a datamodule you can't supply dataloaders

        model = model or self.lightning_module

        self.state = TrainerState.PREDICTING
        self.predicting = True

        if dataloaders and datamodule:
            raise MisconfigurationException(
                'You cannot pass dataloaders to trainer.predict if you supply a datamodule.'
            )

        # Attach datamodule to get setup/prepare_data added to model before the call to it below
        self.data_connector.attach_datamodule(model, datamodule)
        #  Attach dataloaders (if given)
        self.data_connector.attach_dataloaders(model, predict_dataloaders=dataloaders)

        results = self.fit(model)

        assert self.state.stopped
        self.predicting = False

        return results

    def tune(
        self,
        model: LightningModule,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        r"""
        Runs routines to tune hyperparameters before training.

        Args:
            datamodule: A instance of :class:`LightningDataModule`.

            model: Model to tune.

            train_dataloader: A Pytorch DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            val_dataloaders: Either a single Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped

        """
        self.state = TrainerState.TUNING
        self.tuning = True

        self.tuner.tune(model, train_dataloader, val_dataloaders, datamodule)

        assert self.state.stopped
        self.tuning = False

    def call_setup_hook(self, model: LightningModule) -> None:
        assert self.state.running, f"TrainerState: {self.state}"
        # 'fit' is passed for `trainer.tune()` as there aren't "tune_dataloaders"
        state = TrainerState.FITTING if self.state == TrainerState.TUNING else self.state

        if self.datamodule is not None:
            called = getattr(self.datamodule, f'has_setup_{state}')
            if not called:
                self.datamodule.setup(stage=state)

        self.setup(model, stage=state)
        model.setup(stage=state)

    def call_teardown_hook(self, model: LightningModule) -> None:
        if self.state.running:
            state = TrainerState.FITTING if self.state == TrainerState.TUNING else self.state
        else:
            state = None

        self.teardown(stage=state)
        model.teardown(stage=state)

    def _reset_result_and_set_hook_fx_name(self, hook_name):
        # on_before_zero_grad is called within training_step
        if "batch_start" in hook_name or "on_before_zero_grad" in hook_name:
            return True
        model_ref = self.lightning_module
        if model_ref is not None:
            # used to track current hook name called
            model_ref._results = Result()
            model_ref._current_hook_fx_name = hook_name
        return False

    def _cache_logged_metrics(self):
        model_ref = self.lightning_module
        if model_ref is not None:
            # capture logging for this hook
            self.logger_connector.cache_logged_metrics()

    def call_hook(self, hook_name, *args, **kwargs):
        # set hook_name to model + reset Result obj
        skip = self._reset_result_and_set_hook_fx_name(hook_name)

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
