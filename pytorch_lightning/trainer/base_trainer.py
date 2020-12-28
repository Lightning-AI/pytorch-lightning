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
from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from copy import deepcopy
import inspect
import multiprocessing
import os
import platform
from typing import Callable, cast, Iterable, List, Mapping, Optional, Tuple, Type, Union

import torch
from torch import optim, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning import _logger as log
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.accelerator_connector import AcceleratorConnector
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, ProgressBarBase
from pytorch_lightning.core import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.optimizer import is_lightning_optimizer, LightningOptimizer
from pytorch_lightning.core.step_result import EvalResult, Result
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.plugins.plugin_connector import PluginConnector
from pytorch_lightning.profiler import BaseProfiler
from pytorch_lightning.trainer.configuration_validator import ConfigValidator
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.trainer.connectors.debugging_connector import DebuggingConnector
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.connectors.model_connector import ModelConnector
from pytorch_lightning.trainer.connectors.optimizer_connector import OptimizerConnector
from pytorch_lightning.trainer.connectors.precision_connector import PrecisionConnector
from pytorch_lightning.trainer.connectors.profiler_connector import ProfilerConnector
from pytorch_lightning.trainer.connectors.slurm_connector import SLURMConnector
from pytorch_lightning.trainer.connectors.training_trick_connector import TrainingTricksConnector
from pytorch_lightning.trainer.evaluation_loop import EvaluationLoop
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.trainer.training_loop import TrainLoop
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities import (
    argparse_utils,
    DeviceType,
    DistributedType,
    HOROVOD_AVAILABLE,
    rank_zero_warn,
    TPU_AVAILABLE,
)
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.data import has_iterable_dataset, has_len
from pytorch_lightning.utilities.debugging import InternalDebugger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.model_utils import is_overridden

if TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm

if HOROVOD_AVAILABLE:
    import horovod.torch as hvd


class BaseTrainer:

    ###########################
    #                         #
    #    Trainer Attributes   #
    #                         #
    ###########################

    current_epoch: int
    on_gpu: bool
    log_gpu_memory: ...
    logger: Union[LightningLoggerBase, bool]
    global_step: int
    global_rank: int
    use_dp: bool
    use_ddp: bool
    use_horovod: bool
    use_ddp2: bool
    use_tpu: bool
    num_sanity_val_steps: float
    precision: int
    should_stop: bool
    move_metrics_to_cpu: bool
    logger_connector: LoggerConnector
    _state: TrainerState
    fast_dev_run: Union[int, bool]
    model: LightningModule
    data_parallel_device_ids: Optional[List[int]]
    _progress_bar_callback: ProgressBarBase
    _default_root_dir: str
    _weights_save_path: str
    model_connector: ModelConnector
    checkpoint_connector: CheckpointConnector
    callbacks: List[Callback]
    val_check_interval: float
    tpu_local_core_rank: int
    train_dataloader: DataLoader
    overfit_batches: Union[int, float]
    num_training_batches: Union[int, float]
    val_check_batch: Union[int, float]
    val_dataloaders: List[DataLoader]
    num_val_batches: List[Union[int, float]]
    test_dataloaders: List[DataLoader]
    num_test_batches: List[Union[int, float]]
    limit_train_batches: Union[int, float]
    limit_val_batches: Union[int, float]
    limit_test_batches: Union[int, float]
    replace_sampler_ddp: bool
    accelerator_backend: Accelerator
    num_nodes: int
    num_processes: int
    distributed_backend: Optional[str]
    dev_debugger: InternalDebugger
    _distrib_type: DistributedType
    _device_type: DeviceType
    shown_warnings: set
    dev_debugger: InternalDebugger
    config_validator: ConfigValidator
    data_connector: DataConnector
    optimizer_connector: OptimizerConnector
    accelerator_connector: AcceleratorConnector
    logger_connector: LoggerConnector
    model_connector: ModelConnector
    precision_connector: PrecisionConnector
    callback_connector: CallbackConnector
    debugging_connector: DebuggingConnector
    training_tricks_connector: TrainingTricksConnector
    profile_connector: ProfilerConnector
    checkpoint_connector: CheckpointConnector
    slurm_connector: SLURMConnector
    tuner: Tuner
    evaluation_loop: EvaluationLoop
    train_loop: TrainLoop
    plugin_connector: PluginConnector
    fast_dev_run: Union[bool, int]
    profiler: Union[BaseProfiler, bool, str]
    datamodule: LightningDataModule

    ###########################
    #                         #
    #    Trainer Functions    #
    #                         #
    ###########################

    def fit(self, *_, **__):
        raise NotImplementedError

    def tune(self, *_, **__):
        raise NotImplementedError

    def test(self, *_, **__):
        raise NotImplementedError

    ###########################
    #                         #
    #    TrainerProperties    #
    #                         #
    ###########################

    @property
    def log_dir(self):
        if self.checkpoint_callback is not None:
            dirpath = self.checkpoint_callback.dirpath
            dirpath = os.path.split(dirpath)[0]
        elif self.logger is not None:
            if isinstance(self.logger, TensorBoardLogger):
                dirpath = self.logger.log_dir
            else:
                dirpath = self.logger.save_dir
        else:
            dirpath = self._default_root_dir

        if self.accelerator_backend is not None:
            dirpath = self.accelerator_backend.broadcast(dirpath)
        return dirpath

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
    def logged_metrics(self):
        return self.logger_connector.logged_metrics

    @logged_metrics.setter
    def logged_metrics(self, x):
        self.logger_connector.logged_metrics = x

    @property
    def progress_bar_metrics(self):
        return self.logger_connector.progress_bar_metrics

    @progress_bar_metrics.setter
    def progress_bar_metrics(self, x):
        self.logger_connector.progress_bar_metrics = x

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
        init_signature = inspect.signature(cls)

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
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
        return argparse_utils.from_argparse_args(cls, args, **kwargs)

    @classmethod
    def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        return argparse_utils.parse_argparser(cls, arg_parser)

    @classmethod
    def match_env_arguments(cls) -> Namespace:
        return argparse_utils.parse_env_variables(cls)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        return argparse_utils.add_argparse_args(cls, parent_parser)

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
        ref_model = cast(LightningModule, ref_model)
        return dict(**ref_model.get_progress_bar_dict(), **self.logger_connector.progress_bar_metrics)

    @property
    def disable_validation(self) -> bool:
        """ Check if validation is disabled during training. """
        return not self.enable_validation

    @property
    def enable_validation(self) -> bool:
        """ Check if we should run validation during training. """
        model_ref = self.model_connector.get_model()
        val_loop_enabled = is_overridden('validation_step', model_ref) and self.limit_val_batches > 0
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

    @property
    def checkpoint_callback(self) -> Optional[ModelCheckpoint]:
        """
        The first checkpoint callback in the Trainer.callbacks list, or ``None`` if
        no checkpoint callbacks exist.
        """
        callbacks = self.checkpoint_callbacks
        return callbacks[0] if len(callbacks) > 0 else None

    @property
    def checkpoint_callbacks(self) -> List[ModelCheckpoint]:
        """ A list of all instances of ModelCheckpoint found in the Trainer.callbacks list. """
        return [c for c in self.callbacks if isinstance(c, ModelCheckpoint)]

    def save_checkpoint(self, filepath, weights_only: bool = False):
        self.checkpoint_connector.save_checkpoint(filepath, weights_only)

    def get_model(self):
        return self.model_connector.get_model()

    def __getstate__(self):
        # unwrap optimizer
        self.optimizers = [opt._optimizer if is_lightning_optimizer(opt) else opt for opt in self.optimizers]
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        # wrap optimizers in enable_pl_optimzer is True
        self.convert_to_lightning_optimizers()

    @property
    def require_distributed_sampler(self):
        if self.accelerator_backend is not None:
            return self.accelerator_backend.require_distributed_sampler
        return self.use_ddp or self.use_ddp2 or self.use_horovod or self.use_tpu

    @property
    def distributed_sampler_kwargs(self):
        if self.accelerator_backend is not None:
            return self.accelerator_backend.distributed_sampler_kwargs

        if self.use_tpu:
            kwargs = dict(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())

        elif self.use_horovod:
            kwargs = dict(num_replicas=hvd.size(), rank=hvd.rank())

        else:
            world_size = {
                "ddp": self.num_nodes * self.num_processes,
                "ddp_spawn": self.num_nodes * self.num_processes,
                "ddp2": self.num_nodes,
                "ddp_cpu": self.num_processes * self.num_nodes
            }
            assert self.distributed_backend is not None
            kwargs = dict(num_replicas=world_size[self.distributed_backend], rank=self.global_rank)

        return kwargs

    ################################
    #   TrainerInternalLogicMixin  #
    ################################

    def run_evaluation(self, test_mode: bool = False, max_batches=None):

        # used to know if we are logging for val, test + reset cached results
        self.logger_connector.set_stage(test_mode, reset=True)

        # bookkeeping
        self.evaluation_loop.testing = test_mode

        # prepare dataloaders
        dataloaders, max_batches = self.evaluation_loop.get_evaluation_dataloaders(max_batches)

        # check if we want to skip this evaluation
        if self.evaluation_loop.should_skip_evaluation(dataloaders, max_batches):
            return [], []

        # ref model
        model = self.get_model()

        # enable eval mode + no grads
        self.evaluation_loop.on_evaluation_model_eval()
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
                with self.profiler.profile("evaluation_step_and_end"):
                    output = self.evaluation_loop.evaluation_step(test_mode, batch, batch_idx, dataloader_idx)
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
                output.detach()
                if self.move_metrics_to_cpu:
                    output.cpu()
            elif isinstance(output, dict):
                output = recursive_detach(output, to_cpu=self.move_metrics_to_cpu)
            elif isinstance(output, torch.Tensor) and output.is_cuda and self.move_metrics_to_cpu:
                output = output.cpu()
            outputs.append(output)
        return outputs

    def run_test(self):
        # only load test dataloader for testing
        # self.reset_test_dataloader(ref_model)
        with self.profiler.profile("run_test_evaluation"):
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
                    _, _, _, callback_metrics, _ = self.process_dict_result(eval_results)
                self.logger_connector.callback_metrics = callback_metrics

            self.on_sanity_check_end()
            self.running_sanity_check = False

    def _test_using_best_weights(self, ckpt_path, test_dataloaders):
        model = self.get_model()

        # if user requests the best checkpoint but we don't have it, error
        if ckpt_path == 'best' and not self.checkpoint_callback.best_model_path:
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
            if self.accelerator_backend is not None and not self.use_tpu:
                self.accelerator_backend.barrier()

            ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
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

    def _test_given_model(self, model, test_dataloaders):

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
        self.setup(model, stage_name)
        model.setup(stage_name)

    def _reset_result_and_set_hook_fx_name(self, hook_name):
        # on_before_zero_grad is called within training_step
        if "batch_start" in hook_name or "on_before_zero_grad" in hook_name:
            return True
        model_ref = self.get_model()
        if model_ref is not None:
            # used to track current hook name called
            model_ref._results = Result()
            model_ref._current_hook_fx_name = hook_name
        return False

    def _cache_logged_metrics(self):
        model_ref = self.get_model()
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
            model_ref = self.get_model()
            if is_overridden(hook_name, model_ref):
                hook_fx = getattr(model_ref, hook_name)
                output = hook_fx(*args, **kwargs)

            # if the PL module doesn't have the hook then call the accelator
            # used to auto-reduce things for the user with Results obj
            elif hasattr(self.accelerator_backend, hook_name):
                accelerator_hook = getattr(self.accelerator_backend, hook_name)
                output = accelerator_hook(*args, **kwargs)

        if not skip:
            self._cache_logged_metrics()
        return output

    @staticmethod
    def available_plugins():
        """
            List of all available plugins that can be string arguments to the trainer.
            Returns: List of all available plugins that are supported as string arguments.
        """
        return PluginConnector.available_plugins()

    #################################
    #                               #
    #    TrainerCallbackHookMixin   #
    #                               #
    #################################

    def setup(self, model, stage: str):
        """Called in the beginning of fit and test"""
        for callback in self.callbacks:
            callback.setup(self, model, stage)

    def teardown(self, stage: str):
        """Called at the end of fit and test"""
        for callback in self.callbacks:
            callback.teardown(self, self.get_model(), stage)

    def on_init_start(self):
        """Called when the trainer initialization begins, model has not yet been set."""
        for callback in self.callbacks:
            callback.on_init_start(self)

    def on_init_end(self):
        """Called when the trainer initialization ends, model has not yet been set."""
        for callback in self.callbacks:
            callback.on_init_end(self)

    def on_fit_start(self):
        """Called when the trainer initialization begins, model has not yet been set."""
        for callback in self.callbacks:
            callback.on_fit_start(self, self.get_model())

    def on_fit_end(self):
        """Called when the trainer initialization begins, model has not yet been set."""
        for callback in self.callbacks:
            callback.on_fit_end(self, self.get_model())

    def on_sanity_check_start(self):
        """Called when the validation sanity check starts."""
        for callback in self.callbacks:
            callback.on_sanity_check_start(self, self.get_model())

    def on_sanity_check_end(self):
        """Called when the validation sanity check ends."""
        for callback in self.callbacks:
            callback.on_sanity_check_end(self, self.get_model())

    def on_train_epoch_start(self):
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_train_epoch_start(self, self.get_model())

    def on_train_epoch_end(self, outputs):
        """Called when the epoch ends."""
        for callback in self.callbacks:
            callback.on_train_epoch_end(self, self.get_model(), outputs)

    def on_validation_epoch_start(self):
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_validation_epoch_start(self, self.get_model())

    def on_validation_epoch_end(self):
        """Called when the epoch ends."""
        for callback in self.callbacks:
            callback.on_validation_epoch_end(self, self.get_model())

    def on_test_epoch_start(self):
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_test_epoch_start(self, self.get_model())

    def on_test_epoch_end(self):
        """Called when the epoch ends."""
        for callback in self.callbacks:
            callback.on_test_epoch_end(self, self.get_model())

    def on_epoch_start(self):
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_epoch_start(self, self.get_model())

    def on_epoch_end(self):
        """Called when the epoch ends."""
        for callback in self.callbacks:
            callback.on_epoch_end(self, self.get_model())

    def on_train_start(self):
        """Called when the train begins."""
        for callback in self.callbacks:
            callback.on_train_start(self, self.get_model())

    def on_train_end(self):
        """Called when the train ends."""
        for callback in self.callbacks:
            callback.on_train_end(self, self.get_model())

    def on_pretrain_routine_start(self, model):
        """Called when the train begins."""
        for callback in self.callbacks:
            callback.on_pretrain_routine_start(self, model)

    def on_pretrain_routine_end(self, model):
        """Called when the train ends."""
        for callback in self.callbacks:
            callback.on_pretrain_routine_end(self, model)

    def on_batch_start(self):
        """Called when the training batch begins."""
        for callback in self.callbacks:
            callback.on_batch_start(self, self.get_model())

    def on_batch_end(self):
        """Called when the training batch ends."""
        for callback in self.callbacks:
            callback.on_batch_end(self, self.get_model())

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        """Called when the training batch begins."""
        for callback in self.callbacks:
            callback.on_train_batch_start(self, self.get_model(), batch, batch_idx, dataloader_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Called when the training batch ends."""
        for callback in self.callbacks:
            callback.on_train_batch_end(self, self.get_model(), outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        """Called when the validation batch begins."""
        for callback in self.callbacks:
            callback.on_validation_batch_start(self, self.get_model(), batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        for callback in self.callbacks:
            callback.on_validation_batch_end(self, self.get_model(), outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        """Called when the test batch begins."""
        for callback in self.callbacks:
            callback.on_test_batch_start(self, self.get_model(), batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        for callback in self.callbacks:
            callback.on_test_batch_end(self, self.get_model(), outputs, batch, batch_idx, dataloader_idx)

    def on_validation_start(self):
        """Called when the validation loop begins."""
        for callback in self.callbacks:
            callback.on_validation_start(self, self.get_model())

    def on_validation_end(self):
        """Called when the validation loop ends."""
        for callback in self.callbacks:
            callback.on_validation_end(self, self.get_model())

    def on_test_start(self):
        """Called when the test begins."""
        for callback in self.callbacks:
            callback.on_test_start(self, self.get_model())

    def on_test_end(self):
        """Called when the test ends."""
        for callback in self.callbacks:
            callback.on_test_end(self, self.get_model())

    def on_keyboard_interrupt(self):
        """Called when the training is interrupted by KeyboardInterrupt."""
        for callback in self.callbacks:
            callback.on_keyboard_interrupt(self, self.get_model())

    def on_save_checkpoint(self):
        """Called when saving a model checkpoint."""
        callback_states = {}
        for callback in self.callbacks:
            callback_class = type(callback)
            state = callback.on_save_checkpoint(self, self.get_model())
            if state:
                callback_states[callback_class] = state
        return callback_states

    def on_load_checkpoint(self, checkpoint):
        """Called when loading a model checkpoint."""
        callback_states = checkpoint.get('callbacks')
        for callback in self.callbacks:
            state = callback_states.get(type(callback))
            if state:
                state = deepcopy(state)
                callback.on_load_checkpoint(state)

    def on_after_backward(self):
        """
        Called after loss.backward() and before optimizers do anything.
        """
        for callback in self.callbacks:
            callback.on_after_backward(self, self.get_model())

    def on_before_zero_grad(self, optimizer):
        """
        Called after optimizer.step() and before optimizer.zero_grad().
        """
        for callback in self.callbacks:
            callback.on_before_zero_grad(self, self.get_model(), optimizer)

    ###############################
    #                             #
    #    TrainerModelHooksMixin   #
    #                             #
    ###############################

    def is_function_implemented(self, f_name, model=None):
        if model is None:
            model = self.get_model()
        f_op = getattr(model, f_name, None)
        return callable(f_op)

    def has_arg(self, f_name, arg_name):
        model = self.get_model()
        f_op = getattr(model, f_name, None)
        return arg_name in inspect.signature(f_op).parameters

    ##############################
    #   TrainerOptimizersMixin   #
    ##############################

    def init_optimizers(self, model: LightningModule) -> Tuple[List, List, List]:
        optim_conf = model.configure_optimizers()
        if optim_conf is None:
            rank_zero_warn(
                '`LightningModule.configure_optimizers` returned `None`, this fit will run with no optimizer',
                UserWarning,
            )
            optim_conf = _MockOptimizer()

        optimizers, lr_schedulers, optimizer_frequencies = [], [], []
        monitor = None

        # single output, single optimizer
        if isinstance(optim_conf, Optimizer):
            optimizers = [optim_conf]
        # two lists, optimizer + lr schedulers
        elif isinstance(optim_conf, (list, tuple)) and len(optim_conf) == 2 and isinstance(optim_conf[0], list):
            opt, sch = optim_conf
            optimizers = opt
            lr_schedulers = sch if isinstance(sch, list) else [sch]
        # single dictionary
        elif isinstance(optim_conf, dict):
            optimizers = [optim_conf["optimizer"]]
            monitor = optim_conf.get('monitor', None)
            lr_schedulers = [optim_conf["lr_scheduler"]] if "lr_scheduler" in optim_conf else []
        # multiple dictionaries
        elif isinstance(optim_conf, (list, tuple)) and all(isinstance(d, dict) for d in optim_conf):
            optimizers = [opt_dict["optimizer"] for opt_dict in optim_conf]
            lr_schedulers = [opt_dict["lr_scheduler"] for opt_dict in optim_conf if "lr_scheduler" in opt_dict]
            optimizer_frequencies = [
                opt_dict["frequency"] for opt_dict in optim_conf if opt_dict.get("frequency", None) is not None
            ]
            # assert that if frequencies are present, they are given for all optimizers
            if optimizer_frequencies and len(optimizer_frequencies) != len(optimizers):
                raise ValueError("A frequency must be given to each optimizer.")
        # single list or tuple, multiple optimizer
        elif isinstance(optim_conf, (list, tuple)):
            optimizers = list(optim_conf)
        # unknown configuration
        else:
            raise MisconfigurationException(
                'Unknown configuration for model optimizers.'
                ' Output from `model.configure_optimizers()` should either be:\n'
                ' * `torch.optim.Optimizer`\n'
                ' * [`torch.optim.Optimizer`]\n'
                ' * ([`torch.optim.Optimizer`], [`torch.optim.lr_scheduler`])\n'
                ' * {"optimizer": `torch.optim.Optimizer`, (optional) "lr_scheduler": `torch.optim.lr_scheduler`}\n'
                ' * A list of the previously described dict format, with an optional "frequency" key (int)'
            )
        lr_schedulers = self.configure_schedulers(lr_schedulers, monitor=monitor)

        return optimizers, lr_schedulers, optimizer_frequencies

    def convert_to_lightning_optimizers(self):
        def _convert_to_lightning_optimizer(trainer, optimizer):
            if not isinstance(optimizer, LightningOptimizer):
                optimizer = LightningOptimizer(optimizer)
            optimizer._on_trainer_init(trainer)
            return optimizer

        if self._enable_pl_optimizer:
            self.optimizers = [_convert_to_lightning_optimizer(self, opt) for opt in self.optimizers]

    def configure_schedulers(self, schedulers: list, monitor: Optional[str] = None):
        # Convert each scheduler into dict structure with relevant information
        lr_schedulers = []
        default_config = {
            'scheduler': None,
            'name': None,  # no custom name
            'interval': 'epoch',  # after epoch is over
            'frequency': 1,  # every epoch/batch
            'reduce_on_plateau': False,  # most often not ReduceLROnPlateau scheduler
            'monitor': monitor,  # value to monitor for ReduceLROnPlateau
            'strict': True,  # enforce that the monitor exists for ReduceLROnPlateau
        }
        for scheduler in schedulers:
            if isinstance(scheduler, dict):
                # check provided keys
                extra_keys = [k for k in scheduler.keys() if k not in default_config.keys()]
                if extra_keys:
                    rank_zero_warn(f'Found unsupported keys in the lr scheduler dict: {extra_keys}', RuntimeWarning)
                if 'scheduler' not in scheduler:
                    raise MisconfigurationException(
                        'The lr scheduler dict must have the key "scheduler" with its item being an lr scheduler'
                    )
                scheduler['reduce_on_plateau'] = isinstance(
                    scheduler['scheduler'], optim.lr_scheduler.ReduceLROnPlateau
                )
                if scheduler['reduce_on_plateau'] and scheduler.get('monitor', None) is None:
                    raise MisconfigurationException(
                        'The lr scheduler dict must include a monitor when a `ReduceLROnPlateau` scheduler is used.'
                        ' For example: {"optimizer": optimizer, "lr_scheduler":'
                        ' {"scheduler": scheduler, "monitor": "your_loss"}}'
                    )
                lr_schedulers.append({**default_config, **scheduler})
            elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if monitor is None:
                    raise MisconfigurationException(
                        '`configure_optimizers` must include a monitor when a `ReduceLROnPlateau` scheduler is used.'
                        ' For example: {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "metric_to_track"}'
                    )
                lr_schedulers.append(
                    {**default_config, 'scheduler': scheduler, 'reduce_on_plateau': True, 'monitor': monitor}
                )
            elif isinstance(scheduler, optim.lr_scheduler._LRScheduler):
                lr_schedulers.append({**default_config, 'scheduler': scheduler})
            else:
                raise ValueError(f'The provided lr scheduler "{scheduler}" is invalid')
        return lr_schedulers

    def reinit_scheduler_properties(self, optimizers: list, schedulers: list):
        # Reinitialize optimizer.step properties added by schedulers
        for scheduler in schedulers:
            scheduler = scheduler['scheduler']

            for optimizer in optimizers:
                # check that we dont mix users optimizers and schedulers
                if scheduler.optimizer == optimizer:
                    # Find the mro belonging to the base lr scheduler class
                    for i, mro in enumerate(scheduler.__class__.__mro__):
                        if mro in (optim.lr_scheduler._LRScheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            idx = i
                            state = scheduler.state_dict()
                        else:
                            state = None

                scheduler.__class__.__mro__[idx].__init__(scheduler, optimizer)
                if state is not None:
                    scheduler.load_state_dict(state)

    ###########################
    #   TrainerLoggingMixin   #
    ###########################

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def process_dict_result(self, output, train=False):
        """Reduces output according to the training mode.

        Separates loss from logging and progress bar metrics
        """
        # --------------------
        # WARN DEPRECATED KEYS
        # --------------------
        # TODO: 1.0.0 remove
        if isinstance(output, dict):
            for k, v in output.items():
                if k in ['log', 'progress_bar']:
                    m = inspect.cleandoc(
                        f"""The {{{k}:dict keyword}} was deprecated in 0.9.1 and will be removed in 1.0.0
                        Please use self.log(...) inside the lightningModule instead.

                        # log on a step or aggregate epoch metric to the logger and/or progress bar
                        # (inside LightningModule)
                        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
                    """)
                    rank_zero_warn(m)

        # --------------------------
        # handle single scalar only
        # --------------------------
        # single scalar returned from a xx_step
        if isinstance(output, torch.Tensor):
            progress_bar_metrics = {}
            log_metrics = {}
            callback_metrics = {}
            hiddens = None
            return output, progress_bar_metrics, log_metrics, callback_metrics, hiddens

        # ---------------
        # EXTRACT CALLBACK KEYS
        # ---------------
        # all keys not progress_bar or log are candidates for callbacks
        callback_metrics = {}
        if isinstance(output, Mapping):
            for k, v in output.items():
                if k not in ['progress_bar', 'log', 'hiddens']:
                    callback_metrics[k] = v

        if train and (self.use_dp or self.use_ddp2):
            num_gpus = self.num_gpus
            callback_metrics = self.reduce_distributed_output(callback_metrics, num_gpus)

        # ---------------
        # EXTRACT PROGRESS BAR KEYS
        # ---------------
        try:
            progress_output = output['progress_bar']

            # reduce progress metrics for progress bar when using dp
            if train and (self.use_dp or self.use_ddp2):
                num_gpus = self.num_gpus
                progress_output = self.reduce_distributed_output(progress_output, num_gpus)

            progress_bar_metrics = progress_output
        except Exception:
            progress_bar_metrics = {}

        # ---------------
        # EXTRACT LOGGING KEYS
        # ---------------
        # extract metrics to log to experiment
        try:
            log_output = output['log']

            # reduce progress metrics for progress bar when using dp
            if train and (self.use_dp or self.use_ddp2):
                num_gpus = self.num_gpus
                log_output = self.reduce_distributed_output(log_output, num_gpus)

            log_metrics = log_output
        except Exception:
            log_metrics = {}

        # ---------------
        # EXTRACT LOSS
        # ---------------
        # if output dict doesn't have the keyword loss
        # then assume the output=loss if scalar
        loss = None
        if train:
            try:
                loss = output['loss']
            except Exception as exp:
                if isinstance(output, torch.Tensor):
                    loss = output
                else:
                    raise RuntimeError(
                        'No `loss` value in the dictionary returned from `model.training_step()`.'
                    ) from exp

            # when using dp need to reduce the loss
            if self.use_dp or self.use_ddp2:
                loss = self.reduce_distributed_output(loss, self.num_gpus)

        # ---------------
        # EXTRACT HIDDEN
        # ---------------
        hiddens = output.get('hiddens', None) if isinstance(output, Mapping) else None

        # use every metric passed in as a candidate for callback
        callback_metrics.update(progress_bar_metrics)
        callback_metrics.update(log_metrics)

        # detach all metrics for callbacks to prevent memory leaks
        # no .item() because it will slow things down
        callback_metrics = recursive_detach(callback_metrics)
        progress_bar_metrics = recursive_detach(progress_bar_metrics)
        log_metrics = recursive_detach(log_metrics)

        return loss, progress_bar_metrics, log_metrics, callback_metrics, hiddens

    def reduce_distributed_output(self, output, num_gpus):
        if num_gpus <= 1:
            return output

        # when using DP, we get one output per gpu
        # average outputs and return
        if isinstance(output, torch.Tensor):
            return output.mean()

        for k, v in output.items():
            # recurse on nested dics
            if isinstance(output[k], dict):
                output[k] = self.reduce_distributed_output(output[k], num_gpus)

            # compute the average of scalars
            elif isinstance(output[k], list):
                output[k] = sum(output[k]) / len(output[k])

            # do nothing when there's a scalar
            elif isinstance(output[k], torch.Tensor) and output[k].dim() == 0:
                pass

            # do not reduce metrics that have batch size > num gpus
            elif output[k].size(0) <= num_gpus:
                output[k] = torch.mean(output[k])

        return output

    ###################################
    #                                 #
    #    TrainerTrainingTricksMixin   #
    #                                 #
    ###################################

    def print_nan_gradients(self) -> None:
        model = self.get_model()
        for param in model.parameters():
            if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                log.info(param, param.grad)

    def detect_nan_tensors(self, loss: Tensor) -> None:
        model = self.get_model()

        # check if loss is nan
        if not torch.isfinite(loss).all():
            raise ValueError(
                'The loss returned in `training_step` is nan or inf.'
            )
        # check if a network weight is nan
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                self.print_nan_gradients()
                raise ValueError(
                    f'Detected nan and/or inf values in `{name}`.'
                    ' Check your forward pass for numerically unstable operations.'
                )

    ###############################
    #   TrainerDataLoadingMixin   #
    ###############################

    def _worker_check(self, dataloader: DataLoader, name: str) -> None:
        on_windows = platform.system() == 'Windows'

        # ddp_spawn + num_workers > 0 don't mix! tell the user
        is_dataloader = isinstance(dataloader, DataLoader)
        using_spawn = self.distributed_backend == "ddp_spawn"
        if is_dataloader and not on_windows:
            if dataloader.num_workers > 0 and using_spawn:
                rank_zero_warn('Dataloader(num_workers>0) and ddp_spawn do not mix well!'
                               ' Your performance might suffer dramatically.'
                               ' Please consider setting accelerator=ddp to use num_workers > 0'
                               ' (this is a bottleneck of Python .spawn() and PyTorch')

            elif dataloader.num_workers == 0 and using_spawn:
                rank_zero_warn('You are using `accelerator=ddp_spawn` with num_workers=0.'
                               ' For much faster performance, switch to `accelerator=ddp`'
                               ' and set `num_workers>0`')

            elif dataloader.num_workers <= 2 and multiprocessing.cpu_count() > 2 and not using_spawn:
                num_cpus = multiprocessing.cpu_count()
                rank_zero_warn(f'The dataloader, {name}, does not have many workers which may be a bottleneck.'
                               ' Consider increasing the value of the `num_workers` argument`'
                               f' (try {num_cpus} which is the number of cpus on this machine)'
                               ' in the `DataLoader` init to improve performance.')

    def auto_add_sampler(self, dataloader: DataLoader, shuffle: bool) -> DataLoader:

        # don't do anything if it's not a dataloader
        is_dataloader = isinstance(dataloader, DataLoader)
        # don't manipulate iterable datasets
        is_iterable_ds = has_iterable_dataset(dataloader)

        if not is_dataloader or is_iterable_ds:
            return dataloader

        need_dist_sampler = self.require_distributed_sampler and not isinstance(dataloader.sampler, DistributedSampler)
        if self.replace_sampler_ddp and need_dist_sampler:
            if not isinstance(dataloader.sampler, (SequentialSampler, RandomSampler)):
                raise MisconfigurationException(
                    'You seem to have configured a sampler in your DataLoader. This will be replaced '
                    ' by `DistributedSampler` since `replace_sampler_ddp` is True and you are using'
                    ' distributed training. Either remove the sampler from your DataLoader or set'
                    ' `replace_sampler_ddp`=False if you want to use your custom sampler.')

            # replace with distributed sampler
            sampler = self._get_distributed_sampler(dataloader, shuffle)
            dataloader = self.replace_sampler(dataloader, sampler)

        return dataloader

    def replace_sampler(self, dataloader, sampler):
        skip_keys = ['sampler', 'batch_sampler', 'dataset_kind']

        dl_args = {
            k: v for k, v in dataloader.__dict__.items() if not k.startswith('_') and k not in skip_keys
        }

        dl_args['sampler'] = sampler
        dl_args['shuffle'] = False
        multiprocessing_context = dataloader.multiprocessing_context
        dataloader = type(dataloader)(**dl_args)
        dataloader.multiprocessing_context = multiprocessing_context
        return dataloader

    def _get_distributed_sampler(self, dataloader, shuffle):
        kwargs = self.distributed_sampler_kwargs
        kwargs['shuffle'] = shuffle and not self.overfit_batches
        sampler = DistributedSampler(dataloader.dataset, **kwargs)
        return sampler

    def reset_train_dataloader(self, model: LightningModule) -> None:
        """Resets the train dataloader and initialises required variables
        (number of batches, when to validate, etc.).

        Args:
            model: The current `LightningModule`
        """
        self.train_dataloader = self.request_dataloader(model.train_dataloader)
        if (self.overfit_batches > 0):
            if hasattr(self.train_dataloader, 'sampler') and isinstance(self.train_dataloader.sampler, RandomSampler):
                rank_zero_warn('You requested to overfit but enabled training dataloader shuffling.'
                               ' We are turning it off for you.')
                self.train_dataloader = self.replace_sampler(
                    self.train_dataloader, SequentialSampler(self.train_dataloader.dataset))

        # debugging
        self.dev_debugger.track_load_dataloader_call('train_dataloader', dataloaders=[self.train_dataloader])

        self.num_training_batches = 0

        # automatically add samplers
        self.train_dataloader = self.auto_add_sampler(self.train_dataloader, shuffle=True)

        self.num_training_batches = len(self.train_dataloader) if has_len(self.train_dataloader) else float('inf')
        self._worker_check(self.train_dataloader, 'train dataloader')

        if isinstance(self.limit_train_batches, int) or self.limit_train_batches == 0.0:
            self.num_training_batches = min(self.num_training_batches, int(self.limit_train_batches))
        elif self.num_training_batches != float('inf'):
            self.num_training_batches = int(self.num_training_batches * self.limit_train_batches)
        elif self.limit_train_batches != 1.0:
            raise MisconfigurationException(
                'When using an IterableDataset for `limit_train_batches`,'
                ' `Trainer(limit_train_batches)` must be `0.0`, `1.0` or an int. An int k specifies'
                ' `num_training_batches` to use.')

        # determine when to check validation
        # if int passed in, val checks that often
        # otherwise, it checks in [0, 1.0] % range of a training epoch
        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
            if self.val_check_batch > self.num_training_batches:
                raise ValueError(
                    f'`val_check_interval` ({self.val_check_interval}) must be less than or equal '
                    f'to the number of the training batches ({self.num_training_batches}). '
                    'If you want to disable validation set `limit_val_batches` to 0.0 instead.')
        else:
            if not has_len(self.train_dataloader):
                if self.val_check_interval == 1.0:
                    self.val_check_batch = float('inf')
                else:
                    raise MisconfigurationException(
                        'When using an IterableDataset for `train_dataloader`,'
                        ' `Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies'
                        ' checking validation every k training batches.')
            else:
                self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
                self.val_check_batch = max(1, self.val_check_batch)

    def _reset_eval_dataloader(
            self,
            model: LightningModule,
            mode: str
    ) -> Tuple[List[Union[int, float]], List[DataLoader]]:
        """Generic method to reset a dataloader for evaluation.

        Args:
            model: The current `LightningModule`
            mode: Either `'val'` or `'test'`

        Returns:
            Tuple (num_batches, dataloaders)
        """
        # always get the loaders first so we can count how many there are
        loader_name = f'{mode}_dataloader'
        dataloaders = self.request_dataloader(getattr(model, loader_name))

        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        # when overfitting use the training loader as val and test
        # duplicate it the numb of times needed to match the train loaders
        if self.overfit_batches > 0:
            num_loaders = len(dataloaders)
            train_dataloader = self.request_dataloader(getattr(model, 'train_dataloader'))
            dataloaders = [deepcopy(train_dataloader) for _ in range(num_loaders)]

        self.dev_debugger.track_load_dataloader_call(loader_name, dataloaders=dataloaders)

        for loader_i in range(len(dataloaders)):
            loader = dataloaders[loader_i]

            # shuffling in val and test set is bad practice
            if mode in ('val', 'test') and hasattr(loader, 'sampler') and isinstance(loader.sampler, RandomSampler):

                # when overfitting, the dataloader should not have sampler
                if self.overfit_batches > 0:
                    rank_zero_warn('You requested to overfit but enabled test/val dataloader shuffling.'
                                   ' We are turning it off for you.')
                    dataloaders[loader_i] = self.replace_sampler(loader, SequentialSampler(loader.dataset))

                else:
                    rank_zero_warn(f'Your {mode}_dataloader has `shuffle=True`, it is best practice to turn'
                                   ' this off for validation and test dataloaders.')

        if any([dl is None for dl in dataloaders]):
            rank_zero_warn("One of given dataloaders is None and it will be skipped.")

        # add samplers
        dataloaders = [self.auto_add_sampler(dl, shuffle=False) for dl in dataloaders if dl is not None]

        loader_num_batches = []

        # determine number of batches
        # datasets could be none, 1 or 2+
        if len(dataloaders) != 0:
            for i, dataloader in enumerate(dataloaders):
                num_batches = len(dataloader) if has_len(dataloader) else float('inf')
                self._worker_check(dataloader, f'{mode} dataloader {i}')

                # percent or num_steps
                limit_eval_batches = getattr(self, f'limit_{mode}_batches')

                # limit num batches either as a percent or num steps
                if isinstance(limit_eval_batches, int) or limit_eval_batches == 0.0:
                    num_batches = min(num_batches, int(limit_eval_batches))
                elif num_batches != float('inf'):
                    num_batches = int(num_batches * limit_eval_batches)
                elif limit_eval_batches != 1.0:
                    raise MisconfigurationException(
                        'When using an IterableDataset for `limit_{mode}_batches`,'
                        f' `Trainer(limit_{mode}_batches)` must be `0.0`, `1.0` or an int. An int k specifies'
                        f' `num_{mode}_batches` to use.')

                if num_batches == 0 and limit_eval_batches > 0.0 and isinstance(limit_eval_batches, float):
                    min_pct = 1.0 / len(dataloader)
                    raise MisconfigurationException(
                        f'you requested to check {limit_eval_batches} of the {mode} dataloader but'
                        f' {limit_eval_batches}*{num_batches} < 1. Please increase the limit_{mode}_batches.'
                        f' Try at least limit_{mode}_batches={min_pct}'
                    )

                loader_num_batches.append(num_batches)

        return loader_num_batches, dataloaders

    def reset_val_dataloader(self, model: LightningModule) -> None:
        """Resets the validation dataloader and determines the number of batches.

        Args:
            model: The current `LightningModule`
        """
        has_loader = is_overridden('val_dataloader', model)
        has_step = is_overridden('validation_step', model)
        if has_loader and has_step:
            self.num_val_batches, self.val_dataloaders = self._reset_eval_dataloader(model, 'val')

    def reset_test_dataloader(self, model) -> None:
        """Resets the validation dataloader and determines the number of batches.

        Args:
            model: The current `LightningModule`
        """
        has_loader = is_overridden('test_dataloader', model)
        has_step = is_overridden('test_step', model)
        if has_loader and has_step:
            self.num_test_batches, self.test_dataloaders =\
                self._reset_eval_dataloader(model, 'test')

    def request_dataloader(self, dataloader_fx: Callable) -> DataLoader:
        """Handles downloading data in the GPU or TPU case.

        Args:
            dataloader_fx: The bound dataloader getter

        Returns:
            The dataloader
        """
        dataloader = dataloader_fx()
        dataloader = self._flatten_dl_only(dataloader)

        if self.accelerator_backend is not None:
            self.accelerator_backend.barrier('get_dataloaders')
        return dataloader

    def _flatten_dl_only(self, dataloaders):
        # handles user error when they return:
        # return dl1, dl2  vs  return (dl1, dl2)
        if isinstance(dataloaders, tuple):
            all_dls = [isinstance(x, Iterable) for x in dataloaders]
            all_dls = all(all_dls)
            if all_dls:
                dataloaders = list(dataloaders)

        return dataloaders

    #######################################
    #                                     #
    #    DeprecatedDistDeviceAttributes   #
    #                                     #
    #######################################

    _distrib_type: DistributedType
    _device_type: DeviceType
    num_gpus: int

    @property
    def on_cpu(self) -> bool:
        # rank_zero_warn("Internal: `on_cpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._device_type == DeviceType.CPU

    @on_cpu.setter
    def on_cpu(self, val: bool) -> None:
        # rank_zero_warn("Internal: `on_cpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        if val:
            self._device_type = DeviceType.CPU

    @property
    def on_tpu(self) -> bool:
        # rank_zero_warn("Internal: `on_tpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._device_type == DeviceType.TPU

    @on_tpu.setter
    def on_tpu(self, val: bool) -> None:
        # rank_zero_warn("Internal: `on_tpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        # todo add logic that it cannot be set if TPU is missing
        if val:
            self._device_type = DeviceType.TPU

    @property
    def use_tpu(self) -> bool:
        # rank_zero_warn("Internal: `use_tpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._device_type == DeviceType.TPU

    @use_tpu.setter
    def use_tpu(self, val: bool) -> None:
        # rank_zero_warn("Internal: `use_tpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        # todo add logic that it cannot be set if TPU is missing
        if val:
            self._device_type = DeviceType.TPU

    @property
    def on_gpu(self) -> bool:
        # rank_zero_warn("Internal: `on_gpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._device_type == DeviceType.GPU

    @on_gpu.setter
    def on_gpu(self, val: bool) -> None:
        # rank_zero_warn("Internal: `on_gpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        # todo add logic that it cannot be set if GPU is missing
        if val:
            self._device_type = DeviceType.GPU

    @property
    def use_dp(self) -> bool:
        # rank_zero_warn("Internal: `use_dp` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._distrib_type == DistributedType.DP

    @use_dp.setter
    def use_dp(self, val: bool) -> None:
        # rank_zero_warn("Internal: `use_dp` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        if val:
            self._distrib_type = DistributedType.DP

    @property
    def use_ddp(self) -> bool:
        # rank_zero_warn("Internal: `use_ddp` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._distrib_type == DistributedType.DDP

    @use_ddp.setter
    def use_ddp(self, val: bool) -> None:
        # rank_zero_warn("Internal: `use_ddp` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        if val:
            self._distrib_type = DistributedType.DDP

    @property
    def use_ddp2(self) -> bool:
        # rank_zero_warn("Internal: `use_ddp2` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        return self._device_type and self._distrib_type == DistributedType.DDP2

    @use_ddp2.setter
    def use_ddp2(self, val: bool) -> None:
        # rank_zero_warn("Internal: `use_ddp2` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning)
        if val:
            self._distrib_type = DistributedType.DDP2

    @property
    def use_horovod(self) -> bool:
        # rank_zero_warn(
        #     "Internal: `use_horovod` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning
        # )
        return self._device_type and self._distrib_type == DistributedType.HOROVOD

    @use_horovod.setter
    def use_horovod(self, val: bool) -> None:
        # rank_zero_warn(
        #     "Internal: `use_horovod` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning
        # )
        if val:
            self._distrib_type = DistributedType.HOROVOD

    @property
    def use_single_gpu(self) -> bool:
        # rank_zero_warn(
        #     "Internal: `use_single_gpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning,
        # )
        # todo, limiting to exclude DDP2 is not clear but it comes from connectors...
        return (self._device_type and self._device_type == DeviceType.GPU
                and self.num_gpus == 1
                and self._distrib_type not in (DistributedType.DDP2, ))

    @use_single_gpu.setter
    def use_single_gpu(self, val: bool) -> None:
        # rank_zero_warn(
        #     "Internal: `use_single_gpu` is deprecated in v1.1 and will be removed in v1.2.", DeprecationWarning,
        # )
        if val:
            self._device_type = DeviceType.GPU


class _MockOptimizer(Optimizer):
    """The `_MockOptimizer` will be used inplace of an optimizer in the event that `None`
    is returned from `configure_optimizers`.
    """

    def __init__(self):
        super().__init__([torch.zeros(1)], {})

    def add_param_group(self, param_group):
        pass  # Do Nothing

    def load_state_dict(self, state_dict):
        pass  # Do Nothing

    def state_dict(self):
        return {}  # Return Empty

    def step(self, closure=None):
        if closure is not None:
            closure()

    def zero_grad(self):
        pass  # Do Nothing

    def __repr__(self):
        return 'No Optimizer'
