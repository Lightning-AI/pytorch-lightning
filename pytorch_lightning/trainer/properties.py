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
from abc import ABC
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import cast, List, Optional, Type, TypeVar, Union

import torch
from torch.optim import Optimizer

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ProgressBarBase
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import ParallelPlugin, PrecisionPlugin, TrainingTypePlugin
from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.states import RunningStage, TrainerState, TrainerStatus
from pytorch_lightning.trainer.training_loop import TrainLoop
from pytorch_lightning.utilities import DeviceType, DistributedType, rank_zero_warn
from pytorch_lightning.utilities.argparse import (
    add_argparse_args,
    from_argparse_args,
    parse_argparser,
    parse_env_variables,
)
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.model_helpers import is_overridden


class TrainerProperties(ABC):

    _default_root_dir: str
    _lightning_optimizers = None
    _progress_bar_callback: ProgressBarBase
    _weights_save_path: str

    accelerator_connector: AcceleratorConnector
    callbacks: List[Callback]
    checkpoint_connector: CheckpointConnector
    limit_val_batches: int
    logger: LightningLoggerBase
    logger_connector: LoggerConnector
    state: TrainerState
    train_loop: TrainLoop
    """
    Accelerator properties
    """

    @property
    def accelerator(self) -> Accelerator:
        return self.accelerator_connector.accelerator

    @property
    def distributed_backend(self) -> Optional[str]:
        # for backward compatibility
        return self.accelerator_connector.distributed_backend

    @property
    def training_type_plugin(self) -> TrainingTypePlugin:
        return self.accelerator.training_type_plugin

    @property
    def precision_plugin(self) -> PrecisionPlugin:
        return self.accelerator.precision_plugin

    @property
    def global_rank(self) -> int:
        return self.accelerator.training_type_plugin.global_rank

    @property
    def local_rank(self) -> int:
        # some training types define a local rank
        return getattr(self.accelerator.training_type_plugin, "local_rank", 0)

    @property
    def node_rank(self) -> int:
        # some training types define a local rank
        return getattr(self.accelerator.training_type_plugin, "node_rank", 0)

    @property
    def world_size(self) -> int:
        # some training types define a world size
        return getattr(self.accelerator.training_type_plugin, "world_size", 1)

    @property
    def should_rank_save_checkpoint(self) -> bool:
        return self.accelerator.training_type_plugin.should_rank_save_checkpoint

    @property
    def _distrib_type(self) -> DistributedType:
        return self.accelerator_connector._distrib_type

    @property
    def _device_type(self) -> DeviceType:
        return self.accelerator_connector._device_type

    @property
    def num_nodes(self) -> int:
        return self.accelerator_connector.num_nodes

    @property
    def num_processes(self) -> int:
        return self.accelerator_connector.num_processes

    @property
    def root_gpu(self) -> Optional[int]:
        return self.accelerator_connector.root_gpu

    @property
    def tpu_cores(self) -> int:
        return self.accelerator_connector.tpu_cores

    @property
    def num_gpus(self) -> int:
        return self.accelerator_connector.num_gpus

    @property
    def data_parallel_device_ids(self) -> Optional[List[int]]:
        return self.accelerator_connector.parallel_device_ids

    @property
    def lightning_module(self) -> LightningModule:
        return self.accelerator.lightning_module

    @property
    def optimizers(self) -> Optional[List[Optimizer]]:
        return self.accelerator.optimizers

    @optimizers.setter
    def optimizers(self, new_optims: Optional[List[Optimizer]]) -> None:
        # Necessary to rewrap optimizers to lightning
        # They will be re-created when accessing
        # the `lightning_optimizers` trainer property
        self._lightning_optimizers = None

        self.accelerator.optimizers = new_optims

    @property
    def lr_schedulers(self) -> Optional[list]:
        return self.accelerator.lr_schedulers

    @lr_schedulers.setter
    def lr_schedulers(self, new_schedulers: Optional[list]) -> None:
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
        return self.accelerator_connector.gpus

    @property
    def model(self) -> torch.nn.Module:
        """
        The LightningModule, but possibly wrapped into DataParallel or DistributedDataParallel.
        To access the pure LightningModule, use
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.lightning_module` instead.
        """
        return self.accelerator.model

    @model.setter
    def model(self, model: torch.nn.Module) -> None:
        """
        Setter for the model, pass-through to accelerator and plugin where the model reference is stored.
        Used by the Tuner to reset the state of Trainer and Accelerator.

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
        else:
            dirpath = getattr(self.logger, 'log_dir' if isinstance(self.logger, TensorBoardLogger) else 'save_dir')

        dirpath = self.accelerator.broadcast(dirpath)
        return dirpath

    @property
    def use_amp(self) -> bool:
        return self.precision == 16

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def slurm_job_id(self) -> Optional[int]:
        job_id = os.environ.get('SLURM_JOB_ID')
        if job_id:
            try:
                job_id = int(job_id)
            except ValueError:
                job_id = None

        # in interactive mode, don't make logs use the same job id
        in_slurm_interactive_mode = os.environ.get('SLURM_JOB_NAME') == 'bash'
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
            DistributedType.DP, DistributedType.DDP, DistributedType.DDP_SPAWN, DistributedType.DDP2
        )

    @property
    def progress_bar_callback(self) -> Optional[ProgressBarBase]:
        return self._progress_bar_callback

    @property
    def progress_bar_dict(self) -> dict:
        """ Read-only for progress bar metrics. """
        ref_model = self.lightning_module
        ref_model = cast(LightningModule, ref_model)

        standard_metrics = ref_model.get_progress_bar_dict()
        logged_metrics = self.progress_bar_metrics
        duplicates = list(standard_metrics.keys() & logged_metrics.keys())
        if duplicates:
            rank_zero_warn(
                f"The progress bar already tracks a metric with the name(s) '{', '.join(duplicates)}' and"
                f" `self.log('{duplicates[0]}', ..., prog_bar=True)` will overwrite this value. "
                f" If this is undesired, change the name or override `get_progress_bar_dict()`"
                f" in `LightingModule`.", UserWarning
            )
        all_metrics = dict(**standard_metrics)
        all_metrics.update(**logged_metrics)
        return all_metrics

    @property
    def disable_validation(self) -> bool:
        """ Check if validation is disabled during training. """
        return not self.enable_validation

    @property
    def enable_validation(self) -> bool:
        """ Check if we should run validation during training. """
        model_ref = self.lightning_module
        val_loop_enabled = is_overridden('validation_step', model_ref) and self.limit_val_batches > 0
        return val_loop_enabled

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
    def early_stopping_callback(self) -> Optional[EarlyStopping]:
        """
        The first :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
        callback in the Trainer.callbacks list, or ``None`` if it doesn't exist.
        """
        callbacks = self.early_stopping_callbacks
        return callbacks[0] if len(callbacks) > 0 else None

    @property
    def early_stopping_callbacks(self) -> List[EarlyStopping]:
        """
        A list of all instances of :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
        found in the Trainer.callbacks list.
        """
        return [c for c in self.callbacks if isinstance(c, EarlyStopping)]

    @property
    def prediction_writer_callbacks(self) -> List[BasePredictionWriter]:
        """
        A list of all instances of :class:`~pytorch_lightning.callbacks.prediction_writer.BasePredictionWriter`
        found in the Trainer.callbacks list.
        """
        return [cb for cb in self.callbacks if isinstance(cb, BasePredictionWriter)]

    @property
    def checkpoint_callback(self) -> Optional[ModelCheckpoint]:
        """
        The first :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`
        callback in the Trainer.callbacks list, or ``None`` if it doesn't exist.
        """
        callbacks = self.checkpoint_callbacks
        return callbacks[0] if len(callbacks) > 0 else None

    @property
    def checkpoint_callbacks(self) -> List[ModelCheckpoint]:
        """
        A list of all instances of :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`
        found in the Trainer.callbacks list.
        """
        return [c for c in self.callbacks if isinstance(c, ModelCheckpoint)]

    @property
    def resume_from_checkpoint(self) -> Optional[Union[str, Path]]:
        return self.checkpoint_connector.resume_checkpoint_path

    def save_checkpoint(self, filepath, weights_only: bool = False) -> None:
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
            if name.startswith('DEPRECATED') and isinstance(val, (tuple, list)):
                depr_arg_names.extend(val)
        return depr_arg_names

    @classmethod
    def from_argparse_args(cls: Type['_T'], args: Union[Namespace, ArgumentParser], **kwargs) -> '_T':
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
        return self.train_loop.global_step

    @property
    def current_epoch(self) -> int:
        return self.train_loop.current_epoch

    @property
    def max_epochs(self) -> Optional[int]:
        return self.train_loop.max_epochs

    @property
    def min_epochs(self) -> Optional[int]:
        return self.train_loop.min_epochs

    @property
    def max_steps(self) -> Optional[int]:
        return self.train_loop.max_steps

    @property
    def min_steps(self) -> Optional[int]:
        return self.train_loop.min_steps

    """
    Logging properties
    """

    @property
    def callback_metrics(self) -> dict:
        return self.logger_connector.callback_metrics

    @callback_metrics.setter
    def callback_metrics(self, x: dict) -> None:
        self.logger_connector.callback_metrics = x

    @property
    def logged_metrics(self) -> dict:
        return self.logger_connector.logged_metrics

    @logged_metrics.setter
    def logged_metrics(self, x: dict) -> None:
        self.logger_connector.logged_metrics = x

    @property
    def progress_bar_metrics(self) -> dict:
        return self.logger_connector.progress_bar_metrics

    @progress_bar_metrics.setter
    def progress_bar_metrics(self, x: dict) -> None:
        self.logger_connector.progress_bar_metrics = x

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


# Used to represent the concrete type TrainerProperties class methods are called on.
_T = TypeVar('_T', bound=TrainerProperties)
