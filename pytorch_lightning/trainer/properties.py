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
from typing import Any, cast, List, Optional, Type, TypeVar, Union

import torch

from pytorch_lightning.accelerators.accelerator_connector import BackendConnector
from pytorch_lightning.accelerators.legacy.accelerator import Accelerator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ProgressBarBase
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities import _HOROVOD_AVAILABLE, _TPU_AVAILABLE, DeviceType, DistributedType, rank_zero_warn
from pytorch_lightning.utilities.argparse import (
    add_argparse_args,
    from_argparse_args,
    parse_argparser,
    parse_env_variables,
)
from pytorch_lightning.utilities.cloud_io import get_filesystem

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities.model_helpers import is_overridden


class TrainerProperties(ABC):

    precision: int
    logger_connector: LoggerConnector
    _state: TrainerState
    global_rank: int
    fast_dev_run: Union[int, bool]
    _device_type: DeviceType
    _distrib_type: DistributedType
    model: LightningModule
    data_parallel_device_ids: Optional[List[int]]
    _progress_bar_callback: ProgressBarBase
    limit_val_batches: int
    _default_root_dir: str
    _weights_save_path: str
    accelerator_backend: Accelerator
    num_nodes: int
    num_processes: int
    accelerator_connector: BackendConnector
    _lightning_optimizers = None

    @property
    def accelerator(self):
        return self.accelerator_connector.accelerator

    @property
    def accelerator_backend(self):
        # for backward compatibility
        return self.accelerator

    @property
    def distributed_backend(self):
        # for backward compatibility
        return self.accelerator_connector.distributed_backend

    @property
    def training_type_plugin(self):
        return self.accelerator.training_type_plugin

    @property
    def precision_plugin(self):
        return self.accelerator.precision_plugin

    @property
    def global_rank(self):
        return self.accelerator.training_type_plugin.global_rank

    @property
    def local_rank(self):
        # some training types define a local rank
        return getattr(self.accelerator.training_type_plugin, "local_rank", 0)

    @property
    def node_rank(self):
        # some training types define a local rank
        return getattr(self.accelerator.training_type_plugin, "node_rank", 0)

    @property
    def world_size(self):
        # some training types define a world size
        return getattr(self.accelerator.training_type_plugin, "world_size", 1)

    @property
    def _distrib_type(self):
        return self.accelerator_connector._distrib_type

    @property
    def _device_type(self):
        return self.accelerator_connector._device_type

    @property
    def num_nodes(self):
        return self.accelerator_connector.num_nodes

    @property
    def num_processes(self):
        return self.accelerator_connector.num_processes

    @property
    def root_gpu(self):
        return self.accelerator_connector.root_gpu

    @property
    def tpu_cores(self) -> int:
        return self.accelerator_connector.tpu_cores

    @property
    def num_gpus(self) -> int:
        return self.accelerator_connector.num_gpus

    @property
    def data_parallel_device_ids(self):
        return self.accelerator_connector.parallel_device_ids

    @property
    def log_dir(self):
        if self.logger is None:
            dirpath = self.default_root_dir
        else:
            dirpath = getattr(self.logger, 'log_dir' if isinstance(self.logger, TensorBoardLogger) else 'save_dir')

        dirpath = self.training_type_plugin.broadcast(dirpath)
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
    def from_argparse_args(cls: Type['_T'], args: Union[Namespace, ArgumentParser], **kwargs) -> '_T':
        return from_argparse_args(cls, args, **kwargs)

    @classmethod
    def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        return parse_argparser(cls, arg_parser)

    @classmethod
    def match_env_arguments(cls) -> Namespace:
        return parse_env_variables(cls)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        return add_argparse_args(cls, parent_parser)

    @property
    def gpus(self) -> Optional[Union[List[int], str, int]]:
        return self.accelerator_connector.gpus

    @property
    def data_parallel(self) -> bool:
        return self._distrib_type in (
            DistributedType.DP, DistributedType.DDP, DistributedType.DDP_SPAWN, DistributedType.DDP2
        )

    @property
    def progress_bar_callback(self):
        return self._progress_bar_callback

    @property
    def progress_bar_dict(self) -> dict:
        """ Read-only for progress bar metrics. """
        ref_model = self.get_model()
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
        model_ref = self.get_model()
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

    def save_checkpoint(self, filepath, weights_only: bool = False):
        self.checkpoint_connector.save_checkpoint(filepath, weights_only)

    @property
    def model(self) -> Any:
        """
        The LightningModule, but possibly wrapped into DataParallel or DistributedDataParallel.
        To access the pure LightningModule, use
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.lightning_module` instead.
        """
        return self.accelerator.model

    @model.setter
    def model(self, model: torch.nn.Module):
        """
        Setter for the model, pass-through to accelerator and plugin where the model reference is stored.
        Used by the Tuner to reset the state of Trainer and Accelerator.

        Args:
            model: The LightningModule, possibly wrapped into DataParallel or DistributedDataParallel, depending
                on the backend.
        """
        self.accelerator.model = model

    def get_model(self):
        # TODO: rename this to lightning_module (see training type plugin)
        # backward compatible
        return self.lightning_module

    @property
    def lightning_optimizers(self):
        if self._lightning_optimizers is None:
            self.convert_to_lightning_optimizers()
        return self._lightning_optimizers

    @property
    def lightning_module(self):
        return self.training_type_plugin.lightning_module

    @property
    def optimizers(self):
        return self.accelerator.optimizers

    @optimizers.setter
    def optimizers(self, new_optims):
        self.accelerator.optimizers = new_optims

    @property
    def lr_schedulers(self):
        return self.accelerator.lr_schedulers

    @lr_schedulers.setter
    def lr_schedulers(self, new_schedulers):
        self.accelerator.lr_schedulers = new_schedulers

    @property
    def optimizer_frequencies(self):
        return self.accelerator.optimizer_frequencies

    @optimizer_frequencies.setter
    def optimizer_frequencies(self, new_freqs):
        self.accelerator.optimizer_frequencies = new_freqs

    @property
    def amp_backend(self):
        return self.accelerator.amp_backend

    @property
    def precision(self):
        return self.accelerator.precision

    @property
    def scaler(self):
        return self.accelerator.scaler

    # TODO: refactor this so that it can be done in LightningOptimizer
    def __getstate__(self):
        # remove lightning_optimizers
        self._lightning_optimizers = None
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    @property
    def require_distributed_sampler(self):
        if self.accelerator_backend is not None:
            return self.accelerator_backend.require_distributed_sampler
        return self._distrib_type in (
            DistributedType.HOROVOD, DistributedType.DDP, DistributedType.DDP_SPAWN, DistributedType.DDP2
        ) or self._device_type == DeviceType.TPU

    @property
    def distributed_sampler_kwargs(self):
        if self.accelerator_backend is not None:
            return self.training_type_plugin.distributed_sampler_kwargs

        # TODO: make sure the cases below are handled by the training_type_plugin
        if self._device_type == DeviceType.TPU:
            kwargs = dict(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())

        elif self._distrib_type == DistributedType.HOROVOD:
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


# Used to represent the concrete type TrainerProperties class methods are called on.
_T = TypeVar('_T', bound=TrainerProperties)
