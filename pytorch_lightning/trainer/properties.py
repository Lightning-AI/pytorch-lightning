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
from abc import ABC
from argparse import ArgumentParser, Namespace
import inspect
import os
from typing import cast, List, Optional, Type, TypeVar, Union

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint, ProgressBarBase
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.optimizer import is_lightning_optimizer
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.connectors.model_connector import ModelConnector
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities import argparse_utils, HOROVOD_AVAILABLE, rank_zero_warn, TPU_AVAILABLE
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.model_utils import is_overridden

if TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm

if HOROVOD_AVAILABLE:
    import horovod.torch as hvd


class TrainerProperties(ABC):

    precision: int
    logger_connector: LoggerConnector
    _state: TrainerState
    global_rank: int
    fast_dev_run: Union[int, bool]
    use_dp: bool
    use_ddp: bool
    use_ddp2: bool
    model: LightningModule
    data_parallel_device_ids: Optional[List[int]]
    _progress_bar_callback: ProgressBarBase
    limit_val_batches: int
    _default_root_dir: str
    _weights_save_path: str
    accelerator_backend: Accelerator
    logger: LightningLoggerBase
    model_connector: ModelConnector
    checkpoint_connector: CheckpointConnector
    callbacks: List[Callback]
    _lightning_optimizers = None

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
    def from_argparse_args(cls: Type['_T'], args: Union[Namespace, ArgumentParser], **kwargs) -> '_T':
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

    def get_model(self):
        return self.model_connector.get_model()

    @property
    def lightning_optimizers(self):
        if self._lightning_optimizers is None:
            self.convert_to_lightning_optimizers()
        return self._lightning_optimizers

    def __getstate__(self):
        # remove lightning_optimizers
        self._lightning_optimizers = None
        return self.__dict__

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


# Used to represent the concrete type TrainerProperties class methods are called on.
_T = TypeVar('_T', bound=TrainerProperties)
