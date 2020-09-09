from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.trainer.logger_connector import LoggerConnector
from pytorch_lightning.trainer.states import TrainerState
from typing import List, Optional, Union
from pytorch_lightning.utilities import argparse_utils
from argparse import ArgumentParser, Namespace
from abc import ABC
import inspect
import os
from pytorch_lightning.utilities.model_utils import is_overridden
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ProgressBarBase
from pytorch_lightning.trainer.model_connector import ModelConnector


class TrainerProperties(ABC):

    precision: int
    logger_connector: LoggerConnector
    _state: TrainerState
    global_rank: int
    fast_dev_run: bool
    use_dp: bool
    use_ddp: bool
    use_ddp2: bool
    model: LightningModule
    data_parallel_device_ids: Optional[List[int]]
    _progress_bar_callback: ProgressBarBase
    limit_val_batches: int
    _default_root_dir: str
    _weights_save_path: str
    model_connector: ModelConnector

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
    def from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs) -> 'Trainer':
        return argparse_utils.from_argparse_args(cls, args, **kwargs)

    @classmethod
    def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        return argparse_utils.parse_argparser(cls, arg_parser)

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
