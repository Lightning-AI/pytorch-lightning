import os
from abc import ABC, abstractmethod
from typing import Union, List


from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, ProgressBarBase, ProgressBar
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class TrainerCallbackConfigMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    callbacks: List[Callback]
    default_root_dir: str
    logger: Union[LightningLoggerBase, bool]
    weights_save_path: str
    ckpt_path: str
    checkpoint_callback: ModelCheckpoint
    progress_bar_refresh_rate: int
    process_position: int

    @property
    @abstractmethod
    def slurm_job_id(self) -> int:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def save_checkpoint(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def configure_checkpoint_callback(self):
        """
        Weight path set in this priority:
        Checkpoint_callback's path (if passed in).
        User provided weights_saved_path
        Otherwise use os.getcwd()
        """
        ckpt_path = self.default_root_dir
        if self.checkpoint_callback:
            # init a default one
            if self.logger is not None:
                save_dir = (getattr(self.logger, 'save_dir', None) or
                            getattr(self.logger, '_save_dir', None) or
                            self.default_root_dir)

                # weights_save_path overrides anything
                if self.weights_save_path is not None:
                    save_dir = self.weights_save_path

                version = self.logger.version if isinstance(
                    self.logger.version, str) else f'version_{self.logger.version}'
                ckpt_path = os.path.join(
                    save_dir,
                    self.logger.name,
                    version,
                    "checkpoints"
                )
            else:
                ckpt_path = os.path.join(self.default_root_dir, "checkpoints")

            # when no val step is defined, use 'loss' otherwise 'val_loss'
            train_step_only = not self.is_overridden('validation_step')
            monitor_key = 'loss' if train_step_only else 'val_loss'

            if self.checkpoint_callback is True:
                os.makedirs(ckpt_path, exist_ok=True)
                self.checkpoint_callback = ModelCheckpoint(
                    filepath=ckpt_path,
                    monitor=monitor_key
                )
            # If user specified None in filepath, override with runtime default
            elif isinstance(self.checkpoint_callback, ModelCheckpoint) \
                    and self.checkpoint_callback.dirpath is None:
                self.checkpoint_callback.dirpath = ckpt_path
                self.checkpoint_callback.filename = '{epoch}'
                os.makedirs(self.checkpoint_callback.dirpath, exist_ok=True)
        elif self.checkpoint_callback is False:
            self.checkpoint_callback = None

        self.ckpt_path = ckpt_path

        if self.checkpoint_callback:
            # set the path for the callbacks
            self.checkpoint_callback.save_function = self.save_checkpoint

            # if checkpoint callback used, then override the weights path
            self.weights_save_path = self.checkpoint_callback.dirpath

        # if weights_save_path is still none here, set to current working dir
        if self.weights_save_path is None:
            self.weights_save_path = self.default_root_dir

    def configure_early_stopping(self, early_stop_callback):
        if early_stop_callback is True or None:
            self.early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=3,
                strict=True,
                verbose=True,
                mode='min'
            )
            self.enable_early_stop = True
        elif not early_stop_callback:
            self.early_stop_callback = None
            self.enable_early_stop = False
        else:
            self.early_stop_callback = early_stop_callback
            self.enable_early_stop = True

    def configure_progress_bar(self):
        progress_bars = [c for c in self.callbacks if isinstance(c, ProgressBarBase)]
        if len(progress_bars) > 1:
            raise MisconfigurationException(
                'You added multiple progress bar callbacks to the Trainer, but currently only one'
                ' progress bar is supported.'
            )
        elif len(progress_bars) == 1:
            self.progress_bar_callback = progress_bars[0]
        elif self.progress_bar_refresh_rate > 0:
            self.progress_bar_callback = ProgressBar(
                refresh_rate=self.progress_bar_refresh_rate,
                process_position=self.process_position,
            )
            self.callbacks.append(self.progress_bar_callback)
        else:
            self.progress_bar_callback = None
