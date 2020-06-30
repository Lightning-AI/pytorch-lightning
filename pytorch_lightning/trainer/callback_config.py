import os
from abc import ABC, abstractmethod
from typing import List, Callable, Optional


from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, ProgressBarBase, ProgressBar
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class TrainerCallbackConfigMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    callbacks: List[Callback]
    default_root_dir: str
    logger: LightningLoggerBase
    weights_save_path: Optional[str]
    ckpt_path: str
    checkpoint_callback: Optional[ModelCheckpoint]

    @property
    @abstractmethod
    def slurm_job_id(self) -> int:
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def save_checkpoint(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    @abstractmethod
    def is_overridden(self, *args):
        """Warning: this is just empty shell for code implemented in other class."""

    def configure_checkpoint_callback(self, checkpoint_callback):
        """
        Weight path set in this priority:
        Checkpoint_callback's path (if passed in).
        User provided weights_saved_path
        Otherwise use os.getcwd()
        """
        if checkpoint_callback is True:
            # when no val step is defined, use 'loss' otherwise 'val_loss'
            train_step_only = not self.is_overridden('validation_step')
            monitor_key = 'loss' if train_step_only else 'val_loss'
            checkpoint_callback = ModelCheckpoint(
                filepath=None,
                monitor=monitor_key
            )
        elif checkpoint_callback is False:
            checkpoint_callback = None

        if checkpoint_callback:
            checkpoint_callback.save_function = self.save_checkpoint

        # if weights_save_path is still none here, set to current working dir
        if self.weights_save_path is None:
            self.weights_save_path = self.default_root_dir

        return checkpoint_callback

    def configure_early_stopping(self, early_stop_callback):
        if early_stop_callback is True or None:
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=3,
                strict=True,
                verbose=True,
                mode='min'
            )
        elif not early_stop_callback:
            early_stop_callback = None
        else:
            early_stop_callback = early_stop_callback
        return early_stop_callback

    def configure_progress_bar(self, refresh_rate=1, process_position=0):
        progress_bars = [c for c in self.callbacks if isinstance(c, ProgressBarBase)]
        if len(progress_bars) > 1:
            raise MisconfigurationException(
                'You added multiple progress bar callbacks to the Trainer, but currently only one'
                ' progress bar is supported.'
            )
        elif len(progress_bars) == 1:
            progress_bar_callback = progress_bars[0]
        elif refresh_rate > 0:
            progress_bar_callback = ProgressBar(
                refresh_rate=refresh_rate,
                process_position=process_position,
            )
            self.callbacks.append(progress_bar_callback)
        else:
            progress_bar_callback = None

        return progress_bar_callback
