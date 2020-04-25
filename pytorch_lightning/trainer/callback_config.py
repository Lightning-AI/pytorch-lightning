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

    def configure_checkpoint_callback(self,
                                      checkpoint_callback,
                                      default_root_dir,
                                      logger,
                                      weights_save_path):
        """
        Weight path set in this priority:
        Checkpoint_callback's path (if passed in).
        User provided weights_saved_path
        Otherwise use os.getcwd()
        """
        ckpt_path = default_root_dir
        if checkpoint_callback:
            # init a default one
            if logger is not None:
                save_dir = (getattr(logger, 'save_dir', None) or
                            getattr(logger, '_save_dir', None) or
                            default_root_dir)

                # weights_save_path overrides anything
                if weights_save_path is not None:
                    save_dir = weights_save_path

                version = self.logger.version if isinstance(
                    self.logger.version, str) else f'version_{self.logger.version}'
                ckpt_path = os.path.join(
                    save_dir,
                    self.logger.name,
                    version,
                    "checkpoints"
                )
            else:
                ckpt_path = os.path.join(default_root_dir, "checkpoints")

            # when no val step is defined, use 'loss' otherwise 'val_loss'
            train_step_only = not self.is_overridden('validation_step')
            monitor_key = 'loss' if train_step_only else 'val_loss'

            if checkpoint_callback is True:
                os.makedirs(ckpt_path, exist_ok=True)
                checkpoint_callback = ModelCheckpoint(
                    filepath=ckpt_path,
                    monitor=monitor_key
                )
            # If user specified None in filepath, override with runtime default
            elif isinstance(checkpoint_callback, ModelCheckpoint) \
                    and checkpoint_callback.dirpath is None:
                checkpoint_callback.dirpath = ckpt_path
                checkpoint_callback.filename = '{epoch}'
                os.makedirs(checkpoint_callback.dirpath, exist_ok=True)
        elif checkpoint_callback is False:
            checkpoint_callback = None

        self.ckpt_path = ckpt_path

        if checkpoint_callback:
            # set the path for the callbacks
            checkpoint_callback.save_function = self.save_checkpoint

            # if checkpoint callback used, then override the weights path
            self.weights_save_path = self.checkpoint_callback.dirpath

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
            # TODO remove this attribute
            self.enable_early_stop = True
        elif not early_stop_callback:
            early_stop_callback = None
            self.enable_early_stop = False
        else:
            early_stop_callback = early_stop_callback
            self.enable_early_stop = True
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
