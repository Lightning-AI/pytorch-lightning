import os
from abc import ABC, abstractmethod
from typing import Union

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import LightningLoggerBase


class TrainerCallbackConfigMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    #  the proper values/initialisation should be done in child class
    default_root_dir: str
    logger: Union[LightningLoggerBase, bool]
    weights_save_path: str
    ckpt_path: str
    checkpoint_callback: ModelCheckpoint

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
        if self.checkpoint_callback is True:
            # init a default one
            if self.logger is not None:
                save_dir = (getattr(self.logger, 'save_dir', None) or
                            getattr(self.logger, '_save_dir', None) or
                            self.default_root_dir)

                # weights_save_path overrides anything
                if self.weights_save_path is not None:
                    save_dir = self.weights_save_path

                ckpt_path = os.path.join(
                    save_dir,
                    self.logger.name,
                    f'version_{self.logger.version}',
                    "checkpoints"
                )
            else:
                ckpt_path = os.path.join(self.default_root_dir, "checkpoints")

            # when no val step is defined, use 'loss' otherwise 'val_loss'
            train_step_only = not self.is_overriden('validation_step')
            monitor_key = 'loss' if train_step_only else 'val_loss'

            self.ckpt_path = ckpt_path
            os.makedirs(ckpt_path, exist_ok=True)
            self.checkpoint_callback = ModelCheckpoint(
                filepath=ckpt_path,
                monitor=monitor_key
            )
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
