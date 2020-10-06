import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ProgressBarBase, ProgressBar
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class CallbackConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(
            self,
            callbacks,
            early_stop_callback,  # TODO: Deprecated in v0.10.0. Remove in v1.0
            checkpoint_callback,
            progress_bar_refresh_rate,
            process_position,
            default_root_dir,
            weights_save_path,
            resume_from_checkpoint
    ):
        self.trainer.resume_from_checkpoint = resume_from_checkpoint

        # init folder paths for checkpoint + weights save callbacks
        self.trainer._default_root_dir = default_root_dir or os.getcwd()
        self.trainer._weights_save_path = weights_save_path or self.trainer._default_root_dir

        # init callbacks
        self.trainer.callbacks = callbacks or []

        # configure early stop callback
        # creates a default one if none passed in
        # TODO: Deprecated in v0.10.0. Remove early stopping default configuration in v1.0
        early_stop_callback = self.configure_early_stopping(early_stop_callback)
        if early_stop_callback:
            self.trainer.callbacks.append(early_stop_callback)

        # configure checkpoint callback
        # it is important that this is the last callback to run
        # pass through the required args to figure out defaults
        checkpoint_callback = self.init_default_checkpoint_callback(checkpoint_callback)
        if checkpoint_callback:
            self.trainer.callbacks.append(checkpoint_callback)

        # TODO refactor codebase (tests) to not directly reach into these callbacks
        self.trainer.checkpoint_callback = checkpoint_callback

        # TODO: Deprecated in v0.10.0. Remove in v1.0
        self.trainer.early_stop_callback = early_stop_callback

        # init progress bar
        self.trainer._progress_bar_callback = self.configure_progress_bar(
            progress_bar_refresh_rate, process_position
        )

    def init_default_checkpoint_callback(self, checkpoint_callback):
        if checkpoint_callback is True:
            checkpoint_callback = ModelCheckpoint(filepath=None)
        elif checkpoint_callback is False:
            checkpoint_callback = None
        if checkpoint_callback:
            checkpoint_callback.save_function = self.trainer.save_checkpoint

        return checkpoint_callback

    # TODO: Deprecated in v0.10.0. Remove this method in v1.0
    def configure_early_stopping(self, early_stop_callback):
        if early_stop_callback is True or None:
            rank_zero_warn(
                "Trainer(early_stop_callback=True) is deprecated since v0.10.0 and will be"
                " removed in v1.0."
                " Configure the EarlyStopping callback class and add it to the list of callbacks:"
                " Trainer(callbacks=[EarlyStopping(...)])."
            )
            early_stop_callback = EarlyStopping(
                monitor='early_stop_on',
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
        progress_bars = [c for c in self.trainer.callbacks if isinstance(c, ProgressBarBase)]
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
            self.trainer.callbacks.append(progress_bar_callback)
        else:
            progress_bar_callback = None

        return progress_bar_callback
