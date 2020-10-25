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
import os
from abc import ABC
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ProgressBarBase, ProgressBar
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class CallbackConnector:

    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(
            self,
            callbacks,
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

        # configure checkpoint callback
        # it is important that this is the last callback to run
        # pass through the required args to figure out defaults
        checkpoint_callback = self.init_default_checkpoint_callback(checkpoint_callback)
        if checkpoint_callback:
            self.trainer.callbacks.append(checkpoint_callback)

        # TODO refactor codebase (tests) to not directly reach into these callbacks
        self.trainer.checkpoint_callback = checkpoint_callback

        # init progress bar
        self.trainer._progress_bar_callback = self.configure_progress_bar(
            progress_bar_refresh_rate, process_position
        )

    def init_default_checkpoint_callback(self, checkpoint_callback):
        if checkpoint_callback is True:
            checkpoint_callback = ModelCheckpoint(dirpath=None, filename=None)
        elif checkpoint_callback is False:
            checkpoint_callback = None
        if checkpoint_callback:
            checkpoint_callback.save_function = self.trainer.save_checkpoint

        return checkpoint_callback

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

    ############################################
    #                                          #
    #  Start  CallbackConnectorLoggingMixin    #
    #                                          #
    #############################################

    # This part class helps to summarize logging logic for Pytorch LightningModule
    # using self.log functions for `pytorch_lightning.callbacks.CallBack`

    @staticmethod
    def validate_callback_logging_arguments(current_hook_fx_name: str = None, on_step: bool = None,
                                            on_epoch: bool = None) -> None:
        if current_hook_fx_name is None:
            return current_hook_fx_name
        
        current_callback_hook_auth_args = getattr(CallbackConnector, f"_{current_hook_fx_name}_log")()

        if current_callback_hook_auth_args is not None:
            m = "{} function supports only {} in {}. Provided {}"
            if on_step not in current_callback_hook_auth_args["on_step"]:
                msg = m.format(current_hook_fx_name, "on_step", current_callback_hook_auth_args["on_step"], on_step)
                raise MisconfigurationException(msg)

            if on_epoch not in current_callback_hook_auth_args["on_epoch"]:
                msg = m.format(current_hook_fx_name, "on_epoch", current_callback_hook_auth_args["on_epoch"], on_epoch)
                raise MisconfigurationException(msg)
        else:
            raise MisconfigurationException(
                f"{current_hook_fx_name} function doesn't support logging using self.log() yet."
            )

    @staticmethod
    def _setup_log():
        """Called when fit or test begins"""
        return None

    @staticmethod
    def _teardown_log():
        """Called at the end of fit and test"""
        return None

    @staticmethod
    def _on_init_start_log():
        """Called when the trainer initialization begins, model has not yet been set."""
        return None

    @staticmethod
    def _on_init_end_log():
        """Called when the trainer initialization ends, model has not yet been set."""
        return None

    @staticmethod
    def _on_fit_start_log():
        """Called when the trainer initialization begins, model has not yet been set."""
        return None

    @staticmethod
    def _on_fit_end_log():
        """Called when the trainer initialization begins, model has not yet been set."""
        return None

    @staticmethod
    def _on_sanity_check_start_log():
        """Called when the validation sanity check starts."""
        return None

    @staticmethod
    def _on_sanity_check_end_log():
        """Called when the validation sanity check ends."""
        return None

    @staticmethod
    def _on_train_epoch_start_log():
        """Called when the epoch begins."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_train_epoch_end_log():
        """Called when the epoch ends."""
        return {"on_step": [False], "on_epoch": [False, True]}

    @staticmethod
    def _on_validation_epoch_start_log():
        """Called when the epoch begins."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_validation_epoch_end_log():
        """Called when the epoch ends."""
        return {"on_step": [False], "on_epoch": [False, True]}

    @staticmethod
    def _on_test_epoch_start_log():
        """Called when the epoch begins."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_test_epoch_end_log():
        """Called when the epoch ends."""
        return {"on_step": [False], "on_epoch": [False, True]}

    @staticmethod
    def _on_epoch_start_log():
        """Called when the epoch begins."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_epoch_end_log():
        """Called when the epoch ends."""
        return {"on_step": [False], "on_epoch": [False, True]}

    @staticmethod
    def _on_train_start_log():
        """Called when the train begins."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_train_end_log():
        """Called when the train ends."""
        return None

    @staticmethod
    def _on_pretrain_routine_start_log():
        """Called when the train begins."""
        return None

    @staticmethod
    def _on_pretrain_routine_end_log():
        """Called when the train ends."""
        return None

    @staticmethod
    def _on_batch_start_log():
        """Called when the training batch begins."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_batch_end_log():
        """Called when the training batch ends."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_train_batch_start_log():
        """Called when the training batch begins."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_train_batch_end_log():
        """Called when the training batch ends."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_validation_batch_start_log():
        """Called when the validation batch begins."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_validation_batch_end_log():
        """Called when the validation batch ends."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_test_batch_start_log():
        """Called when the test batch begins."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_test_batch_end_log():
        """Called when the test batch ends."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_validation_start_log():
        """Called when the validation loop begins."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_validation_end_log():
        """Called when the validation loop ends."""
        return {"on_step": [False], "on_epoch": [False, True]}

    @staticmethod
    def _on_test_start_log():
        """Called when the test begins."""
        return {"on_step": [False, True], "on_epoch": [False, True]}

    @staticmethod
    def _on_test_end_log():
        """Called when the test ends."""
        return None

    @staticmethod
    def _on_keyboard_interrupt_log():
        """Called when the training is interrupted by KeyboardInterrupt."""
        return None

    @staticmethod
    def _on_save_checkpoint_log():
        """Called when saving a model checkpoint."""
        return None

    @staticmethod
    def _on_load_checkpoint_log():
        """Called when loading a model checkpoint."""
        return None

    ############################################
    #                                          #
    #    End  CallbackConnectorLoggingMixin    #
    #                                          #
    #############################################
