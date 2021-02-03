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

from pytorch_lightning.utilities.exceptions import MisconfigurationException


class CallbackHookNameValidator:

    @staticmethod
    def check_logging_in_callbacks(
        current_hook_fx_name: str = None, on_step: bool = None, on_epoch: bool = None
    ) -> None:
        if current_hook_fx_name is None:
            return

        internal_func = getattr(CallbackHookNameValidator, f"_{current_hook_fx_name}_log", None)

        if internal_func is None:
            return

        current_callback_hook_auth_args = internal_func()

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
    def _on_before_accelerator_backend_setup_log():
        """Called before accelerator is being setup"""
        return None

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
        return None

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
