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
from copy import deepcopy
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Type

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from pytorch_lightning.utilities.warnings import WarningCache

warning_cache = WarningCache()


class TrainerCallbackHookMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    # the proper values/initialisation should be done in child class
    callbacks: List[Callback] = []
    lightning_module: LightningModule

    def on_before_accelerator_backend_setup(self, model: LightningModule) -> None:
        """Called at the beginning of fit (train + validate), validate, test, or predict, or tune."""
        for callback in self.callbacks:
            callback.on_before_accelerator_backend_setup(self, model)

    def configure_sharded_model(self, model: LightningModule) -> None:
        """Called at the beginning of fit (train + validate), validate, test, or predict, or tune."""
        for callback in self.callbacks:
            callback.on_configure_sharded_model(self, model)

    def setup(self, model: LightningModule, stage: Optional[str]) -> None:
        """Called at the beginning of fit (train + validate), validate, test, or predict, or tune."""
        for callback in self.callbacks:
            callback.setup(self, model, stage)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Called at the end of fit (train + validate), validate, test, or predict, or tune."""
        for callback in self.callbacks:
            callback.teardown(self, self.lightning_module, stage)

    def on_init_start(self):
        """Called when the trainer initialization begins, model has not yet been set."""
        for callback in self.callbacks:
            callback.on_init_start(self)

    def on_init_end(self):
        """Called when the trainer initialization ends, model has not yet been set."""
        for callback in self.callbacks:
            callback.on_init_end(self)

    def on_fit_start(self):
        """Called when the trainer initialization begins, model has not yet been set."""
        for callback in self.callbacks:
            callback.on_fit_start(self, self.lightning_module)

    def on_fit_end(self):
        """Called when the trainer initialization begins, model has not yet been set."""
        for callback in self.callbacks:
            callback.on_fit_end(self, self.lightning_module)

    def on_sanity_check_start(self):
        """Called when the validation sanity check starts."""
        for callback in self.callbacks:
            callback.on_sanity_check_start(self, self.lightning_module)

    def on_sanity_check_end(self):
        """Called when the validation sanity check ends."""
        for callback in self.callbacks:
            callback.on_sanity_check_end(self, self.lightning_module)

    def on_train_epoch_start(self):
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_train_epoch_start(self, self.lightning_module)

    def on_train_epoch_end(self, outputs: EPOCH_OUTPUT):
        """Called when the epoch ends.

        Args:
            outputs: List of outputs on each ``train`` epoch
        """
        for callback in self.callbacks:
            if is_param_in_hook_signature(callback.on_train_epoch_end, "outputs"):
                warning_cache.warn(
                    "The signature of `Callback.on_train_epoch_end` has changed in v1.3."
                    " `outputs` parameter has been removed."
                    " Support for the old signature will be removed in v1.5", DeprecationWarning
                )
                callback.on_train_epoch_end(self, self.lightning_module, outputs)
            else:
                callback.on_train_epoch_end(self, self.lightning_module)

    def on_validation_epoch_start(self):
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_validation_epoch_start(self, self.lightning_module)

    def on_validation_epoch_end(self):
        """Called when the validation epoch ends."""
        for callback in self.callbacks:
            callback.on_validation_epoch_end(self, self.lightning_module)

    def on_test_epoch_start(self):
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_test_epoch_start(self, self.lightning_module)

    def on_test_epoch_end(self):
        """Called when the test epoch ends."""
        for callback in self.callbacks:
            callback.on_test_epoch_end(self, self.lightning_module)

    def on_predict_epoch_start(self) -> None:
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_predict_epoch_start(self, self.lightning_module)

    def on_predict_epoch_end(self, outputs: List[Any]) -> None:
        """Called when the epoch ends."""
        for callback in self.callbacks:
            callback.on_predict_epoch_end(self, self.lightning_module, outputs)

    def on_epoch_start(self):
        """Called when either of train/val/test epoch begins."""
        for callback in self.callbacks:
            callback.on_epoch_start(self, self.lightning_module)

    def on_epoch_end(self):
        """Called when either of train/val/test epoch ends."""
        for callback in self.callbacks:
            callback.on_epoch_end(self, self.lightning_module)

    def on_train_start(self):
        """Called when the train begins."""
        for callback in self.callbacks:
            callback.on_train_start(self, self.lightning_module)

    def on_train_end(self):
        """Called when the train ends."""
        for callback in self.callbacks:
            callback.on_train_end(self, self.lightning_module)

    def on_pretrain_routine_start(self) -> None:
        """Called when the pre-train routine begins."""
        for callback in self.callbacks:
            callback.on_pretrain_routine_start(self, self.lightning_module)

    def on_pretrain_routine_end(self) -> None:
        """Called when the pre-train routine ends."""
        for callback in self.callbacks:
            callback.on_pretrain_routine_end(self, self.lightning_module)

    def on_batch_start(self):
        """Called when the training batch begins."""
        for callback in self.callbacks:
            callback.on_batch_start(self, self.lightning_module)

    def on_batch_end(self):
        """Called when the training batch ends."""
        for callback in self.callbacks:
            callback.on_batch_end(self, self.lightning_module)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        """Called when the training batch begins."""
        for callback in self.callbacks:
            callback.on_train_batch_start(self, self.lightning_module, batch, batch_idx, dataloader_idx)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch, batch_idx, dataloader_idx):
        """Called when the training batch ends."""
        for callback in self.callbacks:
            callback.on_train_batch_end(self, self.lightning_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        """Called when the validation batch begins."""
        for callback in self.callbacks:
            callback.on_validation_batch_start(self, self.lightning_module, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, outputs: STEP_OUTPUT, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        for callback in self.callbacks:
            callback.on_validation_batch_end(self, self.lightning_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        """Called when the test batch begins."""
        for callback in self.callbacks:
            callback.on_test_batch_start(self, self.lightning_module, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        for callback in self.callbacks:
            callback.on_test_batch_end(self, self.lightning_module, outputs, batch, batch_idx, dataloader_idx)

    def on_predict_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Called when the predict batch begins."""
        for callback in self.callbacks:
            callback.on_predict_batch_start(self, self.lightning_module, batch, batch_idx, dataloader_idx)

    def on_predict_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Called when the predict batch ends."""
        for callback in self.callbacks:
            callback.on_predict_batch_end(self, self.lightning_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_start(self):
        """Called when the validation loop begins."""
        for callback in self.callbacks:
            callback.on_validation_start(self, self.lightning_module)

    def on_validation_end(self):
        """Called when the validation loop ends."""
        for callback in self.callbacks:
            callback.on_validation_end(self, self.lightning_module)

    def on_test_start(self):
        """Called when the test begins."""
        for callback in self.callbacks:
            callback.on_test_start(self, self.lightning_module)

    def on_test_end(self):
        """Called when the test ends."""
        for callback in self.callbacks:
            callback.on_test_end(self, self.lightning_module)

    def on_predict_start(self) -> None:
        """Called when predict begins."""
        for callback in self.callbacks:
            callback.on_predict_start(self, self.lightning_module)

    def on_predict_end(self) -> None:
        """Called when predict ends."""
        for callback in self.callbacks:
            callback.on_predict_end(self, self.lightning_module)

    def on_keyboard_interrupt(self):
        """Called when the training is interrupted by KeyboardInterrupt."""
        for callback in self.callbacks:
            callback.on_keyboard_interrupt(self, self.lightning_module)

    @staticmethod
    def __is_old_signature_on_save_checkpoint(fn: Callable) -> bool:
        parameters = list(signature(fn).parameters)
        return len(parameters) == 2 and parameters[1] != "args"

    @staticmethod
    def __is_old_signature_on_load_checkpoint(fn: Callable) -> bool:
        parameters = list(signature(fn).parameters)
        return len(parameters) == 1 and parameters[0] != "args"

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[Type, dict]:
        """Called when saving a model checkpoint."""
        callback_states = {}
        for callback in self.callbacks:
            if self.__is_old_signature_on_save_checkpoint(callback.on_save_checkpoint):
                rank_zero_deprecation(
                    "`Callback.on_save_checkpoint` signature has changed in v1.3."
                    " A `checkpoint` parameter has been added."
                    " Support for the old signature will be removed in v1.5"
                )
                state = callback.on_save_checkpoint(self, self.lightning_module)  # noqa: parameter-unfilled
            else:
                state = callback.on_save_checkpoint(self, self.lightning_module, checkpoint)
            if state:
                callback_states[type(callback)] = state
        return callback_states

    def on_load_checkpoint(self, checkpoint):
        """Called when loading a model checkpoint."""

        # Todo: the `callback_states` are dropped with TPUSpawn as they
        # can't be saved using `xm.save`
        # https://github.com/pytorch/xla/issues/2773
        callback_states = checkpoint.get('callbacks')

        if callback_states is None:
            return

        current_callbacks_type = {type(cb) for cb in self.callbacks}
        saved_callbacks_type = set(callback_states.keys())
        difference = saved_callbacks_type.difference(current_callbacks_type)
        if difference:
            rank_zero_warn(
                "Be aware that when using ``resume_from_checkpoint``, "
                "callbacks used to create the checkpoint need to be provided. "
                f"Please, add the following callbacks: {list(difference)}. ", UserWarning
            )

        for callback in self.callbacks:
            state = callback_states.get(type(callback))
            if state:
                state = deepcopy(state)
                if self.__is_old_signature_on_load_checkpoint(callback.on_load_checkpoint):
                    rank_zero_deprecation(
                        "`Callback.on_load_checkpoint` signature has changed in v1.3."
                        " `trainer` and `pl_module` parameters have been added."
                        " Support for the old signature will be removed in v1.5"
                    )
                    callback.on_load_checkpoint(state)  # noqa: parameter-unfilled
                else:
                    callback.on_load_checkpoint(self, self.lightning_module, state)

    def on_after_backward(self):
        """
        Called after loss.backward() and before optimizers do anything.
        """
        for callback in self.callbacks:
            callback.on_after_backward(self, self.lightning_module)

    def on_before_zero_grad(self, optimizer):
        """
        Called after optimizer.step() and before optimizer.zero_grad().
        """
        for callback in self.callbacks:
            callback.on_before_zero_grad(self, self.lightning_module, optimizer)
