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
from typing import Any, Dict, List, Optional, Type, Union

from packaging.version import Version
from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT


class TrainerCallbackHookMixin(ABC):
    r"""
    .. deprecated:: v1.6
        The `TrainerCallbackHookMixin` class was deprecated in v1.6 and will be removed in v1.8.
    """

    # this is just a summary on variables used in this abstract class,
    # the proper values/initialisation should be done in child class
    callbacks: List[Callback] = []
    lightning_module: "pl.LightningModule"

    def on_before_accelerator_backend_setup(self) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_before_accelerator_backend_setup` was deprecated in v1.6
            and will be removed in v1.8.

        Called at the beginning of fit (train + validate), validate, test, or predict, or tune.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_before_accelerator_backend_setup` was deprecated in v1.6 "
            "and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_before_accelerator_backend_setup(self, self.lightning_module)

    def on_configure_sharded_model(self) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_configure_sharded_model` was deprecated in v1.6 and will be removed in v1.8.

        Called at the beginning of fit (train + validate), validate, test, or predict, or tune.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_configure_sharded_model` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_configure_sharded_model(self, self.lightning_module)

    def setup(self, stage: Optional[str]) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.setup` was deprecated in v1.6 and will be removed in v1.8.

        Called at the beginning of fit (train + validate), validate, test, or predict, or tune.
        """
        rank_zero_deprecation("`TrainerCallbackHookMixin.setup` was deprecated in v1.6 and will be removed in v1.8.")
        for callback in self.callbacks:
            callback.setup(self, self.lightning_module, stage=stage)

    def teardown(self, stage: Optional[str] = None) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.teardown` was deprecated in v1.6 and will be removed in v1.8.

        Called at the end of fit (train + validate), validate, test, or predict, or tune.
        """
        rank_zero_deprecation("`TrainerCallbackHookMixin.teardown` was deprecated in v1.6 and will be removed in v1.8.")
        for callback in self.callbacks:
            callback.teardown(self, self.lightning_module, stage=stage)

    def on_init_start(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_init_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the trainer initialization begins, model has not yet been set.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_init_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_init_start(self)

    def on_init_end(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_init_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the trainer initialization ends, model has not yet been set.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_init_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_init_end(self)

    def on_fit_start(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_fit_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when fit begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_fit_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_fit_start(self, self.lightning_module)

    def on_fit_end(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_fit_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when fit ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_fit_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_fit_end(self, self.lightning_module)

    def on_sanity_check_start(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_sanity_check_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the validation sanity check starts.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_sanity_check_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_sanity_check_start(self, self.lightning_module)

    def on_sanity_check_end(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_sanity_check_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the validation sanity check ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_sanity_check_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_sanity_check_end(self, self.lightning_module)

    def on_train_epoch_start(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_train_epoch_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the epoch begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_train_epoch_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_train_epoch_start(self, self.lightning_module)

    def on_train_epoch_end(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_train_epoch_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the epoch ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_train_epoch_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_train_epoch_end(self, self.lightning_module)

    def on_validation_epoch_start(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_validation_epoch_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the epoch begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_validation_epoch_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_validation_epoch_start(self, self.lightning_module)

    def on_validation_epoch_end(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_validation_epoch_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the validation epoch ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_validation_epoch_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_validation_epoch_end(self, self.lightning_module)

    def on_test_epoch_start(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_test_epoch_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the epoch begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_test_epoch_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_test_epoch_start(self, self.lightning_module)

    def on_test_epoch_end(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_test_epoch_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the test epoch ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_test_epoch_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_test_epoch_end(self, self.lightning_module)

    def on_predict_epoch_start(self) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_predict_epoch_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the epoch begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_predict_epoch_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_predict_epoch_start(self, self.lightning_module)

    def on_predict_epoch_end(self, outputs: List[Any]) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_predict_epoch_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the epoch ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_predict_epoch_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_predict_epoch_end(self, self.lightning_module, outputs)

    def on_epoch_start(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_epoch_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when either of train/val/test epoch begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_epoch_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_epoch_start(self, self.lightning_module)

    def on_epoch_end(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_epoch_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when either of train/val/test epoch ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_epoch_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_epoch_end(self, self.lightning_module)

    def on_train_start(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_train_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the train begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_train_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_train_start(self, self.lightning_module)

    def on_train_end(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_train_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the train ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_train_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_train_end(self, self.lightning_module)

    def on_pretrain_routine_start(self) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_pretrain_routine_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the pre-train routine begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_pretrain_routine_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_pretrain_routine_start(self, self.lightning_module)

    def on_pretrain_routine_end(self) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_pretrain_routine_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the pre-train routine ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_pretrain_routine_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_pretrain_routine_end(self, self.lightning_module)

    def on_batch_start(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_batch_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the training batch begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_batch_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_batch_start(self, self.lightning_module)

    def on_batch_end(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_batch_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the training batch ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_batch_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_batch_end(self, self.lightning_module)

    def on_train_batch_start(self, batch, batch_idx):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_train_batch_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the training batch begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_train_batch_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_train_batch_start(self, self.lightning_module, batch, batch_idx)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch, batch_idx):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_train_batch_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the training batch ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_train_batch_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_train_batch_end(self, self.lightning_module, outputs, batch, batch_idx)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_validation_batch_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the validation batch begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_validation_batch_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_validation_batch_start(self, self.lightning_module, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, outputs: STEP_OUTPUT, batch, batch_idx, dataloader_idx):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_validation_batch_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the validation batch ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_validation_batch_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_validation_batch_end(self, self.lightning_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_test_batch_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the test batch begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_test_batch_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_test_batch_start(self, self.lightning_module, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch, batch_idx, dataloader_idx):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_test_batch_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the test batch ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_test_batch_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_test_batch_end(self, self.lightning_module, outputs, batch, batch_idx, dataloader_idx)

    def on_predict_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_predict_batch_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the predict batch begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_predict_batch_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_predict_batch_start(self, self.lightning_module, batch, batch_idx, dataloader_idx)

    def on_predict_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_predict_batch_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the predict batch ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_predict_batch_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_predict_batch_end(self, self.lightning_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_start(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_validation_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the validation loop begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_validation_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_validation_start(self, self.lightning_module)

    def on_validation_end(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_validation_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the validation loop ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_validation_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_validation_end(self, self.lightning_module)

    def on_test_start(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_test_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when the test begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_test_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_test_start(self, self.lightning_module)

    def on_test_end(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_test_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when the test ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_test_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_test_end(self, self.lightning_module)

    def on_predict_start(self) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_predict_start` was deprecated in v1.6 and will be removed in v1.8.

        Called when predict begins.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_predict_start` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_predict_start(self, self.lightning_module)

    def on_predict_end(self) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_predict_end` was deprecated in v1.6 and will be removed in v1.8.

        Called when predict ends.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_predict_end` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_predict_end(self, self.lightning_module)

    def on_exception(self, exception: BaseException) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_exception` was deprecated in v1.6 and will be removed in v1.8.

        Called when any trainer execution is interrupted by an exception.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_exception` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_exception(self, self.lightning_module, exception)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, dict]:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_save_checkpoint` was deprecated in v1.6 and will be removed in v1.8.

        Called when saving a model checkpoint.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_save_checkpoint` was deprecated in v1.6 and will be removed in v1.8."
        )
        callback_states = {}
        for callback in self.callbacks:
            state = callback.on_save_checkpoint(self, self.lightning_module, checkpoint)
            if state:
                callback_states[callback.state_key] = state
        return callback_states

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_load_checkpoint` was deprecated in v1.6 and will be removed in v1.8.

        Called when loading a model checkpoint.
        """
        # Todo: the `callback_states` are dropped with TPUSpawn as they
        # can't be saved using `xm.save`
        # https://github.com/pytorch/xla/issues/2773
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_load_checkpoint` was deprecated in v1.6 and will be removed in v1.8."
        )
        callback_states: Dict[Union[Type, str], Dict] = checkpoint.get("callbacks")

        if callback_states is None:
            return

        is_legacy_ckpt = Version(checkpoint["pytorch-lightning_version"]) < Version("1.5.0dev")
        current_callbacks_keys = {cb._legacy_state_key if is_legacy_ckpt else cb.state_key for cb in self.callbacks}
        difference = callback_states.keys() - current_callbacks_keys
        if difference:
            rank_zero_warn(
                "Be aware that when using `ckpt_path`,"
                " callbacks used to create the checkpoint need to be provided during `Trainer` instantiation."
                f" Please add the following callbacks: {list(difference)}.",
            )

        for callback in self.callbacks:
            state = callback_states.get(callback.state_key, callback_states.get(callback._legacy_state_key))
            if state:
                state = deepcopy(state)
                callback.on_load_checkpoint(self, self.lightning_module, state)

    def on_before_backward(self, loss: Tensor) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_before_backward` was deprecated in v1.6 and will be removed in v1.8.

        Called before ``loss.backward()``.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_before_backward` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_before_backward(self, self.lightning_module, loss)

    def on_after_backward(self):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_after_backward` was deprecated in v1.6 and will be removed in v1.8.

        Called after loss.backward() and before optimizers do anything.
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_after_backward` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_after_backward(self, self.lightning_module)

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_before_optimizer_step` was deprecated in v1.6 and will be removed in v1.8.

        Called after on_after_backward() once the gradient is accumulated and before optimizer.step().
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_before_optimizer_step` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_before_optimizer_step(self, self.lightning_module, optimizer, optimizer_idx)

    def on_before_zero_grad(self, optimizer):
        r"""
        .. deprecated:: v1.6
            `TrainerCallbackHookMixin.on_before_zero_grad` was deprecated in v1.6 and will be removed in v1.8.

        Called after optimizer.step() and before optimizer.zero_grad().
        """
        rank_zero_deprecation(
            "`TrainerCallbackHookMixin.on_before_zero_grad` was deprecated in v1.6 and will be removed in v1.8."
        )
        for callback in self.callbacks:
            callback.on_before_zero_grad(self, self.lightning_module, optimizer)
