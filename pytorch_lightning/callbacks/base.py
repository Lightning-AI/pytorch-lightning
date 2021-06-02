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
r"""
Abstract base class used to build new callbacks.

"""

import abc
from typing import Any, Dict, List, Optional

from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class Callback(abc.ABC):
    r"""
    Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks
    """

    def on_configure_sharded_model(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called before configure sharded model"""

    def on_before_accelerator_backend_setup(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called before accelerator is being setup"""
        pass

    def setup(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', stage: Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune begins"""
        pass

    def teardown(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', stage: Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune ends"""
        pass

    def on_init_start(self, trainer: 'pl.Trainer') -> None:
        """Called when the trainer initialization begins, model has not yet been set."""
        pass

    def on_init_end(self, trainer: 'pl.Trainer') -> None:
        """Called when the trainer initialization ends, model has not yet been set."""
        pass

    def on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when fit begins"""
        pass

    def on_fit_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when fit ends"""
        pass

    def on_sanity_check_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the validation sanity check starts."""
        pass

    def on_sanity_check_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the validation sanity check ends."""
        pass

    def on_train_batch_start(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch begins."""
        pass

    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        pass

    def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the train epoch begins."""
        pass

    def on_train_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', unused: Optional = None
    ) -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """
        pass

    def on_validation_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the val epoch begins."""
        pass

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the val epoch ends."""
        pass

    def on_test_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the test epoch begins."""
        pass

    def on_test_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the test epoch ends."""
        pass

    def on_predict_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the predict epoch begins."""
        pass

    def on_predict_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs: List[Any]) -> None:
        """Called when the predict epoch ends."""
        pass

    def on_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when either of train/val/test epoch begins."""
        pass

    def on_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when either of train/val/test epoch ends."""
        pass

    def on_batch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the training batch begins."""
        pass

    def on_validation_batch_start(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch begins."""
        pass

    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""
        pass

    def on_test_batch_start(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch begins."""
        pass

    def on_test_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends."""
        pass

    def on_predict_batch_start(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the predict batch begins."""
        pass

    def on_predict_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends."""
        pass

    def on_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the training batch ends."""
        pass

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the train begins."""
        pass

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the train ends."""
        pass

    def on_pretrain_routine_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the pretrain routine begins."""
        pass

    def on_pretrain_routine_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the pretrain routine ends."""
        pass

    def on_validation_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the validation loop ends."""
        pass

    def on_test_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the test begins."""
        pass

    def on_test_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the test ends."""
        pass

    def on_predict_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the predict begins."""
        pass

    def on_predict_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when predict ends."""
        pass

    def on_keyboard_interrupt(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called when the training is interrupted by ``KeyboardInterrupt``."""
        pass

    def on_save_checkpoint(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', checkpoint: Dict[str, Any]
    ) -> dict:
        """
        Called when saving a model checkpoint, use to persist state.

        Args:
            trainer: the current :class:`~pytorch_lightning.trainer.Trainer` instance.
            pl_module: the current :class:`~pytorch_lightning.core.lightning.LightningModule` instance.
            checkpoint: the checkpoint dictionary that will be saved.

        Returns:
            The callback state.
        """
        pass

    def on_load_checkpoint(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', callback_state: Dict[str, Any]
    ) -> None:
        """Called when loading a model checkpoint, use to reload state.

        Args:
            trainer: the current :class:`~pytorch_lightning.trainer.Trainer` instance.
            pl_module: the current :class:`~pytorch_lightning.core.lightning.LightningModule` instance.
            callback_state: the callback state returned by ``on_save_checkpoint``.

        Note:
            The ``on_load_checkpoint`` won't be called with an undefined state.
            If your ``on_load_checkpoint`` hook behavior doesn't rely on a state,
            you will still need to override ``on_save_checkpoint`` to return a ``dummy state``.
        """
        pass

    def on_after_backward(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        """Called after ``loss.backward()`` and before optimizers do anything."""
        pass

    def on_before_zero_grad(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', optimizer: Optimizer) -> None:
        """Called after ``optimizer.step()`` and before ``optimizer.zero_grad()``."""
        pass
