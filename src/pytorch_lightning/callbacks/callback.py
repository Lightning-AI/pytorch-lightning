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
Base class used to build new callbacks.

"""

from typing import Any, Dict, List, Optional, Type

from torch import Tensor
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class Callback:
    r"""
    Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks
    """

    @property
    def state_key(self) -> str:
        """Identifier for the state of the callback.

        Used to store and retrieve a callback's state from the checkpoint dictionary by
        ``checkpoint["callbacks"][state_key]``. Implementations of a callback need to provide a unique state key if 1)
        the callback has state and 2) it is desired to maintain the state of multiple instances of that callback.
        """
        return self.__class__.__qualname__

    @property
    def _legacy_state_key(self) -> Type["Callback"]:
        """State key for checkpoints saved prior to version 1.5.0."""
        return type(self)

    def _generate_state_key(self, **kwargs: Any) -> str:
        """Formats a set of key-value pairs into a state key string with the callback class name prefixed. Useful
        for defining a :attr:`state_key`.

        Args:
            **kwargs: A set of key-value pairs. Must be serializable to :class:`str`.
        """
        return f"{self.__class__.__qualname__}{repr(kwargs)}"

    def on_configure_sharded_model(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        r"""
        .. deprecated:: v1.6
            This callback hook was deprecated in v1.6 and will be removed in v1.8. Use `setup()` instead.

        Called before configure sharded model.
        """

    def on_before_accelerator_backend_setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        r"""
        .. deprecated:: v1.6
            This callback hook was deprecated in v1.6 and will be removed in v1.8. Use ``setup()`` instead.

        Called before accelerator is being setup.
        """

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune begins."""

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune ends."""

    def on_init_start(self, trainer: "pl.Trainer") -> None:
        r"""
        .. deprecated:: v1.6
            This callback hook was deprecated in v1.6 and will be removed in v1.8.

        Called when the trainer initialization begins, model has not yet been set.
        """

    def on_init_end(self, trainer: "pl.Trainer") -> None:
        r"""
        .. deprecated:: v1.6
            This callback hook was deprecated in v1.6 and will be removed in v1.8.

        Called when the trainer initialization ends, model has not yet been set.
        """

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit begins."""

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit ends."""

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the validation sanity check starts."""

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the validation sanity check ends."""

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch begins."""

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch ends."""

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch begins."""

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the val epoch begins."""

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the val epoch ends."""

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the test epoch begins."""

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the test epoch ends."""

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the predict epoch begins."""

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: List[Any]) -> None:
        """Called when the predict epoch ends."""

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        r"""
        .. deprecated:: v1.6
            This callback hook was deprecated in v1.6 and will be removed in v1.8. Use
            ``on_<train/validation/test>_epoch_start`` instead.

        Called when either of train/val/test epoch begins.
        """

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        r"""
        .. deprecated:: v1.6
            This callback hook was deprecated in v1.6 and will be removed in v1.8. Use
            ``on_<train/validation/test>_epoch_end`` instead.

        Called when either of train/val/test epoch ends.
        """

    def on_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        r"""
        .. deprecated:: v1.6
            This callback hook was deprecated in v1.6 and will be removed in v1.8. Use
            ``on_train_batch_start`` instead.

        Called when the training batch begins.
        """

    def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        r"""
        .. deprecated:: v1.6
            This callback hook was deprecated in v1.6 and will be removed in v1.8. Use
            ``on_train_batch_end`` instead.

        Called when the training batch ends.
        """

    def on_validation_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the validation batch begins."""

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""

    def on_test_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the test batch begins."""

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends."""

    def on_predict_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the predict batch begins."""

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends."""

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train begins."""

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train ends."""

    def on_pretrain_routine_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        r"""
        .. deprecated:: v1.6

            This callback hook was deprecated in v1.6 and will be removed in v1.8. Use ``on_fit_start`` instead.

        Called when the pretrain routine begins.
        """

    def on_pretrain_routine_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        r"""
        .. deprecated:: v1.6

            This callback hook was deprecated in v1.6 and will be removed in v1.8. Use ``on_fit_start`` instead.

        Called when the pretrain routine ends.
        """

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the validation loop begins."""

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the validation loop ends."""

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the test begins."""

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the test ends."""

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the predict begins."""

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when predict ends."""

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        """Called when any trainer execution is interrupted by an exception."""

    def state_dict(self) -> Dict[str, Any]:
        """Called when saving a checkpoint, implement to generate callback's ``state_dict``.

        Returns:
            A dictionary containing callback state.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload callback state given callback's ``state_dict``.

        Args:
            state_dict: the callback state returned by ``state_dict``.
        """
        pass

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> Optional[dict]:
        r"""
        Called when saving a checkpoint to give you a chance to store anything else you might want to save.

        Args:
            trainer: the current :class:`~pytorch_lightning.trainer.Trainer` instance.
            pl_module: the current :class:`~pytorch_lightning.core.module.LightningModule` instance.
            checkpoint: the checkpoint dictionary that will be saved.

        Returns:
            None or the callback state. Support for returning callback state will be removed in v1.8.

        .. deprecated:: v1.6
            Returning a value from this method was deprecated in v1.6 and will be removed in v1.8.
            Implement ``Callback.state_dict`` instead to return state.
            In v1.8 ``Callback.on_save_checkpoint`` can only return None.
        """

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]
    ) -> None:
        r"""
        Called when loading a model checkpoint, use to reload state.

        Args:
            trainer: the current :class:`~pytorch_lightning.trainer.Trainer` instance.
            pl_module: the current :class:`~pytorch_lightning.core.module.LightningModule` instance.
            callback_state: the callback state returned by ``on_save_checkpoint``.

        Note:
            The ``on_load_checkpoint`` won't be called with an undefined state.
            If your ``on_load_checkpoint`` hook behavior doesn't rely on a state,
            you will still need to override ``on_save_checkpoint`` to return a ``dummy state``.

        .. deprecated:: v1.6
            This callback hook will change its signature and behavior in v1.8.
            If you wish to load the state of the callback, use ``Callback.load_state_dict`` instead.
            In v1.8 ``Callback.on_load_checkpoint(checkpoint)`` will receive the entire loaded
            checkpoint dictionary instead of only the callback state from the checkpoint.
        """

    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: Tensor) -> None:
        """Called before ``loss.backward()``."""

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called after ``loss.backward()`` and before optimizers are stepped."""

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer, opt_idx: int
    ) -> None:
        """Called before ``optimizer.step()``."""

    def on_before_zero_grad(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer) -> None:
        """Called before ``optimizer.zero_grad()``."""
