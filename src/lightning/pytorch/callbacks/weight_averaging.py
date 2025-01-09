# Copyright The Lightning AI team.
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
Weight Averaging Callback
^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import itertools
from copy import deepcopy
from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor
from torch.optim.swa_utils import AveragedModel

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities.types import STEP_OUTPUT


def _return_true(x: int) -> bool:
    return True


def _return_false(x: int) -> bool:
    return False


class WeightAveraging(Callback):
    r"""A callback that updates an averaged model for Stochastic Weight Averaging (SWA) or Exponential Moving Average
    (EMA) after each training step.

    The user should provide either `update_on_step` or `update_on_epoch`, a function that determines when the average
    model should be updated. If neither function is provided, the average model will be updated after every optimizer
    step.

    During validation and after the training finishes, the current model parameters will be replaced with the averaged
    values.

    Args:
        device: If provided, the :class:`AveragedModel` will be stored on the ``device``. If ``None`` the device will be
            inferred from the original model.
        avg_fn: The averaging function used to update the parameters. The function must take in an
            :class:`AveragedModel` parameter, a current model parameter, and the number of models already averaged. If
            ``None``, an equally weighted average will be used.
        update_on_step: A function that takes the number of optimizer steps taken, and returns ``True`` if the average
            model should be updated.
        update_on_epoch: A function that takes the zero-based epoch number, and returns ``True`` if the average model
            should be updated.

    """

    def __init__(
        self,
        device: Optional[Union[torch.device, int]] = torch.device("cpu"),
        avg_fn: Optional[Callable[[Tensor, Tensor, Union[Tensor, int]], Tensor]] = None,
        update_on_step: Optional[Callable[[int], bool]] = None,
        update_on_epoch: Optional[Callable[[int], bool]] = None,
    ):
        self._device = device
        self._avg_fn = avg_fn

        if (update_on_step is None) and (update_on_epoch is None):
            self._update_on_step: Callable[[int], bool] = _return_true
            self._update_on_epoch: Callable[[int], bool] = _return_false
        else:
            self._update_on_step = _return_false if update_on_step is None else update_on_step
            self._update_on_epoch = _return_false if update_on_epoch is None else update_on_epoch

        self._average_model: Optional[AveragedModel] = None

        # Number of optimizer steps taken, when the average model was last updated. Initializing this with zero ensures
        # that the average model will be first updated after the first optimizer step, which takes place after N batches
        # when using accumulate_grad_batches=N.
        self._latest_update_step = 0
        # The epoch after which the average model was last updated. The first epoch is 0, so initializing this to a
        # negative value means that if update_on_step(0) returns True, the first update is after the first epoch.
        self._latest_update_epoch = -1

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins.

        Creates an :class:`AveragedModel` when fit begins.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.
            stage: The :class:`~lightning.pytorch.trainer.trainer.Trainer` state.

        """
        if stage == "fit":
            device = self._device or pl_module.device
            self._average_model = AveragedModel(model=pl_module, device=device, avg_fn=self._avg_fn, use_buffers=True)

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when a training batch ends.

        Updates the :class:`AveragedModel` parameters, if requested by ``update_on_step()``.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.
            outputs: Outputs from the training batch.
            batch: The training batch.
            batch_idx: Index of the training batch.

        """
        if self._update_on_step(trainer.global_step) and (trainer.global_step > self._latest_update_step):
            assert self._average_model is not None
            self._average_model.update_parameters(pl_module)
            self._latest_update_step = trainer.global_step

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when a training epoch ends.

        Updates the :class:`AveragedModel` parameters, if requested by ``update_on_epoch()``.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        if self._update_on_epoch(trainer.current_epoch) and (trainer.current_epoch > self._latest_update_epoch):
            assert self._average_model is not None
            self._average_model.update_parameters(pl_module)
            self._latest_update_epoch = trainer.current_epoch

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when training ends.

        Transfers parameters from the :class:`AveragedModel` to the current model.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        assert self._average_model is not None
        self._copy_average_to_current(pl_module)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when a validation epoch begins.

        Transfers parameter values from the :class:`AveragedModel` to the current model.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        if self._average_model is not None:
            rank_zero_info("Loading the average model parameters for validation.")
            self._swap_models(pl_module)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when a validation epoch ends.

        Recovers the current model parameters from the :class:`AveragedModel`.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        if self._average_model is not None:
            rank_zero_info("Recovering the current model parameters after validation.")
            self._swap_models(pl_module)

    def state_dict(self) -> dict[str, Any]:
        """Called when saving a checkpoint.

        Creates a ``state_dict`` of the callback state.

        Returns:
            A dictionary containing the callback state.

        """
        return {"latest_update_step": self._latest_update_step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint.

        Reloads the callback state given a ``state_dict``.

        Args:
            state_dict: A dictionary containing the callback state.

        """
        self._latest_update_step = state_dict["latest_update_step"]

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: dict[str, Any]
    ) -> None:
        r"""Called when saving a checkpoint.

        Moves the current model state to the key ``current_model_state``, and places the average model state in
        ``state_dict`` instead. Any other state variables of the ``AveragedModel`` will be saved in
        ``averaging_state``.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.
            checkpoint: The checkpoint dictionary that will be saved.

        """
        if self._average_model is None:
            raise Exception("Trying to save a checkpoint, but no average model (outside fit). Don't know what to do.")

        rank_zero_info("The average model parameters will be saved to the state_dict in the checkpoint.")
        average_model_state = self._average_model.state_dict()
        checkpoint["current_model_state"] = checkpoint["state_dict"]
        checkpoint["state_dict"] = {
            name[7:]: value for name, value in average_model_state.items() if name.startswith("module.")
        }
        checkpoint["averaging_state"] = {
            name: value for name, value in average_model_state.items() if not name.startswith("module.")
        }

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: dict[str, Any]
    ) -> None:
        r"""Called when loading a model checkpoint.

        Loads the current model and the :class:`AveragedModel` parameters from the checkpoint.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.
            checkpoint: The full checkpoint dictionary that got loaded by the Trainer.

        """
        if self._average_model is None:
            raise Exception("Trying to load a checkpoint, but no average model (outside fit). Don't know what to do.")

        if ("current_model_state" in checkpoint) and ("averaging_state" in checkpoint):
            rank_zero_info("Found current_model_state in the checkpoint. This will be used to initialize the model.")
            average_model_state = {"module." + name: value for name, value in checkpoint["state_dict"].items()}
            average_model_state |= checkpoint["averaging_state"]
            self._average_model.load_state_dict(average_model_state)
            checkpoint["state_dict"] = checkpoint["current_model_state"]
        else:
            rank_zero_warn(
                "The checkpoint was not created with WeightAveraging. Both the current and the average model will be "
                "initialized with state_dict."
            )
            self._average_model.module.load_state_dict(deepcopy(checkpoint["state_dict"]), strict=False)

    def _swap_models(self, pl_module: "pl.LightningModule") -> None:
        """Swaps the parameter values of the current model and the :class:`AveragedModel`.

        Args:
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        assert self._average_model is not None
        average_params = itertools.chain(self._average_model.module.parameters(), self._average_model.module.buffers())
        current_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            tmp = average_param.data.clone()
            average_param.data.copy_(current_param.data)
            current_param.data.copy_(tmp)

    def _copy_average_to_current(self, pl_module: "pl.LightningModule") -> None:
        """Copies the parameter values from the :class:`AveragedModel` to the current model.

        Args:
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        assert self._average_model is not None
        average_params = itertools.chain(self._average_model.module.parameters(), self._average_model.module.buffers())
        current_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            current_param.data.copy_(average_param.data)
