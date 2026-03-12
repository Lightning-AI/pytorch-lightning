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
Gradient Accumulator
====================

Change gradient accumulation factor according to scheduling.
Trainer also calls ``optimizer.step()`` for the last indivisible step number.

"""

from typing import Any, Literal

from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class GradientAccumulationScheduler(Callback):
    r"""Change gradient accumulation factor according to scheduling.

    Args:
        scheduling: Scheduling in format ``{threshold: accumulation_factor}``. When ``mode="epoch"``,
            keys are zero-indexed epoch numbers. When ``mode="step"``, keys are global step numbers.
        mode: Whether to schedule by ``"epoch"`` or ``"step"``. Defaults to ``"epoch"`` for
            backward compatibility.

    Note:
        The argument scheduling is a dictionary. When ``mode="epoch"``, each key represents an epoch
        and its associated accumulation factor value (epochs are zero-indexed). When ``mode="step"``,
        each key represents a global training step. For example, if you want to change the accumulation
        factor after 4 epochs, use ``scheduling={4: factor}`` with ``mode="epoch"``; for step-based
        scheduling use e.g. ``scheduling={0: 8, 1000: 4, 5000: 1}`` with ``mode="step"``.

    Raises:
        TypeError:
            If ``scheduling`` is an empty ``dict``,
            or not all keys and values of ``scheduling`` are integers.
        MisconfigurationException:
            If ``mode`` is not ``"epoch"`` or ``"step"``, or if keys/values are invalid.
        IndexError:
            If minimal threshold is less than 0.

    Example::

        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import GradientAccumulationScheduler

        # Epoch-based: from epoch 5, accumulate every 2 batches (use 4 for zero-indexed).
        >>> accumulator = GradientAccumulationScheduler(scheduling={4: 2})
        >>> trainer = Trainer(callbacks=[accumulator])

        # Step-based: for single-epoch pretraining, schedule by global step.
        >>> accumulator = GradientAccumulationScheduler(
        ...     scheduling={0: 8, 1000: 4, 5000: 1},
        ...     mode="step",
        ... )
        >>> trainer = Trainer(callbacks=[accumulator])

    """

    def __init__(self, scheduling: dict[int, int], mode: Literal["epoch", "step"] = "epoch"):
        super().__init__()

        if mode not in ("epoch", "step"):
            raise MisconfigurationException(
                f"`mode` must be 'epoch' or 'step'. Got {mode!r}."
            )

        if not scheduling:  # empty dict error
            raise TypeError("Empty dict cannot be interpreted correct")

        threshold_name = "Epoch" if mode == "epoch" else "Step"
        if any(not isinstance(key, int) or key < 0 for key in scheduling):
            raise MisconfigurationException(
                f"{threshold_name} should be an int greater than or equal to 0. Got {list(scheduling.keys())}."
            )

        if any(not isinstance(value, int) or value < 1 for value in scheduling.values()):
            raise MisconfigurationException(
                f"Accumulation factor should be an int greater than 0. Got {list(scheduling.values())}."
            )

        minimal_threshold = min(scheduling.keys())
        if minimal_threshold < 0:
            raise IndexError(
                f"{threshold_name}s are non-negative, {minimal_threshold} cannot be interpreted correct"
            )
        if minimal_threshold != 0:  # if user didn't define first threshold accumulation factor
            scheduling = {**scheduling, 0: 1}

        self.scheduling = scheduling
        self.mode = mode
        self.epochs = sorted(self.scheduling.keys())

    def going_to_accumulate_grad_batches(self) -> bool:
        return any(v > 1 for v in self.scheduling.values())

    def get_accumulate_grad_batches(self, epoch: int) -> int:
        accumulate_grad_batches = 1
        for iter_epoch in reversed(self.epochs):
            if epoch >= iter_epoch:
                accumulate_grad_batches = self.scheduling[iter_epoch]
                break
        return accumulate_grad_batches

    @override
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Performns a configuration validation before training starts and raises errors for incompatible settings."""

        if not pl_module.automatic_optimization:
            raise RuntimeError(
                """Automatic gradient accumulation and the `GradientAccumulationScheduler` is not supported for
                manual optimization. Please remove the callback or switch to automatic optimization."""
            )

        overridden_optimizer_step = is_overridden("optimizer_step", pl_module)
        overridden_optimizer_zero_grad = is_overridden("optimizer_zero_grad", pl_module)
        going_to_accumulate_grad_batches = self.going_to_accumulate_grad_batches()
        has_overridden_optimization_functions = overridden_optimizer_step or overridden_optimizer_zero_grad
        if has_overridden_optimization_functions and going_to_accumulate_grad_batches:
            rank_zero_warn(
                "When using `Trainer(accumulate_grad_batches != 1)` and overriding"
                " `LightningModule.optimizer_{step,zero_grad}`, the hooks will not be called on every batch"
                " (rather, they are called on every optimization step)."
            )

        # local import to avoid circular import
        from lightning.pytorch.strategies import DeepSpeedStrategy

        if isinstance(trainer.strategy, DeepSpeedStrategy):
            raise RuntimeError(
                f"The `{type(trainer.strategy).__name__}` does not support `accumulate_grad_batches` changing"
                " between epochs."
            )
        if trainer.accumulate_grad_batches != 1:
            raise ValueError(
                "You have set `accumulate_grad_batches` and are using the `GradientAccumulationScheduler`"
                " callback. Either remove `accumulate_grad_batches` from the Trainer or remove the callback."
            )

        if self.mode == "step":
            trainer.accumulate_grad_batches = self.get_accumulate_grad_batches(0)

    @override
    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        if self.mode == "epoch":
            trainer.accumulate_grad_batches = self.get_accumulate_grad_batches(trainer.current_epoch)

    @override
    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        if self.mode == "step":
            trainer.accumulate_grad_batches = self.get_accumulate_grad_batches(trainer.global_step)
