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

from typing import Any, Dict

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class GradientAccumulationScheduler(Callback):
    r"""
    Change gradient accumulation factor according to scheduling.

    Args:
        scheduling: scheduling in format {epoch: accumulation_factor}

    Note:
        The argument scheduling is a dictionary. Each key represent an epoch and
        its associated accumulation factor value.
        Warning: Epoch are zero-indexed c.f it means if you want to change
        the accumulation factor after 4 epochs, set ``Trainer(accumulate_grad_batches={4: factor})``
        or ``GradientAccumulationScheduler(scheduling={4: factor})``.
        For more info check the example below.

    Raises:
        TypeError:
            If ``scheduling`` is an empty ``dict``,
            or not all keys and values of ``scheduling`` are integers.
        IndexError:
            If ``minimal_epoch`` is less than 0.

    Example::

        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import GradientAccumulationScheduler

        # from epoch 5, it starts accumulating every 2 batches. Here we have 4 instead of 5
        # because epoch (key) should be zero-indexed.
        >>> accumulator = GradientAccumulationScheduler(scheduling={4: 2})
        >>> trainer = Trainer(callbacks=[accumulator])

        # alternatively, pass the scheduling dict directly to the Trainer
        >>> trainer = Trainer(accumulate_grad_batches={4: 2})
    """

    def __init__(self, scheduling: Dict[int, int]):
        super().__init__()

        if not scheduling:  # empty dict error
            raise TypeError("Empty dict cannot be interpreted correct")

        if any(not isinstance(key, int) or key < 0 for key in scheduling):
            raise MisconfigurationException(
                f"Epoch should be an int greater than or equal to 0. Got {list(scheduling.keys())}."
            )

        if any(not isinstance(value, int) or value < 1 for value in scheduling.values()):
            raise MisconfigurationException(
                f"Accumulation factor should be an int greater than 0. Got {list(scheduling.values())}."
            )

        minimal_epoch = min(scheduling.keys())
        if minimal_epoch < 0:
            raise IndexError(f"Epochs indexing from 1, epoch {minimal_epoch} cannot be interpreted correct")
        if minimal_epoch != 0:  # if user didnt define first epoch accumulation factor
            scheduling.update({0: 1})

        self.scheduling = scheduling
        self.epochs = sorted(scheduling.keys())

    def going_to_accumulate_grad_batches(self) -> bool:
        return any(v > 1 for v in self.scheduling.values())

    def get_accumulate_grad_batches(self, epoch: int) -> int:
        accumulate_grad_batches = 1
        for iter_epoch in reversed(self.epochs):
            if epoch >= iter_epoch:
                accumulate_grad_batches = self.scheduling[iter_epoch]
                break
        return accumulate_grad_batches

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        trainer.accumulate_grad_batches = self.get_accumulate_grad_batches(trainer.current_epoch)
