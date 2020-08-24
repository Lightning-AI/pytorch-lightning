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
Gradient Accumulator
====================

Change gradient accumulation factor according to scheduling.
Trainer also calls ``optimizer.step()`` for the last indivisible step number.

"""

from pytorch_lightning.callbacks.base import Callback


class GradientAccumulationScheduler(Callback):
    r"""
    Change gradient accumulation factor according to scheduling.

    Args:
        scheduling: scheduling in format {epoch: accumulation_factor}

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import GradientAccumulationScheduler

        # at epoch 5 start accumulating every 2 batches
        >>> accumulator = GradientAccumulationScheduler(scheduling={5: 2})
        >>> trainer = Trainer(callbacks=[accumulator])

        # alternatively, pass the scheduling dict directly to the Trainer
        >>> trainer = Trainer(accumulate_grad_batches={5: 2})
    """

    def __init__(self, scheduling: dict):
        super().__init__()

        if not scheduling:  # empty dict error
            raise TypeError("Empty dict cannot be interpreted correct")

        for key in scheduling:
            if not isinstance(key, int) or not isinstance(scheduling[key], int):
                raise TypeError("All epoches and accumulation factor must be integers")

        minimal_epoch = min(scheduling.keys())
        if minimal_epoch < 0:
            raise IndexError(f"Epochs indexing from 1, epoch {minimal_epoch} cannot be interpreted correct")
        if minimal_epoch != 0:  # if user didnt define first epoch accumulation factor
            scheduling.update({0: 1})

        self.scheduling = scheduling
        self.epochs = sorted(scheduling.keys())

    def on_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        for i in reversed(range(len(self.epochs))):
            if epoch >= self.epochs[i]:
                trainer.accumulate_grad_batches = self.scheduling.get(self.epochs[i])
                break
