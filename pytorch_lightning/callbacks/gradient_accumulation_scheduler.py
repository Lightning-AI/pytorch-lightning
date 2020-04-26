r"""
Gradient Accumulator
====================

Change gradient accumulation factor according to scheduling.

"""

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_warn


class GradientAccumulationScheduler(Callback):
    r"""
    Change gradient accumulation factor according to scheduling.

    Args:
        scheduling: scheduling in format {epoch: accumulation_factor}

            .. warning::
                Epochs indexing starts from "1" until v0.6.x,
                but will start from "0" in v0.8.0.

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
        rank_zero_warn('Epochs indexing of `scheduling` starts from "1" until v0.6.x,'
                       ' but will start from "0" in v0.8.0.', DeprecationWarning)
        if minimal_epoch < 1:
            msg = f"Epochs indexing from 1, epoch {minimal_epoch} cannot be interpreted correct"
            raise IndexError(msg)
        if minimal_epoch != 1:  # if user didnt define first epoch accumulation factor
            scheduling.update({1: 1})

        self.scheduling = scheduling
        self.epochs = sorted(scheduling.keys())

    def on_epoch_start(self, trainer, pl_module):
        # indexing epochs from 1 (until v0.6.x)
        # In v0.8.0, ` + 1` should be removed.
        epoch = trainer.current_epoch + 1
        for i in reversed(range(len(self.epochs))):
            if epoch >= self.epochs[i]:
                trainer.accumulate_grad_batches = self.scheduling.get(self.epochs[i])
                break
