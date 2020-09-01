from pytorch_lightning.trainer.supporters import Accumulator
import numpy as np


class TrainLoop:

    def __init__(self, trainer):
        self.trainer = trainer
        self.should_check_val = False
        self.early_stopping_accumulator = None
        self.checkpoint_accumulator = None

    @property
    def num_optimizers(self):
        num_optimizers = len(self.get_optimizers_iterable())
        return num_optimizers

    def on_train_epoch_start(self):
        # hook
        self.trainer.call_hook('on_epoch_start')
        self.trainer.call_hook('on_train_epoch_start')

        # bookkeeping
        self.should_check_val = False

        # structured result accumulators for callbacks
        self.early_stopping_accumulator = Accumulator()
        self.checkpoint_accumulator = Accumulator()

    def get_optimizers_iterable(self):
        if not self.trainer.optimizer_frequencies:
            # call training_step once per optimizer
            return list(enumerate(self.trainer.optimizers))

        optimizer_freq_cumsum = np.cumsum(self.trainer.optimizer_frequencies)
        optimizers_loop_length = optimizer_freq_cumsum[-1]
        current_place_in_loop = self.trainer.total_batch_idx % optimizers_loop_length

        # find optimzier index by looking for the first {item > current_place} in the cumsum list
        opt_idx = np.argmax(optimizer_freq_cumsum > current_place_in_loop)
        return [(opt_idx, self.trainer.optimizers[opt_idx])]