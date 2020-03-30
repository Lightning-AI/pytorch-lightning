r"""
Callback Base
=============
    Abstract base class used to build new callbacks.
"""

import abc


class Callback(abc.ABC):
    r"""
    Abstract base class used to build new callbacks.
    """

    def on_init_start(self, trainer):
        """Called when the trainer initialization begins, model has not yet been set."""
        pass

    def on_init_end(self, trainer):
        """Called when the trainer initialization ends, model has not yet been set."""
        pass

    def on_epoch_start(self, trainer, pl_module):
        """Called when the epoch begins."""
        pass

    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        pass

    def on_batch_start(self, trainer, pl_module):
        """Called when the training batch begins."""
        pass

    def on_batch_end(self, trainer, pl_module):
        """Called when the training batch ends."""
        pass

    def on_train_start(self, trainer, pl_module):
        """Called when the train begins."""
        pass

    def on_train_end(self, trainer, pl_module):
        """Called when the train ends."""
        pass

    def on_validation_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        pass

    def on_test_start(self, trainer, pl_module):
        """Called when the test begins."""
        pass

    def on_test_end(self, trainer, pl_module):
        """Called when the test ends."""
        pass
