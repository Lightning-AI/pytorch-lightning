from abc import ABC

from pytorch_lightning.callbacks import Callback


class TrainerCallbackHookMixin(ABC):

    def __init__(self):
        # this is just a summary on variables used in this abstract class,
        # the proper values/initialisation should be done in child class
        self.callbacks: list[Callback] = []

    def on_init_begin(self):
        """Called when the trainer initialization begins."""
        for callback in self.callbacks:
            callback.set_trainer(self)
            callback.on_init_begin()

    def on_init_end(self):
        """Called when the trainer initialization ends."""
        for callback in self.callbacks:
            callback.on_init_end()

    def on_fit_begin(self):
        """Called when the fit begins."""
        for callback in self.callbacks:
            callback.on_fit_begin()

    def on_fit_end(self):
        """Called when the fit ends."""
        for callback in self.callbacks:
            callback.on_fit_end()

    def on_epoch_begin(self):
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        """Called when the epoch ends."""
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_train_begin(self):
        """Called when the train begins."""
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        """Called when the train ends."""
        for callback in self.callbacks:
            callback.on_train_end()

    def on_batch_begin(self):
        """Called when the training batch begins."""
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self):
        """Called when the training batch ends."""
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_validation_begin(self):
        """Called when the validation loop begins."""
        for callback in self.callbacks:
            callback.on_validation_begin()

    def on_validation_end(self):
        """Called when the validation loop ends."""
        for callback in self.callbacks:
            callback.on_validation_end()

    def on_test_begin(self):
        """Called when the test begins."""
        for callback in self.callbacks:
            callback.on_test_begin()

    def on_test_end(self):
        """Called when the test ends."""
        for callback in self.callbacks:
            callback.on_test_end()
