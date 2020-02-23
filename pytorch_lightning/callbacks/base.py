"""
Callbacks
=========

Callbacks supported by Lightning
"""

import abc


_NO_TRAINER_ERROR_MSG = ".set_trainer() should be called after the callback initialization"


class Callback(abc.ABC):
    """Abstract base class used to build new callbacks."""

    def __init__(self):
        self._trainer = None

    @property
    def trainer(self):
        assert self._trainer is not None, _NO_TRAINER_ERROR_MSG
        return self._trainer

    @property
    def module(self):
        return self._trainer.get_model()

    @property
    def default_save_path(self):
        """Trainer default save path.
        """
        return self._trainer.default_save_path

    @property
    def rank(self):
        """Current trainer rank.
        """
        return self._trainer.proc_rank

    def set_trainer(self, trainer):
        """Make a link to the trainer, so different things like `trainer.current_epoch`,
        `trainer.batch_idx`, `trainer.global_step` can be used."""
        self._trainer = trainer

    def on_init_begin(self):
        """Called when the trainer initialization begins."""
        pass

    def on_init_end(self):
        """Called when the trainer initialization ends."""
        pass

    def on_fit_begin(self):
        """Called when the fit begins."""
        pass

    def on_fit_end(self):
        """Called when the fit ends."""
        pass

    def on_epoch_begin(self):
        """Called when the epoch begins."""
        pass

    def on_epoch_end(self):
        """Called when the epoch ends."""
        pass

    def on_batch_begin(self):
        """Called when the training batch begins."""
        pass

    def on_batch_end(self):
        """Called when the training batch ends."""
        pass

    def on_train_begin(self):
        """Called when the train begins."""
        pass

    def on_train_end(self):
        """Called when the train ends."""
        pass

    def on_validation_begin(self):
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self):
        """Called when the validation loop ends."""
        pass

    def on_test_begin(self):
        """Called when the test begins."""
        pass

    def on_test_end(self):
        """Called when the test ends."""
        pass
