from abc import ABC, abstractmethod


class LightningBackend(ABC):
    """Interface for all Lightning backends"""

    def __init__(self, trainer):
        self._trainer = trainer

    @abstractmethod
    def setup(self, model):
        """Setting-up all needed attributes."""
        self._model = model
        # call setup after the ddp process has connected
        self._trainer.call_setup_hook(model)

    @abstractmethod
    def train(self, *args):
        """Perform the training."""

    def teardown(self):
        """Optional clenaing after process ends."""
