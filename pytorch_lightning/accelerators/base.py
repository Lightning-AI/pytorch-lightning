from abc import ABC, abstractmethod


class LightningBackend(ABC):
    """Interface for all Lightning backends"""

    def __init__(self, trainer):
        self.trainer = trainer

    def setup(self, model, *args):
        """Setting-up all needed attributes."""
        self.model = model
        # call setup after the ddp process has connected
        self.trainer.call_setup_hook(model)

    @abstractmethod
    def train(self, *args):
        """Perform the training."""

    def teardown(self, *args):
        """Optional cleaning after process ends."""
