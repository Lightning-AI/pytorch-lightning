import contextlib

import torch


class Plugin(object):
    """Basic Plugin class to derive precision and training type plugins from."""

    def connect(self, model: torch.nn.Module, *args, **kwargs):
        """Connects the plugin with the accelerator (and thereby with trainer and model).
        Will be called by the accelerator.
        """
        pass

    def pre_optimizer_step(self, optimizer: torch.optim.Optimizer, optimizer_idx: int):
        """Hook to do something before each optimizer step."""
        pass

    def post_optimizer_step(self, optimizer: torch.optim.Optimizer, optimizer_idx: int):
        """Hook to do something after each optimizer step."""
        pass

    def pre_training(self):
        """Hook to do something before the training starts."""
        pass

    def post_training(self):
        """Hook to do something after the training finishes."""
        pass

    @contextlib.contextmanager
    def train_step_context(self):
        """A contextmanager for the trainstep"""
        yield

    @contextlib.contextmanager
    def val_step_context(self):
        """A contextmanager for the validation step"""
        yield

    @contextlib.contextmanager
    def test_step_context(self):
        """A contextmanager for the teststep"""
        yield