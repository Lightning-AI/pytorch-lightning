"""Models for testing."""

import torch

from .base import (LightningTestModelBase,
                   LightningTestModelBaseWithoutDataloader)
from .mixins import (
    LightningValidationStepMixin,
    LightningValidationMixin,
    LightningValidationStepMultipleDataloadersMixin,
    LightningValidationMultipleDataloadersMixin,
    LightningTestStepMixin,
    LightningTestMixin,
    LightningTestStepMultipleDataloadersMixin,
    LightningTestMultipleDataloadersMixin,
)


class LightningTestModel(LightningValidationMixin, LightningTestMixin, LightningTestModelBase):
    """
    Most common test case. Validation and test dataloaders.
    """

    def on_training_metrics(self, logs):
        logs['some_tensor_to_test'] = torch.rand(1)
