"""Models for testing."""

import torch

from .base import TestModelWithDataloader, TestModelWithoutDataloader
from .mixins import (
    LightningValidationStepMixin,
    LightningValidationMixin,
    LightningValidationStepMultipleDataloadersMixin,
    LightningValidationMultipleDataloadersMixin,
    LightningTestStepMixin,
    LightningTestMixin,
    LightningTestStepMultipleDataloadersMixin,
    LightningTestMultipleDataloadersMixin,
    LightningTestFitSingleTestDataloadersMixin,
    LightningTestFitMultipleTestDataloadersMixin,
    LightningValStepFitSingleDataloaderMixin,
    LightningValStepFitMultipleDataloadersMixin
)


class LightningTestModel(LightningValidationMixin, LightningTestMixin, TestModelWithDataloader):
    """
    Most common test case. Validation and test dataloaders.
    """

    def on_training_metrics(self, logs):
        logs['some_tensor_to_test'] = torch.rand(1)
