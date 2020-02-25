"""Models for testing."""

import torch

from .base import TestModelBase
from .mixins import (
    LightEmptyTestStep,
    LightValidationStepMixin,
    LightValidationMixin,
    LightValidationStepMultipleDataloadersMixin,
    LightValidationMultipleDataloadersMixin,
    LightTestStepMixin,
    LightTestMixin,
    LightTestStepMultipleDataloadersMixin,
    LightTestMultipleDataloadersMixin,
    LightTestFitSingleTestDataloadersMixin,
    LightTestFitMultipleTestDataloadersMixin,
    LightValStepFitSingleDataloaderMixin,
    LightValStepFitMultipleDataloadersMixin,
    LightTrainDataloader,
    LightTestDataloader,
)


class LightningTestModel(LightTrainDataloader,
                         LightValidationMixin,
                         LightTestMixin,
                         TestModelBase):
    """Most common test case. Validation and test dataloaders."""

    def on_training_metrics(self, logs):
        logs['some_tensor_to_test'] = torch.rand(1)
