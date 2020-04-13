"""Models for testing."""

import torch

from tests.base.models import TrialModelBase, DictHparamsModel
from tests.base.mixins import (
    LightEmptyTstStep,
    LightValStepMixin,
    LightValMixin,
    LightValStepMultipleDataloadersMixin,
    LightValMultipleDataloadersMixin,
    LightTstStepMixin,
    LightTstMixin,
    LightTstStepMultipleDataloadersMixin,
    LightTstMultipleDataloadersMixin,
    LightTstStepSingleTstDataloadersMixin,
    LightTstStepMultipleTstDataloadersMixin,
    LightValStepFitSingleDataloaderMixin,
    LightValStepFitMultipleDataloadersMixin,
    LightTrnDataloader,
    LightValDataloader,
    LightTstDataloader,
    LightInfTrnDataloader,
    LightInfValDataloader,
    LightInfTstDataloader,
    LightOptimizerWithSchedulingMixin,
    LightMultipleOptimizersWithSchedulingMixin,
    LightOptimizersWithMixedSchedulingMixin,
    LightReduceLROnPlateauMixin,
    LightNoneOptimizerMixin,
    LightZeroLenDataloader
)


class LightningTrialModel(LightTrnDataloader,
                          LightValMixin,
                          LightTstMixin,
                          TrialModelBase):
    """Most common test case. Validation and test dataloaders."""

    def on_training_metrics(self, logs):
        logs['some_tensor_to_test'] = torch.rand(1)
