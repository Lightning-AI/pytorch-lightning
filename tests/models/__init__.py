"""Models for testing."""

import torch

from .base import TestModelBase, DictHparamsModel
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
    LightTestOptimizerWithSchedulingMixin,
    LightTestMultipleOptimizersWithSchedulingMixin,
    LightTestOptimizersWithMixedSchedulingMixin,
    LightTestReduceLROnPlateauMixin
)


class LightningTestModel(LightTrainDataloader,
                         LightValidationMixin,
                         LightTestMixin,
                         TestModelBase):
    """Most common test case. Validation and test dataloaders."""

    def on_training_metrics(self, logs):
        logs['some_tensor_to_test'] = torch.rand(1)


class LightningTestModelWithoutHyperparametersArg(LightningTestModel):
    """ without hparams argument in constructor """

    def __init__(self):
        import tests.models.utils as tutils

        # the user loads the hparams in some other way
        hparams = tutils.get_hparams()
        super().__init__(hparams)


class LightningTestModelWithUnusedHyperparametersArg(LightningTestModelWithoutHyperparametersArg):
    """ has hparams argument in constructor but is not used """

    def __init__(self, hparams):
        super().__init__()
