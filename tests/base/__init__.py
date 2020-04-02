"""Models for testing."""

import torch

from tests.base.models import TestModelBase, DictHparamsModel
from tests.base.mixins import (
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
    LightValidationDataloader,
    LightTestDataloader,
    LightInfTrainDataloader,
    LightInfValDataloader,
    LightInfTestDataloader,
    LightTestOptimizerWithSchedulingMixin,
    LightTestMultipleOptimizersWithSchedulingMixin,
    LightTestOptimizersWithMixedSchedulingMixin,
    LightTestReduceLROnPlateauMixin,
    LightTestNoneOptimizerMixin,
    LightZeroLenDataloader
)


class LightningTestModel(LightTrainDataloader,
                         LightValidationMixin,
                         LightTestMixin,
                         TestModelBase):
    """Most common test case. Validation and test dataloaders."""

    def on_training_metrics(self, logs):
        logs['some_tensor_to_test'] = torch.rand(1)


class LightningTestModelWithoutHyperparametersArg(LightningTestModel):
    """Without hparams argument in constructor """

    def __init__(self):
        import tests.base.utils as tutils

        # the user loads the hparams in some other way
        hparams = tutils.get_default_hparams()
        super().__init__(hparams)


class LightningTestModelWithUnusedHyperparametersArg(LightningTestModelWithoutHyperparametersArg):
    """It has hparams argument in constructor but is not used."""

    def __init__(self, hparams):
        super().__init__()
