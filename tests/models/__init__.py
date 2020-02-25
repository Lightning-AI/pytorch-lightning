"""Models for testing."""

import torch

from .base import LightningTestModelBase, LightningTestModelBaseWithoutDataloader
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


class LightningTestModel(LightningValidationMixin, LightningTestMixin, LightningTestModelBase):
    """
    Most common test case. Validation and test dataloaders.
    """

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
