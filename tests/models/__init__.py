"""Models for testing."""
import sys
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

# the PL examples use torchvision, which is not installed in the test environment
# we redirect the torchvision imports in the examples to our custom MNIST
sys.modules['torchvision'] = __import__(
    'tests.mocks.torchvision', fromlist=['torchvision'])
sys.modules['torchvision.transforms'] = __import__(
    'tests.mocks.torchvision.transforms', fromlist=['transforms']
)
sys.modules['torchvision.datasets'] = __import__(
    'tests.mocks.torchvision.mnist', fromlist=['mnist']
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
