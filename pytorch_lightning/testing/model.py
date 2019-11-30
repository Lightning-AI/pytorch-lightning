import torch

from .model_base import LightningTestModelBase
from .model_mixins import LightningValidationMixin, LightningTestMixin


class LightningTestModel(LightningValidationMixin, LightningTestMixin, LightningTestModelBase):
    """
    Most common test case. Validation and test dataloaders
    """

    def on_training_metrics(self, logs):
        logs['some_tensor_to_test'] = torch.rand(1)
