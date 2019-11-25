import torch

from .lm_test_module_base import LightningTestModelBase
from .lm_test_module_mixins import LightningValidationMixin, LightningTestMixin


class LightningTestModel(LightningValidationMixin, LightningTestMixin, LightningTestModelBase):
    """
    Most common test case. Validation and test dataloaders
    """

    def on_training_metrics(self, logs):
        logs['some_tensor_to_test'] = torch.rand(1)
