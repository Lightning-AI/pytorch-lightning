"""Models for testing."""

import torch

from tests.base.eval_model_template import EvalModelTemplate
from tests.base.models import TestModelBase, DictHparamsModel


class LightningTestModel(LightTrainDataloader,
                         LightValidationMixin,
                         LightTestMixin,
                         TestModelBase):
    """Most common test case. Validation and test dataloaders."""

    def on_training_metrics(self, logs):
        logs['some_tensor_to_test'] = torch.rand(1)
