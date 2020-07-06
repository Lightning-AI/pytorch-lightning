import pytest
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

from pytorch_lightning.inference.infer import (
    Inference,
    IterativeInference,
    TempScalingInference
)


class MixedDtypeModel(LightningModule):
    """ The parameters and inputs of this model have different dtypes. """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 20)   # expects dtype long as input
        self.reduce = nn.Linear(20, 1)      # dtype: float
        self.example_input_array = torch.tensor([[0, 2, 1], [3, 5, 3]])  # dtype: long

    def forward(self, x):
        return self.reduce(self.embed(x))


@pytest.fixture
def random():
    torch.manual_seed(0)

@pytest.mark.parametrize('infer_mode', ['eval', 'train'])
def test_inference(infer_mode):
    model = MixedDtypeModel()
    assert isinstance(Inference(model, infer_mode=infer_mode)(model.example_input_array), torch.Tensor)