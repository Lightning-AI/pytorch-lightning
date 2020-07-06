import pytest
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

from pytorch_lightning.inference.infer import (
    Inference,
    IterativeInference,
    TempScalingInference
)


class UnorderedModel(LightningModule):
    """ A model in which the layers not defined in order of execution """

    def __init__(self):
        super().__init__()
        # note: the definition order is intentionally scrambled for this test
        self.layer2 = nn.Linear(10, 2)
        self.combine = nn.Linear(7, 9)
        self.layer1 = nn.Linear(3, 5)
        self.relu = nn.ReLU()
        # this layer is unused, therefore input-/output shapes are unknown
        self.unused = nn.Conv2d(1, 1, 1)

        self.example_input_array = (torch.rand(2, 3), torch.rand(2, 10))

    def forward(self, x, y):
        out1 = self.layer1(x)
        out2 = self.layer2(y)
        out = self.relu(torch.cat((out1, out2), 1))
        out = self.combine(out)
        return out


@pytest.fixture
def random():
    torch.manual_seed(0)

@pytest.mark.parametrize('infer_mode', ['eval', 'train'])
def test_inference(infer_mode):
    model = UnorderedModel()
    assert isinstance(Inference(model, infer_mode=infer_mode)(model.example_input_array), torch.Tensor)