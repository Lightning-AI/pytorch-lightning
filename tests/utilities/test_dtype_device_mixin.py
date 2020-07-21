import torch
import torch.nn as nn

from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
from tests.base import EvalModelTemplate


class Model(EvalModelTemplate):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc = Accuracy()


def test_submodules_device(tmpdir):

    model = Model()
    assert model.device == torch.device('cpu')
    model = model.to('cuda')
    assert model.device == model.acc.device == torch.device('cuda')