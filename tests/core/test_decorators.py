import pytest
import torch

from pytorch_lightning.core.decorators import auto_move_data
from tests.base import EvalModelTemplate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.parametrize(['src_device', 'dest_device'], [
    pytest.param(torch.device('cpu'), torch.device('cpu')),
    pytest.param(torch.device('cpu', 0), torch.device('cuda', 0)),
    pytest.param(torch.device('cuda', 0), torch.device('cpu')),
    pytest.param(torch.device('cuda', 0), torch.device('cuda', 0)),
])
def test_auto_move_data(src_device, dest_device):
    """ Test that the decorator moves the data to the device the model is on. """

    class CurrentModel(EvalModelTemplate):
        pass

    # apply the decorator
    CurrentModel.forward = auto_move_data(CurrentModel.forward)

    model = CurrentModel()
    model = model.to(dest_device)
    model.prepare_data()
    loader = model.train_dataloader()
    x, y, = next(iter(loader))
    x = x.flatten(1)

    # test that data on source device gets moved to destination device
    x = x.to(src_device)
    assert model(x).device == dest_device, "Automoving data to same device as model failed"
