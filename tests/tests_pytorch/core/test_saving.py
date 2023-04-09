import pytest
import lightning.pytorch as pl

import torch

from lightning.pytorch.demos.boring_classes import BoringModel
from tests_pytorch.helpers.runif import RunIf


def create_boring_checkpoint():
    model = BoringModel()
    trainer = pl.Trainer(accelerator='auto', max_epochs=1)
    trainer.fit(model)
    trainer.save_checkpoint("./boring.ckpt")


@pytest.mark.parametrize("map_location", ('cpu', torch.device('cpu'), lambda storage, loc: storage, {'cpu': 'cpu'}))
def test_load_from_checkpoint_map_location_cpu(map_location):
    create_boring_checkpoint()
    model = BoringModel.load_from_checkpoint("./boring.ckpt", map_location=map_location)
    assert model.device.type == 'cpu'


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("map_location", ('cuda', torch.device('cuda'), lambda storage, loc: storage.cuda(), {'cpu': 'cuda'}))
def test_load_from_checkpoint_map_location_gpu(map_location):
    model = BoringModel.load_from_checkpoint("./boring.ckpt", map_location=map_location)
    assert model.device.type == 'cuda'
