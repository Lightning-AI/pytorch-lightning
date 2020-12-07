import os
import pytest
import torch
import torch.nn as nn

from tests.base.boring_model import BoringModel
from tests.base.develop_utils import set_random_master_port
from pytorch_lightning import Trainer, seed_everything


class AllGatherModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

        self.layer1 = torch.nn.Linear(32, 2)
        self.layer2 = torch.nn.Linear(32, 2)
        self.layer3 = torch.nn.Linear(32, 2)
        self.layer4 = torch.nn.Linear(32, 2)

    def forward(self, x):
        # no grad cases
        tensor1 = self.layer1(x)
        tensor2 = self.layer2(x)

        tensor1_gathered = self.all_gather(tensor1)
        tensor2_gathered = self.all_gather(tensor2)

        assert torch.sum(tensor1_gathered[self.global_rank] - tensor1) == 0
        assert torch.sum(tensor2_gathered[self.global_rank] - tensor2) == 0

        # with grad cases
        tensor3 = self.layer3(x)
        tensor4 = self.layer4(x)

        tensor3_gathered = self.all_gather(tensor3)
        tensor4_gathered = self.all_gather(tensor4)

        assert torch.sum(tensor3_gathered[self.global_rank] - tensor3) == 0
        assert torch.sum(tensor4_gathered[self.global_rank] - tensor4) == 0

        # test for grads

        return self.layer(x)


# TODO: horovod and TPU
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not available on windows")
@pytest.mark.parametrize("accelerator", ['ddp', 'ddp_cpu', 'ddp_spawn'])
def test_all_gather(accelerator):
    gpus = 2

    seed_everything(234)
    set_random_master_port()

    model = AllGatherModel()
    train_dataloader = model.train_dataloader()

    trainer = Trainer(
        gpus=gpus,
        accelerator=accelerator,
        max_epochs=1,
        max_steps=3,
        num_sanity_val_steps=0,
    )

    result = trainer.fit(model, train_dataloader)
    assert result == 1, "All gather op fails in Lightning Module"
