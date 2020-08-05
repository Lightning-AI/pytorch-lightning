import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import Trainer, seed_everything, LightningModule, TrainResult
from pytorch_lightning.utilities import FLOAT16_EPSILON
from tests.base.datamodules import MNISTDataModule
from tests.base.develop_utils import set_random_master_port


class SyncBNModule(LightningModule):
    def __init__(self, gpu_count=1, **kwargs):
        super().__init__()

        self.gpu_count = gpu_count
        self.bn_targets = None
        if 'bn_targets' in kwargs:
            self.bn_targets = kwargs['bn_targets']

        self.linear = nn.Linear(28 * 28, 10)
        self.bn_layer = nn.BatchNorm1d(28 * 28)

    def forward(self, x, batch_idx):
        with torch.no_grad():
            out_bn = self.bn_layer(x.view(x.size(0), -1))

            if self.bn_targets:
                bn_target = self.bn_targets[batch_idx]

                # executes on both GPUs
                bn_target = bn_target[self.trainer.local_rank::self.gpu_count]
                bn_target = bn_target.to(out_bn.device)
                assert torch.sum(torch.abs(bn_target - out_bn)) < FLOAT16_EPSILON

        out = self.linear(out_bn)

        return out, out_bn

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat, _ = self(x, batch_idx)
        loss = F.cross_entropy(y_hat, y)

        return TrainResult(loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.linear.parameters(), lr=0.02)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_sync_batchnorm_ddp(tmpdir):
    seed_everything(234)
    set_random_master_port()

    # define datamodule and dataloader
    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup(stage=None)

    train_dataloader = dm.train_dataloader()
    model = SyncBNModule()

    bn_outputs = []

    # shuffle is false by default
    for batch_idx, batch in enumerate(train_dataloader):
        x, _ = batch

        _, out_bn = model.forward(x, batch_idx)
        bn_outputs.append(out_bn)

        # get 3 steps
        if batch_idx == 2:
            break

    bn_outputs = [x.cuda() for x in bn_outputs]

    # reset datamodule
    # batch-size = 16 because 2 GPUs in DDP
    dm = MNISTDataModule(batch_size=16, dist_sampler=True)
    dm.prepare_data()
    dm.setup(stage=None)

    model = SyncBNModule(gpu_count=2, bn_targets=bn_outputs)

    trainer = Trainer(
        gpus=2,
        num_nodes=1,
        distributed_backend='ddp_spawn',
        max_epochs=1,
        max_steps=3,
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        replace_sampler_ddp=False,
    )

    result = trainer.fit(model, dm)
    assert result == 1, "Sync batchnorm failing with DDP"
