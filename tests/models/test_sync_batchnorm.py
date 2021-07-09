# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.utilities import FLOAT16_EPSILON
from tests.helpers.datamodules import MNISTDataModule
from tests.helpers.runif import RunIf
from tests.helpers.utils import set_random_master_port


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

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.linear.parameters(), lr=0.02)


# TODO: Fatal Python error: Bus error
@pytest.mark.skip(reason="Fatal Python error: Bus error")
@RunIf(min_gpus=2, special=True)
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
    ddp = DDPSpawnPlugin(
        parallel_devices=[torch.device("cuda", 0), torch.device("cuda", 1)],
        num_nodes=1,
        sync_batchnorm=True,
        cluster_environment=LightningEnvironment(),
        find_unused_parameters=True
    )

    trainer = Trainer(
        default_root_dir=tmpdir,
        gpus=2,
        num_nodes=1,
        accelerator='ddp_spawn',
        max_epochs=1,
        max_steps=3,
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        replace_sampler_ddp=False,
        plugins=[ddp]
    )

    trainer.fit(model, dm)
    assert trainer.state.finished, "Sync batchnorm failing with DDP"
