# Copyright The Lightning AI team.
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

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader, DistributedSampler

from parity_pytorch import RunIf


class SyncBNModule(LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.bn_layer = nn.BatchNorm1d(1)
        self.linear = nn.Linear(1, 10)
        self.bn_outputs = []

    def on_train_start(self) -> None:
        assert isinstance(self.bn_layer, torch.nn.modules.batchnorm.SyncBatchNorm)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            out_bn = self.bn_layer(batch)
        self.bn_outputs.append(out_bn.detach())
        out = self.linear(out_bn)
        return out.sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.02)

    def train_dataloader(self):
        dataset = torch.arange(64, dtype=torch.float).view(-1, 1)
        # we need to set a distributed sampler ourselves to force shuffle=False
        sampler = DistributedSampler(
            dataset, num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, shuffle=False
        )
        return DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)


@RunIf(min_cuda_gpus=2, standalone=True)
def test_sync_batchnorm_parity(tmpdir):
    """Test parity between 1) Training a synced batch-norm layer on 2 GPUs with batch size B per device 2) Training a
    batch-norm layer on CPU with twice the batch size."""
    seed_everything(3)
    # 2 GPUS, batch size = 4 per GPU => total batch size = 8
    model = SyncBNModule(batch_size=4)
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        devices=2,
        max_steps=3,
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        deterministic=True,
        benchmark=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)

    # the strategy is responsible for tearing down the batch norm wrappers
    assert not isinstance(model.bn_layer, torch.nn.modules.batchnorm.SyncBatchNorm)
    assert isinstance(model.bn_layer, torch.nn.modules.batchnorm._BatchNorm)

    bn_outputs = torch.stack(model.bn_outputs)  # 2 x 4 x 1 on each GPU
    bn_outputs_multi_device = trainer.strategy.all_gather(bn_outputs).cpu()  # 2 x 2 x 4 x 1

    if trainer.global_rank == 0:
        # pretend we are now training on a single GPU/process
        # (we are reusing the rank 0 from the previous training)

        # 1 GPU, batch size = 8 => total batch size = 8
        bn_outputs_single_device = _train_single_process_sync_batchnorm(batch_size=8, num_steps=3)

        gpu0_outputs = bn_outputs_multi_device[0]  # 2 x 4 x 1
        gpu1_outputs = bn_outputs_multi_device[1]  # 2 x 4 x 1
        slice0 = bn_outputs_single_device[:, 0::2]
        slice1 = bn_outputs_single_device[:, 1::2]

        assert torch.allclose(gpu0_outputs, slice0)
        assert torch.allclose(gpu1_outputs, slice1)


def _train_single_process_sync_batchnorm(batch_size, num_steps):
    seed_everything(3)
    dataset = torch.arange(64, dtype=torch.float).view(-1, 1)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    model = SyncBNModule(batch_size=batch_size)
    optimizer = model.configure_optimizers()
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        loss = model.training_step(batch, batch)
        loss.backward()
        optimizer.step()
        if batch_idx == num_steps - 1:
            break

    return torch.stack(model.bn_outputs)  # num_steps x batch_size x 1
