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
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

class TestModel(LightningModule):
    def __init__(self):
        super(TestModel, self).__init__()

        self.first = nn.Conv2d(1, 1, 3)
        self.second = nn.Conv2d(1, 1, 3)
        self.loss = nn.L1Loss()

    def train_dataloader(self):
        xs, ys = torch.ones(16, 1, 10, 10), torch.ones(16, 1, 6, 6)*5
        ds = TensorDataset(xs, ys)
        return DataLoader(ds)

    def forward(self, xs):
        out = self.first(xs)
        out = self.second(out)
        return out

    def configure_optimizers(self):
        first = torch.optim.SGD(self.first.parameters(), lr=0.01)
        second = torch.optim.Adam(self.second.parameters(), lr=0.01)
        return [first, second]

    def training_step(self, batch, batch_idx, optimizer_idx):
        xs, ys = batch
        out = self.forward(xs)
        return {'loss': self.loss(out, ys)}


def test_grad_norm_track():
    model = TestModel()
    trainer = Trainer(track_grad_norm=2,
                      log_every_n_steps=1,
                      fast_dev_run=True)

    trainer.fit(model)

    first_optimizer = ('grad_2.0_norm_first.weight', 'grad_2.0_norm_first.bias', 'grad_2.0_norm_total')
    assert all([trainer.logged_metrics[key] > 0.0 for key in first_optimizer]), f'Grad norm not logged for first optimizer'

    second_optimizer = ('grad_2.0_norm_second.weight', 'grad_2.0_norm_second.bias', 'grad_2.0_norm_total')
    assert all([trainer.logged_metrics[key] > 0.0 for key in second_optimizer]), f'Grad norm not logged for second optimizer'

