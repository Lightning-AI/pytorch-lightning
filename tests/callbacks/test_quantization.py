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
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import StaticModelQuantization
from pytorch_lightning.metrics.functional import mean_absolute_error
from tests.base import BoringModel
from tests.base.simple_model import RandomDataset

TARGET_VALUE = 100

class QuantModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict()
        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()
        self.layers["mlp_1"] = nn.Linear(32, 32)
        self.layers["mlp_1a"] = torch.nn.ReLU()
        self.layers["mlp_2"] = nn.Linear(32, 64)
        self.layers["mlp_2a"] = torch.nn.ReLU()
        self.layers["mlp_3"] = nn.Linear(64, 64)
        self.layers["mlp_3a"] = torch.nn.ReLU()
        self.layers["mlp_4"] = nn.Linear(64, 32)
        self.layers["mlp_4a"] = torch.nn.ReLU()
        self.layers["mlp_5"] = nn.Linear(32, 2)

    def forward(self, x):
        # x = self.quant(x)
        for n in sorted(self.layers):
            x = self.layers[n](x)
        # x = self.dequant(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.layers.parameters(), lr=0.01)
        return [optimizer], []

    def targets(self, output):
        return torch.ones_like(output) * TARGET_VALUE

    def metric(self, output):
        return mean_absolute_error(output, self.targets(output)).item()

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        loss = torch.nn.functional.mse_loss(output, self.targets(output))
        self.log('train_MAE', self.metric(output), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.log('val_MAE', self.metric(output), prog_bar=True)

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64))



def test_static_quantization(tmpdir):

    model = QuantModel()
    # model.validation_step = None
    model.test_step = None

    org_size = model.model_size()

    qcb = StaticModelQuantization()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=10,
        max_epochs=15,
        callbacks=[qcb],
    )
    trainer.fit(model)

    mae = torch.mean(torch.tensor([model.metric(model(x)) for x in RandomDataset(32, 64)]))
    print(mae)
    mae = torch.mean(torch.tensor([model.metric(qcb.qmodel(x.type(torch.float32))) for x in RandomDataset(32, 64)]))
    print(mae)

    # todo: test that the trained model is smaller then initial
    # todo: test that the test score is almost the same as with pure training
