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
from torch import nn

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import StaticModelQuantization
from tests.base import BoringModel


class QuantModel(BoringModel):

    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleDict()

        self.layer["mlp_1"] = nn.Linear(32, 32)
        self.layer["mlp_2"] = nn.Linear(32, 32)
        self.layer["mlp_3"] = nn.Linear(32, 32)
        self.layer["mlp_4"] = nn.Linear(32, 32)
        self.layer["mlp_5"] = nn.Linear(32, 2)

    def forward(self, x):
        m = self.layer
        x = m["mlp_1"](x)
        x = m["mlp_2"](x)
        x = m["mlp_3"](x)
        x = m["mlp_4"](x)
        return m["mlp_5"](x)

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}


def train_with_quantization_callback(tmpdir):

    model = QuantModel()
    model.validation_step = None
    model.test_step = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=10,
        max_epochs=2,
        callbacks=[StaticModelQuantization],
    )
    trainer.fit(model)
