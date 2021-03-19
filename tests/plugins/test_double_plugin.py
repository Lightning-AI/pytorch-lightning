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
import pickle

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DoublePrecisionPlugin
from tests.helpers import BoringModel


class DoublePrecisionBoringModel(BoringModel):

    def training_step(self, batch, batch_idx):
        assert batch.dtype == torch.float32
        output = self((batch, torch.ones_like(batch).long()))  # Add some non floating-point data
        loss = self.loss(batch, output)
        return {"loss": loss}

    def on_fit_start(self):
        assert self.layer.weight.dtype == torch.float64

    def forward(self, x):
        try:
            x, ones = x  # training
            assert ones.dtype == torch.long
        except ValueError:
            pass  # test / val
        assert x.dtype == torch.float64
        return super().forward(x)

    def on_after_backward(self):
        assert self.layer.weight.grad.dtype == torch.float64


def test_double_precision(tmpdir):
    model = DoublePrecisionBoringModel()
    original_forward = model.forward

    trainer = Trainer(
        max_epochs=2,
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_test_batches=2,
        limit_val_batches=2,
        precision=64,
        log_every_n_steps=1,
    )
    trainer.fit(model)

    assert model.forward == original_forward


def test_double_precision_pickle(tmpdir):
    double_precision = DoublePrecisionPlugin()
    pickle.dumps(double_precision)
