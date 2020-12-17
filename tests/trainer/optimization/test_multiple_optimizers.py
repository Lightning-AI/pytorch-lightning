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
"""
Tests to ensure that the behaviours related to multiple optimizers works
"""
import torch

import pytorch_lightning as pl
from tests.base.boring_model import BoringModel


def test_unbalanced_logging_with_multiple_optimizers(tmpdir):
    """
    This tests ensures reduction works in un-balanced logging settings
    """
    class TestModel(BoringModel):

        loss_1 = []
        loss_2 = []

        def training_step(self, batch, batch_idx, optimizer_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            if optimizer_idx == 0 and self.trainer.global_step > 10:
                self.log("loss_1", loss, on_epoch=True, prog_bar=True)
                self.loss_1.append(loss.detach().clone())
            elif optimizer_idx == 1:
                self.log("loss_2", loss, on_epoch=True, prog_bar=True)
                self.loss_2.append(loss.detach().clone())
            return {"loss": loss}

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.001)
            optimizer2 = torch.optim.SGD(self.layer.parameters(), lr=0.001)
            return [optimizer, optimizer2]

    model = TestModel()
    model.training_epoch_end = None

    # Initialize a trainer
    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )

    trainer.fit(model)

    assert torch.equal(trainer.callback_metrics["loss_2_step"], model.loss_2[-1])
    assert torch.equal(trainer.callback_metrics["loss_1_step"], model.loss_1[-1])
    # test loss are properly reduced
    assert torch.abs(trainer.callback_metrics["loss_2_epoch"] - torch.FloatTensor(model.loss_2).mean()) < 1e-6
    assert torch.abs(trainer.callback_metrics["loss_1_epoch"] - torch.FloatTensor(model.loss_1).mean()) < 1e-6
