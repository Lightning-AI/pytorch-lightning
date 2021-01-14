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
    This tests ensures reduction works in unbalanced logging settings,
    even when a Callback also logs.
    """
    class TestModel(BoringModel):
        actual = {0: [], 1: []}

        def training_step(self, batch, batch_idx, optimizer_idx):
            out = super().training_step(batch, batch_idx)
            loss = out["loss"]
            self.log(f"loss_{optimizer_idx}", loss, on_epoch=True)
            self.actual[optimizer_idx].append(loss)
            return out

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.001)
            optimizer2 = torch.optim.SGD(self.layer.parameters(), lr=0.001)
            return [optimizer, optimizer2]

    model = TestModel()
    model.training_epoch_end = None

    class TestCallback(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, *args):
            pl_module.log("test", -1)

    # Initialize a trainer
    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        callbacks=[TestCallback()]
    )
    trainer.fit(model)

    for k, v in model.actual.items():
        assert torch.equal(trainer.callback_metrics[f"loss_{k}_step"], v[-1])
        # test loss is properly reduced
        torch.testing.assert_allclose(trainer.callback_metrics[f"loss_{k}_epoch"], torch.tensor(v).mean())
