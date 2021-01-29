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

from pytorch_lightning import Trainer
from tests.base.boring_model import BoringModel


def test_multiple_optimizers(tmpdir):
    """
    Tests that only training_step can be used
    """
    class TestModel(BoringModel):
        def on_train_epoch_start(self) -> None:
            self.opt_0_seen = False
            self.opt_1_seen = False

        def training_step(self, batch, batch_idx, optimizer_idx):
            if optimizer_idx == 0:
                self.opt_0_seen = True
            elif optimizer_idx == 1:
                self.opt_1_seen = True
            else:
                raise Exception('should only have two optimizers')

            self.training_step_called = True
            loss = self.step(batch[0])
            return loss

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer, optimizer_2

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)
    assert model.opt_0_seen
    assert model.opt_1_seen


def test_multiple_optimizers_manual(tmpdir):
    """
    Tests that only training_step can be used
    """
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def on_train_epoch_start(self) -> None:
            self.opt_0_seen = False
            self.opt_1_seen = False

        def training_step(self, batch, batch_idx, optimizer_idx):
            # manual
            (opt_a, opt_b) = self.optimizers()
            loss_1 = self.step(batch[0])

            # fake generator
            self.manual_backward(loss_1, opt_a)
            opt_a.step()
            opt_a.zero_grad()

            # fake discriminator
            loss_2 = self.step(batch[0])
            self.manual_backward(loss_2, opt_b)
            opt_b.step()
            opt_b.zero_grad()

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer, optimizer_2

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)
