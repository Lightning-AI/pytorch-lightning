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
"""Tests to ensure that the behaviours related to multiple optimizers works."""
import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.demos.boring_classes import BoringModel


class MultiOptModel(BoringModel):
    def configure_optimizers(self):
        opt_a = torch.optim.SGD(self.layer.parameters(), lr=0.001)
        opt_b = torch.optim.SGD(self.layer.parameters(), lr=0.001)
        return opt_a, opt_b


def test_multiple_optimizers_automatic_optimization_raises():
    """Test that multiple optimizers in automatic optimization is not allowed."""

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            return super().training_step(batch, batch_idx)

    model = TestModel()
    model.automatic_optimization = True

    trainer = pl.Trainer()
    with pytest.raises(RuntimeError, match="Remove the `optimizer_idx` argument from `training_step`"):
        trainer.fit(model)

    class TestModel(BoringModel):
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters()), torch.optim.Adam(self.parameters())

    model = TestModel()
    model.automatic_optimization = True

    trainer = pl.Trainer()
    with pytest.raises(RuntimeError, match="multiple optimizers is only supported with manual optimization"):
        trainer.fit(model)


def test_multiple_optimizers_manual(tmpdir):
    class TestModel(MultiOptModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            self.training_step_called = True

            # manual optimization
            opt_a, opt_b = self.optimizers()
            loss_1 = self.step(batch[0])

            # fake generator
            self.manual_backward(loss_1)
            opt_a.step()
            opt_a.zero_grad()

            # fake discriminator
            loss_2 = self.step(batch[0])
            self.manual_backward(loss_2)
            opt_b.step()
            opt_b.zero_grad()

    model = TestModel()
    model.val_dataloader = None

    trainer = pl.Trainer(
        default_root_dir=tmpdir, limit_train_batches=2, max_epochs=1, log_every_n_steps=1, enable_model_summary=False
    )
    trainer.fit(model)

    assert model.training_step_called
