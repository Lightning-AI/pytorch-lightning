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
from tests.helpers.boring_model import BoringModel


class MultiOptModel(BoringModel):

    def configure_optimizers(self):
        opt_a = torch.optim.SGD(self.layer.parameters(), lr=0.001)
        opt_b = torch.optim.SGD(self.layer.parameters(), lr=0.001)
        return opt_a, opt_b


def test_unbalanced_logging_with_multiple_optimizers(tmpdir):
    """
    This tests ensures reduction works in unbalanced logging settings,
    even when a Callback also logs.
    """

    class TestModel(MultiOptModel):

        actual = {0: [], 1: []}

        def training_step(self, batch, batch_idx, optimizer_idx):
            out = super().training_step(batch, batch_idx)
            loss = out["loss"]
            self.log(f"loss_{optimizer_idx}", loss, on_epoch=True)
            self.actual[optimizer_idx].append(loss)
            return out

    model = TestModel()
    model.training_epoch_end = None

    class TestCallback(pl.Callback):

        def on_train_batch_end(self, trainer, pl_module, output, batch, batch_idx, dl_idx):
            # when this is called, the EpochResultStore state has not been reset yet because we are still
            # "INSIDE_BATCH_TRAIN_LOOP" and the LoggerConnector runs its `on_train_batch_end` after the
            # Callback (see `TrainLoop.on_train_batch_end`). For this reason, opt_idx here is the index
            # of the last optimizer updated (the second, index 1). This produced a KeyError as reported in #5459
            pl_module.log("test_train_batch_end", trainer.logger_connector.cached_results._opt_idx)

    # Initialize a trainer
    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=5,
        limit_val_batches=5,
        callbacks=[TestCallback()],
        weights_summary=None,
    )
    trainer.fit(model)

    for k, v in model.actual.items():
        assert torch.equal(trainer.callback_metrics[f"loss_{k}_step"], v[-1])
        # test loss is properly reduced
        torch.testing.assert_allclose(trainer.callback_metrics[f"loss_{k}_epoch"], torch.tensor(v).mean())

    assert trainer.callback_metrics["test_train_batch_end"] == len(model.optimizers()) - 1


def test_multiple_optimizers(tmpdir):

    class TestModel(MultiOptModel):

        seen = [False, False]

        def training_step(self, batch, batch_idx, optimizer_idx):
            self.seen[optimizer_idx] = True
            return super().training_step(batch, batch_idx)

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer, optimizer_2

    model = TestModel()
    model.val_dataloader = None

    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    assert all(model.seen)


def test_multiple_optimizers_manual(tmpdir):

    class TestModel(MultiOptModel):

        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx, optimizer_idx):
            self.training_step_called = True

            # manual optimization
            opt_a, opt_b = self.optimizers()
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

    model = TestModel()
    model.val_dataloader = None

    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    assert model.training_step_called
