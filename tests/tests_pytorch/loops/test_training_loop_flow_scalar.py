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
import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.loops.optimization.automatic import Closure
from lightning.pytorch.trainer.states import RunningStage
from lightning_utilities.test.warning import no_warning_call
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from tests_pytorch.helpers.deterministic_model import DeterministicModel


def test__training_step__flow_scalar(tmpdir):
    """Tests that only training_step can be used."""

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return acc

        def backward(self, loss):
            return LightningModule.backward(self, loss)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called


def test__training_step__tr_batch_end__flow_scalar(tmpdir):
    """Tests that only training_step can be used."""

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            self.out = acc
            return acc

        def on_train_batch_end(self, tr_step_output, *_):
            assert self.count_num_graphs({"loss": tr_step_output}) == 0

        def backward(self, loss):
            return LightningModule.backward(self, loss)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called


def test__training_step__epoch_end__flow_scalar(tmpdir):
    """Tests that only training_step can be used."""

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx

            self.training_step_called = True
            return acc

        def backward(self, loss):
            return LightningModule.backward(self, loss)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called

    # assert epoch end metrics were added
    assert len(trainer.callback_metrics) == 0
    assert len(trainer.progress_bar_metrics) == 0

    trainer.state.stage = RunningStage.TRAINING
    # make sure training outputs what is expected
    kwargs = {"batch": next(iter(model.train_dataloader())), "batch_idx": 0}
    train_step_out = trainer.fit_loop.epoch_loop.automatic_optimization.run(trainer.optimizers[0], 0, kwargs)

    assert isinstance(train_step_out["loss"], Tensor)
    assert train_step_out["loss"].item() == 171

    # make sure the optimizer closure returns the correct things
    opt_closure = trainer.fit_loop.epoch_loop.automatic_optimization._make_closure(kwargs, trainer.optimizers[0], 0)
    opt_closure_result = opt_closure()
    assert opt_closure_result.item() == 171


def test_train_step_no_return(tmpdir):
    """Tests that only training_step raises a warning when nothing is returned in case of automatic_optimization."""

    class TestModel(BoringModel):
        def training_step(self, batch):
            self.training_step_called = True
            loss = self.step(batch[0])
            self.log("a", loss, on_step=True, on_epoch=True)

        def validation_step(self, batch, batch_idx):
            self.validation_step_called = True

    model = TestModel()
    trainer_args = {"default_root_dir": tmpdir, "fast_dev_run": 2}
    trainer = Trainer(**trainer_args)

    Closure.warning_cache.clear()

    with pytest.warns(UserWarning, match=r"training_step` returned `None"):
        trainer.fit(model)

    assert model.training_step_called
    assert model.validation_step_called

    model = TestModel()
    model.automatic_optimization = False
    trainer = Trainer(**trainer_args)

    Closure.warning_cache.clear()

    with no_warning_call(UserWarning, match=r"training_step` returned `None"):
        trainer.fit(model)


def test_training_step_no_return_when_even(tmpdir):
    """Tests correctness when some training steps have been skipped."""

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.training_step_called = True
            loss = self.step(batch[0])
            self.log("a", loss, on_step=True, on_epoch=True)
            return loss if batch_idx % 2 else None

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=4,
        limit_val_batches=1,
        max_epochs=4,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )

    Closure.warning_cache.clear()

    with pytest.warns(UserWarning, match=r".*training_step` returned `None.*"):
        trainer.fit(model)

    trainer.state.stage = RunningStage.TRAINING

    # manually check a few batches
    for batch_idx, batch in enumerate(model.train_dataloader()):
        kwargs = {"batch": batch, "batch_idx": batch_idx}
        out = trainer.fit_loop.epoch_loop.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs)
        if not batch_idx % 2:
            assert out == {}


def test_training_step_none_batches(tmpdir):
    """Tests correctness when the train dataloader gives None for some steps."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def collate_none_when_even(self, batch):
            result = None if self.counter % 2 == 0 else default_collate(batch)
            self.counter += 1
            return result

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 4), collate_fn=self.collate_none_when_even)

        def on_train_batch_end(self, outputs, batch, batch_idx):
            if batch_idx % 2 == 0:
                assert outputs is None
            else:
                assert outputs

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_val_batches=1,
        max_epochs=4,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )

    with pytest.warns(UserWarning, match=r".*train_dataloader yielded None.*"):
        trainer.fit(model)
