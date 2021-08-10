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
Test logging in the training loop
"""

import collections
import itertools
from re import escape

import numpy as np
import pytest
import torch
from torchmetrics import Accuracy

import pytorch_lightning as pl
from pytorch_lightning import callbacks, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel, RandomDictDataset
from tests.helpers.runif import RunIf


def test__training_step__log(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            loss = out["loss"]

            # -----------
            # default
            # -----------
            self.log("default", loss)

            # -----------
            # logger
            # -----------
            # on_step T on_epoch F
            self.log("l_s", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

            # on_step F on_epoch T
            self.log("l_e", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            # on_step T on_epoch T
            self.log("l_se", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

            # -----------
            # pbar
            # -----------
            # on_step T on_epoch F
            self.log("p_s", loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)

            # on_step F on_epoch T
            self.log("p_e", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

            # on_step T on_epoch T
            self.log("p_se", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)

            return loss

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
        callbacks=[ModelCheckpoint(monitor="l_se")],
    )
    trainer.fit(model)

    logged_metrics = set(trainer.logged_metrics)
    assert logged_metrics == {"epoch", "default", "l_e", "l_s", "l_se_step", "l_se_epoch"}

    pbar_metrics = set(trainer.progress_bar_metrics)
    assert pbar_metrics == {"p_e", "p_s", "p_se_step", "p_se_epoch"}

    assert set(trainer.callback_metrics) == (logged_metrics | pbar_metrics | {"p_se", "l_se"}) - {"epoch"}


def test__training_step__epoch_end__log(tmpdir):
    """
    Tests that training_epoch_end can log
    """

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            loss = out["loss"]
            self.log("a", loss, on_step=True, on_epoch=True)
            self.log_dict({"a1": loss, "a2": loss})
            return out

        def training_epoch_end(self, outputs):
            self.log("b1", outputs[0]["loss"])
            self.log("b", outputs[0]["loss"], on_epoch=True, prog_bar=True, logger=True)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    logged_metrics = set(trainer.logged_metrics)
    assert logged_metrics == {"epoch", "a_step", "a_epoch", "b", "b1", "a1", "a2"}

    pbar_metrics = set(trainer.progress_bar_metrics)
    assert pbar_metrics == {"b"}

    assert set(trainer.callback_metrics) == (logged_metrics | pbar_metrics | {"a"}) - {"epoch"}


@pytest.mark.parametrize(["batches", "log_interval", "max_epochs"], [(1, 1, 1), (64, 32, 2)])
def test__training_step__step_end__epoch_end__log(tmpdir, batches, log_interval, max_epochs):
    """
    Tests that training_step_end and training_epoch_end can log
    """

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = self.step(batch[0])
            self.log("a", loss, on_step=True, on_epoch=True)
            return loss

        def training_step_end(self, out):
            self.log("b", out, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return out

        def training_epoch_end(self, outputs):
            self.log("c", outputs[0]["loss"], on_epoch=True, prog_bar=True, logger=True)
            self.log("d/e/f", 2)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=batches,
        limit_val_batches=batches,
        max_epochs=max_epochs,
        log_every_n_steps=log_interval,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure all the metrics are available for callbacks
    logged_metrics = set(trainer.logged_metrics)
    assert logged_metrics == {"a_step", "a_epoch", "b_step", "b_epoch", "c", "d/e/f", "epoch"}

    pbar_metrics = set(trainer.progress_bar_metrics)
    assert pbar_metrics == {"c", "b_epoch", "b_step"}

    assert set(trainer.callback_metrics) == (logged_metrics | pbar_metrics | {"a", "b"}) - {"epoch"}


@pytest.mark.parametrize(
    ["batches", "fx", "result"], [(3, min, 0), (3, torch.max, 2), (11, max, 10), (5, "avg", 2), (5, "SUM", 10)]
)
def test__training_step__log_max_reduce_fx(tmpdir, batches, fx, result):
    """
    Tests that log works correctly with different tensor types
    """

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch[0])
            self.log("foo", torch.tensor(batch_idx, dtype=torch.long), on_step=False, on_epoch=True, reduce_fx=fx)
            return acc

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("bar", torch.tensor(batch_idx).float(), on_step=False, on_epoch=True, reduce_fx=fx)
            return {"x": loss}

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=batches,
        limit_val_batches=batches,
        max_epochs=2,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure types are correct
    assert trainer.logged_metrics["foo"] == result
    assert trainer.logged_metrics["bar"] == result


def test_different_batch_types_for_sizing(tmpdir):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            assert isinstance(batch, dict)
            a = batch["a"]
            acc = self.step(a)
            self.log("a", {"d1": 2, "d2": torch.tensor(1)}, on_step=True, on_epoch=True)
            return acc

        def validation_step(self, batch, batch_idx):
            assert isinstance(batch, dict)
            a = batch["a"]
            output = self.layer(a)
            loss = self.loss(batch, output)
            self.log("n", {"d3": 2, "d4": torch.tensor(1)}, on_step=True, on_epoch=True)
            return {"x": loss}

        def train_dataloader(self):
            return torch.utils.data.DataLoader(RandomDictDataset(32, 64), batch_size=32)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(RandomDictDataset(32, 64), batch_size=32)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=2,
        max_epochs=1,
        weights_summary=None,
        fast_dev_run=True,
    )
    trainer.fit(model)

    assert set(trainer.logged_metrics) == {"a_step", "a_epoch", "n_step", "n_epoch", "epoch"}


def test_log_works_in_train_callback(tmpdir):
    """
    Tests that log can be called within callback
    """

    class TestCallback(callbacks.Callback):

        count = 0
        choices = [False, True]

        # used to compute expected values
        logged_values = collections.defaultdict(list)
        call_counter = collections.Counter()
        logged_arguments = {}

        def make_logging(self, pl_module, func_name, on_steps, on_epochs, prob_bars):
            self.call_counter.update([func_name])

            for idx, (on_step, on_epoch, prog_bar) in enumerate(itertools.product(on_steps, on_epochs, prob_bars)):
                fx = f"{func_name}_{idx}"
                pl_module.log(fx, self.count, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)
                self.logged_values[fx].append(self.count)
                self.logged_arguments[fx] = {"on_step": on_step, "on_epoch": on_epoch, "prog_bar": prog_bar}
                self.count += 1

        def on_train_start(self, _, pl_module):
            self.make_logging(pl_module, "on_train_start", on_steps=[False], on_epochs=[True], prob_bars=self.choices)

        def on_epoch_start(self, _, pl_module):
            self.make_logging(
                pl_module, "on_epoch_start", on_steps=self.choices, on_epochs=[True], prob_bars=self.choices
            )

        def on_train_epoch_start(self, _, pl_module):
            self.make_logging(
                pl_module, "on_train_epoch_start", on_steps=self.choices, on_epochs=[True], prob_bars=self.choices
            )

        def on_batch_end(self, _, pl_module):
            self.make_logging(
                pl_module, "on_batch_end", on_steps=self.choices, on_epochs=self.choices, prob_bars=self.choices
            )

        def on_train_batch_end(self, _, pl_module, *__):
            self.make_logging(
                pl_module, "on_train_batch_end", on_steps=self.choices, on_epochs=self.choices, prob_bars=self.choices
            )

        def on_train_epoch_end(self, _, pl_module):
            self.make_logging(
                pl_module, "on_train_epoch_end", on_steps=[False], on_epochs=[True], prob_bars=self.choices
            )

        def on_epoch_end(self, _, pl_module):
            self.make_logging(pl_module, "on_epoch_end", on_steps=[False], on_epochs=[True], prob_bars=self.choices)

    class TestModel(BoringModel):
        seen_losses = []

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)["loss"]
            self.seen_losses.append(loss)
            self.log("train_loss", loss, prog_bar=True)
            return {"loss": loss}

    model = TestModel()
    cb = TestCallback()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        max_epochs=1,
        callbacks=[cb],
    )
    trainer.fit(model)

    # Make sure the func_name output equals the average from all logged values when on_epoch true
    assert trainer.progress_bar_dict["train_loss"] == model.seen_losses[-1]
    assert trainer.callback_metrics["train_loss"] == model.seen_losses[-1]

    assert cb.call_counter == {
        "on_train_start": 1,
        "on_epoch_start": 1,
        "on_train_epoch_start": 1,
        "on_train_batch_end": 2,
        "on_batch_end": 2,
        "on_train_epoch_end": 1,
        "on_epoch_end": 1,
    }

    def get_expected(on_epoch, values):
        reduction = np.mean if on_epoch else np.max
        return reduction(values)

    for fx, value in trainer.callback_metrics.items():
        actual = value.item()
        if fx not in cb.logged_arguments:
            continue
        on_epoch = cb.logged_arguments[fx]["on_epoch"]
        values = cb.logged_values[fx]
        expected = get_expected(on_epoch, values)
        assert actual == expected

    for fx, attrs in cb.logged_arguments.items():
        should_include = attrs["prog_bar"] and attrs["on_step"] ^ attrs["on_epoch"]
        is_included = fx in trainer.progress_bar_metrics
        assert is_included if should_include else not is_included


@pytest.mark.parametrize("gpus", [None, pytest.param(1, marks=RunIf(min_gpus=1))])
def test_logging_sync_dist_true(tmpdir, gpus):
    """
    Tests to ensure that the sync_dist flag works (should just return the original value)
    """
    fake_result = 1

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("foo", fake_result, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
            self.log("foo_2", 2, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            self.log("bar", fake_result, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
            return super().validation_step(batch, batch_idx)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=2,
        weights_summary=None,
        gpus=gpus,
    )
    trainer.fit(model)

    assert trainer.logged_metrics["foo"] == fake_result
    assert trainer.logged_metrics["foo_2"] == 2
    assert trainer.logged_metrics["bar"] == fake_result


@RunIf(min_gpus=2, special=True)
def test_logging_sync_dist_true_ddp(tmpdir):
    """
    Tests to ensure that the sync_dist flag works with ddp
    """

    class TestLoggingSyncDistModel(BoringModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch[0])
            self.log("foo", 1, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="SUM")
            self.log("cho", acc, on_step=False, on_epoch=True)
            return acc

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("bar", 2, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="AVG")
            return {"x": loss}

    model = TestLoggingSyncDistModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=2,
        weights_summary=None,
        accelerator="ddp",
        gpus=2,
        profiler="pytorch",
    )
    trainer.fit(model)

    assert trainer.logged_metrics["foo"] == 2
    assert trainer.logged_metrics["bar"] == 2


def test_progress_bar_dict_contains_values_on_train_epoch_end(tmpdir):
    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", torch.tensor(self.current_epoch), on_step=False, on_epoch=True, prog_bar=True)
            return super().training_step(*args)

        def on_train_epoch_end(self, *_):
            self.log(
                "foo_2", torch.tensor(self.current_epoch), prog_bar=True, on_epoch=True, sync_dist=True, reduce_fx="sum"
            )
            self.on_train_epoch_end_called = True

        def on_epoch_end(self):
            assert self.trainer.progress_bar_dict["foo"] == self.current_epoch
            assert self.trainer.progress_bar_dict["foo_2"] == self.current_epoch
            self.on_epoch_end_called = True

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=0,
        checkpoint_callback=False,
        logger=False,
        weights_summary=None,
        progress_bar_refresh_rate=0,
    )
    model = TestModel()
    trainer.fit(model)
    assert model.on_train_epoch_end_called
    assert model.on_epoch_end_called


def test_logging_in_callbacks_with_log_function(tmpdir):
    """
    Tests ensure self.log can be used directly in callbacks.
    """

    class LoggingCallback(callbacks.Callback):
        def on_train_start(self, trainer, pl_module):
            self.log("on_train_start", 1)

        def on_train_epoch_start(self, trainer, pl_module):
            self.log("on_train_epoch_start", 2)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            self.log("on_train_batch_end", 3)

        def on_batch_end(self, trainer, pl_module):
            self.log("on_batch_end", 4)

        def on_epoch_end(self, trainer, pl_module):
            self.log("on_epoch_end", 5)

        def on_train_epoch_end(self, trainer, pl_module, outputs):
            self.log("on_train_epoch_end", 6)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
        callbacks=[LoggingCallback()],
    )
    trainer.fit(model)

    expected = {
        "on_train_start": 1,
        "on_train_epoch_start": 2,
        "on_train_batch_end": 3,
        "on_batch_end": 4,
        "on_epoch_end": 5,
        "on_train_epoch_end": 6,
    }
    assert trainer.callback_metrics == expected


@RunIf(min_gpus=1)
def test_metric_are_properly_reduced(tmpdir):
    class TestingModel(BoringModel):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.val_acc = pl.metrics.Accuracy()

        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            self.log("train_loss", output["loss"])
            return output

        def validation_step(self, batch, batch_idx):
            preds = torch.tensor([[0.9, 0.1]], device=self.device)
            targets = torch.tensor([1], device=self.device)
            if batch_idx < 8:
                preds = torch.tensor([[0.1, 0.9]], device=self.device)
            self.val_acc(preds, targets)
            self.log("val_acc", self.val_acc, on_step=True, on_epoch=True)
            return super().validation_step(batch, batch_idx)

    early_stop = EarlyStopping(monitor="val_acc", mode="max")

    checkpoint = ModelCheckpoint(monitor="val_acc", save_last=True, save_top_k=2, mode="max")

    model = TestingModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        gpus=1,
        max_epochs=2,
        limit_train_batches=5,
        limit_val_batches=32,
        callbacks=[early_stop, checkpoint],
    )
    trainer.fit(model)

    assert trainer.callback_metrics["val_acc"] == 8 / 32.0
    assert "train_loss" in trainer.callback_metrics


@pytest.mark.parametrize(
    "value", [None, dict(a=None), dict(a=dict(b=None)), dict(a=dict(b=1)), "foo", [1, 2, 3], (1, 2, 3), [[1, 2], 3]]
)
def test_log_none_raises(tmpdir, value):
    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", value)

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    model = TestModel()
    match = escape(f"self.log(foo, {value})` was called")
    with pytest.raises(ValueError, match=match):
        trainer.fit(model)


def test_logging_raises(tmpdir):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("foo/dataloader_idx_0", -1)

    trainer = Trainer(default_root_dir=tmpdir)
    model = TestModel()
    with pytest.raises(MisconfigurationException, match="`self.log` with the key `foo/dataloader_idx_0`"):
        trainer.fit(model)

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("foo", Accuracy())

    trainer = Trainer(default_root_dir=tmpdir)
    model = TestModel()
    with pytest.raises(MisconfigurationException, match="fix this by setting an attribute for the metric in your"):
        trainer.fit(model)

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.bar = Accuracy()

        def training_step(self, batch, batch_idx):
            self.log("foo", Accuracy())

    trainer = Trainer(default_root_dir=tmpdir)
    model = TestModel()
    with pytest.raises(
        MisconfigurationException,
        match=r"`self.log\(foo, ..., metric_attribute=name\)` where `name` is one of \['bar'\]",
    ):
        trainer.fit(model)

    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", -1, prog_bar=False)
            self.log("foo", -1, prog_bar=True)
            return super().training_step(*args)

    trainer = Trainer(default_root_dir=tmpdir)
    model = TestModel()
    with pytest.raises(MisconfigurationException, match=r"self.log\(foo, ...\)` twice in `training_step`"):
        trainer.fit(model)

    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", -1, reduce_fx=torch.argmax)
            return super().training_step(*args)

    trainer = Trainer(default_root_dir=tmpdir)
    model = TestModel()
    with pytest.raises(MisconfigurationException, match=r"reduce_fx={min,max,mean,sum}\)` are currently supported"):
        trainer.fit(model)


def test_sanity_metrics_are_reset(tmpdir):
    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            output = super().validation_step(batch, batch_idx)
            if self.trainer.sanity_checking:
                self.log("val_loss", output["x"], prog_bar=True, logger=True)
            return output

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            if batch_idx == 0:
                assert self.trainer.progress_bar_metrics == {}
                assert self.trainer.logged_metrics == {}
                assert self.trainer.callback_metrics == {}
            self.log("train_loss", loss, prog_bar=True, logger=True)
            return loss

    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, limit_train_batches=1, limit_val_batches=2, num_sanity_val_steps=2
    )
    trainer.fit(TestModel())

    assert "val_loss" not in trainer.progress_bar_metrics


@RunIf(min_gpus=2)
def test_log_gpu_memory_without_logging_on_step(tmpdir):

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        log_gpu_memory="all",
        log_every_n_steps=1,
        gpus=[1],
    )
    trainer.fit(model)

    assert "gpu_id: 1/memory.used (MB)" in trainer.logged_metrics
