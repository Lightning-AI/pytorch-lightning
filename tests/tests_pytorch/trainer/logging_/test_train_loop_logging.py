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
"""Test logging in the training loop."""

import collections
import itertools
from re import escape
from unittest import mock
from unittest.mock import call

import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer, callbacks
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset, RandomDictDataset
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.trainer.states import RunningStage
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_0_11 as _TM_GE_0_11
from lightning_utilities.test.warning import no_warning_call
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from tests_pytorch.helpers.runif import RunIf


def test__training_step__log(tmp_path):
    """Tests that only training_step can be used."""

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
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        enable_model_summary=False,
        callbacks=[ModelCheckpoint(monitor="l_se")],
    )
    trainer.fit(model)

    logged_metrics = set(trainer.logged_metrics)
    assert logged_metrics == {"default", "l_e", "l_s", "l_se_step", "l_se_epoch"}

    pbar_metrics = set(trainer.progress_bar_metrics)
    assert pbar_metrics == {"p_e", "p_s", "p_se_step", "p_se_epoch"}

    assert set(trainer.callback_metrics) == (logged_metrics | pbar_metrics | {"p_se", "l_se"})
    assert all(isinstance(v, Tensor) for v in trainer.callback_metrics.values())
    assert all(isinstance(v, Tensor) for v in trainer.logged_metrics.values())
    assert all(isinstance(v, float) for v in trainer.progress_bar_metrics.values())


def test__training_step__epoch_end__log(tmp_path):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            loss = out["loss"]
            self.log("a", loss, on_step=True, on_epoch=True)
            self.log_dict({"a1": loss, "a2": loss})
            return out

        def on_train_epoch_end(self):
            self.log("b1", torch.tensor(1.0))
            self.log("b", torch.tensor(2.0), on_epoch=True, prog_bar=True, logger=True)

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)

    logged_metrics = set(trainer.logged_metrics)
    assert logged_metrics == {"a_step", "a_epoch", "b", "b1", "a1", "a2"}

    pbar_metrics = set(trainer.progress_bar_metrics)
    assert pbar_metrics == {"b"}

    assert set(trainer.callback_metrics) == (logged_metrics | pbar_metrics | {"a"})
    assert all(isinstance(v, Tensor) for v in trainer.callback_metrics.values())
    assert all(isinstance(v, Tensor) for v in trainer.logged_metrics.values())
    assert all(isinstance(v, float) for v in trainer.progress_bar_metrics.values())


@pytest.mark.parametrize(
    ("batches", "fx", "result"), [(3, min, 0), (3, torch.max, 2), (11, max, 10), (5, "avg", 2), (5, "SUM", 10)]
)
def test__training_step__log_max_reduce_fx(tmp_path, batches, fx, result):
    """Tests that log works correctly with different tensor types."""

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch[0])
            self.log("foo", torch.tensor(batch_idx, dtype=torch.long), on_step=False, on_epoch=True, reduce_fx=fx)
            return acc

        def validation_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("bar", torch.tensor(batch_idx).float(), on_step=False, on_epoch=True, reduce_fx=fx)
            return {"x": loss}

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=batches,
        limit_val_batches=batches,
        max_epochs=2,
        enable_model_summary=False,
    )
    trainer.fit(model)

    # make sure types are correct
    assert trainer.logged_metrics["foo"] == result
    assert trainer.logged_metrics["bar"] == result


def test_different_batch_types_for_sizing(tmp_path):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            assert isinstance(batch, dict)
            a = batch["a"]
            acc = self.step(a)
            self.log("a", 2, on_step=True, on_epoch=True)
            return acc

        def validation_step(self, batch, batch_idx):
            assert isinstance(batch, dict)
            loss = self.step(batch["a"])
            self.log("n", 3, on_step=True, on_epoch=True)
            return {"x": loss}

        def train_dataloader(self):
            return torch.utils.data.DataLoader(RandomDictDataset(32, 64), batch_size=32)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(RandomDictDataset(32, 64), batch_size=32)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=2,
        max_epochs=1,
        enable_model_summary=False,
        fast_dev_run=True,
    )
    trainer.fit(model)

    assert set(trainer.logged_metrics) == {"a_step", "a_epoch", "n_step", "n_epoch"}


def test_log_works_in_train_callback(tmp_path):
    """Tests that log can be called within callback."""

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
                if not on_step and not on_epoch:
                    with pytest.raises(MisconfigurationException, match="is not useful"):
                        pl_module.log(fx, self.count, on_step=on_step, on_epoch=on_epoch)
                    continue
                pl_module.log(fx, self.count, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)
                self.logged_values[fx].append(self.count)
                self.logged_arguments[fx] = {"on_step": on_step, "on_epoch": on_epoch, "prog_bar": prog_bar}
                self.count += 1

        def on_train_start(self, _, pl_module):
            self.make_logging(pl_module, "on_train_start", on_steps=[False], on_epochs=[True], prob_bars=self.choices)

        def on_train_epoch_start(self, _, pl_module):
            self.make_logging(
                pl_module, "on_train_epoch_start", on_steps=[False], on_epochs=[True], prob_bars=self.choices
            )

        def on_train_batch_start(self, _, pl_module, *__):
            self.make_logging(
                pl_module, "on_train_batch_start", on_steps=self.choices, on_epochs=self.choices, prob_bars=self.choices
            )

        def on_train_batch_end(self, _, pl_module, *__):
            self.make_logging(
                pl_module, "on_train_batch_end", on_steps=self.choices, on_epochs=self.choices, prob_bars=self.choices
            )

        def on_train_epoch_end(self, _, pl_module):
            self.make_logging(
                pl_module, "on_train_epoch_end", on_steps=[False], on_epochs=[True], prob_bars=self.choices
            )

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
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        max_epochs=1,
        callbacks=[cb],
    )

    trainer.fit(model)

    # Make sure the func_name output equals the average from all logged values when on_epoch true
    assert trainer.progress_bar_callback.get_metrics(trainer, model)["train_loss"] == model.seen_losses[-1]
    assert trainer.callback_metrics["train_loss"] == model.seen_losses[-1]

    assert cb.call_counter == {
        "on_train_start": 1,
        "on_train_epoch_start": 1,
        "on_train_batch_start": 2,
        "on_train_batch_end": 2,
        "on_train_epoch_end": 1,
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


class LoggingSyncDistModel(BoringModel):
    def __init__(self, fake_result):
        super().__init__()
        self.fake_result = fake_result

    @property
    def rank(self) -> int:
        return self.trainer.global_rank

    def training_step(self, batch, batch_idx):
        value = self.fake_result + self.rank
        self.log("foo", value, on_step=True, on_epoch=False, sync_dist=True, reduce_fx="sum")
        self.log("foo_2", 2, on_step=True, on_epoch=False, sync_dist=True, reduce_fx="sum")
        self.log("foo_3", 2, on_step=True, on_epoch=False, sync_dist=True, reduce_fx="mean")
        self.log("foo_4", value, on_step=True, on_epoch=False, sync_dist=True, reduce_fx="mean")
        self.log("foo_5", batch_idx + self.rank, on_step=True, on_epoch=False, sync_dist=True, reduce_fx="max")

        self.log("foo_6", value, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("foo_7", 2, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("foo_8", 2, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("foo_9", value, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("foo_10", batch_idx, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="max")
        self.log("foo_11", batch_idx + self.rank, on_step=True, on_epoch=True, sync_dist=True, reduce_fx="mean")
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.log("bar", self.fake_result, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("bar_2", self.fake_result, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("bar_3", batch_idx + self.rank, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="max")
        return super().validation_step(batch, batch_idx)


@pytest.mark.parametrize(
    ("devices", "accelerator"),
    [
        (1, "cpu"),
        (2, "cpu"),
        pytest.param(2, "gpu", marks=RunIf(min_cuda_gpus=2)),
    ],
)
def test_logging_sync_dist_true(tmp_path, devices, accelerator):
    """Tests to ensure that the sync_dist flag works (should just return the original value)"""
    fake_result = 1
    model = LoggingSyncDistModel(fake_result)

    use_multiple_devices = devices > 1
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        limit_train_batches=3,
        limit_val_batches=3,
        enable_model_summary=False,
        strategy="ddp_spawn" if use_multiple_devices else "auto",
        accelerator=accelerator,
        devices=devices,
    )
    trainer.fit(model)

    total = fake_result * devices + 1
    metrics = trainer.callback_metrics
    assert metrics["foo"] == total if use_multiple_devices else fake_result
    assert metrics["foo_2"] == 2 * devices
    assert metrics["foo_3"] == 2
    assert metrics["foo_4"] == total / devices if use_multiple_devices else 1
    assert metrics["foo_5"] == fake_result * 2 + 1 if use_multiple_devices else fake_result * 2
    assert metrics["foo_6"] == (0 + 1 + 1 + 2 + 2 + 3) if use_multiple_devices else fake_result * 3 * 2
    assert metrics["foo_7"] == 2 * devices * 3
    assert metrics["foo_8"] == 2
    assert metrics["foo_9"] == (fake_result * 2 + 1) / devices if use_multiple_devices else fake_result
    assert metrics["foo_10"] == 2
    assert metrics["foo_11_step"] == (2 + 3) / 2 if use_multiple_devices else fake_result * 2
    assert metrics["foo_11"] == (0 + 1 + 1 + 2 + 2 + 3) / (devices * 3) if use_multiple_devices else fake_result
    assert metrics["bar"] == fake_result * 3 * devices
    assert metrics["bar_2"] == fake_result
    assert metrics["bar_3"] == 2 + int(use_multiple_devices)


@RunIf(min_cuda_gpus=2, standalone=True)
def test_logging_sync_dist_true_ddp(tmp_path):
    """Tests to ensure that the sync_dist flag works with ddp."""

    class TestLoggingSyncDistModel(BoringModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch[0])
            self.log("foo", 1, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="SUM")
            self.log("cho", acc, on_step=False, on_epoch=True)
            return acc

        def validation_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("bar", 2, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="AVG")
            return {"x": loss}

    model = TestLoggingSyncDistModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=2,
        strategy="ddp",
        accelerator="gpu",
        devices=2,
        profiler="pytorch",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)

    assert trainer.logged_metrics["foo"] == 2
    assert trainer.logged_metrics["bar"] == 2


def test_progress_bar_metrics_contains_values_on_train_epoch_end(tmp_path: str):
    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", torch.tensor(self.current_epoch), on_step=False, on_epoch=True, prog_bar=True)
            return super().training_step(*args)

        def on_train_epoch_end(self, *_):
            self.log(
                "foo_2", torch.tensor(self.current_epoch), prog_bar=True, on_epoch=True, sync_dist=True, reduce_fx="sum"
            )
            self.on_train_epoch_end_called = True

    class TestProgressBar(TQDMProgressBar):
        def get_metrics(self, trainer: Trainer, model: LightningModule):
            items = super().get_metrics(trainer, model)
            items.pop("v_num", None)
            return items

        def on_train_end(self, trainer: Trainer, model: LightningModule):
            metrics = self.get_metrics(trainer, model)
            assert metrics["foo"] == self.trainer.current_epoch - 1
            assert metrics["foo_2"] == self.trainer.current_epoch - 1
            model.callback_on_train_end_called = True

    progress_bar = TestProgressBar()
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[progress_bar],
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )
    model = TestModel()
    trainer.fit(model)
    assert model.on_train_epoch_end_called
    assert model.callback_on_train_end_called


def test_logging_in_callbacks_with_log_function(tmp_path):
    """Tests ensure self.log can be used directly in callbacks."""

    class LoggingCallback(callbacks.Callback):
        def on_train_start(self, trainer, pl_module):
            self.log("on_train_start", 1)

        def on_train_epoch_start(self, trainer, pl_module):
            self.log("on_train_epoch_start", 2)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.log("on_train_batch_end", 3)

        def on_train_epoch_end(self, trainer, pl_module):
            self.log("on_train_epoch_end", 5)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        enable_model_summary=False,
        callbacks=[LoggingCallback()],
    )
    trainer.fit(model)

    expected = {
        "on_train_start": 1,
        "on_train_epoch_start": 2,
        "on_train_batch_end": 3,
        "on_train_epoch_end": 5,
    }
    assert trainer.callback_metrics == expected


# mps not yet supported by torchmetrics, see https://github.com/Lightning-AI/metrics/issues/1044
@pytest.mark.parametrize(
    "accelerator",
    [
        pytest.param("gpu", marks=RunIf(min_cuda_gpus=1)),
        "cpu",
    ],
)
def test_metric_are_properly_reduced(tmp_path, accelerator):
    class TestingModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.val_acc = Accuracy(task="multiclass", num_classes=2) if _TM_GE_0_11 else Accuracy()

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
        default_root_dir=tmp_path,
        accelerator=accelerator,
        devices=1,
        max_epochs=2,
        limit_train_batches=5,
        limit_val_batches=32,
        callbacks=[early_stop, checkpoint],
    )
    trainer.fit(model)

    assert trainer.callback_metrics["val_acc"] == 8 / 32.0
    assert "train_loss" in trainer.callback_metrics


@pytest.mark.parametrize(
    "value",
    [None, {"a": None}, {"a": 1}, {"a": {"b": None}}, {"a": {"b": 1}}, "foo", [1, 2, 3], (1, 2, 3), [[1, 2], 3]],
)
def test_log_invalid_raises(tmp_path, value):
    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", value)

    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)
    model = TestModel()
    match = escape(f"self.log(foo, {value})` was called")
    with pytest.raises(ValueError, match=match):
        trainer.fit(model)


def test_log_tensor_and_clone_no_torch_warning(tmp_path):
    """Regression test for issue https://github.com/Lightning-AI/lightning/issues/14594."""

    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", torch.tensor(1))
            return super().training_step(*args)

    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)
    model = TestModel()
    match = r"recommended.*.clone\(\).detach\(\)"
    with no_warning_call(UserWarning, match=match):
        trainer.fit(model)


def test_logging_raises(tmp_path):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("foo/dataloader_idx_0", -1)

    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=0,
        max_epochs=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )
    model = TestModel()
    with pytest.raises(MisconfigurationException, match="`self.log` with the key `foo/dataloader_idx_0`"):
        trainer.fit(model)

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("foo", Accuracy(task="multiclass", num_classes=2) if _TM_GE_0_11 else Accuracy())

    model = TestModel()
    with pytest.raises(MisconfigurationException, match="fix this by setting an attribute for the metric in your"):
        trainer.fit(model)

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.bar = Accuracy(task="multiclass", num_classes=2) if _TM_GE_0_11 else Accuracy()

        def training_step(self, batch, batch_idx):
            self.log("foo", Accuracy(task="multiclass", num_classes=2) if _TM_GE_0_11 else Accuracy())

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

    model = TestModel()
    with pytest.raises(MisconfigurationException, match=r"self.log\(foo, ...\)` twice in `training_step`"):
        trainer.fit(model)

    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", -1, reduce_fx=torch.argmax)
            return super().training_step(*args)

    model = TestModel()
    with pytest.raises(MisconfigurationException, match=r"reduce_fx={min,max,mean,sum}\)` are supported"):
        trainer.fit(model)

    class TestModel(BoringModel):
        def on_train_start(self):
            self.log("foo", torch.tensor([1.0, 2.0]))

    model = TestModel()
    with pytest.raises(ValueError, match="tensor must have a single element"):
        trainer.fit(model)


def test_sanity_metrics_are_reset(tmp_path):
    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            output = super().validation_step(batch, batch_idx)
            if self.trainer.sanity_checking:
                self.log("val_loss", output["x"], prog_bar=True, logger=True)
            return output

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)["loss"]
            if batch_idx == 0:
                assert self.trainer.progress_bar_metrics == {}
                assert self.trainer.logged_metrics == {}
                assert self.trainer.callback_metrics == {}
            self.log("train_loss", loss, prog_bar=True, logger=True)
            return loss

    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=1, limit_train_batches=1, limit_val_batches=2, num_sanity_val_steps=2
    )
    trainer.fit(TestModel())

    assert "val_loss" not in trainer.progress_bar_metrics


def test_on_epoch_logging_with_sum_and_on_batch_start(tmp_path):
    class TestModel(BoringModel):
        def on_train_epoch_end(self):
            self.log("on_train_epoch_end", 3.0, reduce_fx="mean")
            assert self.trainer._results["on_train_epoch_end.on_train_epoch_end"].value == 3.0
            assert all(v == 3 for v in self.trainer.callback_metrics.values())

        def on_validation_epoch_end(self):
            self.log("on_validation_epoch_end", 3.0, reduce_fx="mean")
            assert self.trainer._results["on_validation_epoch_end.on_validation_epoch_end"].value == 3.0
            assert all(v == 3 for v in self.trainer.callback_metrics.values())

        def on_train_batch_start(self, *_):
            self.log("on_train_batch_start", 1.0, on_step=False, on_epoch=True, reduce_fx="sum")

        def on_train_batch_end(self, *_):
            self.log("on_train_batch_end", 1.0, on_step=False, on_epoch=True, reduce_fx="sum")

        def on_validation_batch_start(self, *_):
            self.log("on_validation_batch_start", 1.0, reduce_fx="sum")

        def on_validation_batch_end(self, *_):
            self.log("on_validation_batch_end", 1.0, reduce_fx="sum")

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        limit_train_batches=3,
        limit_val_batches=3,
        num_sanity_val_steps=3,
        max_epochs=1,
    )
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)


@mock.patch("lightning.pytorch.loggers.TensorBoardLogger.log_metrics")
def test_log_metrics_epoch_step_values(mock_log_metrics, tmp_path):
    """Tests the default epoch and step values logged."""

    class MyModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("foo", 0.0, on_step=True, on_epoch=True)
            return super().training_step(batch, batch_idx)

    model = MyModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=0,
        max_epochs=2,
        log_every_n_steps=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=TensorBoardLogger(tmp_path),
    )
    trainer.fit(model)

    mock_log_metrics.assert_has_calls([
        call(metrics={"foo_step": 0.0, "epoch": 0}, step=0),
        call(metrics={"foo_step": 0.0, "epoch": 0}, step=1),
        call(metrics={"foo_epoch": 0.0, "epoch": 0}, step=1),
        call(metrics={"foo_step": 0.0, "epoch": 1}, step=2),
        call(metrics={"foo_step": 0.0, "epoch": 1}, step=3),
        call(metrics={"foo_epoch": 0.0, "epoch": 1}, step=3),
    ])


@mock.patch("lightning.pytorch.loggers.TensorBoardLogger.log_metrics")
def test_log_on_train_start(mock_log_metrics, tmp_path):
    """Tests that logged metrics on_train_start get reset after the first epoch."""

    class MyModel(BoringModel):
        def on_train_start(self):
            self.log("foo", 123)

    model = MyModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=0,
        max_epochs=2,
        log_every_n_steps=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=TensorBoardLogger(tmp_path),
    )
    trainer.fit(model)

    assert mock_log_metrics.mock_calls == [call(metrics={"foo": 123.0, "epoch": 0}, step=0)]
    assert trainer.max_epochs > 1


def test_unsqueezed_tensor_logging():
    model = BoringModel()
    trainer = Trainer()
    trainer.state.stage = RunningStage.TRAINING
    model._current_fx_name = "training_step"
    model.trainer = trainer
    model.log("foo", Tensor([1.2]), on_epoch=True)
    assert trainer.callback_metrics["foo"].ndim == 0
