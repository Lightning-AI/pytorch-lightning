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
"""Test logging in the evaluation loop."""
import collections
import itertools
from unittest import mock
from unittest.mock import call

import numpy as np
import pytest
import torch

from pytorch_lightning import callbacks, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel, RandomDataset
from tests.helpers.runif import RunIf


def test__validation_step__log(tmpdir):
    """Tests that validation_step can log."""

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            self.log("a", out["loss"], on_step=True, on_epoch=True)
            self.log("a2", 2)
            return out

        def validation_step(self, batch, batch_idx):
            out = super().validation_step(batch, batch_idx)
            self.log("b", out["x"], on_step=True, on_epoch=True)
            return out

    model = TestModel()
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)

    assert set(trainer.logged_metrics) == {"a2", "a_step", "a_epoch", "b_step", "b_epoch"}

    # we don't want to enable val metrics during steps because it is not something that users should do
    # on purpose DO NOT allow b_step... it's silly to monitor val step metrics
    assert set(trainer.callback_metrics) == {"a", "a2", "b", "a_epoch", "b_epoch", "a_step"}


def test__validation_step__epoch_end__log(tmpdir):
    """Tests that validation_epoch_end can log."""

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            self.log("a", out["loss"])
            self.log("b", out["loss"], on_step=True, on_epoch=True)
            return out

        def validation_step(self, batch, batch_idx):
            out = super().validation_step(batch, batch_idx)
            self.log("c", out["x"])
            self.log("d", out["x"], on_step=True, on_epoch=True)
            return out

        def validation_epoch_end(self, outputs):
            self.log("g", torch.tensor(2, device=self.device), on_epoch=True)

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)

    # make sure all the metrics are available for loggers
    assert set(trainer.logged_metrics) == {"a", "b_step", "b_epoch", "c", "d_step", "d_epoch", "g"}

    assert not trainer.progress_bar_metrics

    # we don't want to enable val metrics during steps because it is not something that users should do
    assert set(trainer.callback_metrics) == {"a", "b", "b_epoch", "c", "d", "d_epoch", "g", "b_step"}


@pytest.mark.parametrize(["batches", "log_interval", "max_epochs"], [(1, 1, 1), (64, 32, 2)])
def test_eval_epoch_logging(tmpdir, batches, log_interval, max_epochs):
    class TestModel(BoringModel):
        def validation_epoch_end(self, outputs):
            self.log("c", torch.tensor(2), on_epoch=True, prog_bar=True, logger=True)
            self.log("d/e/f", 2)

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=batches,
        limit_val_batches=batches,
        max_epochs=max_epochs,
        log_every_n_steps=log_interval,
        enable_model_summary=False,
    )
    trainer.fit(model)

    # assert the loggers received the expected number
    logged_metrics = set(trainer.logged_metrics)
    assert logged_metrics == {"c", "d/e/f"}

    pbar_metrics = set(trainer.progress_bar_metrics)
    assert pbar_metrics == {"c"}

    # make sure all the metrics are available for callbacks
    callback_metrics = set(trainer.callback_metrics)
    assert callback_metrics == (logged_metrics | pbar_metrics)


def test_eval_float_logging(tmpdir):
    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("a", 12.0)
            return {"x": loss}

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)

    assert set(trainer.logged_metrics) == {"a"}


def test_eval_logging_auto_reduce(tmpdir):
    class TestModel(BoringModel):
        val_losses = []
        manual_epoch_end_mean = None

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.val_losses.append(loss)
            self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
            return {"x": loss}

        def validation_epoch_end(self, outputs) -> None:
            for passed_in, manually_tracked in zip(outputs, self.val_losses):
                assert passed_in["x"] == manually_tracked
            self.manual_epoch_end_mean = torch.stack([x["x"] for x in outputs]).mean()

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=3,
        limit_val_batches=3,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model)

    # make sure all the metrics are available for callbacks
    assert set(trainer.callback_metrics) == {"val_loss", "val_loss_epoch"}

    # make sure values are correct
    assert trainer.logged_metrics["val_loss_epoch"] == model.manual_epoch_end_mean
    assert trainer.callback_metrics["val_loss_epoch"] == model.manual_epoch_end_mean
    assert trainer.callback_metrics["val_loss"] == model.manual_epoch_end_mean
    assert trainer.logged_metrics["val_loss_step"] == model.val_losses[-1]


@pytest.mark.parametrize(["batches", "log_interval", "max_epochs"], [(1, 1, 1), (64, 32, 2)])
def test_eval_epoch_only_logging(tmpdir, batches, log_interval, max_epochs):
    """Tests that test_epoch_end can be used to log, and we return them in the results."""

    class TestModel(BoringModel):
        def test_epoch_end(self, outputs):
            self.log("c", torch.tensor(2))
            self.log("d/e/f", 2)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=max_epochs,
        limit_test_batches=batches,
        log_every_n_steps=log_interval,
        enable_model_summary=False,
    )
    results = trainer.test(model)

    assert len(results) == 1
    assert results[0] == {"c": torch.tensor(2), "d/e/f": 2}


@pytest.mark.parametrize("suffix", (False, True))
def test_multi_dataloaders_add_suffix_properly(tmpdir, suffix):
    class TestModel(BoringModel):
        def test_step(self, batch, batch_idx, dataloader_idx=0):
            out = super().test_step(batch, batch_idx)
            self.log("test_loss", out["y"], on_step=True, on_epoch=True)
            return out

        def test_dataloader(self):
            if suffix:
                return [
                    torch.utils.data.DataLoader(RandomDataset(32, 64)),
                    torch.utils.data.DataLoader(RandomDataset(32, 64)),
                ]
            return super().test_dataloader()

    model = TestModel()
    model.test_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=0,
        limit_val_batches=0,
        limit_test_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    results = trainer.test(model)

    for i, r in enumerate(results):
        expected = {"test_loss", "test_loss_epoch"}
        if suffix:
            expected = {e + f"/dataloader_idx_{i}" for e in expected}
        assert set(r) == expected


def test_log_works_in_val_callback(tmpdir):
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

        def on_validation_start(self, _, pl_module):
            self.make_logging(
                pl_module, "on_validation_start", on_steps=[False], on_epochs=[True], prob_bars=self.choices
            )

        def on_epoch_start(self, trainer, pl_module):
            if trainer.validating:
                self.make_logging(
                    pl_module, "on_epoch_start", on_steps=[False], on_epochs=[True], prob_bars=self.choices
                )

        def on_validation_epoch_start(self, _, pl_module):
            self.make_logging(
                pl_module, "on_validation_epoch_start", on_steps=[False], on_epochs=[True], prob_bars=self.choices
            )

        def on_validation_batch_end(self, _, pl_module, *__):
            self.make_logging(
                pl_module,
                "on_validation_batch_end",
                on_steps=self.choices,
                on_epochs=self.choices,
                prob_bars=self.choices,
            )

        def on_epoch_end(self, trainer, pl_module):
            if trainer.validating:
                self.make_logging(pl_module, "on_epoch_end", on_steps=[False], on_epochs=[True], prob_bars=self.choices)

        def on_validation_epoch_end(self, _, pl_module):
            self.make_logging(
                pl_module, "on_validation_epoch_end", on_steps=[False], on_epochs=[True], prob_bars=self.choices
            )

    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            loss = super().validation_step(batch, batch_idx)["x"]
            self.log("val_loss", loss)

    model = TestModel()
    model.validation_epoch_end = None
    cb = TestCallback()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=4,
        num_sanity_val_steps=0,
        max_epochs=1,
        callbacks=[cb],
    )
    trainer.fit(model)

    assert cb.call_counter == {
        "on_validation_batch_end": 4,
        "on_validation_start": 1,
        "on_epoch_start": 1,
        "on_validation_epoch_start": 1,
        "on_validation_epoch_end": 1,
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


def test_log_works_in_test_callback(tmpdir):
    """Tests that log can be called within callback."""

    class TestCallback(callbacks.Callback):

        # helpers
        count = 0
        choices = [False, True]

        # used to compute expected values
        callback_funcs_called = collections.defaultdict(list)
        funcs_called_count = collections.defaultdict(int)
        funcs_attr = {}

        def make_logging(self, pl_module, func_name, on_steps, on_epochs, prob_bars):
            original_func_name = func_name[:]
            self.funcs_called_count[original_func_name] += 1

            for idx, (on_step, on_epoch, prog_bar) in enumerate(itertools.product(on_steps, on_epochs, prob_bars)):
                func_name = original_func_name[:]
                custom_func_name = f"{idx}_{func_name}"

                if not on_step and not on_epoch:
                    with pytest.raises(MisconfigurationException, match="is not useful"):
                        pl_module.log(custom_func_name, self.count, on_step=on_step, on_epoch=on_epoch)
                    continue
                pl_module.log(custom_func_name, self.count, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)

                num_dl_ext = ""
                if pl_module._current_dataloader_idx is not None:
                    dl_idx = pl_module._current_dataloader_idx
                    num_dl_ext = f"/dataloader_idx_{dl_idx}"
                    func_name += num_dl_ext

                # catch information for verification
                self.callback_funcs_called[func_name].append([self.count])
                self.funcs_attr[custom_func_name + num_dl_ext] = {
                    "on_step": on_step,
                    "on_epoch": on_epoch,
                    "prog_bar": prog_bar,
                    "func_name": func_name,
                }
                if on_step and on_epoch:
                    self.funcs_attr[f"{custom_func_name}_step" + num_dl_ext] = {
                        "on_step": True,
                        "on_epoch": False,
                        "prog_bar": prog_bar,
                        "func_name": func_name,
                    }

                    self.funcs_attr[f"{custom_func_name}_epoch" + num_dl_ext] = {
                        "on_step": False,
                        "on_epoch": True,
                        "prog_bar": prog_bar,
                        "func_name": func_name,
                    }

        def on_test_start(self, _, pl_module):
            self.make_logging(pl_module, "on_test_start", on_steps=[False], on_epochs=[True], prob_bars=self.choices)

        def on_epoch_start(self, trainer, pl_module):
            if trainer.testing:
                self.make_logging(
                    pl_module, "on_epoch_start", on_steps=[False], on_epochs=[True], prob_bars=self.choices
                )

        def on_test_epoch_start(self, _, pl_module):
            self.make_logging(
                pl_module, "on_test_epoch_start", on_steps=[False], on_epochs=[True], prob_bars=self.choices
            )

        def on_test_batch_end(self, _, pl_module, *__):
            self.make_logging(
                pl_module, "on_test_batch_end", on_steps=self.choices, on_epochs=self.choices, prob_bars=self.choices
            )

        def on_test_epoch_end(self, _, pl_module):
            self.make_logging(
                pl_module, "on_test_epoch_end", on_steps=[False], on_epochs=[True], prob_bars=self.choices
            )

    num_dataloaders = 2

    class TestModel(BoringModel):
        seen_losses = {i: [] for i in range(num_dataloaders)}

        def test_step(self, batch, batch_idx, dataloader_idx=None):
            loss = super().test_step(batch, batch_idx)["y"]
            self.log("test_loss", loss)
            self.seen_losses[dataloader_idx].append(loss)

        def test_dataloader(self):
            return [torch.utils.data.DataLoader(RandomDataset(32, 64)) for _ in range(num_dataloaders)]

    model = TestModel()
    model.test_epoch_end = None
    cb = TestCallback()
    trainer = Trainer(
        default_root_dir=tmpdir, limit_test_batches=2, num_sanity_val_steps=0, max_epochs=2, callbacks=[cb]
    )
    trainer.test(model)

    assert cb.funcs_called_count["on_test_start"] == 1
    assert cb.funcs_called_count["on_test_epoch_start"] == 1
    assert cb.funcs_called_count["on_test_batch_end"] == 4
    assert cb.funcs_called_count["on_test_epoch_end"] == 1

    callback_metrics_keys = list(trainer.callback_metrics)
    for func_name in cb.callback_funcs_called.keys():
        is_in = False
        for callback_metrics_key in callback_metrics_keys:
            if func_name in callback_metrics_key:
                is_in = True
        assert is_in, (func_name, callback_metrics_keys)

    def get_expected(on_epoch, values):
        reduction = np.mean if on_epoch else np.max
        return reduction(values)

    # Make sure the func_name output equals the average from all logged values when on_epoch true
    for dl_idx in range(num_dataloaders):
        key = f"test_loss/dataloader_idx_{dl_idx}"
        assert key in trainer.callback_metrics
        assert torch.stack(model.seen_losses[dl_idx]).mean() == trainer.callback_metrics.pop(key)

    for func_name, output_value in trainer.callback_metrics.items():
        output_value = output_value.item()
        func_attr = cb.funcs_attr[func_name]
        original_values = cb.callback_funcs_called[func_attr["func_name"]]
        expected_output = get_expected(func_attr["on_epoch"], original_values)
        assert output_value == expected_output

    for fx, attrs in cb.funcs_attr.items():
        should_include = attrs["prog_bar"] and attrs["on_step"] ^ attrs["on_epoch"]
        is_included = fx in trainer.progress_bar_metrics
        assert is_included if should_include else not is_included


@mock.patch("pytorch_lightning.loggers.TensorBoardLogger.log_metrics")
def test_validation_step_log_with_tensorboard(mock_log_metrics, tmpdir):
    """This tests make sure we properly log_metrics to loggers."""

    class ExtendedModel(BoringModel):

        val_losses = []

        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("train_loss", loss)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.val_losses.append(loss)
            self.log("valid_loss_0", loss, on_step=True, on_epoch=True)
            self.log("valid_loss_1", loss, on_step=False, on_epoch=True)
            self.log("valid_loss_2", loss, on_step=True, on_epoch=False)
            return {"val_loss": loss}  # not added to callback_metrics

        def test_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("test_loss", loss)
            return {"y": loss}

    model = ExtendedModel()
    model.validation_epoch_end = None

    # Initialize a trainer
    trainer = Trainer(
        default_root_dir=tmpdir,
        logger=TensorBoardLogger(tmpdir),
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=2,
    )

    # Train the model ⚡
    trainer.fit(model)

    # hp_metric + 2 steps + epoch + 2 steps + epoch
    expected_num_calls = 1 + 2 + 1 + 2 + 1

    assert set(trainer.callback_metrics) == {
        "train_loss",
        "valid_loss_0_epoch",
        "valid_loss_0",
        "valid_loss_1",
    }
    assert len(mock_log_metrics.mock_calls) == expected_num_calls
    assert mock_log_metrics.mock_calls[0] == call({"hp_metric": -1}, 0)

    def get_metrics_at_idx(idx):
        mock_calls = list(mock_log_metrics.mock_calls)
        if isinstance(mock_calls[idx].kwargs, dict):
            return mock_calls[idx].kwargs["metrics"]
        return mock_calls[idx][2]["metrics"]

    expected = {"valid_loss_0_step", "valid_loss_2"}
    assert set(get_metrics_at_idx(1)) == expected
    assert set(get_metrics_at_idx(2)) == expected

    assert get_metrics_at_idx(1)["valid_loss_0_step"] == model.val_losses[2]
    assert get_metrics_at_idx(2)["valid_loss_0_step"] == model.val_losses[3]

    assert set(get_metrics_at_idx(3)) == {"valid_loss_0_epoch", "valid_loss_1", "epoch"}

    assert get_metrics_at_idx(3)["valid_loss_1"] == torch.stack(model.val_losses[2:4]).mean()

    expected = {"valid_loss_0_step", "valid_loss_2"}
    assert set(get_metrics_at_idx(4)) == expected
    assert set(get_metrics_at_idx(5)) == expected

    assert get_metrics_at_idx(4)["valid_loss_0_step"] == model.val_losses[4]
    assert get_metrics_at_idx(5)["valid_loss_0_step"] == model.val_losses[5]

    assert set(get_metrics_at_idx(6)) == {"valid_loss_0_epoch", "valid_loss_1", "epoch"}

    assert get_metrics_at_idx(6)["valid_loss_1"] == torch.stack(model.val_losses[4:]).mean()

    results = trainer.test(model)
    assert set(trainer.callback_metrics) == {
        "test_loss",
    }
    assert set(results[0]) == {"test_loss"}


def test_logging_dict_on_validation_step(tmpdir):
    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            loss = super().validation_step(batch, batch_idx)
            loss = loss["x"]
            metrics = {
                "loss": loss,
                "loss_1": loss,
            }
            self.log("val_metrics", metrics)

        validation_epoch_end = None

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
    )

    trainer.fit(model)


@pytest.mark.parametrize("val_check_interval", [0.5, 1.0])
def test_multiple_dataloaders_reset(val_check_interval, tmpdir):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            value = 1 + batch_idx
            if self.current_epoch != 0:
                value *= 10
            self.log("batch_idx", value, on_step=True, on_epoch=True, prog_bar=True)
            return out

        def training_epoch_end(self, outputs):
            metrics = self.trainer.progress_bar_metrics
            v = 15 if self.current_epoch == 0 else 150
            assert metrics["batch_idx_epoch"] == (v / 5.0)

        def validation_step(self, batch, batch_idx, dataloader_idx):
            value = (1 + batch_idx) * (1 + dataloader_idx)
            if self.current_epoch != 0:
                value *= 10
            self.log("val_loss", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return value

        def validation_epoch_end(self, outputs):
            if self.current_epoch == 0:
                assert sum(outputs[0]) / 5 == 3
                assert sum(outputs[1]) / 5 == 6
            else:
                assert sum(outputs[0]) / 5 == 30
                assert sum(outputs[1]) / 5 == 60

            tot_loss = torch.mean(torch.tensor(outputs, dtype=torch.float))
            if self.current_epoch == 0:
                assert tot_loss == (3 + 6) / 2
            else:
                assert tot_loss == (30 + 60) / 2
            assert self.trainer._results["validation_step.val_loss.0"].cumulated_batch_size == 5
            assert self.trainer._results["validation_step.val_loss.1"].cumulated_batch_size == 5

        def val_dataloader(self):
            return [super().val_dataloader(), super().val_dataloader()]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=5,
        limit_val_batches=5,
        num_sanity_val_steps=0,
        val_check_interval=val_check_interval,
        max_epochs=3,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)


@pytest.mark.parametrize(
    ["kwargs", "expected"],
    [
        ({"dl_idx": 0, "metrics": {"acc": 123}}, {"acc": 123}),
        (
            {"dl_idx": 0, "metrics": {"acc/dataloader_idx_0": 123, "acc/dataloader_idx_1": 321}},
            {"acc/dataloader_idx_0": 123},
        ),
        (
            {"dl_idx": 10, "metrics": {"acc/dataloader_idx_1": 123, "acc/dataloader_idx_10": 321}},
            {"acc/dataloader_idx_10": 321},
        ),
        (
            {"dl_idx": 3, "metrics": {"top_3_acc/dataloader_idx_0": 123, "top_3_acc/dataloader_idx_3": 321}},
            {"top_3_acc/dataloader_idx_3": 321},
        ),
        # theoretical case, as `/dataloader_idx_3` would have been added
        ({"dl_idx": 3, "metrics": {"top_3_acc": 123}}, {"top_3_acc": 123}),
    ],
)
def test_filter_metrics_for_dataloader(kwargs, expected):
    """Logged metrics should only include metrics from the concerned dataloader."""
    actual = LoggerConnector._filter_metrics_for_dataloader(**kwargs)
    assert actual == expected


@RunIf(min_gpus=1)
def test_evaluation_move_metrics_to_cpu_and_outputs(tmpdir):
    class TestModel(BoringModel):
        def validation_step(self, *args):
            x = torch.tensor(2.0, requires_grad=True, device=self.device)
            y = x * 2
            assert x.requires_grad is True
            assert y.grad_fn is None  # disabled by validation

            self.log("foo", y)
            return y

        def validation_epoch_end(self, outputs):
            # the step outputs were not moved
            assert all(o.device == self.device for o in outputs), outputs
            # but the logging results were
            assert self.trainer.callback_metrics["foo"].device.type == "cpu"

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, limit_val_batches=2, move_metrics_to_cpu=True, gpus=1)
    trainer.validate(model, verbose=False)
