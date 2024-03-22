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
"""Test logging in the evaluation loop."""

import collections
import itertools
import os
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from unittest.mock import ANY, call

import numpy as np
import pytest
import torch
from lightning.fabric.utilities.imports import _PYTHON_GREATER_EQUAL_3_8_0
from lightning.pytorch import Trainer, callbacks
from lightning.pytorch.callbacks.progress.rich_progress import _RICH_AVAILABLE
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loops import _EvaluationLoop
from lightning.pytorch.trainer.states import RunningStage
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch import Tensor

from tests_pytorch.helpers.runif import RunIf

if _RICH_AVAILABLE:
    from rich import get_console


def test__validation_step__log(tmp_path):
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
    trainer = Trainer(
        default_root_dir=tmp_path,
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
    assert all(isinstance(v, Tensor) for v in trainer.callback_metrics.values())
    assert all(isinstance(v, Tensor) for v in trainer.logged_metrics.values())
    assert all(isinstance(v, float) for v in trainer.progress_bar_metrics.values())


def test__validation_step__epoch_end__log(tmp_path):
    """Tests that on_validation_epoch_end can log."""

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

        def on_validation_epoch_end(self):
            self.log("g", torch.tensor(2, device=self.device), on_epoch=True)

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmp_path,
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
    assert all(isinstance(v, Tensor) for v in trainer.callback_metrics.values())
    assert all(isinstance(v, Tensor) for v in trainer.logged_metrics.values())
    assert all(isinstance(v, float) for v in trainer.progress_bar_metrics.values())


@pytest.mark.parametrize(("batches", "log_interval", "max_epochs"), [(1, 1, 1), (64, 32, 2)])
def test_eval_epoch_logging(tmp_path, batches, log_interval, max_epochs):
    class TestModel(BoringModel):
        def on_validation_epoch_end(self):
            self.log("c", torch.tensor(2), on_epoch=True, prog_bar=True, logger=True)
            self.log("d/e/f", 2)

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmp_path,
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
    assert all(isinstance(v, Tensor) for v in trainer.callback_metrics.values())
    assert all(isinstance(v, Tensor) for v in trainer.logged_metrics.values())
    assert all(isinstance(v, float) for v in trainer.progress_bar_metrics.values())


def test_eval_float_logging(tmp_path):
    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("a", 12.0)
            return {"x": loss}

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)

    assert set(trainer.logged_metrics) == {"a"}


def test_eval_logging_auto_reduce(tmp_path):
    class TestModel(BoringModel):
        val_losses = []
        manual_epoch_end_mean = None

        def validation_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.val_losses.append(loss)
            self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
            return {"x": loss}

        def on_validation_epoch_end(self) -> None:
            self.manual_epoch_end_mean = torch.stack(self.val_losses).mean()

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
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
    assert torch.allclose(trainer.logged_metrics["val_loss_epoch"], model.manual_epoch_end_mean)
    assert torch.allclose(trainer.callback_metrics["val_loss_epoch"], model.manual_epoch_end_mean)
    assert torch.allclose(trainer.callback_metrics["val_loss"], model.manual_epoch_end_mean)
    assert trainer.logged_metrics["val_loss_step"] == model.val_losses[-1]


@pytest.mark.parametrize(("batches", "log_interval", "max_epochs"), [(1, 1, 1), (64, 32, 2)])
def test_eval_epoch_only_logging(tmp_path, batches, log_interval, max_epochs):
    """Tests that on_test_epoch_end can be used to log, and we return them in the results."""

    class TestModel(BoringModel):
        def on_test_epoch_end(self):
            self.log("c", torch.tensor(2))
            self.log("d/e/f", 2)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=max_epochs,
        limit_test_batches=batches,
        log_every_n_steps=log_interval,
        enable_model_summary=False,
    )
    results = trainer.test(model)

    assert len(results) == 1
    assert results[0] == {"c": torch.tensor(2), "d/e/f": 2}


@pytest.mark.parametrize("suffix", [False, True])
def test_multi_dataloaders_add_suffix_properly(tmp_path, suffix):
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

    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=0,
        limit_val_batches=0,
        limit_test_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    results = trainer.test(model)

    for i, r in enumerate(results):
        expected = {"test_loss_epoch"}
        if suffix:
            expected = {e + f"/dataloader_idx_{i}" for e in expected}
        assert set(r) == expected


def test_log_works_in_val_callback(tmp_path):
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

        def on_validation_epoch_end(self, _, pl_module):
            self.make_logging(
                pl_module, "on_validation_epoch_end", on_steps=[False], on_epochs=[True], prob_bars=self.choices
            )

    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            loss = super().validation_step(batch, batch_idx)["x"]
            self.log("val_loss", loss)

    model = TestModel()
    cb = TestCallback()
    trainer = Trainer(
        default_root_dir=tmp_path,
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
        "on_validation_epoch_start": 1,
        "on_validation_epoch_end": 1,
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


def test_log_works_in_test_callback(tmp_path):
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
                dl_idx = pl_module.trainer._results.dataloader_idx
                if dl_idx is not None:
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

        def test_step(self, batch, batch_idx, dataloader_idx=0):
            loss = super().test_step(batch, batch_idx)["y"]
            self.log("test_loss", loss)
            self.seen_losses[dataloader_idx].append(loss)

        def test_dataloader(self):
            return [torch.utils.data.DataLoader(RandomDataset(32, 64)) for _ in range(num_dataloaders)]

    model = TestModel()
    cb = TestCallback()
    trainer = Trainer(
        default_root_dir=tmp_path, limit_test_batches=2, num_sanity_val_steps=0, max_epochs=2, callbacks=[cb]
    )
    trainer.test(model)

    assert cb.funcs_called_count["on_test_start"] == 1
    assert cb.funcs_called_count["on_test_epoch_start"] == 1
    assert cb.funcs_called_count["on_test_batch_end"] == 4
    assert cb.funcs_called_count["on_test_epoch_end"] == 1

    callback_metrics = trainer.callback_metrics
    for func_name in cb.callback_funcs_called:
        for key in callback_metrics:
            if func_name in key:
                break
        else:
            pytest.fail(f"{func_name}, {list(callback_metrics)}")

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


@mock.patch("lightning.pytorch.loggers.TensorBoardLogger.log_metrics")
def test_validation_step_log_with_tensorboard(mock_log_metrics, tmp_path):
    """This tests make sure we properly log_metrics to loggers."""

    class ExtendedModel(BoringModel):
        val_losses = []

        def __init__(self, some_val=7):
            super().__init__()
            self.save_hyperparameters()

        def training_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("train_loss", loss)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.val_losses.append(loss)
            self.log("valid_loss_0", loss, on_step=True, on_epoch=True)
            self.log("valid_loss_1", loss, on_step=False, on_epoch=True)
            self.log("valid_loss_2", loss, on_step=True, on_epoch=False)
            return {"val_loss": loss}  # not added to callback_metrics

        def test_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("test_loss", loss)
            return {"y": loss}

    model = ExtendedModel()

    # Initialize a trainer
    trainer = Trainer(
        default_root_dir=tmp_path,
        logger=TensorBoardLogger(tmp_path),
        limit_train_batches=1,
        limit_val_batches=2,
        limit_test_batches=2,
        log_every_n_steps=1,
        max_epochs=2,
    )

    # Train the model ⚡
    trainer.fit(model)

    assert set(trainer.callback_metrics) == {
        "train_loss",
        "valid_loss_0_epoch",
        "valid_loss_0",
        "valid_loss_1",
    }
    assert mock_log_metrics.mock_calls == [
        call({"hp_metric": -1}, 0),
        call(metrics={"train_loss": ANY, "epoch": 0}, step=0),
        call(metrics={"valid_loss_0_step": ANY, "valid_loss_2": ANY}, step=0),
        call(metrics={"valid_loss_0_step": ANY, "valid_loss_2": ANY}, step=1),
        call(metrics={"valid_loss_0_epoch": ANY, "valid_loss_1": ANY, "epoch": 0}, step=0),
        call(metrics={"train_loss": ANY, "epoch": 1}, step=1),
        call(metrics={"valid_loss_0_step": ANY, "valid_loss_2": ANY}, step=2),
        call(metrics={"valid_loss_0_step": ANY, "valid_loss_2": ANY}, step=3),
        call(metrics={"valid_loss_0_epoch": ANY, "valid_loss_1": ANY, "epoch": 1}, step=1),
    ]

    def get_metrics_at_idx(idx):
        mock_call = mock_log_metrics.mock_calls[idx]
        return mock_call.kwargs["metrics"] if _PYTHON_GREATER_EQUAL_3_8_0 else mock_call[2]["metrics"]

    assert get_metrics_at_idx(2)["valid_loss_0_step"] == model.val_losses[2]
    assert get_metrics_at_idx(3)["valid_loss_0_step"] == model.val_losses[3]
    assert get_metrics_at_idx(4)["valid_loss_1"] == torch.stack(model.val_losses[2:4]).mean()
    assert get_metrics_at_idx(6)["valid_loss_0_step"] == model.val_losses[4]
    assert get_metrics_at_idx(7)["valid_loss_0_step"] == model.val_losses[5]
    assert get_metrics_at_idx(8)["valid_loss_1"] == torch.stack(model.val_losses[4:]).mean()

    results = trainer.test(model)
    assert set(trainer.callback_metrics) == {"test_loss"}
    assert set(results[0]) == {"test_loss"}


@pytest.mark.parametrize("val_check_interval", [0.5, 1.0])
def test_multiple_dataloaders_reset(val_check_interval, tmp_path):
    class TestModel(BoringModel):
        val_outputs = [[], []]

        def training_step(self, batch, batch_idx):
            out = super().training_step(batch, batch_idx)
            value = 1 + batch_idx
            if self.current_epoch != 0:
                value *= 10
            self.log("batch_idx", value, on_step=True, on_epoch=True, prog_bar=True)
            return out

        def on_train_epoch_end(self):
            metrics = self.trainer.progress_bar_metrics
            v = 15 if self.current_epoch == 0 else 150
            assert metrics["batch_idx_epoch"] == (v / 5.0)

        def validation_step(self, batch, batch_idx, dataloader_idx):
            value = (1 + batch_idx) * (1 + dataloader_idx)
            if self.current_epoch != 0:
                value *= 10
            self.val_outputs[dataloader_idx].append(value)
            self.log("val_loss", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        def on_validation_epoch_end(self):
            outputs = self.val_outputs
            self.val_outputs = [[], []]

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
        default_root_dir=tmp_path,
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
    "accelerator",
    [
        pytest.param("cuda", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps", marks=RunIf(mps=True)),
    ],
)
def test_metrics_and_outputs_device(tmp_path, accelerator):
    class TestModel(BoringModel):
        outputs = []

        def on_before_backward(self, loss: Tensor) -> None:
            # the loss should be on the correct device before backward
            assert loss.device.type == accelerator

        def validation_step(self, *args):
            x = torch.tensor(2.0, requires_grad=True, device=self.device)
            y = x * 2
            assert x.requires_grad is True
            assert y.grad_fn is None  # disabled by validation
            self.log("foo", y)
            self.outputs.append(y)
            return y

        def on_validation_epoch_end(self):
            # the step outputs were not moved after returning them
            assert all(o.device == self.device for o in self.outputs)
            # and the logged metrics aren't
            assert self.trainer.callback_metrics["foo"].device.type == accelerator

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True, accelerator=accelerator, devices=1)
    trainer.fit(model)
    trainer.validate(model, verbose=False)


def test_logging_results_with_no_dataloader_idx(tmp_path):
    num_dataloaders = 2
    log_common_same_val = {"test_log_common": 789}
    log_common_diff_val = "test_log_common_diff_value"
    log_key_no_dl_idx = "test_log_no_dl_idx_{}"
    log_key_dl0 = {"test_log_a_class": 123}
    log_key_dl1 = {"test_log_b_class": 456}

    class CustomBoringModel(BoringModel):
        def test_step(self, batch, batch_idx, dataloader_idx):
            self.log_dict(log_common_same_val)
            self.log(log_common_diff_val, dataloader_idx + 1)
            self.log(
                log_key_no_dl_idx.format(dataloader_idx),
                321 * (dataloader_idx + 1),
                add_dataloader_idx=False,
            )
            self.log_dict(log_key_dl0 if dataloader_idx == 0 else log_key_dl1, add_dataloader_idx=False)

        def test_dataloader(self):
            return [torch.utils.data.DataLoader(RandomDataset(32, 64)) for _ in range(num_dataloaders)]

    model = CustomBoringModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)
    results = trainer.test(model)

    assert len(results) == num_dataloaders
    assert results[0] == {
        "test_log_common/dataloader_idx_0": 789.0,
        "test_log_common_diff_value/dataloader_idx_0": 1.0,
        "test_log_no_dl_idx_0": 321,
        "test_log_a_class": 123.0,
    }
    assert results[1] == {
        "test_log_common/dataloader_idx_1": 789.0,
        "test_log_common_diff_value/dataloader_idx_1": 2.0,
        "test_log_no_dl_idx_1": 321 * 2,
        "test_log_b_class": 456.0,
    }


@mock.patch("lightning.pytorch.loggers.TensorBoardLogger.log_metrics")
def test_logging_multi_dataloader_on_epoch_end(mock_log_metrics, tmp_path):
    class CustomBoringModel(BoringModel):
        outputs = [[], []]

        def test_step(self, batch, batch_idx, dataloader_idx):
            value = dataloader_idx + 1
            self.log("foo", value)
            self.outputs[dataloader_idx].append(value)
            return value

        def on_test_epoch_end(self):
            self.log("foobar", sum(sum(o) for o in self.outputs))

        def test_dataloader(self):
            return [super().test_dataloader(), super().test_dataloader()]

    model = CustomBoringModel()
    trainer = Trainer(default_root_dir=tmp_path, limit_test_batches=1, logger=TensorBoardLogger(tmp_path))
    results = trainer.test(model)

    # what's logged in `on_test_epoch_end` gets included in the results of each dataloader
    assert results == [{"foo/dataloader_idx_0": 1, "foobar": 3}, {"foo/dataloader_idx_1": 2, "foobar": 3}]
    cb_metrics = set(trainer.callback_metrics)
    assert cb_metrics == {"foo/dataloader_idx_0", "foo/dataloader_idx_1", "foobar"}

    mock_call = mock_log_metrics.mock_calls[0]
    logged_metrics = mock_call.kwargs["metrics"] if _PYTHON_GREATER_EQUAL_3_8_0 else mock_call[2]["metrics"]
    cb_metrics.add("epoch")
    assert set(logged_metrics) == cb_metrics


inputs0 = ([{"log": torch.tensor(5)}, {"no_log": torch.tensor(6)}], RunningStage.TESTING.value)
expected0 = """
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0             DataLoader 1
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           log                       5
         no_log                                               6
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""

inputs1 = (
    [
        {
            "value": torch.tensor(2),
            "performance": {"log:1": torch.tensor(0), "log2": torch.tensor(3), "log3": torch.tensor(7)},
            "extra": {"log3": torch.tensor(7)},
        },
        {"different value": torch.tensor(1.5), "tes:t": {"no_log1": torch.tensor(6), "no_log2": torch.tensor(1)}},
    ],
    RunningStage.TESTING.value,
)
expected1 = """
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0             DataLoader 1
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     different value                                         1.5
       extra:log3                    7
    performance:log2                 3
    performance:log3                 7
    performance:log:1                0
      tes:t:no_log1                                           6
      tes:t:no_log2                                           1
          value                      2
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""

inputs2 = (
    [
        {f"a {'really ' * 11}long metric name": torch.tensor(5)},
        {f"a {'really ' * 11}long metric name": torch.tensor([[6]])},
    ],
    RunningStage.VALIDATING.value,
)
expected2 = """
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                      Validate metric                                               DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
a really really really really really really really really re                             5
            ally really really long metric name
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                      Validate metric                                               DataLoader 1
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
a really really really really really really really really re                             6
            ally really really long metric name
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""

inputs3 = ([{f"log/dataloader_idx_{i}": torch.tensor(5)} for i in range(5)], "foobar")
expected3 = """
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      Foobar metric            DataLoader 0             DataLoader 1             DataLoader 2
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           log                       5                        5                        5
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      Foobar metric            DataLoader 3             DataLoader 4
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           log                       5                        5
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""

inputs4 = ([{}], "foo")
expected4 = ""


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param(inputs0, expected0, id="case0"),
        pytest.param(inputs1, expected1, id="case1"),
        pytest.param(inputs2, expected2, id="case2"),
        pytest.param(inputs3, expected3, id="case3"),
        pytest.param(inputs4, expected4, id="empty case"),
    ],
)
def test_native_print_results(monkeypatch, inputs, expected):
    import lightning.pytorch.loops.evaluation_loop as imports

    monkeypatch.setattr(imports, "_RICH_AVAILABLE", False)

    with redirect_stdout(StringIO()) as out:
        _EvaluationLoop._print_results(*inputs)
    expected = expected[1:]  # remove the initial line break from the """ string
    assert out.getvalue().replace(os.linesep, "\n") == expected.lstrip()


@pytest.mark.parametrize("encoding", ["latin-1", "utf-8"])
def test_native_print_results_encodings(monkeypatch, encoding):
    import lightning.pytorch.loops.evaluation_loop as imports

    monkeypatch.setattr(imports, "_RICH_AVAILABLE", False)

    out = mock.Mock()
    out.encoding = encoding
    with redirect_stdout(out) as out:
        _EvaluationLoop._print_results(*inputs0)

    # Attempt to encode everything the file is told to write with the given encoding
    for call_ in out.method_calls:
        name, args, _ = call_
        if name == "write":
            args[0].encode(encoding)


expected0 = """
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃       Test metric       ┃      DataLoader 0       ┃       DataLoader 1       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│           log           │            5            │                          │
│         no_log          │                         │            6             │
└─────────────────────────┴─────────────────────────┴──────────────────────────┘
"""

expected1 = """
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃       Test metric       ┃      DataLoader 0       ┃       DataLoader 1       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│     different value     │                         │           1.5            │
│       extra:log3        │            7            │                          │
│    performance:log2     │            3            │                          │
│    performance:log3     │            7            │                          │
│    performance:log:1    │            0            │                          │
│      tes:t:no_log1      │                         │            6             │
│      tes:t:no_log2      │                         │            1             │
│          value          │            2            │                          │
└─────────────────────────┴─────────────────────────┴──────────────────────────┘
"""

expected2 = """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Validate metric            ┃             DataLoader 0              ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ a really really really really really │                   5                   │
│  really really really really really  │                                       │
│       really long metric name        │                                       │
└──────────────────────────────────────┴───────────────────────────────────────┘
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           Validate metric            ┃             DataLoader 1              ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ a really really really really really │                   6                   │
│  really really really really really  │                                       │
│       really long metric name        │                                       │
└──────────────────────────────────────┴───────────────────────────────────────┘
"""

expected3 = """
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃   Foobar metric   ┃   DataLoader 0    ┃   DataLoader 1    ┃   DataLoader 2   ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│        log        │         5         │         5         │        5         │
└───────────────────┴───────────────────┴───────────────────┴──────────────────┘
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      Foobar metric      ┃      DataLoader 3       ┃       DataLoader 4       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│           log           │            5            │            5             │
└─────────────────────────┴─────────────────────────┴──────────────────────────┘
"""


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        pytest.param(inputs0, expected0, id="case0"),
        pytest.param(inputs1, expected1, id="case1"),
        pytest.param(inputs2, expected2, id="case2"),
        pytest.param(inputs3, expected3, id="case3"),
        pytest.param(inputs4, expected4, id="empty case"),
    ],
)
@RunIf(skip_windows=True, rich=True)
def test_rich_print_results(inputs, expected):
    console = get_console()
    with console.capture() as capture:
        _EvaluationLoop._print_results(*inputs)
    expected = expected[1:]  # remove the initial line break from the """ string
    assert capture.get() == expected.lstrip()


@mock.patch("lightning.pytorch.loggers.TensorBoardLogger.log_metrics")
@pytest.mark.parametrize("num_dataloaders", [1, 2])
def test_eval_step_logging(mock_log_metrics, tmp_path, num_dataloaders):
    """Test that eval step during fit/validate/test is updated correctly."""

    class CustomBoringModel(BoringModel):
        def validation_step(self, batch, batch_idx, dataloader_idx=None):
            self.log(f"val_log_{self.trainer.state.fn.value}", batch_idx, on_step=True, on_epoch=False)

        def test_step(self, batch, batch_idx, dataloader_idx=None):
            self.log("test_log", batch_idx, on_step=True, on_epoch=False)

        def val_dataloader(self):
            return [super().val_dataloader()] * num_dataloaders

        def test_dataloader(self):
            return [super().test_dataloader()] * num_dataloaders

    limit_batches = 4
    max_epochs = 3
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=max_epochs,
        limit_train_batches=1,
        limit_val_batches=limit_batches,
        limit_test_batches=limit_batches,
        logger=TensorBoardLogger(tmp_path),
    )
    model = CustomBoringModel()

    def get_suffix(dl_idx):
        return f"/dataloader_idx_{dl_idx}" if num_dataloaders == 2 else ""

    eval_steps = range(limit_batches)
    trainer.fit(model)
    fit_calls = [
        call(metrics={f"val_log_fit{get_suffix(dl_idx)}": float(step)}, step=step + (limit_batches * epoch))
        for epoch in range(max_epochs)
        for dl_idx in range(num_dataloaders)
        for step in eval_steps
    ]
    assert mock_log_metrics.mock_calls == fit_calls

    mock_log_metrics.reset_mock()
    trainer.validate(model)
    validate_calls = [
        call(metrics={f"val_log_validate{get_suffix(dl_idx)}": float(val)}, step=val)
        for dl_idx in range(num_dataloaders)
        for val in eval_steps
    ]
    assert mock_log_metrics.mock_calls == validate_calls

    mock_log_metrics.reset_mock()
    trainer.test(model)
    test_calls = [
        call(metrics={f"test_log{get_suffix(dl_idx)}": float(val)}, step=val)
        for dl_idx in range(num_dataloaders)
        for val in eval_steps
    ]
    assert mock_log_metrics.mock_calls == test_calls
