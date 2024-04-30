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
import math
import os
import pickle
import sys
from collections import defaultdict
from typing import Union
from unittest import mock
from unittest.mock import ANY, PropertyMock, call

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar, TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.logger import DummyLogger
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf
from torch.utils.data.dataloader import DataLoader


class MockTqdm(Tqdm):
    def __init__(self, *args, **kwargs):
        self.n_values = []
        self.total_values = []
        self.descriptions = []
        super().__init__(*args, **kwargs)
        self.__n = 0
        self.__total = 0
        # again to reset additions from `super().__init__`
        self.n_values = []
        self.total_values = []
        self.descriptions = []

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        self.__n = value

        # track the changes in the `n` value
        if not len(self.n_values) or value != self.n_values[-1]:
            self.n_values.append(value)

    @property
    def total(self):
        return self.__total

    @total.setter
    def total(self, value):
        self.__total = value
        self.total_values.append(value)

    def set_description(self, *args, **kwargs):
        super().set_description(*args, **kwargs)
        self.descriptions.append(self.desc)


@pytest.mark.parametrize(
    "pbar",
    [
        # won't print but is still set
        TQDMProgressBar(refresh_rate=0),
        TQDMProgressBar(),
    ],
)
def test_tqdm_progress_bar_on(tmp_path, pbar):
    """Test different ways the progress bar can be turned on."""
    trainer = Trainer(default_root_dir=tmp_path, callbacks=pbar)

    progress_bars = [c for c in trainer.callbacks if isinstance(c, ProgressBar)]
    assert len(progress_bars) == 1
    assert progress_bars[0] is trainer.progress_bar_callback


def test_tqdm_progress_bar_off(tmp_path):
    """Test turning the progress bar off."""
    trainer = Trainer(default_root_dir=tmp_path, enable_progress_bar=False)
    progress_bars = [c for c in trainer.callbacks if isinstance(c, ProgressBar)]
    assert not len(progress_bars)


def test_tqdm_progress_bar_misconfiguration():
    """Test that Trainer doesn't accept multiple progress bars."""
    # Trainer supports only a single progress bar callback at the moment
    callbacks = [TQDMProgressBar(), TQDMProgressBar(), ModelCheckpoint(dirpath="../trainer")]
    with pytest.raises(MisconfigurationException, match=r"^You added multiple progress bar callbacks"):
        Trainer(callbacks=callbacks)

    with pytest.raises(MisconfigurationException, match=r"enable_progress_bar=False` but found `TQDMProgressBar"):
        Trainer(callbacks=TQDMProgressBar(), enable_progress_bar=False)


@pytest.mark.parametrize("num_dl", [1, 2])
def test_tqdm_progress_bar_totals(tmp_path, num_dl):
    """Test that the progress finishes with the correct total steps processed."""

    class CustomModel(BoringModel):
        def _get_dataloaders(self):
            dls = [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]
            return dls[0] if num_dl == 1 else dls

        def val_dataloader(self):
            return self._get_dataloaders()

        def test_dataloader(self):
            return self._get_dataloaders()

        def predict_dataloader(self):
            return self._get_dataloaders()

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            return

        def test_step(self, batch, batch_idx, dataloader_idx=0):
            return

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            return

    model = CustomModel()

    # check the sanity dataloaders
    num_sanity_val_steps = 4
    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=1, limit_train_batches=0, num_sanity_val_steps=num_sanity_val_steps
    )
    pbar = trainer.progress_bar_callback
    with mock.patch("lightning.pytorch.callbacks.progress.tqdm_progress.Tqdm", MockTqdm):
        trainer.fit(model)

    expected_sanity_steps = [num_sanity_val_steps] * num_dl
    assert not pbar.val_progress_bar.leave
    assert trainer.num_sanity_val_batches == expected_sanity_steps
    assert pbar.val_progress_bar.total_values == expected_sanity_steps
    assert pbar.val_progress_bar.n_values == list(range(num_sanity_val_steps + 1)) * num_dl
    assert pbar.val_progress_bar.descriptions == [f"Sanity Checking DataLoader {i}: " for i in range(num_dl)]

    # fit
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    pbar = trainer.progress_bar_callback
    with mock.patch("lightning.pytorch.callbacks.progress.tqdm_progress.Tqdm", MockTqdm):
        trainer.fit(model)

    n = trainer.num_training_batches
    m = trainer.num_val_batches
    assert len(trainer.train_dataloader) == n
    # train progress bar should have reached the end
    assert pbar.train_progress_bar.total == n
    assert pbar.train_progress_bar.n == n
    assert pbar.train_progress_bar.leave

    # check val progress bar total
    assert pbar.val_progress_bar.total_values == m
    assert pbar.val_progress_bar.n_values == list(range(m[0] + 1)) * num_dl
    assert pbar.val_progress_bar.descriptions == [f"Validation DataLoader {i}: " for i in range(num_dl)]
    assert not pbar.val_progress_bar.leave

    # validate
    with mock.patch("lightning.pytorch.callbacks.progress.tqdm_progress.Tqdm", MockTqdm):
        trainer.validate(model)
    assert trainer.num_val_batches == m
    assert pbar.val_progress_bar.total_values == m
    assert pbar.val_progress_bar.n_values == list(range(m[0] + 1)) * num_dl
    assert pbar.val_progress_bar.descriptions == [f"Validation DataLoader {i}: " for i in range(num_dl)]

    # test
    with mock.patch("lightning.pytorch.callbacks.progress.tqdm_progress.Tqdm", MockTqdm):
        trainer.test(model)
    assert pbar.test_progress_bar.leave
    k = trainer.num_test_batches
    assert pbar.test_progress_bar.total_values == k
    assert pbar.test_progress_bar.n_values == list(range(k[0] + 1)) * num_dl
    assert pbar.test_progress_bar.descriptions == [f"Testing DataLoader {i}: " for i in range(num_dl)]
    assert pbar.test_progress_bar.leave

    # predict
    with mock.patch("lightning.pytorch.callbacks.progress.tqdm_progress.Tqdm", MockTqdm):
        trainer.predict(model)
    assert pbar.predict_progress_bar.leave
    k = trainer.num_predict_batches
    assert pbar.predict_progress_bar.total_values == k
    assert pbar.predict_progress_bar.n_values == list(range(k[0] + 1)) * num_dl
    assert pbar.predict_progress_bar.descriptions == [f"Predicting DataLoader {i}: " for i in range(num_dl)]
    assert pbar.predict_progress_bar.leave


def test_tqdm_progress_bar_fast_dev_run(tmp_path):
    model = BoringModel()

    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)

    trainer.fit(model)

    pbar = trainer.progress_bar_callback

    assert pbar.val_progress_bar.n == 1
    assert pbar.val_progress_bar.total == 1

    # the train progress bar should display 1 batch
    assert pbar.train_progress_bar.total == 1
    assert pbar.train_progress_bar.n == 1

    trainer.validate(model)

    # the validation progress bar should display 1 batch
    assert pbar.val_progress_bar.total == 1
    assert pbar.val_progress_bar.n == 1

    trainer.test(model)

    # the test progress bar should display 1 batch
    assert pbar.test_progress_bar.total == 1
    assert pbar.test_progress_bar.n == 1


@pytest.mark.parametrize("refresh_rate", [0, 1, 50])
def test_tqdm_progress_bar_progress_refresh(tmp_path, refresh_rate: int):
    """Test that the three progress bars get correctly updated when using different refresh rates."""
    model = BoringModel()

    class CurrentProgressBar(TQDMProgressBar):
        train_batches_seen = 0
        val_batches_seen = 0
        test_batches_seen = 0

        def on_train_batch_end(self, *args):
            super().on_train_batch_end(*args)
            self.train_batches_seen += 1

        def on_validation_batch_end(self, *args):
            super().on_validation_batch_end(*args)
            self.val_batches_seen += 1

        def on_test_batch_end(self, *args):
            super().on_test_batch_end(*args)
            self.test_batches_seen += 1

    pbar = CurrentProgressBar(refresh_rate=refresh_rate)
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[pbar],
        limit_train_batches=1.0,
        num_sanity_val_steps=2,
        max_epochs=3,
    )
    assert trainer.progress_bar_callback.refresh_rate == refresh_rate

    trainer.fit(model)
    assert pbar.train_batches_seen == 3 * pbar.train_progress_bar.total
    assert pbar.val_batches_seen == 3 * pbar.val_progress_bar.total + trainer.num_sanity_val_steps
    assert pbar.test_batches_seen == 0

    trainer.validate(model)
    assert pbar.train_batches_seen == 3 * pbar.train_progress_bar.total
    assert pbar.val_batches_seen == 4 * pbar.val_progress_bar.total + trainer.num_sanity_val_steps
    assert pbar.test_batches_seen == 0

    trainer.test(model)
    assert pbar.train_batches_seen == 3 * pbar.train_progress_bar.total
    assert pbar.val_batches_seen == 4 * pbar.val_progress_bar.total + trainer.num_sanity_val_steps
    assert pbar.test_batches_seen == pbar.test_progress_bar.total


@pytest.mark.parametrize("limit_val_batches", [0, 5])
def test_num_sanity_val_steps_progress_bar(tmp_path, limit_val_batches: int):
    """Test val_progress_bar total with 'num_sanity_val_steps' Trainer argument."""

    class CurrentProgressBar(TQDMProgressBar):
        val_pbar_total = 0
        sanity_pbar_total = 0

        def on_sanity_check_end(self, *args):
            self.sanity_pbar_total = self.val_progress_bar.total
            super().on_sanity_check_end(*args)

        def on_validation_epoch_end(self, *args):
            self.val_pbar_total = self.val_progress_bar.total
            super().on_validation_epoch_end(*args)

    model = BoringModel()
    pbar = CurrentProgressBar()
    num_sanity_val_steps = 2

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_train_batches=1,
        limit_val_batches=limit_val_batches,
        callbacks=[pbar],
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model)

    assert pbar.sanity_pbar_total == min(num_sanity_val_steps, limit_val_batches)
    assert pbar.val_pbar_total == limit_val_batches


def test_tqdm_progress_bar_default_value(tmp_path):
    """Test that a value of None defaults to refresh rate 1."""
    trainer = Trainer(default_root_dir=tmp_path)
    assert trainer.progress_bar_callback.refresh_rate == 1


@mock.patch.dict(os.environ, {"COLAB_GPU": "1"})
def test_tqdm_progress_bar_value_on_colab(tmp_path):
    """Test that Trainer will override the default in Google COLAB."""
    trainer = Trainer(default_root_dir=tmp_path)
    assert trainer.progress_bar_callback.refresh_rate == 20

    trainer = Trainer(default_root_dir=tmp_path, callbacks=TQDMProgressBar())
    assert trainer.progress_bar_callback.refresh_rate == 20

    trainer = Trainer(default_root_dir=tmp_path, callbacks=TQDMProgressBar(refresh_rate=19))
    assert trainer.progress_bar_callback.refresh_rate == 19


@pytest.mark.parametrize(
    ("refresh_rate", "env_value", "expected"),
    [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
        (2, 1, 2),
        (1, 2, 2),
    ],
)
def test_tqdm_progress_bar_refresh_rate_via_env_variable(refresh_rate, env_value, expected):
    with mock.patch.dict(os.environ, {"TQDM_MINITERS": str(env_value)}):
        bar = TQDMProgressBar(refresh_rate=refresh_rate)
    assert bar.refresh_rate == expected


@pytest.mark.parametrize(
    ("train_batches", "val_batches", "refresh_rate", "train_updates", "val_updates"),
    [
        (2, 3, 1, [0, 1, 2], [0, 1, 2, 3]),
        (0, 0, 3, None, None),
        (1, 0, 3, [0, 1], None),
        (1, 1, 3, [0, 1], [0, 1]),
        (5, 0, 3, [0, 3, 5], None),
        (5, 2, 3, [0, 3, 5], [0, 2]),
        (5, 2, 6, [0, 5], [0, 2]),
    ],
)
def test_train_progress_bar_update_amount(
    tmp_path, train_batches: int, val_batches: int, refresh_rate: int, train_updates, val_updates
):
    """Test that the train progress updates with the correct amount together with the val progress.

    At the end of the epoch, the progress must not overshoot if the number of steps is not divisible by the refresh
    rate.

    """
    model = BoringModel()
    progress_bar = TQDMProgressBar(refresh_rate=refresh_rate)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        callbacks=[progress_bar],
        logger=False,
        enable_checkpointing=False,
    )
    with mock.patch("lightning.pytorch.callbacks.progress.tqdm_progress.Tqdm", MockTqdm):
        trainer.fit(model)
    if train_batches > 0:
        assert progress_bar.train_progress_bar.n_values == train_updates
    if val_batches > 0:
        assert progress_bar.val_progress_bar.n_values == val_updates


@pytest.mark.parametrize(
    ("test_batches", "refresh_rate", "updates"), [(1, 3, [0, 1]), (3, 1, [0, 1, 2, 3]), (5, 3, [0, 3, 5])]
)
def test_test_progress_bar_update_amount(tmp_path, test_batches: int, refresh_rate: int, updates: list):
    """Test that test progress updates with the correct amount."""
    model = BoringModel()
    progress_bar = TQDMProgressBar(refresh_rate=refresh_rate)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_test_batches=test_batches,
        callbacks=[progress_bar],
        logger=False,
        enable_checkpointing=False,
    )
    with mock.patch("lightning.pytorch.callbacks.progress.tqdm_progress.Tqdm", MockTqdm):
        trainer.test(model)
    assert progress_bar.test_progress_bar.n_values == updates


def test_tensor_to_float_conversion(tmp_path):
    """Check tensor gets converted to float."""

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("a", torch.tensor(0.123), prog_bar=True, on_epoch=False)
            self.log("b", torch.tensor([1]), prog_bar=True, on_epoch=False)
            self.log("c", 2, prog_bar=True, on_epoch=False)
            return super().training_step(batch, batch_idx)

    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=1, limit_train_batches=2, logger=False, enable_checkpointing=False
    )
    trainer.fit(TestModel())

    torch.testing.assert_close(trainer.progress_bar_metrics["a"], 0.123)
    assert trainer.progress_bar_metrics["b"] == 1.0
    assert trainer.progress_bar_metrics["c"] == 2.0
    pbar = trainer.progress_bar_callback.train_progress_bar
    actual = str(pbar.postfix)
    assert actual.endswith("a=0.123, b=1.000, c=2.000"), actual


@pytest.mark.parametrize(
    ("input_num", "expected"),
    [
        (1, "1"),
        (1.0, "1.000"),
        (0.1, "0.100"),
        (1e-3, "0.001"),
        (1e-5, "1e-5"),
        ("1.0", "1.000"),
        ("10000", "10000"),
        ("abc", "abc"),
    ],
)
def test_tqdm_format_num(input_num: Union[str, int, float], expected: str):
    """Check that the specialized tqdm.format_num appends 0 to floats and strings."""
    assert Tqdm.format_num(input_num) == expected


class PrintModel(BoringModel):
    def training_step(self, *args, **kwargs):
        self.print("training_step", end="")
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        self.print("validation_step", file=sys.stderr)
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        self.print("test_step")
        return super().test_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        self.print("predict_step")
        return super().predict_step(*args, **kwargs)


@mock.patch("tqdm.tqdm.write")
def test_tqdm_progress_bar_print(tqdm_write, tmp_path):
    """Test that printing in the LightningModule redirects arguments to the progress bar."""
    model = PrintModel()
    bar = TQDMProgressBar()
    trainer = Trainer(
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        max_steps=1,
        callbacks=[bar],
    )
    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model)
    assert tqdm_write.call_args_list == [
        call("training_step", end=""),
        call("validation_step", file=sys.stderr),
        call("test_step"),
        call("predict_step"),
    ]


@mock.patch("tqdm.tqdm.write")
def test_tqdm_progress_bar_print_no_train(tqdm_write, tmp_path):
    """Test that printing in the LightningModule redirects arguments to the progress bar without training."""
    model = PrintModel()
    bar = TQDMProgressBar()
    trainer = Trainer(
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        max_steps=1,
        callbacks=[bar],
    )

    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)
    assert tqdm_write.call_args_list == [
        call("validation_step", file=sys.stderr),
        call("test_step"),
        call("predict_step"),
    ]


@mock.patch("builtins.print")
@mock.patch("lightning.pytorch.callbacks.progress.tqdm_progress.Tqdm.write")
def test_tqdm_progress_bar_print_disabled(tqdm_write, mock_print, tmp_path):
    """Test that printing in LightningModule goes through built-in print function when progress bar is disabled."""
    model = PrintModel()
    bar = TQDMProgressBar()
    trainer = Trainer(
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        max_steps=1,
        callbacks=[bar],
    )
    bar.disable()
    trainer.fit(model)
    trainer.test(model, verbose=False)
    trainer.predict(model)

    mock_print.assert_has_calls([
        call("training_step", end=""),
        call("validation_step", file=ANY),
        call("test_step"),
        call("predict_step"),
    ])
    tqdm_write.assert_not_called()


def test_tqdm_progress_bar_can_be_pickled(tmp_path):
    bar = TQDMProgressBar()
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[bar],
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        logger=False,
        enable_model_summary=False,
    )
    model = BoringModel()
    pickle.dumps(bar)
    trainer.fit(model)
    pickle.dumps(bar)
    trainer.validate(model)
    pickle.dumps(bar)
    trainer.test(model)
    pickle.dumps(bar)
    trainer.predict(model)
    pickle.dumps(bar)


@pytest.mark.parametrize(
    ("val_check_interval", "train_progress_bar_updates", "val_progress_bar_updates"),
    [(4, [0, 3, 6, 7], [0, 3, 6, 7]), (0.5, [0, 3, 6, 7], [0, 3, 6, 7])],
)
def test_progress_bar_max_val_check_interval(
    tmp_path, val_check_interval, train_progress_bar_updates, val_progress_bar_updates
):
    limit_batches = 7
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
        max_epochs=1,
        enable_model_summary=False,
        val_check_interval=val_check_interval,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        callbacks=TQDMProgressBar(refresh_rate=3),
    )
    with mock.patch("lightning.pytorch.callbacks.progress.tqdm_progress.Tqdm", MockTqdm):
        trainer.fit(model)

    pbar = trainer.progress_bar_callback
    assert pbar.train_progress_bar.n_values == train_progress_bar_updates
    assert pbar.val_progress_bar.n_values == val_progress_bar_updates

    val_check_batch = (
        max(1, int(limit_batches * val_check_interval)) if isinstance(val_check_interval, float) else val_check_interval
    )
    assert trainer.val_check_batch == val_check_batch
    math.ceil(limit_batches // val_check_batch)
    pbar_callback = trainer.progress_bar_callback

    assert pbar_callback.val_progress_bar.n == limit_batches
    assert pbar_callback.val_progress_bar.total == limit_batches
    assert pbar_callback.train_progress_bar.n == limit_batches
    assert pbar_callback.train_progress_bar.total == limit_batches
    assert pbar_callback.is_enabled


@RunIf(min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize("val_check_interval", [0.2, 0.5])
def test_progress_bar_max_val_check_interval_ddp(tmp_path, val_check_interval):
    world_size = 2
    total_train_samples = 16
    train_batch_size = 4
    total_val_samples = 2
    val_batch_size = 1
    train_data = DataLoader(RandomDataset(32, 8), batch_size=train_batch_size)
    val_data = DataLoader(RandomDataset(32, total_val_samples), batch_size=val_batch_size)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
        max_epochs=1,
        val_check_interval=val_check_interval,
        accelerator="gpu",
        devices=world_size,
        strategy="ddp",
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)

    total_train_batches = total_train_samples // (train_batch_size * world_size)
    val_check_batch = max(1, int(total_train_batches * val_check_interval))
    assert trainer.val_check_batch == val_check_batch
    total_val_batches = total_val_samples // (val_batch_size * world_size)
    pbar_callback = trainer.progress_bar_callback

    if trainer.is_global_zero:
        assert pbar_callback.val_progress_bar.n == total_val_batches
        assert pbar_callback.val_progress_bar.total == total_val_batches
        assert pbar_callback.train_progress_bar.n == total_train_batches // world_size
        assert pbar_callback.train_progress_bar.total == total_train_batches // world_size
        assert pbar_callback.is_enabled


def test_get_progress_bar_metrics(tmp_path):
    """Test that the metrics shown in the progress bar can be customized."""

    class TestProgressBar(TQDMProgressBar):
        def get_metrics(self, trainer: Trainer, model: LightningModule):
            items = super().get_metrics(trainer, model)
            items.pop("v_num", None)
            items["my_metric"] = 123
            return items

    progress_bar = TestProgressBar()
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[progress_bar],
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    model = BoringModel()
    trainer.fit(model)
    standard_metrics = progress_bar.get_metrics(trainer, model)
    assert "v_num" not in standard_metrics
    assert "my_metric" in standard_metrics


def test_get_progress_bar_metrics_fast_dev_run(tmp_path):
    """Test that `v_num` does not appear in the progress bar when a dummy logger is used (fast-dev-run)."""
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    model = BoringModel()
    trainer.fit(model)
    standard_metrics = trainer.progress_bar_callback.get_metrics(trainer, model)
    assert isinstance(trainer.logger, DummyLogger)
    assert "v_num" not in standard_metrics


def test_tqdm_progress_bar_correct_value_epoch_end(tmp_path):
    """TQDM counterpart to test_rich_progress_bar::test_rich_progress_bar_correct_value_epoch_end."""

    class MockedProgressBar(TQDMProgressBar):
        calls = defaultdict(list)

        def get_metrics(self, trainer, pl_module):
            items = super().get_metrics(trainer, model)
            del items["v_num"]
            # this is equivalent to mocking `set_postfix` as this method gets called every time
            self.calls[trainer.state.fn].append((
                trainer.state.stage,
                trainer.current_epoch,
                trainer.global_step,
                items,
            ))
            return items

    class MyModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("a", self.global_step, prog_bar=True, on_step=False, on_epoch=True, reduce_fx=max)
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            self.log("b", self.global_step, prog_bar=True, on_step=False, on_epoch=True, reduce_fx=max)
            return super().validation_step(batch, batch_idx)

        def test_step(self, batch, batch_idx):
            self.log("c", self.global_step, prog_bar=True, on_step=False, on_epoch=True, reduce_fx=max)
            return super().test_step(batch, batch_idx)

    model = MyModel()
    pbar = MockedProgressBar()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=2,
        enable_model_summary=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
        callbacks=pbar,
        logger=CSVLogger(tmp_path),
    )

    trainer.fit(model)
    assert pbar.calls["fit"] == [
        ("sanity_check", 0, 0, {"b": 0}),
        ("train", 0, 1, {}),
        ("train", 0, 2, {}),
        ("validate", 0, 2, {"b": 2}),  # validation end
        # epoch end over, `on_epoch=True` metrics are computed
        ("train", 0, 2, {"a": 1, "b": 2}),  # training epoch end
        ("train", 1, 3, {"a": 1, "b": 2}),
        ("train", 1, 4, {"a": 1, "b": 2}),
        ("validate", 1, 4, {"a": 1, "b": 4}),  # validation end
        ("train", 1, 4, {"a": 3, "b": 4}),  # training epoch end
    ]

    trainer.validate(model, verbose=False)
    assert pbar.calls["validate"] == []

    trainer.test(model, verbose=False)
    assert pbar.calls["test"] == []


@mock.patch("lightning.pytorch.trainer.trainer.Trainer.is_global_zero", new_callable=PropertyMock, return_value=False)
def test_tqdm_progress_bar_disabled_when_not_rank_zero(is_global_zero):
    """Test that the progress bar is disabled when not in global rank zero."""
    pbar = TQDMProgressBar()
    model = BoringModel()
    trainer = Trainer(
        callbacks=[pbar],
        fast_dev_run=True,
    )

    pbar.enable()
    trainer.fit(model)
    assert pbar.is_disabled

    pbar.enable()
    trainer.predict(model)
    assert pbar.is_disabled

    pbar.enable()
    trainer.validate(model)
    assert pbar.is_disabled

    pbar.enable()
    trainer.test(model)
    assert pbar.is_disabled
