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
import pickle
from unittest import mock
from unittest.mock import DEFAULT

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ProgressBar, ProgressBarBase, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.utilities.imports import _RICH_AVAILABLE
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


@RunIf(rich=True)
def test_rich_progress_bar_callback():
    trainer = Trainer(callbacks=RichProgressBar())

    progress_bars = [c for c in trainer.callbacks if isinstance(c, ProgressBarBase)]

    assert len(progress_bars) == 1
    assert isinstance(trainer.progress_bar_callback, RichProgressBar)


@RunIf(rich=True)
def test_rich_progress_bar_refresh_rate_enable_disable():
    progress_bar = RichProgressBar(refresh_rate_per_second=1)
    assert progress_bar.is_enabled
    assert not progress_bar.is_disabled
    progress_bar = RichProgressBar(refresh_rate_per_second=0)
    assert not progress_bar.is_enabled
    assert progress_bar.is_disabled


@RunIf(rich=True)
def test_rich_progress_bar_refresh_rate(tmpdir):
    """Test that the refresh rate is set correctly based on the Trainer, and warn if the user sets the argument."""
    trainer = Trainer(default_root_dir=tmpdir)
    assert trainer.progress_bar_callback.refresh_rate_per_second == 10
    assert isinstance(trainer.progress_bar_callback, RichProgressBar)

    trainer = Trainer(default_root_dir=tmpdir, progress_bar_refresh_rate=None)
    assert isinstance(trainer.progress_bar_callback, RichProgressBar)
    assert trainer.progress_bar_callback.refresh_rate_per_second == 10

    with pytest.warns(UserWarning, match="does not support setting the refresh rate via the Trainer."):
        trainer = Trainer(default_root_dir=tmpdir, progress_bar_refresh_rate=19)
    assert isinstance(trainer.progress_bar_callback, ProgressBar)
    assert trainer.progress_bar_callback.refresh_rate == 19


@RunIf(rich=True)
@mock.patch("pytorch_lightning.callbacks.progress.rich_progress.Progress.update")
def test_rich_progress_bar(progress_update, tmpdir):
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        max_steps=1,
        callbacks=RichProgressBar(),
    )

    trainer.fit(model)
    trainer.test(model)
    trainer.predict(model)

    assert progress_update.call_count == 6


def test_rich_progress_bar_import_error():
    if not _RICH_AVAILABLE:
        with pytest.raises(ImportError, match="`RichProgressBar` requires `rich` to be installed."):
            Trainer(callbacks=RichProgressBar())


@RunIf(rich=True)
@mock.patch("pytorch_lightning.callbacks.progress.rich_progress.Progress")
def test_rich_progress_bar_custom_theme(mock_progress, tmpdir):
    """Test to ensure that custom theme styles are used."""
    with mock.patch.multiple(
        "pytorch_lightning.callbacks.progress.rich_progress",
        BarColumn=DEFAULT,
        BatchesProcessedColumn=DEFAULT,
        CustomTimeColumn=DEFAULT,
        ProcessingSpeedColumn=DEFAULT,
    ) as mocks:
        theme = RichProgressBarTheme()

        progress_bar = RichProgressBar(theme=theme)
        progress_bar.on_train_start(Trainer(tmpdir), BoringModel())

        assert progress_bar.theme == theme
        args, kwargs = mocks["BarColumn"].call_args
        assert kwargs["complete_style"] == theme.progress_bar_complete
        assert kwargs["finished_style"] == theme.progress_bar_finished

        args, kwargs = mocks["BatchesProcessedColumn"].call_args
        assert kwargs["style"] == theme.batch_process

        args, kwargs = mocks["CustomTimeColumn"].call_args
        assert kwargs["style"] == theme.time

        args, kwargs = mocks["ProcessingSpeedColumn"].call_args
        assert kwargs["style"] == theme.processing_speed


@RunIf(rich=True)
def test_rich_progress_bar_keyboard_interrupt(tmpdir):
    """Test to ensure that when the user keyboard interrupts, we close the progress bar."""

    class TestModel(BoringModel):
        def on_train_start(self) -> None:
            raise KeyboardInterrupt

    model = TestModel()

    with mock.patch(
        "pytorch_lightning.callbacks.progress.rich_progress.Progress.stop", autospec=True
    ) as mock_progress_stop:
        progress_bar = RichProgressBar()
        trainer = Trainer(
            default_root_dir=tmpdir,
            fast_dev_run=True,
            callbacks=progress_bar,
        )

        trainer.fit(model)
    mock_progress_stop.assert_called_once()
    trainer.progress_bar_callback.teardown(trainer, model)


@RunIf(rich=True)
def test_progress_bar_totals(tmpdir):
    """Test that the progress finishes with the correct total steps processed."""

    model = BoringModel()

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)
    bar = trainer.progress_bar_callback
    assert 0 == bar.total_train_batches
    assert 0 == bar.total_val_batches
    assert 0 == bar.total_test_batches

    trainer.fit(model)

    # check main progress bar total
    n = bar.total_train_batches
    m = bar.total_val_batches

    assert len(trainer.train_dataloader) == n
    assert bar.main_progress_bar.total == n + m

    # check val progress bar total
    assert sum(len(loader) for loader in trainer.val_dataloaders) == m
    assert bar.val_progress_bar.total == m

    # check that the test progress bar is off
    assert bar.total_test_batches == 0

    trainer.validate(model)

    assert bar.val_progress_bar.total == m
    assert bar.val_batch_idx == m

    trainer.test(model)

    # check test progress bar total
    k = bar.total_test_batches
    assert sum(len(loader) for loader in trainer.test_dataloaders) == k
    assert bar.test_progress_bar.total == k
    assert bar.test_batch_idx == k


@RunIf(rich=True)
def test_progress_bar_fast_dev_run(tmpdir):
    model = BoringModel()

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)

    trainer.fit(model)

    bar = trainer.progress_bar_callback

    assert 1 == bar.total_train_batches
    # total val batches are known only after val dataloaders have reloaded

    assert 1 == bar.total_val_batches
    assert 1 == bar.train_batch_idx
    assert 1 == bar.val_batch_idx
    assert 0 == bar.test_batch_idx

    # the main progress bar should display 2 batches (1 train, 1 val)
    assert 2 == bar.main_progress_bar.total

    trainer.validate(model)

    # the validation progress bar should display 1 batch
    assert 1 == bar.val_batch_idx
    assert 1 == bar.val_progress_bar.total

    trainer.test(model)

    # the test progress bar should display 1 batch
    assert 1 == bar.test_batch_idx
    assert 1 == bar.test_progress_bar.total


@RunIf(rich=True)
@pytest.mark.parametrize("limit_val_batches", (0, 5))
def test_num_sanity_val_steps_progress_bar(tmpdir, limit_val_batches: int):
    """Test val_progress_bar total with 'num_sanity_val_steps' Trainer argument."""

    class CurrentProgressBar(RichProgressBar):
        val_pbar_total = 0
        sanity_pbar_total = 0

        def on_sanity_check_end(self, *args):
            super().on_sanity_check_end(*args)
            self.sanity_pbar_total = self.val_sanity_check_bar.total

        def on_validation_epoch_end(self, *args):
            super().on_validation_epoch_end(*args)
            self.val_pbar_total = self.val_progress_bar.total

    model = BoringModel()
    progress_bar = CurrentProgressBar()
    num_sanity_val_steps = 2

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_train_batches=1,
        limit_val_batches=limit_val_batches,
        callbacks=[progress_bar],
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(model)

    assert progress_bar.sanity_pbar_total == min(num_sanity_val_steps, limit_val_batches)
    assert progress_bar.val_pbar_total == limit_val_batches


@RunIf(rich=True)
def test_progress_bar_can_be_pickled():
    bar = RichProgressBar()
    trainer = Trainer(fast_dev_run=True, callbacks=[bar], max_steps=1)
    model = BoringModel()

    pickle.dumps(bar)
    trainer.fit(model)
    pickle.dumps(bar)
    trainer.test(model)
    pickle.dumps(bar)
    trainer.predict(model)
    pickle.dumps(bar)
