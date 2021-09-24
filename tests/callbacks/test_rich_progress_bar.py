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

    with pytest.warns(
        UserWarning, match="``RichProgressBar`` does not support setting the refresh rate via the Trainer. "
    ):
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
def test_rich_progress_bar_custom_theme(tmpdir):
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
        progress_bar.setup(Trainer(tmpdir), BoringModel(), stage=None)

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
