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
from unittest.mock import DEFAULT, Mock

import pytest
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ProgressBarBase, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.utilities.imports import _RICH_AVAILABLE
from tests.helpers.boring_model import BoringModel, RandomDataset, RandomIterableDataset
from tests.helpers.runif import RunIf


@RunIf(rich=True)
def test_rich_progress_bar_callback():
    trainer = Trainer(callbacks=RichProgressBar())

    progress_bars = [c for c in trainer.callbacks if isinstance(c, ProgressBarBase)]

    assert len(progress_bars) == 1
    assert isinstance(trainer.progress_bar_callback, RichProgressBar)


@RunIf(rich=True)
def test_rich_progress_bar_refresh_rate():
    progress_bar = RichProgressBar(refresh_rate_per_second=1)
    assert progress_bar.is_enabled
    assert not progress_bar.is_disabled
    progress_bar = RichProgressBar(refresh_rate_per_second=0)
    assert not progress_bar.is_enabled
    assert progress_bar.is_disabled


@RunIf(rich=True)
@mock.patch("pytorch_lightning.callbacks.progress.rich_progress.Progress.update")
@pytest.mark.parametrize("dataset", [RandomDataset(32, 64), RandomIterableDataset(32, 64)])
def test_rich_progress_bar(progress_update, tmpdir, dataset):
    class TestModel(BoringModel):
        def train_dataloader(self):
            return DataLoader(dataset=dataset)

        def val_dataloader(self):
            return DataLoader(dataset=dataset)

        def test_dataloader(self):
            return DataLoader(dataset=dataset)

        def predict_dataloader(self):
            return DataLoader(dataset=dataset)

    model = TestModel()

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
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)

    assert progress_update.call_count == 8


def test_rich_progress_bar_import_error():
    if not _RICH_AVAILABLE:
        with pytest.raises(ImportError, match="`RichProgressBar` requires `rich` to be installed."):
            Trainer(callbacks=RichProgressBar())


@RunIf(rich=True)
def test_rich_progress_bar_custom_theme(tmpdir):
    """Test to ensure that custom theme styles are used."""
    with mock.patch.multiple(
        "pytorch_lightning.callbacks.progress.rich_progress",
        CustomBarColumn=DEFAULT,
        BatchesProcessedColumn=DEFAULT,
        CustomTimeColumn=DEFAULT,
        ProcessingSpeedColumn=DEFAULT,
    ) as mocks:
        theme = RichProgressBarTheme()

        progress_bar = RichProgressBar(theme=theme)
        progress_bar.on_train_start(Trainer(tmpdir), BoringModel())

        assert progress_bar.theme == theme
        args, kwargs = mocks["CustomBarColumn"].call_args
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


@RunIf(rich=True)
def test_rich_progress_bar_configure_columns():
    from rich.progress import TextColumn

    custom_column = TextColumn("[progress.description]Testing Rich!")

    class CustomRichProgressBar(RichProgressBar):
        def configure_columns(self, trainer):
            return [custom_column]

    progress_bar = CustomRichProgressBar()

    progress_bar._init_progress(Mock())

    assert progress_bar.progress.columns[0] == custom_column
    assert len(progress_bar.progress.columns) == 2


@RunIf(rich=True)
@pytest.mark.parametrize(("leave", "reset_call_count"), ([(True, 0), (False, 5)]))
def test_rich_progress_bar_leave(tmpdir, leave, reset_call_count):
    # Calling `reset` means continuing on the same progress bar.
    model = BoringModel()

    with mock.patch(
        "pytorch_lightning.callbacks.progress.rich_progress.Progress.reset", autospec=True
    ) as mock_progress_reset:
        progress_bar = RichProgressBar(leave=leave)
        trainer = Trainer(
            default_root_dir=tmpdir,
            num_sanity_val_steps=0,
            limit_train_batches=1,
            max_epochs=6,
            callbacks=progress_bar,
        )
        trainer.fit(model)
    assert mock_progress_reset.call_count == reset_call_count
