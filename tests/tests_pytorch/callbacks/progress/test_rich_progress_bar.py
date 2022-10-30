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
from collections import defaultdict
from unittest import mock
from unittest.mock import DEFAULT, Mock

import pytest
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ProgressBarBase, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.demos.boring_classes import BoringModel, RandomDataset, RandomIterableDataset
from tests_pytorch.helpers.runif import RunIf


@RunIf(rich=True)
def test_rich_progress_bar_callback():
    trainer = Trainer(callbacks=RichProgressBar())

    progress_bars = [c for c in trainer.callbacks if isinstance(c, ProgressBarBase)]

    assert len(progress_bars) == 1
    assert isinstance(trainer.progress_bar_callback, RichProgressBar)


@RunIf(rich=True)
def test_rich_progress_bar_refresh_rate_enabled():
    progress_bar = RichProgressBar(refresh_rate=1)
    assert progress_bar.is_enabled
    assert not progress_bar.is_disabled
    progress_bar = RichProgressBar(refresh_rate=0)
    assert not progress_bar.is_enabled
    assert progress_bar.is_disabled


@RunIf(rich=True)
@pytest.mark.parametrize("dataset", [RandomDataset(32, 64), RandomIterableDataset(32, 64)])
def test_rich_progress_bar(tmpdir, dataset):
    class TestModel(BoringModel):
        def train_dataloader(self):
            return DataLoader(dataset=dataset)

        def val_dataloader(self):
            return DataLoader(dataset=dataset)

        def test_dataloader(self):
            return DataLoader(dataset=dataset)

        def predict_dataloader(self):
            return DataLoader(dataset=dataset)

    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        max_epochs=1,
        callbacks=RichProgressBar(),
    )
    model = TestModel()

    with mock.patch("pytorch_lightning.callbacks.progress.rich_progress.Progress.update") as mocked:
        trainer.fit(model)
    # 3 for main progress bar and 1 for val progress bar
    assert mocked.call_count == 4

    with mock.patch("pytorch_lightning.callbacks.progress.rich_progress.Progress.update") as mocked:
        trainer.validate(model)
    assert mocked.call_count == 1

    with mock.patch("pytorch_lightning.callbacks.progress.rich_progress.Progress.update") as mocked:
        trainer.test(model)
    assert mocked.call_count == 1

    with mock.patch("pytorch_lightning.callbacks.progress.rich_progress.Progress.update") as mocked:
        trainer.predict(model)
    assert mocked.call_count == 1


def test_rich_progress_bar_import_error(monkeypatch):
    import pytorch_lightning.callbacks.progress.rich_progress as imports

    monkeypatch.setattr(imports, "_RICH_AVAILABLE", False)
    with pytest.raises(ModuleNotFoundError, match="`RichProgressBar` requires `rich` >= 10.2.2."):
        RichProgressBar()


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
        assert kwargs["complete_style"] == theme.progress_bar
        assert kwargs["finished_style"] == theme.progress_bar_finished

        args, kwargs = mocks["BatchesProcessedColumn"].call_args
        assert kwargs["style"] == theme.batch_progress

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
@pytest.mark.parametrize(("leave", "reset_call_count"), ([(True, 0), (False, 3)]))
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
            limit_val_batches=0,
            max_epochs=4,
            callbacks=progress_bar,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(model)
    assert mock_progress_reset.call_count == reset_call_count


@RunIf(rich=True)
@mock.patch("pytorch_lightning.callbacks.progress.rich_progress.Progress.update")
def test_rich_progress_bar_refresh_rate_disabled(progress_update, tmpdir):
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=4,
        callbacks=RichProgressBar(refresh_rate=0),
    )
    trainer.fit(BoringModel())
    assert progress_update.call_count == 0


@RunIf(rich=True)
@pytest.mark.parametrize(
    "refresh_rate,train_batches,val_batches,expected_call_count",
    [
        (3, 6, 6, 4 + 3),
        (4, 6, 6, 3 + 3),
        (7, 6, 6, 2 + 2),
        (1, 2, 3, 5 + 4),
        (1, 0, 0, 0 + 0),
        (3, 1, 0, 1 + 0),
        (3, 1, 1, 1 + 2),
        (3, 5, 0, 2 + 0),
        (3, 5, 2, 3 + 2),
        (6, 5, 2, 2 + 2),
    ],
)
def test_rich_progress_bar_with_refresh_rate(tmpdir, refresh_rate, train_batches, val_batches, expected_call_count):
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=0,
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        max_epochs=1,
        callbacks=RichProgressBar(refresh_rate=refresh_rate),
    )

    trainer.progress_bar_callback.on_train_start(trainer, model)
    with mock.patch.object(
        trainer.progress_bar_callback.progress, "update", wraps=trainer.progress_bar_callback.progress.update
    ) as progress_update:
        trainer.fit(model)
        assert progress_update.call_count == expected_call_count

    if train_batches > 0:
        fit_main_bar = trainer.progress_bar_callback.progress.tasks[0]
        assert fit_main_bar.completed == train_batches + val_batches
        assert fit_main_bar.total == train_batches + val_batches
        assert fit_main_bar.visible
    if val_batches > 0:
        fit_val_bar = trainer.progress_bar_callback.progress.tasks[1]
        assert fit_val_bar.completed == val_batches
        assert fit_val_bar.total == val_batches
        assert not fit_val_bar.visible


@RunIf(rich=True)
@pytest.mark.parametrize("limit_val_batches", (1, 5))
def test_rich_progress_bar_num_sanity_val_steps(tmpdir, limit_val_batches):
    model = BoringModel()

    progress_bar = RichProgressBar()
    num_sanity_val_steps = 3

    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_train_batches=1,
        limit_val_batches=limit_val_batches,
        max_epochs=1,
        callbacks=progress_bar,
    )

    trainer.fit(model)
    assert progress_bar.progress.tasks[0].completed == min(num_sanity_val_steps, limit_val_batches)
    assert progress_bar.progress.tasks[0].total == min(num_sanity_val_steps, limit_val_batches)


@RunIf(rich=True)
def test_rich_progress_bar_counter_with_val_check_interval(tmpdir):
    """Test the completed and total counter for rich progress bar when using val_check_interval."""
    progress_bar = RichProgressBar()
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        val_check_interval=2,
        max_epochs=1,
        limit_train_batches=7,
        limit_val_batches=4,
        callbacks=[progress_bar],
    )
    trainer.fit(model)

    fit_main_progress_bar = progress_bar.progress.tasks[1]
    assert fit_main_progress_bar.completed == 7 + 3 * 4
    assert fit_main_progress_bar.total == 7 + 3 * 4

    fit_val_bar = progress_bar.progress.tasks[2]
    assert fit_val_bar.completed == 4
    assert fit_val_bar.total == 4

    trainer.validate(model)
    val_bar = progress_bar.progress.tasks[0]
    assert val_bar.completed == 4
    assert val_bar.total == 4


@RunIf(rich=True)
@mock.patch("pytorch_lightning.callbacks.progress.rich_progress._detect_light_colab_theme", return_value=True)
def test_rich_progress_bar_colab_light_theme_update(*_):
    theme = RichProgressBar().theme
    assert theme.description == "black"
    assert theme.batch_progress == "black"
    assert theme.metrics == "black"

    theme = RichProgressBar(theme=RichProgressBarTheme(description="blue", metrics="red")).theme
    assert theme.description == "blue"
    assert theme.batch_progress == "black"
    assert theme.metrics == "red"


@RunIf(rich=True)
def test_rich_progress_bar_metric_display_task_id(tmpdir):
    class CustomModel(BoringModel):
        def training_step(self, *args, **kwargs):
            res = super().training_step(*args, **kwargs)
            self.log("train_loss", res["loss"], prog_bar=True)
            return res

    progress_bar = RichProgressBar()
    model = CustomModel()
    trainer = Trainer(default_root_dir=tmpdir, callbacks=progress_bar, fast_dev_run=True)

    trainer.fit(model)
    main_progress_bar_id = progress_bar.main_progress_bar_id
    val_progress_bar_id = progress_bar.val_progress_bar_id
    rendered = progress_bar.progress.columns[-1]._renderable_cache

    for key in ("loss", "v_num", "train_loss"):
        assert key in rendered[main_progress_bar_id][1]
        assert key not in rendered[val_progress_bar_id][1]


@RunIf(rich=True)
def test_rich_progress_bar_correct_value_epoch_end(tmpdir):
    """Rich counterpart to test_tqdm_progress_bar::test_tqdm_progress_bar_correct_value_epoch_end."""

    class MockedProgressBar(RichProgressBar):
        calls = defaultdict(list)

        def get_metrics(self, trainer, pl_module):
            items = super().get_metrics(trainer, model)
            del items["v_num"]
            del items["loss"]
            # this is equivalent to mocking `set_postfix` as this method gets called every time
            self.calls[trainer.state.fn].append(
                (trainer.state.stage, trainer.current_epoch, trainer.global_step, items)
            )
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
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=2,
        enable_model_summary=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
        callbacks=pbar,
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


@RunIf(rich=True)
def test_rich_progress_bar_padding():
    progress_bar = RichProgressBar()
    trainer = Mock()
    trainer.max_epochs = 1
    progress_bar._trainer = trainer

    train_description = progress_bar._get_train_description(current_epoch=0)
    assert "Epoch 0/0" in train_description
    assert len(progress_bar.validation_description) == len(train_description)


@RunIf(rich=True)
def test_rich_progress_bar_can_be_pickled():
    bar = RichProgressBar()
    trainer = Trainer(
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


@RunIf(rich=True)
def test_rich_progress_bar_reset_bars():
    """Test that the progress bar resets all internal bars when a new trainer stage begins."""
    bar = RichProgressBar()
    assert bar.is_enabled
    assert bar.progress is None
    assert bar._progress_stopped is False

    def _set_fake_bar_ids():
        bar.main_progress_bar_id = 0
        bar.val_sanity_progress_bar_id = 1
        bar.val_progress_bar_id = 2
        bar.test_progress_bar_id = 3
        bar.predict_progress_bar_id = 4

    for stage in ("train", "sanity_check", "validation", "test", "predict"):
        hook_name = f"on_{stage}_start"
        hook = getattr(bar, hook_name)

        _set_fake_bar_ids()  # pretend that bars are initialized from a previous run
        hook(Mock(), Mock())
        bar.teardown(Mock(), Mock(), Mock())

        # assert all bars are reset
        assert bar.main_progress_bar_id is None
        assert bar.val_sanity_progress_bar_id is None
        assert bar.val_progress_bar_id is None
        assert bar.test_progress_bar_id is None
        assert bar.predict_progress_bar_id is None

        # the progress object remains in case we need it for the next stage
        assert bar.progress is not None


@RunIf(rich=True)
def test_rich_progress_bar_disabled(tmpdir):
    """Test that in a disabled bar there are no updates and no internal progress objects."""
    bar = RichProgressBar()
    bar.disable()
    assert bar.is_disabled

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        limit_predict_batches=2,
        max_epochs=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[bar],
    )

    with mock.patch("pytorch_lightning.callbacks.progress.rich_progress.CustomProgress") as mocked:
        trainer.fit(model)
        trainer.validate(model)
        trainer.test(model)
        trainer.predict(model)

    mocked.assert_not_called()
    assert bar.main_progress_bar_id is None
    assert bar.val_sanity_progress_bar_id is None
    assert bar.val_progress_bar_id is None
    assert bar.test_progress_bar_id is None
    assert bar.predict_progress_bar_id is None
