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
import os
import pickle
import sys
from typing import Optional, Union
from unittest import mock
from unittest.mock import ANY, call, Mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, ProgressBarBase
from pytorch_lightning.callbacks.progress import tqdm
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel


@pytest.mark.parametrize(
    'callbacks,refresh_rate', [
        ([], None),
        ([], 1),
        ([], 2),
        ([ProgressBar(refresh_rate=1)], 0),
        ([ProgressBar(refresh_rate=2)], 0),
        ([ProgressBar(refresh_rate=2)], 1),
    ]
)
def test_progress_bar_on(tmpdir, callbacks: list, refresh_rate: Optional[int]):
    """Test different ways the progress bar can be turned on."""

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=callbacks,
        progress_bar_refresh_rate=refresh_rate,
        max_epochs=1,
        overfit_batches=5,
    )

    progress_bars = [c for c in trainer.callbacks if isinstance(c, ProgressBarBase)]
    # Trainer supports only a single progress bar callback at the moment
    assert len(progress_bars) == 1
    assert progress_bars[0] is trainer.progress_bar_callback


@pytest.mark.parametrize(
    'callbacks,refresh_rate', [
        ([], 0),
        ([], False),
        ([ModelCheckpoint(dirpath='../trainer')], 0),
    ]
)
def test_progress_bar_off(tmpdir, callbacks: list, refresh_rate: Union[bool, int]):
    """Test different ways the progress bar can be turned off."""

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=callbacks,
        progress_bar_refresh_rate=refresh_rate,
    )

    progress_bars = [c for c in trainer.callbacks if isinstance(c, ProgressBar)]
    assert 0 == len(progress_bars)
    assert not trainer.progress_bar_callback


def test_progress_bar_misconfiguration():
    """Test that Trainer doesn't accept multiple progress bars."""
    callbacks = [ProgressBar(), ProgressBar(), ModelCheckpoint(dirpath='../trainer')]
    with pytest.raises(MisconfigurationException, match=r'^You added multiple progress bar callbacks'):
        Trainer(callbacks=callbacks)


def test_progress_bar_totals(tmpdir):
    """Test that the progress finishes with the correct total steps processed."""

    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=1,
        max_epochs=1,
    )
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

    # main progress bar should have reached the end (train batches + val batches)
    assert bar.main_progress_bar.n == n + m
    assert bar.train_batch_idx == n

    # val progress bar should have reached the end
    assert bar.val_progress_bar.n == m
    assert bar.val_batch_idx == m

    # check that the test progress bar is off
    assert 0 == bar.total_test_batches
    assert bar.test_progress_bar is None

    trainer.validate(model)

    assert bar.val_progress_bar.total == m
    assert bar.val_progress_bar.n == m
    assert bar.val_batch_idx == m

    trainer.test(model)

    # check test progress bar total
    k = bar.total_test_batches
    assert sum(len(loader) for loader in trainer.test_dataloaders) == k
    assert bar.test_progress_bar.total == k

    # test progress bar should have reached the end
    assert bar.test_progress_bar.n == k
    assert bar.test_batch_idx == k


def test_progress_bar_fast_dev_run(tmpdir):
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )

    trainer.fit(model)

    progress_bar = trainer.progress_bar_callback
    assert 1 == progress_bar.total_train_batches
    # total val batches are known only after val dataloaders have reloaded

    assert 1 == progress_bar.total_val_batches
    assert 1 == progress_bar.train_batch_idx
    assert 1 == progress_bar.val_batch_idx
    assert 0 == progress_bar.test_batch_idx

    # the main progress bar should display 2 batches (1 train, 1 val)
    assert 2 == progress_bar.main_progress_bar.total
    assert 2 == progress_bar.main_progress_bar.n

    trainer.validate(model)

    # the validation progress bar should display 1 batch
    assert 1 == progress_bar.val_batch_idx
    assert 1 == progress_bar.val_progress_bar.total
    assert 1 == progress_bar.val_progress_bar.n

    trainer.test(model)

    # the test progress bar should display 1 batch
    assert 1 == progress_bar.test_batch_idx
    assert 1 == progress_bar.test_progress_bar.total
    assert 1 == progress_bar.test_progress_bar.n


@pytest.mark.parametrize('refresh_rate', [0, 1, 50])
def test_progress_bar_progress_refresh(tmpdir, refresh_rate: int):
    """Test that the three progress bars get correctly updated when using different refresh rates."""

    model = BoringModel()

    class CurrentProgressBar(ProgressBar):

        train_batches_seen = 0
        val_batches_seen = 0
        test_batches_seen = 0

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            super().on_train_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
            assert self.train_batch_idx == trainer.train_loop.batch_idx

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
            assert self.train_batch_idx == trainer.train_loop.batch_idx + 1
            if not self.is_disabled and self.train_batch_idx % self.refresh_rate == 0:
                assert self.main_progress_bar.n == self.train_batch_idx
            self.train_batches_seen += 1

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
            if not self.is_disabled and self.val_batch_idx % self.refresh_rate == 0:
                assert self.val_progress_bar.n == self.val_batch_idx
            self.val_batches_seen += 1

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
            if not self.is_disabled and self.test_batch_idx % self.refresh_rate == 0:
                assert self.test_progress_bar.n == self.test_batch_idx
            self.test_batches_seen += 1

    progress_bar = CurrentProgressBar(refresh_rate=refresh_rate)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[progress_bar],
        progress_bar_refresh_rate=101,  # should not matter if custom callback provided
        limit_train_batches=1.0,
        num_sanity_val_steps=2,
        max_epochs=3,
    )
    assert trainer.progress_bar_callback.refresh_rate == refresh_rate

    trainer.fit(model)
    assert progress_bar.train_batches_seen == 3 * progress_bar.total_train_batches
    assert progress_bar.val_batches_seen == 3 * progress_bar.total_val_batches + trainer.num_sanity_val_steps
    assert progress_bar.test_batches_seen == 0

    trainer.validate(model)
    assert progress_bar.train_batches_seen == 3 * progress_bar.total_train_batches
    assert progress_bar.val_batches_seen == 4 * progress_bar.total_val_batches + trainer.num_sanity_val_steps
    assert progress_bar.test_batches_seen == 0

    trainer.test(model)
    assert progress_bar.train_batches_seen == 3 * progress_bar.total_train_batches
    assert progress_bar.val_batches_seen == 4 * progress_bar.total_val_batches + trainer.num_sanity_val_steps
    assert progress_bar.test_batches_seen == progress_bar.total_test_batches


@pytest.mark.parametrize('limit_val_batches', (0, 5))
def test_num_sanity_val_steps_progress_bar(tmpdir, limit_val_batches: int):
    """
    Test val_progress_bar total with 'num_sanity_val_steps' Trainer argument.
    """

    class CurrentProgressBar(ProgressBar):
        val_pbar_total = 0
        sanity_pbar_total = 0

        def on_sanity_check_end(self, *args):
            self.sanity_pbar_total = self.val_progress_bar.total
            super().on_sanity_check_end(*args)

        def on_validation_epoch_end(self, *args):
            self.val_pbar_total = self.val_progress_bar.total
            super().on_validation_epoch_end(*args)

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


def test_progress_bar_default_value(tmpdir):
    """ Test that a value of None defaults to refresh rate 1. """
    trainer = Trainer(default_root_dir=tmpdir)
    assert trainer.progress_bar_callback.refresh_rate == 1

    trainer = Trainer(default_root_dir=tmpdir, progress_bar_refresh_rate=None)
    assert trainer.progress_bar_callback.refresh_rate == 1


@mock.patch.dict(os.environ, {'COLAB_GPU': '1'})
def test_progress_bar_value_on_colab(tmpdir):
    """ Test that Trainer will override the default in Google COLAB. """
    trainer = Trainer(default_root_dir=tmpdir)
    assert trainer.progress_bar_callback.refresh_rate == 20

    trainer = Trainer(default_root_dir=tmpdir, progress_bar_refresh_rate=None)
    assert trainer.progress_bar_callback.refresh_rate == 20

    trainer = Trainer(default_root_dir=tmpdir, progress_bar_refresh_rate=19)
    assert trainer.progress_bar_callback.refresh_rate == 19


class MockedUpdateProgressBars(ProgressBar):
    """ Mocks the update method once bars get initializied. """

    def _mock_bar_update(self, bar):
        bar.update = Mock(wraps=bar.update)
        return bar

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        return self._mock_bar_update(bar)

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        return self._mock_bar_update(bar)

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        return self._mock_bar_update(bar)


@pytest.mark.parametrize(
    "train_batches,val_batches,refresh_rate,train_deltas,val_deltas", [
        [2, 3, 1, [1, 1, 1, 1, 1], [1, 1, 1]],
        [0, 0, 3, [], []],
        [1, 0, 3, [1], []],
        [1, 1, 3, [2], [1]],
        [5, 0, 3, [3, 2], []],
        [5, 2, 3, [3, 3, 1], [2]],
        [5, 2, 6, [6, 1], [2]],
    ]
)
def test_main_progress_bar_update_amount(
    tmpdir, train_batches: int, val_batches: int, refresh_rate: int, train_deltas: list, val_deltas: list
):
    """
    Test that the main progress updates with the correct amount together with the val progress. At the end of
    the epoch, the progress must not overshoot if the number of steps is not divisible by the refresh rate.
    """
    model = BoringModel()
    progress_bar = MockedUpdateProgressBars(refresh_rate=refresh_rate)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        callbacks=[progress_bar],
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(model)
    progress_bar.main_progress_bar.update.assert_has_calls([call(delta) for delta in train_deltas])
    if val_batches > 0:
        progress_bar.val_progress_bar.update.assert_has_calls([call(delta) for delta in val_deltas])


@pytest.mark.parametrize("test_batches,refresh_rate,test_deltas", [
    [1, 3, [1]],
    [3, 1, [1, 1, 1]],
    [5, 3, [3, 2]],
])
def test_test_progress_bar_update_amount(tmpdir, test_batches: int, refresh_rate: int, test_deltas: list):
    """
    Test that test progress updates with the correct amount.
    """
    model = BoringModel()
    progress_bar = MockedUpdateProgressBars(refresh_rate=refresh_rate)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_test_batches=test_batches,
        callbacks=[progress_bar],
        logger=False,
        checkpoint_callback=False,
    )
    trainer.test(model)
    progress_bar.test_progress_bar.update.assert_has_calls([call(delta) for delta in test_deltas])


def test_tensor_to_float_conversion(tmpdir):
    """Check tensor gets converted to float"""

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            self.log('foo', torch.tensor(0.123), prog_bar=True)
            self.log('bar', {"baz": torch.tensor([1])}, prog_bar=True)
            return super().training_step(batch, batch_idx)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=2,
        logger=False,
        checkpoint_callback=False,
    )
    trainer.fit(TestModel())

    pbar = trainer.progress_bar_callback.main_progress_bar
    actual = str(pbar.postfix)
    assert actual.endswith("foo=0.123, bar={'baz': tensor([1])}")


@pytest.mark.parametrize(
    "input_num, expected", [
        [1, '1'],
        [1.0, '1.000'],
        [0.1, '0.100'],
        [1e-3, '0.001'],
        [1e-5, '1e-5'],
        ['1.0', '1.000'],
        ['10000', '10000'],
        ['abc', 'abc'],
    ]
)
def test_tqdm_format_num(input_num: Union[str, int, float], expected: str):
    """ Check that the specialized tqdm.format_num appends 0 to floats and strings """
    assert tqdm.format_num(input_num) == expected


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


@mock.patch("pytorch_lightning.callbacks.progress.tqdm.write")
def test_progress_bar_print(tqdm_write, tmpdir):
    """ Test that printing in the LightningModule redirects arguments to the progress bar. """
    model = PrintModel()
    bar = ProgressBar()
    trainer = Trainer(
        default_root_dir=tmpdir,
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
    assert tqdm_write.call_count == 4
    assert tqdm_write.call_args_list == [
        call("training_step", end="", file=None, nolock=False),
        call("validation_step", end=os.linesep, file=sys.stderr, nolock=False),
        call("test_step", end=os.linesep, file=None, nolock=False),
        call("predict_step", end=os.linesep, file=None, nolock=False),
    ]


@mock.patch("pytorch_lightning.callbacks.progress.tqdm.write")
def test_progress_bar_print_no_train(tqdm_write, tmpdir):
    """ Test that printing in the LightningModule redirects arguments to the progress bar without training. """
    model = PrintModel()
    bar = ProgressBar()
    trainer = Trainer(
        default_root_dir=tmpdir,
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
    assert tqdm_write.call_count == 3
    assert tqdm_write.call_args_list == [
        call("validation_step", end=os.linesep, file=sys.stderr, nolock=False),
        call("test_step", end=os.linesep, file=None, nolock=False),
        call("predict_step", end=os.linesep, file=None, nolock=False),
    ]


@mock.patch('builtins.print')
@mock.patch("pytorch_lightning.callbacks.progress.tqdm.write")
def test_progress_bar_print_disabled(tqdm_write, mock_print, tmpdir):
    """ Test that printing in LightningModule goes through built-in print function when progress bar is disabled. """
    model = PrintModel()
    bar = ProgressBar()
    trainer = Trainer(
        default_root_dir=tmpdir,
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


def test_progress_bar_can_be_pickled():
    bar = ProgressBar()
    trainer = Trainer(fast_dev_run=True, callbacks=[bar], max_steps=1)
    model = BoringModel()

    pickle.dumps(bar)
    trainer.fit(model)
    pickle.dumps(bar)
    trainer.test(model)
    pickle.dumps(bar)
    trainer.predict(model)
    pickle.dumps(bar)
