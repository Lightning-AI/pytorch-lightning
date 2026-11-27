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
from unittest.mock import ANY, Mock, call

import pytest
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class DummyPredictionWriter(BasePredictionWriter):
    def write_on_batch_end(self, *_, **__):
        pass

    def write_on_epoch_end(self, *_, **__):
        pass


def test_prediction_writer_invalid_write_interval():
    """Test that configuring an unknown interval name raises an error."""
    with pytest.raises(MisconfigurationException, match=r"`write_interval` should be one of \['batch"):
        DummyPredictionWriter("something")


@pytest.mark.parametrize("interval", ["BATCH", "Epoch", "Batch_And_Epoch"])
def test_prediction_writer_case_insensitive_write_interval(interval):
    """Test that case-insensitive `write_interval` values accepted by `LightningEnum` are resolved."""
    writer = DummyPredictionWriter(interval)
    assert writer.interval.value == interval.lower()


def test_prediction_writer_hook_call_intervals(tmp_path):
    """Test that the `write_on_batch_end` and `write_on_epoch_end` hooks get invoked based on the defined interval."""
    DummyPredictionWriter.write_on_batch_end = Mock()
    DummyPredictionWriter.write_on_epoch_end = Mock()

    dataloader = DataLoader(RandomDataset(32, 64))

    model = BoringModel()
    cb = DummyPredictionWriter("batch_and_epoch")
    trainer = Trainer(default_root_dir=tmp_path, logger=False, limit_predict_batches=4, callbacks=cb)
    results = trainer.predict(model, dataloaders=dataloader)
    assert len(results) == 4
    assert cb.write_on_batch_end.call_count == 4
    assert cb.write_on_epoch_end.call_count == 1

    DummyPredictionWriter.write_on_batch_end.reset_mock()
    DummyPredictionWriter.write_on_epoch_end.reset_mock()

    cb = DummyPredictionWriter("batch_and_epoch")
    trainer = Trainer(default_root_dir=tmp_path, logger=False, limit_predict_batches=4, callbacks=cb)
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    assert cb.write_on_batch_end.call_count == 4
    assert cb.write_on_epoch_end.call_count == 1

    DummyPredictionWriter.write_on_batch_end.reset_mock()
    DummyPredictionWriter.write_on_epoch_end.reset_mock()

    cb = DummyPredictionWriter("batch")
    trainer = Trainer(default_root_dir=tmp_path, logger=False, limit_predict_batches=4, callbacks=cb)
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    assert cb.write_on_batch_end.call_count == 4
    assert cb.write_on_epoch_end.call_count == 0

    DummyPredictionWriter.write_on_batch_end.reset_mock()
    DummyPredictionWriter.write_on_epoch_end.reset_mock()

    cb = DummyPredictionWriter("epoch")
    trainer = Trainer(default_root_dir=tmp_path, logger=False, limit_predict_batches=4, callbacks=cb)
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    assert cb.write_on_batch_end.call_count == 0
    assert cb.write_on_epoch_end.call_count == 1


@pytest.mark.parametrize("num_workers", [0, 2])
def test_prediction_writer_batch_indices(num_workers, tmp_path):
    DummyPredictionWriter.write_on_batch_end = Mock()
    DummyPredictionWriter.write_on_epoch_end = Mock()

    dataloader = DataLoader(RandomDataset(32, 64), batch_size=4, num_workers=num_workers)
    model = BoringModel()
    writer = DummyPredictionWriter("batch_and_epoch")
    trainer = Trainer(default_root_dir=tmp_path, logger=False, limit_predict_batches=4, callbacks=writer)
    trainer.predict(model, dataloaders=dataloader)

    writer.write_on_batch_end.assert_has_calls([
        call(trainer, model, ANY, [0, 1, 2, 3], ANY, 0, 0),
        call(trainer, model, ANY, [4, 5, 6, 7], ANY, 1, 0),
        call(trainer, model, ANY, [8, 9, 10, 11], ANY, 2, 0),
        call(trainer, model, ANY, [12, 13, 14, 15], ANY, 3, 0),
    ])

    writer.write_on_epoch_end.assert_has_calls([
        call(trainer, model, ANY, [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]]),
    ])


def test_batch_level_batch_indices(tmp_path):
    """Test that batch_indices are returned when `return_predictions=False`."""
    DummyPredictionWriter.write_on_batch_end = Mock()

    class CustomBoringModel(BoringModel):
        def on_predict_epoch_end(self, *args, **kwargs):
            assert self.trainer.predict_loop.epoch_batch_indices == [[]]

    writer = DummyPredictionWriter("batch")
    model = CustomBoringModel()
    dataloader = DataLoader(RandomDataset(32, 64), batch_size=4)
    trainer = Trainer(default_root_dir=tmp_path, logger=False, limit_predict_batches=4, callbacks=writer)
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)

    writer.write_on_batch_end.assert_has_calls([
        call(trainer, model, ANY, [0, 1, 2, 3], ANY, 0, 0),
        call(trainer, model, ANY, [4, 5, 6, 7], ANY, 1, 0),
        call(trainer, model, ANY, [8, 9, 10, 11], ANY, 2, 0),
        call(trainer, model, ANY, [12, 13, 14, 15], ANY, 3, 0),
    ])
    assert trainer.predict_loop.current_batch_indices == []


def test_prediction_writer_clears_batch_indices_for_non_indexed_dataloader(tmp_path):
    """Test that `current_batch_indices` does not leak from an indexed DataLoader to a non-indexed iterable."""
    DummyPredictionWriter.write_on_batch_end = Mock()

    writer = DummyPredictionWriter("batch")
    model = BoringModel()
    indexed_loader = DataLoader(RandomDataset(32, 8), batch_size=4)
    non_indexed_iterable = iter([indexed_loader.dataset[0].unsqueeze(0), indexed_loader.dataset[1].unsqueeze(0)])
    trainer = Trainer(default_root_dir=tmp_path, logger=False, callbacks=writer)
    trainer.predict(model, dataloaders=[indexed_loader, non_indexed_iterable], return_predictions=False)

    writer.write_on_batch_end.assert_has_calls([
        call(trainer, model, ANY, [0, 1, 2, 3], ANY, 0, 0),
        call(trainer, model, ANY, [4, 5, 6, 7], ANY, 1, 0),
        call(trainer, model, ANY, [], ANY, 0, 1),
        call(trainer, model, ANY, [], ANY, 1, 1),
    ])

