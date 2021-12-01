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
from unittest.mock import Mock

import pytest
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel, RandomDataset


class DummyPredictionWriter(BasePredictionWriter):
    def write_on_batch_end(self, *args, **kwargs):
        pass

    def write_on_epoch_end(self, *args, **kwargs):
        pass


def test_prediction_writer_invalid_write_interval():
    with pytest.raises(MisconfigurationException, match=r"`write_interval` should be one of \['batch"):
        DummyPredictionWriter("something")


def test_prediction_writer_hook_call_intervals(tmpdir):
    DummyPredictionWriter.write_on_batch_end = Mock()
    DummyPredictionWriter.write_on_epoch_end = Mock()

    dataloader = DataLoader(RandomDataset(32, 64))

    model = BoringModel()
    cb = DummyPredictionWriter("batch_and_epoch")
    trainer = Trainer(limit_predict_batches=4, callbacks=cb)
    results = trainer.predict(model, dataloaders=dataloader)
    assert len(results) == 4
    assert cb.write_on_batch_end.call_count == 4
    assert cb.write_on_epoch_end.call_count == 1

    DummyPredictionWriter.write_on_batch_end.reset_mock()
    DummyPredictionWriter.write_on_epoch_end.reset_mock()

    cb = DummyPredictionWriter("batch_and_epoch")
    trainer = Trainer(limit_predict_batches=4, callbacks=cb)
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    assert cb.write_on_batch_end.call_count == 4
    assert cb.write_on_epoch_end.call_count == 1

    DummyPredictionWriter.write_on_batch_end.reset_mock()
    DummyPredictionWriter.write_on_epoch_end.reset_mock()

    cb = DummyPredictionWriter("batch")
    trainer = Trainer(limit_predict_batches=4, callbacks=cb)
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    assert cb.write_on_batch_end.call_count == 4
    assert cb.write_on_epoch_end.call_count == 0

    DummyPredictionWriter.write_on_batch_end.reset_mock()
    DummyPredictionWriter.write_on_epoch_end.reset_mock()

    cb = DummyPredictionWriter("epoch")
    trainer = Trainer(limit_predict_batches=4, callbacks=cb)
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    assert cb.write_on_batch_end.call_count == 0
    assert cb.write_on_epoch_end.call_count == 1
