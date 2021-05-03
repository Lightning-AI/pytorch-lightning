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

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel


def test_prediction_writer(tmpdir):

    class CustomPredictionWriter(BasePredictionWriter):

        def __init__(self, writer_interval: str):
            super().__init__(writer_interval)

            self.write_on_batch_end_called = False
            self.write_on_epoch_end_called = False

        def write_on_batch_end(self, *args, **kwargs):
            self.write_on_batch_end_called = True

        def write_on_epoch_end(self, *args, **kwargs):
            self.write_on_epoch_end_called = True

    with pytest.raises(MisconfigurationException, match=r"`write_interval` should be one of \['batch"):
        CustomPredictionWriter("something")

    model = BoringModel()
    cb = CustomPredictionWriter("batch_and_epoch")
    trainer = Trainer(limit_predict_batches=4, callbacks=cb)
    results = trainer.predict(model, dataloaders=model.train_dataloader())
    assert len(results) == 4
    assert cb.write_on_batch_end_called
    assert cb.write_on_epoch_end_called

    cb = CustomPredictionWriter("batch_and_epoch")
    trainer = Trainer(limit_predict_batches=4, callbacks=cb)
    trainer.predict(model, dataloaders=model.train_dataloader(), return_predictions=False)
    assert cb.write_on_batch_end_called
    assert cb.write_on_epoch_end_called

    cb = CustomPredictionWriter("batch")
    trainer = Trainer(limit_predict_batches=4, callbacks=cb)
    trainer.predict(model, dataloaders=model.train_dataloader(), return_predictions=False)
    assert cb.write_on_batch_end_called
    assert not cb.write_on_epoch_end_called

    cb = CustomPredictionWriter("epoch")
    trainer = Trainer(limit_predict_batches=4, callbacks=cb)
    trainer.predict(model, dataloaders=model.train_dataloader(), return_predictions=False)
    assert not cb.write_on_batch_end_called
    assert cb.write_on_epoch_end_called
