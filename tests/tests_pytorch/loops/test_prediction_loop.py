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
from unittest import mock
from unittest.mock import call

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


def test_prediction_loop_stores_predictions(tmp_path):
    class MyModel(BoringModel):
        def predict_step(self, batch, batch_idx):
            return batch_idx

    model = MyModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_predict_batches=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    predictions = trainer.predict(model, return_predictions=True)
    assert predictions == [0, 1]
    # the predictions are still available
    assert trainer.predict_loop.predictions == predictions

    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_predict_batches=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    predictions = trainer.predict(model, return_predictions=False)
    assert predictions is None
    assert trainer.predict_loop.predictions == []


def test_prediction_loop_batch_sampler_set_epoch_called(tmp_path):
    """Tests that set_epoch is called on the dataloader's batch sampler (if any) during prediction."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_predict_batches=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit_loop.epoch_progress.current.processed = 2

    with mock.patch("lightning.pytorch.overrides.distributed.IndexBatchSamplerWrapper.set_epoch") as set_epoch_mock:
        trainer.predict(model)
    assert set_epoch_mock.mock_calls == [call(2)]
