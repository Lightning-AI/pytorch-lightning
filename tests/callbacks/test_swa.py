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
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.utilities import _PYTORCH_GREATER_EQUAL_1_6_0
from tests.base import BoringModel, RandomDataset

if _PYTORCH_GREATER_EQUAL_1_6_0:
    from pytorch_lightning.callbacks import StochasticWeightAveraging
    from pytorch_lightning.callbacks.swa import LightningAveragedModel


@pytest.mark.skipif(not _PYTORCH_GREATER_EQUAL_1_6_0, reason="SWA available from in PyTorch 1.7.0")
def test_stochastic_weight_averaging_callback(tmpdir):

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer = nn.Sequential(
                nn.Linear(32, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def training_step(self, batch, batch_idx):
            output = self.forward(batch)
            loss = self.loss(batch, output)
            return {"loss": loss}

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

    class SwaCheck(StochasticWeightAveraging):

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            if self._model_contains_batch_norm and trainer.current_epoch == self._max_epochs:
                assert isinstance(pl_module, LightningAveragedModel)

        def on_train_epoch_end(self, trainer, pl_module, *_):
            super().on_train_epoch_end(trainer, pl_module, *_)
            if self._model_contains_batch_norm and trainer.current_epoch == self._max_epochs:
                assert not isinstance(trainer.get_model(), LightningAveragedModel)

    model = TestModel()
    swa_callback = SwaCheck(swa_epoch_start=2, swa_lrs=0.005)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=4,
        limit_train_batches=4,
        callbacks=[swa_callback]
    )
    trainer.fit(model)

    assert swa_callback.swa_model is not None
    assert trainer.model == model
    assert swa_callback.swa_model.n_averaged > 0
