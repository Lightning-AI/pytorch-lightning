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
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from tests.helpers import BoringDataModule, BoringModel


class BatchSizeDataModule(BoringDataModule):

    def __init__(self, batch_size=None):
        super().__init__()
        if batch_size is not None:
            self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.random_train, batch_size=getattr(self, "batch_size", 1))


class BatchSizeModel(BoringModel):

    def __init__(self, batch_size=None):
        super().__init__()
        if batch_size is not None:
            self.batch_size = batch_size


@pytest.mark.parametrize(
    "model,datamodule", [
        (BatchSizeModel(2), None),
        (BatchSizeModel(2), BatchSizeDataModule(2)),
        (BatchSizeModel(2), BatchSizeDataModule(None)),
        (BatchSizeModel(None), BatchSizeDataModule(2)),
    ]
)
def test_scale_batch_size_method_with_model_or_datamodule(tmpdir, model, datamodule):
    """ Test the tuner method `Tuner.scale_batch_size` with a datamodule. """
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=0,
        max_epochs=1,
    )
    tuner = Tuner(trainer)
    new_batch_size = tuner.scale_batch_size(
        model=model, mode="binsearch", init_val=4, max_trials=2, datamodule=datamodule
    )
    assert new_batch_size == 16
    if hasattr(model, "batch_size"):
        assert model.batch_size == 16
    if datamodule is not None and hasattr(datamodule, "batch_size"):
        assert datamodule.batch_size == 16
