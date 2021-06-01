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

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything, Trainer
from tests.helpers.boring_model import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.datasets import SklearnDataset
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel


class IPUModel(BoringModel):

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return loss

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return loss

    def training_epoch_end(self, outputs) -> None:
        pass

    def validation_epoch_end(self, outputs) -> None:
        pass

    def test_epoch_end(self, outputs) -> None:
        pass


class IPUClassificationModel(ClassificationModel):

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        # todo (sean): currently IPU poptorch doesn't implicit convert bools to tensor
        # hence we use an explicit calculation for accuracy here. Once fixed in poptorch
        # we can use the accuracy metric.
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def validation_epoch_end(self, outputs) -> None:
        self.log('val_acc', torch.stack(outputs).mean())

    def test_epoch_end(self, outputs) -> None:
        self.log('test_acc', torch.stack(outputs).mean())


@RunIf(ipu=True)
@pytest.mark.parametrize('ipu_cores', [1, 4])
def test_all_stages(tmpdir, ipu_cores):
    model = IPUModel()
    trainer = Trainer(fast_dev_run=True, accelerator='ipu', ipu_cores=ipu_cores)
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model, model.val_dataloader())


@RunIf(ipu=True)
@pytest.mark.parametrize('ipu_cores', [1, 4])
def test_inference_only(tmpdir, ipu_cores):
    model = IPUModel()

    trainer = Trainer(fast_dev_run=True, accelerator='ipu', ipu_cores=ipu_cores)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model, model.val_dataloader())


def test_optimization(tmpdir):
    seed_everything(42)

    # Override to drop last uneven batch, as IPU poptorch does not support uneven inputs.
    class DataModule(ClassifDataModule):

        def train_dataloader(self):
            return DataLoader(
                SklearnDataset(self.x_train, self.y_train, self._x_type, self._y_type),
                batch_size=self.batch_size,
                drop_last=True
            )

        def val_dataloader(self):
            return DataLoader(
                SklearnDataset(self.x_valid, self.y_valid, self._x_type, self._y_type),
                batch_size=self.batch_size,
                drop_last=True
            )

        def test_dataloader(self):
            return DataLoader(
                SklearnDataset(self.x_test, self.y_test, self._x_type, self._y_type),
                batch_size=self.batch_size,
                drop_last=True
            )

    dm = DataModule(length=1024)
    model = IPUClassificationModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
        deterministic=True,
        ipu_cores=2,
    )

    # fit model
    trainer.fit(model, dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert dm.trainer is not None

    # validate
    result = trainer.validate(datamodule=dm)
    assert dm.trainer is not None
    assert result[0]['val_acc'] > 0.7

    # test
    result = trainer.test(datamodule=dm)
    assert dm.trainer is not None
    test_result = result[0]['test_acc']
    assert test_result > 0.6

    # test saved model
    model_path = os.path.join(tmpdir, 'model.pt')
    trainer.save_checkpoint(model_path)

    model = IPUClassificationModel.load_from_checkpoint(model_path)

    trainer = Trainer(default_root_dir=tmpdir, deterministic=True)

    result = trainer.test(model, dm.test_dataloader())
    saved_result = result[0]['test_acc']
    assert saved_result > 0.6 and (saved_result == test_result)


# todo add test for precision 16 and fully half precision + device iterations
