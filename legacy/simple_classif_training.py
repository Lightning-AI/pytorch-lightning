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

import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything
from pytorch_lightning.callbacks import EarlyStopping

PATH_LEGACY = os.path.dirname(__file__)


class SklearnDataset(Dataset):

    def __init__(self, x, y, x_type, y_type):
        self.x = x
        self.y = y
        self._x_type = x_type
        self._y_type = y_type

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=self._x_type), torch.tensor(self.y[idx], dtype=self._y_type)

    def __len__(self):
        return len(self.y)


class SklearnDataModule(LightningDataModule):

    def __init__(self, sklearn_dataset, x_type, y_type, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size
        self._x, self._y = sklearn_dataset
        self._split_data()
        self._x_type = x_type
        self._y_type = y_type

    def _split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self._x, self._y, test_size=0.20, random_state=42
        )
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            self.x_train, self.y_train, test_size=0.40, random_state=42
        )

    def train_dataloader(self):
        return DataLoader(
            SklearnDataset(self.x_train, self.y_train, self._x_type, self._y_type),
            shuffle=True,
            batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            SklearnDataset(self.x_valid, self.y_valid, self._x_type, self._y_type), batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            SklearnDataset(self.x_test, self.y_test, self._x_type, self._y_type), batch_size=self.batch_size
        )


class ClassifDataModule(SklearnDataModule):

    def __init__(self, num_features=24, length=6000, num_classes=3, batch_size=128):
        data = make_classification(
            n_samples=length,
            n_features=num_features,
            n_classes=num_classes,
            n_clusters_per_class=2,
            n_informative=int(num_features / num_classes),
            random_state=42,
        )
        super().__init__(data, x_type=torch.float32, y_type=torch.long, batch_size=batch_size)


class ClassificationModel(LightningModule):

    def __init__(self, num_features=24, num_classes=3, lr=0.01):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        for i in range(3):
            setattr(self, f"layer_{i}", nn.Linear(num_features, num_features))
            setattr(self, f"layer_{i}a", torch.nn.ReLU())
        setattr(self, "layer_end", nn.Linear(num_features, num_classes))

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_0a(x)
        x = self.layer_1(x)
        x = self.layer_1a(x)
        x = self.layer_2(x)
        x = self.layer_2a(x)
        x = self.layer_end(x)
        logits = F.softmax(x, dim=1)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc(logits, y), prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log('val_loss', F.cross_entropy(logits, y), prog_bar=False)
        self.log('val_acc', self.valid_acc(logits, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log('test_loss', F.cross_entropy(logits, y), prog_bar=False)
        self.log('test_acc', self.test_acc(logits, y), prog_bar=True)


def main_train(dir_path, max_epochs: int = 20):
    seed_everything(42)
    stopping = EarlyStopping(monitor="val_acc", mode="max", min_delta=0.005)
    trainer = pl.Trainer(
        default_root_dir=dir_path,
        gpus=int(torch.cuda.is_available()),
        precision=16 if torch.cuda.is_available() else 32,
        checkpoint_callback=True,
        callbacks=[stopping],
        min_epochs=3,
        max_epochs=max_epochs,
        accumulate_grad_batches=2,
        deterministic=True,
    )

    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer.fit(model, datamodule=dm)
    res = trainer.test(model, datamodule=dm)
    assert res[0]['test_loss'] <= 0.7
    assert res[0]['test_acc'] >= 0.85
    assert trainer.current_epoch < (max_epochs - 1)
    print(res)


if __name__ == '__main__':
    path_dir = os.path.join(PATH_LEGACY, 'checkpoints', str(pl.__version__))
    main_train(path_dir)
