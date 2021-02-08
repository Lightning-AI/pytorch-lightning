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
import torch
from torch.utils.data import Dataset

from pytorch_lightning import Trainer
from tests.helpers.boring_model import BoringModel


class RandomDatasetA(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return torch.zeros(1)

    def __len__(self):
        return self.len


class RandomDatasetB(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return torch.ones(1)

    def __len__(self):
        return self.len


def test_multiple_eval_dataloaders_tuple(tmpdir):

    class TestModel(BoringModel):

        def validation_step(self, batch, batch_idx, dataloader_idx):
            if dataloader_idx == 0:
                assert batch.sum() == 0
            elif dataloader_idx == 1:
                assert batch.sum() == 11
            else:
                raise Exception('should only have two dataloaders')

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def val_dataloader(self):
            dl1 = torch.utils.data.DataLoader(RandomDatasetA(32, 64), batch_size=11)
            dl2 = torch.utils.data.DataLoader(RandomDatasetB(32, 64), batch_size=11)
            return [dl1, dl2]

    model = TestModel()
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)


def test_multiple_eval_dataloaders_list(tmpdir):

    class TestModel(BoringModel):

        def validation_step(self, batch, batch_idx, dataloader_idx):
            if dataloader_idx == 0:
                assert batch.sum() == 0
            elif dataloader_idx == 1:
                assert batch.sum() == 11
            else:
                raise Exception('should only have two dataloaders')

        def val_dataloader(self):
            dl1 = torch.utils.data.DataLoader(RandomDatasetA(32, 64), batch_size=11)
            dl2 = torch.utils.data.DataLoader(RandomDatasetB(32, 64), batch_size=11)
            return dl1, dl2

    model = TestModel()
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)


def test_multiple_optimizers_multiple_dataloaders(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(BoringModel):

        def on_train_epoch_start(self) -> None:
            self.opt_0_seen = False
            self.opt_1_seen = False

        def training_step(self, batch, batch_idx, optimizer_idx):
            if optimizer_idx == 0:
                self.opt_0_seen = True
            elif optimizer_idx == 1:
                self.opt_1_seen = True
            else:
                raise Exception('should only have two optimizers')

            self.training_step_called = True
            loss = self.step(batch[0])
            return loss

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def validation_step(self, batch, batch_idx, dataloader_idx):
            if dataloader_idx == 0:
                assert batch.sum() == 0
            elif dataloader_idx == 1:
                assert batch.sum() == 11
            else:
                raise Exception('should only have two dataloaders')

        def val_dataloader(self):
            dl1 = torch.utils.data.DataLoader(RandomDatasetA(32, 64), batch_size=11)
            dl2 = torch.utils.data.DataLoader(RandomDatasetB(32, 64), batch_size=11)
            return dl1, dl2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer, optimizer_2

    model = TestModel()
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)
    assert model.opt_0_seen
    assert model.opt_1_seen
