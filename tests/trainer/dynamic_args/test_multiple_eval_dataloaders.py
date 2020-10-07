from pytorch_lightning import Trainer
from tests.base.boring_model import BoringModel
import torch
from torch.utils.data import Dataset


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
        row_log_interval=1,
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

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

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
        row_log_interval=1,
        weights_summary=None,
    )

    trainer.fit(model)