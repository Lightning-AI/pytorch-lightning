from lightning.app.utilities.imports import _is_pytorch_lightning_available

if _is_pytorch_lightning_available():
    import torch
    from pytorch_lightning import LightningDataModule, LightningModule
    from pytorch_lightning.cli import LightningCLI
    from torch.utils.data import DataLoader, Dataset

    class RandomDataset(Dataset):
        def __init__(self, size, length):
            self.len = length
            self.data = torch.randn(length, size)

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return self.len

    class BoringDataModule(LightningDataModule):
        def __init__(self, root_folder: str = "./", batch_size: int = 32):
            super().__init__()

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

        def val_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

        def test_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

        def predict_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

    class BoringDataModule2(LightningDataModule):
        def __init__(self, root_folder: str = "./", batch_size: int = 32, num_workers: int = 6):
            super().__init__()

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

        def val_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

        def test_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

        def predict_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

    class BoringModel(LightningModule):
        def __init__(self, hidden_size: int = 16):
            super().__init__()
            self.layer = torch.nn.Linear(32, 2)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, batch_idx):
            loss = self(batch).sum()
            self.log("train_loss", loss)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            loss = self(batch).sum()
            self.log("valid_loss", loss)

        def test_step(self, batch, batch_idx):
            loss = self(batch).sum()
            self.log("test_loss", loss)

        def configure_optimizers(self):
            return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    class BoringModel2(LightningModule):
        def __init__(self, hidden_size: int = 16, batch_norm: bool = False):
            super().__init__()
            self.layer = torch.nn.Linear(32, 2)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, batch_idx):
            loss = self(batch).sum()
            self.log("train_loss", loss)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            loss = self(batch).sum()
            self.log("valid_loss", loss)

        def test_step(self, batch, batch_idx):
            loss = self(batch).sum()
            self.log("test_loss", loss)

        def configure_optimizers(self):
            return torch.optim.SGD(self.layer.parameters(), lr=0.1)


if __name__ == "__main__":
    LightningCLI()
