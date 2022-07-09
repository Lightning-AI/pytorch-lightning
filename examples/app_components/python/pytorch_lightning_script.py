import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64))

    val_dataloader = train_dataloader
    test_dataloader = train_dataloader
    predict_dataloader = train_dataloader


if __name__ == "__main__":
    model = BoringModel()
    trainer = Trainer(max_epochs=1, accelerator="cpu", devices=2, strategy="ddp")
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)
