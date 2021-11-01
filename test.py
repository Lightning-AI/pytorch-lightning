import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer


class RandomDataset(Dataset):
    def __init__(self, size, length):
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

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def on_after_backward(self) -> None:
        print("Engine", {name: p.grad for name, p in self.trainer.model.named_parameters()})
        print("LM", {name: p.grad for name, p in self.named_parameters()})

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        max_epochs=1,
        enable_model_summary=False,
        strategy="deepspeed",
        gpus=1,
    )
    trainer.fit(model, train_dataloaders=train_data)


if __name__ == "__main__":
    run()
