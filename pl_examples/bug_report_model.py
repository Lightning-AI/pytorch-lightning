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
        print(f"training_step, {batch_idx=}: {loss=}")
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def training_epoch_end(self, outputs):
        print("training_epoch_end:", [id(x["loss"]) for x in outputs])


if __name__ == "__main__":
    dl = DataLoader(RandomDataset(32, 100), batch_size=10)

    model = BoringModel()
    trainer = Trainer(max_epochs=1, progress_bar_refresh_rate=0)
    trainer.fit(model, dl)
