import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite

import torch
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringLite(LightningLite):
    def run(self):
        seed_everything(1)

        train_dataloader = DataLoader(RandomDataset(32, 64), batch_size=2)
        train_dataloader = self.setup_dataloaders(train_dataloader)

        model = nn.Linear(32, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        model, optimizer = self.setup(model, optimizer)

        model.train()
        for batch_idx, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = output.sum()
            self.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    BoringLite(accelerator="gpu", devices="2", strategy="bagua").run()
