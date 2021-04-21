import os
import torch
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning import LightningModule, Trainer


class RandomDataset(IterableDataset):
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        return self.gen()


class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        return self.layer(batch)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)

    def train_dataloader(self):
        return DataLoader(RandomDataset(self.gen))

    def gen(self):
        if self.current_epoch == 0:
            # produce data in epoch 0
            yield torch.rand(2)
        # no data otherwise


def run():
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=2,
        weights_summary=None,
    )
    trainer.fit(model)


if __name__ == '__main__':
    run()