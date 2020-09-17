import torch
from torch.nn import Conv2d
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.metrics.regression import MSE
import pytorch_lightning as pl
from pytorch_lightning import Trainer


class MyDataset(Dataset):
    def __init__(self, size=100):
        super(MyDataset, self).__init__()
        self.data = torch.stack([idx * torch.ones(3, 100, 100) for idx in range(size)])
        self.idx_list = []

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_1 = Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1)
        self.loss = MSE()
        self.idx_list = []

    def forward(self, batch):
        return self.conv_1(batch)

    def training_step(self, batch, batch_idx):
        idx = batch[0, 0, 0, 0].detach()
        pred = self.forward(batch)
        loss = self.loss(pred, batch)
        print(batch_idx, idx)
        return {'loss': loss}

    def setup(self, stage):
        self.dataset = MyDataset()

    def train_dataloader(self):
        loader = DataLoader(self.dataset, batch_size=1, num_workers=20, pin_memory=True, shuffle=False)
        return loader

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=0.001)


def main():
    pl_model = MyModel()
    # trainer = Trainer(distributed_backend='ddp', num_nodes=1, gpus=2, overfit_batches=4)
    trainer = Trainer(distributed_backend="ddp", gpus=2, overfit_batches=5, max_epochs=4, check_val_every_n_epoch=100)
    trainer.fit(pl_model)


if __name__ == '__main__':
    main()