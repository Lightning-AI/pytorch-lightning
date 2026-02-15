import torch

import lightning as L
from lightning.pytorch.plugins import DistributedAsyncCheckpointIO

# -- dataset --
x = torch.randn(100, 4)  # random dataset of size 100, containing 4 features
y = torch.randint(0, 2, (100,))  # random labels: {0, 1} (2 classes)

dataset = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, drop_last=True)


# -- lightning module --


class SimpleModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 8)
        self.act = torch.nn.ReLU()
        self.out = torch.nn.Linear(8, 2)  # number of classes 2

    def forward(self, x):
        return self.out(self.act(self.layer(x)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.cross_entropy(y_pred, y)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main():
    model = SimpleModel()
    dist_async_ckpt = DistributedAsyncCheckpointIO()

    trainer = L.Trainer(max_epochs=10, log_every_n_steps=1, plugins=[dist_async_ckpt])

    trainer.fit(
        model=model,
        train_dataloaders=dataloader,
        ckpt_path="lightning_logs/version_1/checkpoints/epoch=9-step=250.ckpt",
    )


if __name__ == "__main__":
    main()
