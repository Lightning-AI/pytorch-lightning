import torch
from pytorch_lightning import LightningModule, Trainer, LightningDataModule


class BoringData(LightningDataModule):
    pass

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

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    model = BoringModel()
    trainer = Trainer()
    trainer.fit(model, BoringData())
    # trainer.fit(model, train_dataloaders=None)
    # trainer.fit(model, datamodule=None)


if __name__ == "__main__":
    run()
