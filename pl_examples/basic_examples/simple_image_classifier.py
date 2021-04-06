from argparse import ArgumentParser
from pprint import pprint

import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin


class LitClassifier(pl.LightningModule):

    def __init__(self, hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitClassifier")
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser


def cli_main():
    pl.seed_everything(1234)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    dm = MNISTDataModule.from_argparse_args(args, num_workers=2)
    model = LitClassifier(args.hidden_dim, args.learning_rate)

    trainer = Trainer(
        max_epochs=10,
        callbacks=[EarlyStopping(monitor="valid_loss", patience=0, min_delta=0.01, verbose=True)],
        limit_train_batches=10,
        num_processes=2,
        accelerator="ddp_cpu",
        # plugins=[DDPPlugin(find_unused_parameters=False)],
    )
    trainer.fit(model, datamodule=dm)
    result = trainer.test(model, datamodule=dm)
    pprint(result)


if __name__ == '__main__':
    cli_main()
