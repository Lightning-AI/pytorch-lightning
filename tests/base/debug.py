import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from tests.base.datasets import TestingMNIST


# from test_models import assert_ok_test_acc, load_model, \
#     clear_save_dir, get_default_testtube_logger, get_default_hparams, init_save_dir, \
#     init_checkpoint_callback, reset_seed, set_random_master_port


class CoolModel(pl.LightningModule):

    def __init(self):
        super().__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x))

    def my_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'training_loss': self.my_loss(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'val_loss': self.my_loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs['val_loss']]).mean()
        return avg_loss

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    def train_dataloader(self):
        return DataLoader(TestingMNIST(train=True, num_samples=100), batch_size=16)

    def val_dataloader(self):
        return DataLoader(TestingMNIST(train=False, num_samples=50), batch_size=16)

    def test_dataloader(self):
        return DataLoader(TestingMNIST(train=False, num_samples=50), batch_size=16)
