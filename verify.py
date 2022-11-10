import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


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

    def on_train_start(self) -> None:
        print("The callback states at the beginning of training are")
        for c in self.trainer.callbacks:
            print(c.state_key, c.state_dict())

    def on_train_end(self) -> None:
        print("The callback states at the end of training are")
        for c in self.trainer.callbacks:
            print(c.state_key, c.state_dict())


    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss1", loss)
        self.log("valid_loss2", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=2)
    val_data = DataLoader(RandomDataset(32, 64), batch_size=2)

    checkpoint1 = ModelCheckpoint(monitor="valid_loss1")  # in v1.7 -> True
    checkpoint2 = ModelCheckpoint(monitor="valid_loss2", save_on_train_epoch_end=False)  # in v1.7 -> False
    checkpoint3 = ModelCheckpoint(monitor="train_loss", save_on_train_epoch_end=True)  # in v1.7 -> True

    #
    # model = BoringModel()
    # trainer = Trainer(
    #     default_root_dir=os.getcwd(),
    #     limit_train_batches=1,
    #     limit_val_batches=1,
    #     limit_test_batches=1,
    #     num_sanity_val_steps=0,
    #     max_epochs=1,
    #     callbacks=[checkpoint1, checkpoint2, checkpoint3],
    #     enable_model_summary=False,
    # )
    # trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)
    name = "check/verify1-7-7.ckpt"
    # trainer.save_checkpoint(name)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        max_epochs=2,
        enable_model_summary=False,
        callbacks=[checkpoint3, checkpoint3],
    )
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data, ckpt_path=name)


"""
ModelCheckpoint{'monitor': 'valid_loss1', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None, 'save_on_train_epoch_end': True} {'monitor': 'valid_loss1', 'best_model_score': tensor(1.5518), 'best_model_path': '/Users/adrian/repositories/lightning/lightning_logs/version_6/checkpoints/epoch=0-step=1-v1.ckpt', 'current_score': tensor(1.5518), 'dirpath': '/Users/adrian/repositories/lightning/lightning_logs/version_6/checkpoints', 'best_k_models': {'/Users/adrian/repositories/lightning/lightning_logs/version_6/checkpoints/epoch=0-step=1-v1.ckpt': tensor(1.5518)}, 'kth_best_model_path': '/Users/adrian/repositories/lightning/lightning_logs/version_6/checkpoints/epoch=0-step=1-v1.ckpt', 'kth_value': tensor(1.5518), 'last_model_path': ''}
ModelCheckpoint{'monitor': 'valid_loss2', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None, 'save_on_train_epoch_end': False} {'monitor': 'valid_loss2', 'best_model_score': tensor(1.5518), 'best_model_path': '/Users/adrian/repositories/lightning/lightning_logs/version_6/checkpoints/epoch=0-step=1.ckpt', 'current_score': tensor(1.5518), 'dirpath': '/Users/adrian/repositories/lightning/lightning_logs/version_6/checkpoints', 'best_k_models': {'/Users/adrian/repositories/lightning/lightning_logs/version_6/checkpoints/epoch=0-step=1.ckpt': tensor(1.5518)}, 'kth_best_model_path': '/Users/adrian/repositories/lightning/lightning_logs/version_6/checkpoints/epoch=0-step=1.ckpt', 'kth_value': tensor(1.5518), 'last_model_path': ''}
ModelCheckpoint{'monitor': 'train_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None, 'save_on_train_epoch_end': True} {'monitor': 'train_loss', 'best_model_score': tensor(1.5103), 'best_model_path': '/Users/adrian/repositories/lightning/lightning_logs/version_6/checkpoints/epoch=0-step=1-v2.ckpt', 'current_score': tensor(1.5103), 'dirpath': '/Users/adrian/repositories/lightning/lightning_logs/version_6/checkpoints', 'best_k_models': {'/Users/adrian/repositories/lightning/lightning_logs/version_6/checkpoints/epoch=0-step=1-v2.ckpt': tensor(1.5103)}, 'kth_best_model_path': '/Users/adrian/repositories/lightning/lightning_logs/version_6/checkpoints/epoch=0-step=1-v2.ckpt', 'kth_value': tensor(1.5103), 'last_model_path': ''}

"""
if __name__ == "__main__":
    run()
