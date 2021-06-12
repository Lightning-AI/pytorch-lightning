import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pl_examples.bug_report_model import ToyTask
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    task = ToyTask()

    dataset = [{"model_input": torch.randn(20, 10), "label": torch.randn(20, 5)} for _ in range(10)]

    train_dataloader = DataLoader(dataset, batch_size=None)
    val_dataloader = DataLoader(dataset, batch_size=None)

    model_checkpoint = ModelCheckpoint(
        save_last=True,
        every_n_val_epochs=1,
    )

    trainer = pl.Trainer(
        gpus=2,
        precision=16,
        max_epochs=4,
        reload_dataloaders_every_epoch=True,
        limit_train_batches=10,
        limit_val_batches=10,
        limit_test_batches=10,
        callbacks=[model_checkpoint],
        resume_from_checkpoint=
        "/home/adrian/repositories/pytorch-lightning/lightning_logs/version_82/checkpoints/last.ckpt",
    )
    trainer.fit(task, train_dataloader)
