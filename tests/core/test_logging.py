import os
import torch
from torch.utils.data import Dataset
from pytorch_lightning import Trainer
from tests.base import BoringModel
from tests.base.boring_model import RandomDictStringDataset


def test_validation_step_with_string_data_logging():
    class TestModel(BoringModel):
        def on_train_epoch_start(self) -> None:
            print("override any method to prove your bug")

        def training_step(self, batch, batch_idx):
            output = self.layer(batch["x"])
            loss = self.loss(batch, output)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch["x"])
            loss = self.loss(batch, output)
            self.log("x", loss)
            return {"x": loss}

    # fake data
    train_data = torch.utils.data.DataLoader(RandomDictStringDataset(32, 64))
    val_data = torch.utils.data.DataLoader(RandomDictStringDataset(32, 64))

    # model
    model = TestModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model, train_data, val_data)
