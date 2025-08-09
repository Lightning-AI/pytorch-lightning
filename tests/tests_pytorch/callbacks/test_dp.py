import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl
from lightning.pytorch.callbacks.differential_privacy import DifferentialPrivacy


class MockDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, input_dim=10, num_classes=2):
        super().__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_classes = num_classes

    def setup(self, stage=None):
        # Generate random data
        X = torch.randn(1000, self.input_dim)
        y = torch.randint(0, self.num_classes, (1000,))
        dataset = TensorDataset(X, y)
        self.train_data, self.val_data = torch.utils.data.random_split(dataset, [800, 200])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)


class SimpleClassifier(pl.LightningModule):
    def __init__(self, input_dim=10, num_classes=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Linear(input_dim, num_classes)
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits.softmax(dim=-1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits.softmax(dim=-1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def test_privacy_callback() -> None:
    """Test on simple classifier.

    We test that:
    * the privacy budget has been spent (`epsilon > 0`);
    * spent budget is  greater than max privacy budget;
    * traininng did not stop because `max_steps` has been reached, but because the total budget has been spent.

    """
    # choose dataset
    datamodule = MockDataModule()

    # choose model: choose a model with more than one optim
    model = SimpleClassifier()

    # init DP callback
    dp_cb = DifferentialPrivacy(budget=0.232, private_dataloader=False)

    # define training
    max_steps = 20
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        max_steps=max_steps,
        callbacks=[dp_cb],
    )
    trainer.fit(model=model, datamodule=datamodule)

    # tests
    epsilon, best_alpha = dp_cb.get_privacy_spent()
    print(f"Total spent budget {epsilon} with alpha: {best_alpha}")
    assert epsilon > 0, f"No privacy budget has been spent: {epsilon}"
    assert epsilon >= dp_cb.budget, (
        f"Spent budget is not greater than max privacy budget: epsilon = {epsilon} and budget = {dp_cb.budget}"
    )
    assert trainer.global_step < max_steps, (
        "Traininng stopped because max_steps has been reached, not because the total budget has been spent."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s"])
