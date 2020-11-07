"""
Example template for defining a Lightning Module with Hydra
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from pytorch_lightning.core import LightningModule
import hydra
import logging

# Hydra configures the Python logging subsystem automatically.
log = logging.getLogger(__name__)


class LightningTemplateModel(LightningModule):
    def __init__(self, model, data, scheduler, opt) -> "LightningTemplateModel":
        # init superclass
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.data = data
        self.opt = opt
        self.scheduler = scheduler
        self.c_d1 = nn.Linear(in_features=self.model.in_features, out_features=self.model.hidden_dim)
        self.c_d1_bn = nn.BatchNorm1d(self.model.hidden_dim)
        self.c_d1_drop = nn.Dropout(self.model.drop_prob)

        self.c_d2 = nn.Linear(in_features=self.model.hidden_dim, out_features=self.model.out_features)

        self.example_input_array = torch.zeros(2, 1, 28, 28)

    def forward(self, x):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        x = self.c_d1(x.view(x.size(0), -1))
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)
        x = self.c_d2(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {"val_loss": val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {"test_loss": test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(x["n_pred"] for x in outputs)
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": val_acc}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(x["n_pred"] for x in outputs)
        tensorboard_logs = {"test_loss": avg_loss, "test_acc": test_acc}
        return {"test_loss": avg_loss, "log": tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = hydra.utils.instantiate(self.opt, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def prepare_data(self):
        transform = transforms.Compose([hydra.utils.instantiate(trans) for trans in self.data.tf])
        self.train_set = hydra.utils.instantiate(self.data.ds, transform=transform, train=True)
        self.test_set = hydra.utils.instantiate(self.data.ds, transform=transform, train=False)

    def train_dataloader(self):
        log.info("Training data loader called.")
        return hydra.utils.instantiate(self.data.dl, dataset=self.train_set)

    def val_dataloader(self):
        log.info("Validation data loader called.")
        return hydra.utils.instantiate(self.data.dl, dataset=self.test_set)

    def test_dataloader(self):
        log.info("Test data loader called.")
        return hydra.utils.instantiate(self.data.dl, dataset=self.test_set)

