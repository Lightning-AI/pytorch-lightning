# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Here are the steps to convert from LightningLite to a LightningModule.

1. Start implementing the ``training_step``, ``forward``, ``train_dataloader`` and ``configure_optimizers``
methods on the LightningLite class.

2. Utilize those methods within the ``run`` method.

3. Finally, switch to LightningModule and validate that your results are still reproducible (next script).

Learn more from the documentation: https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.html.
"""

import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy

from pl_examples.basic_examples.mnist_datamodule import MNIST
from pl_examples.basic_examples.mnist_examples.image_classifier_1_pytorch import Net
from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite


class Lite(LightningLite):
    """Lite is starting to look like a LightningModule."""

    def run(self, hparams):
        self.hparams = hparams
        seed_everything(hparams.seed)  # instead of torch.manual_seed(...)

        self.model = Net()
        [optimizer], [scheduler] = self.configure_optimizers()
        model, optimizer = self.setup(self.model, optimizer)

        if self.is_global_zero:
            # In multi-device training, this code will only run on the first process / GPU
            self.prepare_data()

        train_loader, test_loader = self.setup_dataloaders(self.train_dataloader(), self.train_dataloader())

        self.test_acc = Accuracy().to(self.device)

        # EPOCH LOOP
        for epoch in range(1, hparams.epochs + 1):

            # TRAINING LOOP
            self.model.train()
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.training_step(batch, batch_idx)
                self.backward(loss)
                optimizer.step()

                if (batch_idx == 0) or ((batch_idx + 1) % hparams.log_interval == 0):
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            (batch_idx + 1) * self.hparams.batch_size,
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )
                    if hparams.dry_run:
                        break

            scheduler.step()

            # TESTING LOOP
            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    test_loss += self.test_step(batch, batch_idx)
                    if hparams.dry_run:
                        break

            test_loss = self.all_gather(test_loss).sum() / len(test_loader.dataset)

            print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({self.test_acc.compute():.0f}%)\n")
            self.test_acc.reset()

            if hparams.dry_run:
                break

        if hparams.save_model:
            self.save(model.state_dict(), "mnist_cnn.pt")

    # Methods for the `LightningModule` conversion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Here you compute and return the training loss and compute extra training metrics."""
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        return loss

    def test_step(self, batch, batch_idx):
        """Here you compute and return the testing loss and compute extra testing metrics."""
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        self.test_acc(logits, y.long())
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.model.parameters(), lr=self.hparams.lr)
        return [optimizer], [StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)]

    # Methods for the `LightningDataModule` conversion

    @property
    def transform(self):
        return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self) -> None:
        MNIST("./data", download=True)

    def train_dataloader(self):
        train_dataset = MNIST("./data", train=True, download=False, transform=self.transform)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        test_dataset = MNIST("./data", train=False, download=False, transform=self.transform)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.hparams.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightningLite to LightningModule MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    hparams = parser.parse_args()

    Lite(accelerator="auto", devices="auto").run(hparams)
