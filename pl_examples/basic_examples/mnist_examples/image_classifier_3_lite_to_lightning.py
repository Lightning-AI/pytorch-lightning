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
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy
from torchvision import datasets, transforms

from pl_examples.basic_examples.mnist_examples.image_classifier_1_pytorch import Net
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything
from pytorch_lightning.lite import LightningLite


class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    @property
    def transform(self):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self) -> None:
        datasets.MNIST("./data", download=True)

    def setup(self, stage=None) -> None:
        self.train_dataset = datasets.MNIST("./data", train=True, download=False, transform=self.transform)
        self.test_dataset = datasets.MNIST("./data", train=False, download=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)


class LiftModel(LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y.long())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y.long())
        self.val_acc(logits, y.long())
        return loss

    def configure_optimizers(self):
        return optim.Adadelta(self.parameters(), lr=args.lr)


def train(lite, args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = model.training_step(batch, batch_idx)
        lite.backward(loss)
        optimizer.step()
        if (batch_idx == 0) or ((batch_idx + 1) % args.log_interval == 0):
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(batch[0]),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(lite, args, model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            test_loss += model.test_step(batch, batch_idx)
            if args.dry_run:
                break

    test_loss = lite.all_gather(test_loss).sum() / len(test_loader.dataset)

    if lite.is_global_zero:
        print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({model.val_acc.compute():.0f}%)\n")


class Lite(LightningLite):
    def run(self, args):

        dm = MNISTDataModule(args.batch_size)
        if self.is_global_zero:
            dm.prepare_data()
        dm.setup()
        train_loader, test_loader = self.setup_dataloaders(dm.train_dataloader(), dm.train_dataloader())

        model = LiftModel(Net(), args.lr)
        model, optimizer = self.setup(model, model.configure_optimizers())

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(self, args, model, train_loader, optimizer, epoch)
            test(self, args, model, test_loader)
            scheduler.step()

        if args.save_model and self.is_global_zero:
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="LightningLite MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=14, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
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
    args = parser.parse_args()

    seed_everything(args.seed)

    if torch.cuda.is_available():
        lite_kwargs = {"accelerator": "gpu", "devices": torch.cuda.device_count()}
    else:
        lite_kwargs = {"accelerator": "cpu"}

    if torch.cuda.is_available():
        lite_kwargs = {"accelerator": "gpu", "devices": torch.cuda.device_count()}
    else:
        lite_kwargs = {"accelerator": "cpu"}

    Lite(**lite_kwargs).run(args)
