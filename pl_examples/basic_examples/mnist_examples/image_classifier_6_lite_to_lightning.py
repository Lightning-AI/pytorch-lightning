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
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy
from torchvision import datasets, transforms

from pl_examples.basic_examples.mnist_examples.image_classifier_1_pytorch import Net
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.loops import Loop


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
    def __init__(self, model, lr, gamma):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
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
        optimizer = optim.Adadelta(self.parameters(), lr=self.hparams.lr)
        return optimizer, StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)


class TrainLoop(Loop):
    def __init__(self, lite, args, model, optimizer, scheduler, dataloader):
        super().__init__()
        self.lite = lite
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader

    @property
    def done(self) -> bool:
        return False

    def reset(self):
        self.dataloader_iter = enumerate(self.dataloader)

    def advance(self, epoch) -> None:
        batch_idx, batch = next(self.dataloader_iter)
        self.optimizer.zero_grad()
        loss = self.model.training_step(batch, batch_idx)
        self.lite.backward(loss)
        self.optimizer.zero_grad()

        if (batch_idx == 0) or ((batch_idx + 1) % self.args.log_interval == 0):
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(self.dataloader),
                    len(self.dataloader.dataset),
                    100.0 * batch_idx / len(self.dataloader),
                    loss.item(),
                )
            )

        if self.args.dry_run:
            raise StopIteration

    def on_run_end(self):
        self.scheduler.step()
        self.dataloader_iter = None


class TestLoop(Loop):
    def __init__(self, lite, args, model, dataloader):
        super().__init__()
        self.lite = lite
        self.args = args
        self.model = model
        self.dataloader = dataloader

    @property
    def done(self) -> bool:
        return False

    def reset(self):
        self.dataloader_iter = enumerate(self.dataloader)
        self.test_loss = 0

    def advance(self) -> None:
        batch_idx, batch = next(self.dataloader_iter)
        self.test_loss += self.model.test_step(batch, batch_idx)

        if self.args.dry_run:
            raise StopIteration

    def on_run_end(self):
        test_loss = self.lite.all_gather(self.test_loss).sum() / len(self.dataloader.dataset)

        if self.lite.is_global_zero:
            print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({self.model.val_acc.compute():.0f}%)\n")


class MainLoop(Loop):
    def __init__(self, lite, args, model, datamodule):
        super().__init__()
        self.lite = lite
        self.args = args
        self.model = model
        self.datamodule = datamodule
        self.epoch = 0

    @property
    def done(self) -> bool:
        return self.epoch >= self.args.epochs

    def reset(self):
        pass

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        if self.lite.is_global_zero:
            self.datamodule.prepare_data()
        self.datamodule.setup()

        train_loader, test_loader = self.lite.setup_dataloaders(
            self.datamodule.train_dataloader(), self.datamodule.train_dataloader()
        )

        optimizer, scheduler = self.model.configure_optimizers()
        model, optimizer = self.lite.setup(self.model, optimizer)

        self.train_loop = TrainLoop(self.lite, self.args, model, optimizer, scheduler, train_loader)
        self.test_loop = TestLoop(self.lite, self.args, model, test_loader)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self.train_loop.run(self.epoch)
        self.test_loop.run()
        self.epoch += 1


class Lite(LightningLite):
    def run(self, args):

        model = LiftModel(Net(), args.lr, args.gamma)
        dm = MNISTDataModule(args.batch_size)

        loop = MainLoop(self, args, model, dm)
        loop.run()

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

    Lite(**lite_kwargs).run(args)
