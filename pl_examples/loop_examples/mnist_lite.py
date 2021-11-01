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
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy

from pl_examples.basic_examples.mnist_datamodule import MNIST
from pl_examples.basic_examples.mnist_examples.image_classifier_1_pytorch import Net
from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.loops import Loop


class TrainLoop(Loop):
    def __init__(self, lite, args, model, optimizer, scheduler, dataloader):
        super().__init__()
        self.lite = lite
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.dataloader_iter = None

    @property
    def done(self) -> bool:
        return False

    def reset(self):
        self.dataloader_iter = enumerate(self.dataloader)

    def advance(self, epoch) -> None:
        batch_idx, (data, target) = next(self.dataloader_iter)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        self.lite.backward(loss)
        self.optimizer.step()

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
        self.dataloader_iter = None
        self.accuracy = Accuracy().to(lite.device)
        self.test_loss = 0

    @property
    def done(self) -> bool:
        return False

    def reset(self):
        self.dataloader_iter = enumerate(self.dataloader)
        self.test_loss = 0
        self.accuracy.reset()

    def advance(self) -> None:
        _, (data, target) = next(self.dataloader_iter)
        output = self.model(data)
        self.test_loss += F.nll_loss(output, target)
        self.accuracy(output, target)

        if self.args.dry_run:
            raise StopIteration

    def on_run_end(self):
        test_loss = self.lite.all_gather(self.test_loss).sum() / len(self.dataloader.dataset)

        if self.lite.is_global_zero:
            print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({self.accuracy.compute():.0f}%)\n")


class MainLoop(Loop):
    def __init__(self, lite, args, model, optimizer, scheduler, train_loader, test_loader):
        super().__init__()
        self.lite = lite
        self.args = args
        self.epoch = 0
        self.train_loop = TrainLoop(self.lite, self.args, model, optimizer, scheduler, train_loader)
        self.test_loop = TestLoop(self.lite, self.args, model, test_loader)

    @property
    def done(self) -> bool:
        return self.epoch >= self.args.epochs

    def reset(self):
        pass

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self.train_loop.run(self.epoch)
        self.test_loop.run()

        if self.args.dry_run:
            raise StopIteration

        self.epoch += 1


class Lite(LightningLite):
    def run(self, hparams):
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        if self.is_global_zero:
            MNIST("./data", download=True)
        self.barrier()
        train_dataset = MNIST("./data", train=True, transform=transform)
        test_dataset = MNIST("./data", train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, hparams.batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, hparams.test_batch_size)

        train_loader, test_loader = self.setup_dataloaders(train_loader, test_loader)

        model = Net()
        optimizer = optim.Adadelta(model.parameters(), lr=hparams.lr)

        model, optimizer = self.setup(model, optimizer)
        scheduler = StepLR(optimizer, step_size=1, gamma=hparams.gamma)

        MainLoop(self, hparams, model, optimizer, scheduler, train_loader, test_loader).run()

        if hparams.save_model and self.is_global_zero:
            self.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightningLite MNIST Example with Lightning Loops.")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 14)")
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

    seed_everything(hparams.seed)

    Lite(accelerator="cpu", devices=1).run(hparams)
