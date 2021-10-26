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
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import Accuracy

from pl_examples.basic_examples.mnist_datamodule import MNIST
from pl_examples.basic_examples.mnist_examples.image_classifier_1_pytorch import Net
from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite


def train(lite, args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        lite.backward(loss)
        optimizer.step()
        if (batch_idx == 0) or ((batch_idx + 1) % args.log_interval == 0):
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
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
    acc = Accuracy().to(lite.device)
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            acc.update(output, target)
            if args.dry_run:
                break

    test_loss = lite.all_gather(test_loss).sum() / len(test_loader.dataset)

    if lite.is_global_zero:
        print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({acc.compute():.0f}%)\n")


class Lite(LightningLite):
    def run(self, args):
        train_kwargs = {"batch_size": args.batch_size}
        test_kwargs = {"batch_size": args.test_batch_size}
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train_dataset = MNIST("./data", train=True, download=True, transform=transform)
        test_dataset = MNIST("./data", train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        train_loader, test_loader = self.setup_dataloaders(train_loader, test_loader)

        model = Net()
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        model, optimizer = self.setup(model, optimizer)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(self, args, model, train_loader, optimizer, epoch)
            test(self, args, model, test_loader)
            scheduler.step()

            if args.dry_run:
                break

        if args.save_model and self.is_global_zero:
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":

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
