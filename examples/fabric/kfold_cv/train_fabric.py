# Copyright The Lightning AI team.
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
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from sklearn import model_selection
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import Accuracy
from torchvision.datasets import MNIST

from lightning.fabric import Fabric  # import Fabric
from lightning.fabric import seed_everything

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "..", "Datasets")


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train_dataloader(model, data_loader, optimizer, fabric, epoch, hparams, fold):
    # TRAINING LOOP
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        # NOTE: no need to call `.to(device)` on the data, target
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        fabric.backward(loss)  # instead of loss.backward()

        optimizer.step()
        if (batch_idx == 0) or ((batch_idx + 1) % hparams.log_interval == 0):
            print(
                "Fold {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    fold,
                    epoch,
                    batch_idx * len(data),
                    len(data_loader.dataset),
                    100.0 * batch_idx / len(data_loader),
                    loss.item(),
                )
            )

        if hparams.dry_run:
            break


def validate_dataloader(model, data_loader, fabric, hparams, metric_fn, fold):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            # NOTE: no need to call `.to(device)` on the data, target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()

            # WITHOUT TorchMetrics
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

            # WITH TorchMetrics
            metric_fn(output, target)

            if hparams.dry_run:
                break

    # all_gather is used to aggregated the value across processes
    test_loss = fabric.all_gather(test_loss).sum() / len(data_loader.dataset)

    # val acc
    val_acc = metric_fn.compute()

    print(f"\nFor fold: {fold} Test set: Average loss: {test_loss:.4f}, Accuracy: ({100 * val_acc:.0f}%)\n")
    metric_fn.reset()

    return val_acc


def run(hparams):
    # Create the Lightning Fabric object. The parameters like accelerator, strategy, devices etc. will be proided
    # by the command line. See all options: `lightning run model --help`
    fabric = Fabric()

    seed_everything(hparams.seed)  # instead of torch.manual_seed(...)

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    # This is meant to ensure the data are download only by 1 process.
    if fabric.is_global_zero:
        MNIST(DATASETS_PATH, download=True)
    fabric.barrier()

    # initialize dataset
    dataset = MNIST(DATASETS_PATH, train=True, transform=transform)

    # Loop over different folds
    kfold = model_selection.KFold(n_splits=5)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"Working on fold {fold}")

        # split dataset
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # initialize dataloaders
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams.batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams.batch_size, sampler=val_sampler)

        # don't forget to call `setup_dataloaders` to prepare for dataloaders for distributed training.
        train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

        model = Net()  # remove call to .to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=hparams.lr)

        # don't forget to call `setup` to prepare for model / optimizer for distributed training.
        # the model is moved automatically to the right device.
        model, optimizer = fabric.setup(model, optimizer)

        scheduler = StepLR(optimizer, step_size=1, gamma=hparams.gamma)

        # use torchmetrics instead of manually computing the accuracy
        test_acc = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

        # EPOCH LOOP
        for epoch in range(1, hparams.epochs + 1):

            # TRAINING LOOP
            train_dataloader(model, train_loader, optimizer, fabric, epoch, hparams, fold)

            scheduler.step()

            # VALIDATION LOOP
            validate_dataloader(model, val_loader, fabric, hparams, test_acc, fold)

    # When using distributed training, use `fabric.save`
    # to ensure the current process is allowed to save a checkpoint
    if hparams.save_model:
        fabric.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    # Arguments can be passed in through the CLI as normal and will be parsed here
    # Example:
    # lightning run model image_classifier.py accelerator=cuda --epochs=3
    parser = argparse.ArgumentParser(description="Fabric MNIST K-Fold Cross Validation Example")
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

    run(hparams)
