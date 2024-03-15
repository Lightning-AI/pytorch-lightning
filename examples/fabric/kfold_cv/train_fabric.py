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
from lightning.fabric import Fabric, seed_everything
from sklearn import model_selection
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics.classification import Accuracy
from torchvision.datasets import MNIST

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
        return F.log_softmax(x, dim=1)


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
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)}"
                f" ({100.0 * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

        if hparams.dry_run:
            break


def validate_dataloader(model, data_loader, fabric, hparams, fold, acc_metric):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            # NOTE: no need to call `.to(device)` on the data, target
            output = model(data)
            loss += F.nll_loss(output, target, reduction="sum").item()

            # Accuracy with torchmetrics
            acc_metric.update(output, target)

            if hparams.dry_run:
                break

    # all_gather is used to aggregate the value across processes
    loss = fabric.all_gather(loss).sum() / len(data_loader.dataset)

    # compute acc
    acc = acc_metric.compute() * 100

    print(f"\nFor fold: {fold} Validation set: Average loss: {loss:.4f}, Accuracy: ({acc:.0f}%)\n")
    return acc


def run(hparams):
    # Create the Lightning Fabric object. The parameters like accelerator, strategy, devices etc. will be proided
    # by the command line. See all options: `fabric run --help`
    fabric = Fabric()

    seed_everything(hparams.seed)  # instead of torch.manual_seed(...)

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    # Let rank 0 download the data first, then everyone will load MNIST
    with fabric.rank_zero_first(local=False):  # set `local=True` if your filesystem is not shared between machines
        dataset = MNIST(DATASETS_PATH, train=True, download=True, transform=transform)

    # Loop over different folds (shuffle = False by default so reproducible)
    folds = hparams.folds
    kfold = model_selection.KFold(n_splits=folds)

    # initialize n_splits models and optimizers
    models = [Net() for _ in range(kfold.n_splits)]
    optimizers = [optim.Adadelta(model.parameters(), lr=hparams.lr) for model in models]

    # fabric setup for models and optimizers
    for i in range(kfold.n_splits):
        models[i], optimizers[i] = fabric.setup(models[i], optimizers[i])

    # Accuracy using torchmetrics
    acc_metric = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    # loop over epochs
    for epoch in range(1, hparams.epochs + 1):
        # loop over folds
        epoch_acc = 0
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"Working on fold {fold}")

            # initialize dataloaders based on folds
            batch_size = hparams.batch_size
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_ids))
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_ids))

            # set up dataloaders to move data to the correct device
            train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

            # get model and optimizer for the current fold
            model, optimizer = models[fold], optimizers[fold]

            # train and validate
            train_dataloader(model, train_loader, optimizer, fabric, epoch, hparams, fold)
            epoch_acc += validate_dataloader(model, val_loader, fabric, hparams, fold, acc_metric)
            acc_metric.reset()

        # log epoch metrics
        print(f"Epoch {epoch} - Average acc: {epoch_acc / kfold.n_splits}")

        if hparams.dry_run:
            break

    # When using distributed training, use `fabric.save`
    # to ensure the current process is allowed to save a checkpoint
    if hparams.save_model:
        fabric.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    # Arguments can be passed in through the CLI as normal and will be parsed here
    # Example:
    # fabric run image_classifier.py accelerator=cuda --epochs=3
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
    parser.add_argument("--folds", type=int, default=5, help="number of folds for k-fold cross validation")
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    hparams = parser.parse_args()

    run(hparams)
