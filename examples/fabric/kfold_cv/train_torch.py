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
from torch.utils.data import DataLoader, SubsetRandomSampler
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
        output = F.log_softmax(x, dim=1)
        return output


def train_dataloader(model, data_loader, optimizer, epoch, hparams, fold):
    # TRAINING LOOP
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        # NOTE: no need to call `.to(device)` on the data, target
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

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


def validate_dataloader(model, data_loader, hparams, fold):
    model.eval()
    loss = 0
    correct = 0
    len_ = 0
    with torch.no_grad():
        for data, target in data_loader:
            # NOTE: no need to call `.to(device)` on the data, target
            output = model(data)
            loss += F.nll_loss(output, target, reduction="sum").item()

            # compute accuracy
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            len_ += pred.shape[0]

            if hparams.dry_run:
                break

    # compute average loss
    loss /= len_

    # compute acc
    acc = 100.0 * correct / len_
    print(f"\nFor fold: {fold} Validation set: Average loss: {loss:.4f}, Accuracy: ({acc:.0f}%)\n")
    return acc


def run(hparams):
    torch.manual_seed(hparams.seed)

    use_cuda = torch.cuda.is_available()
    torch.device("cuda" if use_cuda else "cpu")
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    # initialize dataset
    dataset = MNIST(DATASETS_PATH, train=True, transform=transform)

    # Loop over different folds (shuffle = False by default so reproducible)
    kfold = model_selection.KFold(n_splits=5)

    # initialize n_splits models and optimizers
    models = [Net() for _ in range(kfold.n_splits)]
    optimizers = [optim.Adadelta(model.parameters(), lr=hparams.lr) for model in models]

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

            # get model and optimizer for the current fold
            model, optimizer = models[fold], optimizers[fold]

            # train and validate
            train_dataloader(model, train_loader, optimizer, epoch, hparams, fold)
            epoch_acc += validate_dataloader(model, val_loader, hparams, fold)

        # log epoch metrics
        print(f"Epoch {epoch} - Average acc: {epoch_acc / kfold.n_splits}")

        if hparams.dry_run:
            break

    # save model
    if hparams.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch MNIST K-Fold Cross Validation Example")
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
