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

"""Here are 5 required steps to convert to LightningLite.

1. Subclass LightningLite and override its ``run`` method.

2. Move the body of your existing ``run`` function into the ``run`` method.

3. Remove all ``.to``, ``.cuda`` etc calls since LightningLite will take care of it.

4. Apply ``setup`` over each model and optimizers pair, ``setup_dataloaders`` on all your dataloaders,
and replace ``loss.backward()`` with ``self.backward(loss)``.

5. Instantiate your LightningLite and call its ``run`` method.

Learn more from the documentation: https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.html.
"""

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
from pytorch_lightning.lite import LightningLite  # import LightningLite


class Lite(LightningLite):
    def run(self, hparams):
        self.hparams = hparams
        seed_everything(hparams.seed)  # instead of torch.manual_seed(...)

        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        # This is meant to ensure the data are download only by 1 process.
        if self.is_global_zero:
            MNIST("./data", download=True)
        self.barrier()
        train_dataset = MNIST("./data", train=True, transform=transform)
        test_dataset = MNIST("./data", train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=hparams.batch_size,
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size)

        # don't forget to call `setup_dataloaders` to prepare for dataloaders for distributed training.
        train_loader, test_loader = self.setup_dataloaders(train_loader, test_loader)

        model = Net()  # remove call to .to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=hparams.lr)

        # don't forget to call `setup` to prepare for model / optimizer for distributed training.
        # the model is moved automatically to the right device.
        model, optimizer = self.setup(model, optimizer)

        scheduler = StepLR(optimizer, step_size=1, gamma=hparams.gamma)

        # use torchmetrics instead of manually computing the accuracy
        test_acc = Accuracy().to(self.device)

        # EPOCH LOOP
        for epoch in range(1, hparams.epochs + 1):

            # TRAINING LOOP
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # NOTE: no need to call `.to(device)` on the data, target
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                self.backward(loss)  # instead of loss.backward()

                optimizer.step()
                if (batch_idx == 0) or ((batch_idx + 1) % hparams.log_interval == 0):
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )
                    if hparams.dry_run:
                        break

            scheduler.step()

            # TESTING LOOP
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for data, target in test_loader:
                    # NOTE: no need to call `.to(device)` on the data, target
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction="sum").item()

                    # WITHOUT TorchMetrics
                    # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    # correct += pred.eq(target.view_as(pred)).sum().item()

                    # WITH TorchMetrics
                    test_acc(output, target)

                    if hparams.dry_run:
                        break

            # all_gather is used to aggregated the value across processes
            test_loss = self.all_gather(test_loss).sum() / len(test_loader.dataset)

            print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({test_acc.compute():.0f}%)\n")
            test_acc.reset()

            if hparams.dry_run:
                break

        # When using distributed training, use `self.save`
        # to ensure the current process is allowed to save a checkpoint
        if hparams.save_model:
            self.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightningLite MNIST Example")
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
