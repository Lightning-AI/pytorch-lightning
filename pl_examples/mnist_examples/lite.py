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
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from pl_examples.mnist_examples.pytorch import Net, test, train
from pytorch_lightning.lite import LightningLite


class Lite(LightningLite):
    def run(self, args):
        train_kwargs = {"batch_size": args.batch_size}
        test_kwargs = {"batch_size": args.test_batch_size}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST("./data", train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        # this line ensures distributed training works properly with your dataloaders
        train_loader, test_loader = self.setup_dataloaders(train_loader, test_loader)

        model = Net()
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        # this line ensures distributed training works properly the selected strategy.
        model, optimizer = self.setup(model, optimizer)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, self.device, train_loader, optimizer, epoch, compute_backward=self.backward)
            test(model, self.device, test_loader, reduce_loss=self.reduce_loss, should_print=self.should_print)
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")


    def reduce_loss(self, loss):
        return self.all_gather(loss).mean()

    
    def should_print(self):
        return self.is_global_zero


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
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

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        lite_kwargs = {"accelerator": "gpu", "devices": torch.cuda.device_count()}
    else:
        lite_kwargs = {"accelerator": "gpu"}

    Lite(**lite_kwargs).run(args)
