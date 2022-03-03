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

import os
import torch
from torch.nn import functional as F

import pytorch_lightning as pl

from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule

import habana_frameworks.torch.core as htcore
from pytorch_lightning.callbacks import HPUStatsMonitor
from pytorch_lightning.plugins import HPUPrecisionPlugin
from pytorch_lightning.strategies.hpu import HPUStrategy
from pytorch_lightning.strategies.hpu_parallel import HPUParallelStrategy
import argparse

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--hpus', default=1, type=int, metavar='N',
                        help='number of habana accelerator for training (default: 1)')
    parser.add_argument('--hmp', dest='is_hmp', action='store_true', help='enable habana mixed precision mode')
    parser.add_argument('--hmp-bf16', default='', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')

    args = parser.parse_args()

    return args

class LitClassifier(pl.LightningModule):

    def __init__(self):
        super(LitClassifier, self).__init__()

        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def validation_epoch_end(self, outputs) -> None:
        self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True)

    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", torch.stack(outputs).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

if __name__ == "__main__":

    args = parse_args()

    # Init our model
    model = LitClassifier()

    # Init DataLoader from MNIST Dataset
    dm = MNISTDataModule(batch_size=args.batch_size)

    # TBD: import these keys from hmp
    hmp_keys = ["level", "verbose", "bf16_ops", "fp32_ops"]
    hmp_params = dict.fromkeys(hmp_keys)
    hmp_params["level"] = args.hmp_opt_level
    hmp_params["verbose"] = args.hmp_verbose
    hmp_params["bf16_ops"] = args.hmp_bf16 #"./pl_examples/hpu_examples/simple_mnist/ops_bf16_mnist.txt"
    hmp_params["fp32_ops"] = args.hmp_fp32 #"./pl_examples/hpu_examples/simple_mnist/ops_fp32_mnist.txt"

    hpu_stats = HPUStatsMonitor(log_save_dir="habana_ptl_log", exp_name="mnist")

    parallel_devices = args.hpus
    hpustrat_1 = HPUStrategy(
        device=torch.device("hpu"), precision_plugin=HPUPrecisionPlugin(precision=16, hmp_params=hmp_params)
    )
    hpustrat_8 = HPUParallelStrategy(
        parallel_devices=[torch.device("hpu")] * parallel_devices,
        precision_plugin=HPUPrecisionPlugin(precision=16, hmp_params=hmp_params),
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        strategy=hpustrat_8 if (parallel_devices == 8) else hpustrat_1,
        devices=parallel_devices,
        callbacks=[hpu_stats],
        max_epochs=args.epochs,
        default_root_dir=os.getcwd(),
        accelerator="hpu",
    )


    # Train the model ⚡
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
