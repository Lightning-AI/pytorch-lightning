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
import sys

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
from argparse import ArgumentParser

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from tests.base.boring_model import *


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = BoringModel()

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)

if __name__ == '__main__':
    cli_main()
