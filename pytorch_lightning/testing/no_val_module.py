import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
from torchvision import transforms
from test_tube import HyperOptArgumentParser

from pytorch_lightning.root_module.root_module import LightningModule
from pytorch_lightning import data_loader

from .lm_test_module_base import LightningTestModelBase


class NoValModel(LightningTestModelBase):
    """
    Sample model to show how to define a template
    """
    pass
