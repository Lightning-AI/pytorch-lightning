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
from abc import ABC, abstractmethod
from typing import Callable

import torch.nn as nn
from torch.optim import Optimizer
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class ParityModel(ABC, nn.Module):
    """Defines the interface for a model in a Fabric-PyTorch parity test."""

    @abstractmethod
    def get_optimizer(self, *args, **kwargs) -> Optimizer:
        pass

    @abstractmethod
    def get_dataloader(self, *args, **kwargs) -> DataLoader:
        pass

    @abstractmethod
    def get_loss_function(self) -> Callable:
        pass


class ConvNet(ParityModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_optimizer(self):
        return torch.optim.SGD(self.parameters(), lr=0.0001)

    def get_dataloader(self, dataset_size=100, batch_size=4):
        inputs = torch.rand(dataset_size, 3, 32, 32)
        labels = torch.randint(0, 10, (dataset_size, ))
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        return dataloader

    def get_loss_function(self):
        return F.cross_entropy
