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

import torch
import torch.nn.functional as F
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
from lightning.pytorch.utilities.model_helpers import get_torchvision_model
from tests_pytorch import _PATH_DATASETS
from torch.utils.data import DataLoader

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
    from torchvision.datasets import CIFAR10


class ParityModuleCIFAR(LightningModule):
    def __init__(self, backbone="resnet101", hidden_dim=1024, learning_rate=1e-3, weights="DEFAULT"):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_classes = 10
        self.backbone = get_torchvision_model(backbone, weights=weights)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1000, hidden_dim), torch.nn.Linear(hidden_dim, self.num_classes)
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self._loss = []  # needed for checking if the loss is the same as vanilla torch

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        y_hat = self.classifier(y_hat)
        loss = F.cross_entropy(y_hat, y)
        self._loss.append(loss.item())
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return DataLoader(
            CIFAR10(root=_PATH_DATASETS, train=True, download=True, transform=self.transform),
            batch_size=32,
            num_workers=1,
        )
