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
import os

import torch
from torch.nn import functional as F
from torchvision import models, transforms

import pytorch_lightning as pl
from pytorch_lightning.utilities.hpu_datamodule import HPUDataModule

_DATASETS_PATH = "./data"


class RN50Module(pl.LightningModule):
    def __init__(self):
        super().__init__()

        backbone = models.resnet50()
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*layers)

        num_target_classes = 10
        self.classifier = torch.nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def accuracy(self, logits, y):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

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

    def validation_epoch_end(self, outputs) -> None:
        self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True)

    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", torch.stack(outputs).mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def main(args):

    data_path = args.data_path
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    val_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]

    data_module = HPUDataModule(
        train_dir,
        val_dir,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        num_workers=8,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    # Initialize a trainer
    trainer = pl.Trainer(devices=1, accelerator="hpu", max_epochs=1, max_steps=2)

    # Init our model
    model = RN50Module()

    trainer.fit(model, datamodule=data_module)
    trainer.validate(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pytorch Lightning ImageNet Training")
    parser.add_argument("--data-path", default=_DATASETS_PATH, help="dataset")
    args = parser.parse_args()

    main(args)
