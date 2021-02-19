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
"""Computer vision example on Transfer Learning.
This computer vision example illustrates how one could fine-tune a pre-trained
network (by default, a ResNet50 is used) using pytorch-lightning. For the sake
of this example, the 'cats and dogs dataset' (~60MB, see `DATA_URL` below) and
the proposed network (denoted by `TransferLearningModel`, see below) is
trained for 15 epochs.

The training consists of three stages.

From epoch 0 to 4, the feature extractor (the pre-trained network) is frozen except
maybe for the BatchNorm layers (depending on whether `train_bn = True`). The BatchNorm
layers (if `train_bn = True`) and the parameters of the classifier are trained as a
single parameters group with lr = 1e-2.

From epoch 5 to 9, the last two layer groups of the pre-trained network are unfrozen
and added to the optimizer as a new parameter group with lr = 1e-4 (while lr = 1e-3
for the first parameter group in the optimizer).

Eventually, from epoch 10, all the remaining layer groups of the pre-trained network
are unfrozen and added to the optimizer as a third parameter group. From epoch 10,
the parameters of the pre-trained network are trained with lr = 1e-5 while those of
the classifier is trained with lr = 1e-4.

Note:
    See: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""
import argparse
import os
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pytorch_lightning import _logger as log
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities import rank_zero_info

DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

#  --- Finetuning Callback ---


class MilestonesFinetuning(BaseFinetuning):

    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = False):
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule):
        self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        if epoch == self.milestones[0]:
            # unfreeze 5 last layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[-5:], optimizer=optimizer, train_bn=self.train_bn
            )

        elif epoch == self.milestones[1]:
            # unfreeze remaing layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[:-5], optimizer=optimizer, train_bn=self.train_bn
            )


class CatDogImageDataModule(LightningDataModule):

    def __init__(
        self,
        dl_path: Union[str, Path],
        num_workers: int = 0,
        batch_size: int = 8,
    ):
        super().__init__()

        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size

    def prepare_data(self):
        """Download images and prepare images datasets."""
        download_and_extract_archive(url=DATA_URL, download_root=self._dl_path, remove_finished=True)

    @property
    def data_path(self):
        return Path(self._dl_path).joinpath("cats_and_dogs_filtered")

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), self.normalize_transform
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), self.normalize_transform])

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool):
        """Train/validation loaders."""
        if train:
            dataset = self.create_dataset(self.data_path.joinpath("train"), self.train_transform)
        else:
            dataset = self.create_dataset(self.data_path.joinpath("validation"), self.valid_transform)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=train)

    def train_dataloader(self):
        log.info("Training data loaded.")
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info("Validation data loaded.")
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            "--num-workers", default=0, type=int, metavar="W", help="number of CPU workers", dest="num_workers"
        )
        parser.add_argument(
            "--batch-size", default=8, type=int, metavar="W", help="number of sample in a batch", dest="batch_size"
        )
        return parser


#  --- Pytorch-lightning module ---


class TransferLearningModel(pl.LightningModule):

    def __init__(
        self,
        backbone: str = "resnet50",
        train_bn: bool = True,
        milestones: tuple = (5, 10),
        batch_size: int = 32,
        lr: float = 1e-2,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        **kwargs,
    ) -> None:
        """
        Args:
            dl_path: Path where the data will be downloaded
        """
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers

        self.__build_model()

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.save_hyperparameters()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)

        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)

        # 2. Classifier:
        _fc_layers = [
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.Linear(32, 1),
        ]
        self.fc = nn.Sequential(*_fc_layers)

        # 3. Loss:
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x):
        """Forward pass. Returns logits."""

        # 1. Feature extraction:
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)

        # 2. Classifier (returns logits):
        x = self.fc(x)

        return torch.sigmoid(x)

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)

        # 2. Compute loss
        train_loss = self.loss(y_logits, y_true)

        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(y_logits, y_true.int()), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)

        # 2. Compute loss
        self.log("val_loss", self.loss(y_logits, y_true), prog_bar=True)

        # 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(y_logits, y_true.int()), prog_bar=True)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            "--backbone",
            default="resnet50",
            type=str,
            metavar="BK",
            help="Name (as in ``torchvision.models``) of the feature extractor",
        )
        parser.add_argument(
            "--epochs", default=15, type=int, metavar="N", help="total number of epochs", dest="nb_epochs"
        )
        parser.add_argument("--batch-size", default=8, type=int, metavar="B", help="batch size", dest="batch_size")
        parser.add_argument("--gpus", type=int, default=0, help="number of gpus to use")
        parser.add_argument(
            "--lr", "--learning-rate", default=1e-3, type=float, metavar="LR", help="initial learning rate", dest="lr"
        )
        parser.add_argument(
            "--lr-scheduler-gamma",
            default=1e-1,
            type=float,
            metavar="LRG",
            help="Factor by which the learning rate is reduced at each milestone",
            dest="lr_scheduler_gamma",
        )
        parser.add_argument(
            "--train-bn",
            default=False,
            type=bool,
            metavar="TB",
            help="Whether the BatchNorm layers should be trainable",
            dest="train_bn",
        )
        parser.add_argument(
            "--milestones", default=[2, 4], type=list, metavar="M", help="List of two epochs milestones"
        )
        return parser


def main(args: argparse.Namespace) -> None:
    """Train the model.

    Args:
        args: Model hyper-parameters

    Note:
        For the sake of the example, the images dataset will be downloaded
        to a temporary directory.
    """

    datamodule = CatDogImageDataModule(
        dl_path=os.path.join(args.root_data_path, 'data'), batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = TransferLearningModel(**vars(args))
    finetuning_callback = MilestonesFinetuning(milestones=args.milestones)

    trainer = pl.Trainer(
        weights_summary=None,
        progress_bar_refresh_rate=1,
        num_sanity_val_steps=0,
        gpus=args.gpus,
        max_epochs=args.nb_epochs,
        callbacks=[finetuning_callback]
    )

    trainer.fit(model, datamodule=datamodule)


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--root-data-path",
        metavar="DIR",
        type=str,
        default=Path.cwd().as_posix(),
        help="Root directory where to download the data",
        dest="root_data_path",
    )
    parser = TransferLearningModel.add_model_specific_args(parent_parser)
    parser = CatDogImageDataModule.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    cli_lightning_logo()
    main(get_args())
