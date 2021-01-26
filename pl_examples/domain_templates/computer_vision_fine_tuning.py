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
import logging
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator, Optional, Union

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import Module
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo

log = logging.getLogger(__name__)
BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"


#  --- Utility functions ---


def _make_trainable(module: Module) -> None:
    """Unfreezes a given module.

    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: Module, train_bn: bool = True) -> None:
    """Freezes the layers of a given module.

    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module: Module, n: Optional[int] = None, train_bn: bool = True) -> None:
    """Freezes the layers up to index n (if n is not None).

    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)


def filter_params(module: Module, train_bn: bool = True) -> Generator:
    """Yields the trainable parameters of a given module.

    Args:
        module: A given module
        train_bn: If True, leave the BatchNorm layers in training mode

    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, train_bn=train_bn):
                yield param


def _unfreeze_and_add_param_group(
    module: Module, optimizer: Optimizer, lr: Optional[float] = None, train_bn: bool = True
):
    """Unfreezes a module and adds its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
    optimizer.add_param_group(
        {
            "params": filter_params(module=module, train_bn=train_bn),
            "lr": params_lr / 10.0,
        }
    )


#  --- Pytorch-lightning module ---


class TransferLearningModel(pl.LightningModule):
    """Transfer Learning with pre-trained ResNet50.

    >>> with TemporaryDirectory(dir='.') as tmp_dir:
    ...     TransferLearningModel(tmp_dir)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    TransferLearningModel(
      (feature_extractor): Sequential(...)
      (fc): Sequential(...)
    )
    """

    def __init__(
        self,
        dl_path: Union[str, Path],
        backbone: str = "resnet50",
        train_bn: bool = True,
        milestones: tuple = (5, 10),
        batch_size: int = 8,
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
        self.dl_path = dl_path
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers

        self.dl_path = dl_path
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
        self.feature_extractor = torch.nn.Sequential(*_layers)
        freeze(module=self.feature_extractor, train_bn=self.train_bn)

        # 2. Classifier:
        _fc_layers = [torch.nn.Linear(2048, 256), torch.nn.Linear(256, 32), torch.nn.Linear(32, 1)]
        self.fc = torch.nn.Sequential(*_fc_layers)

        # 3. Loss:
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x):
        """Forward pass. Returns logits."""

        # 1. Feature extraction:
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)

        # 2. Classifier (returns logits):
        x = self.fc(x)

        return x

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def train(self, mode=True):
        super().train(mode=mode)

        epoch = self.current_epoch
        if epoch < self.milestones[0] and mode:
            # feature extractor is frozen (except for BatchNorm layers)
            freeze(module=self.feature_extractor, train_bn=self.train_bn)

        elif self.milestones[0] <= epoch < self.milestones[1] and mode:
            # Unfreeze last two layers of the feature extractor
            freeze(module=self.feature_extractor, n=-2, train_bn=self.train_bn)

    def on_epoch_start(self):
        """Use `on_epoch_start` to unfreeze layers progressively."""
        optimizer = self.trainer.optimizers[0]
        if self.current_epoch == self.milestones[0]:
            _unfreeze_and_add_param_group(
                module=self.feature_extractor[-2:], optimizer=optimizer, train_bn=self.train_bn
            )

        elif self.current_epoch == self.milestones[1]:
            _unfreeze_and_add_param_group(
                module=self.feature_extractor[:-2], optimizer=optimizer, train_bn=self.train_bn
            )

    def training_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)

        # 2. Compute loss & accuracy:
        train_loss = self.loss(y_logits, y_true)
        accuracy = self.train_acc(y_logits, y_true)

        # 3. Outputs:
        tqdm_dict = {"train_loss": train_loss}
        self.log_dict(tqdm_dict, prog_bar=True)
        return {"loss": train_loss}

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        train_loss_mean = torch.stack([output["loss"] for output in outputs]).mean()
        train_acc_mean = self.train_acc.compute()
        self.log_dict({"train_loss": train_loss_mean, "train_acc": train_acc_mean, "step": self.current_epoch})

    def validation_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)

        # 2. Compute loss & accuracy:
        val_loss = self.loss(y_logits, y_true)
        accuracy = self.valid_acc(y_logits, y_true)

        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output["val_loss"] for output in outputs]).mean()
        train_acc_mean = self.valid_acc.compute()
        log_dict = {"val_loss": val_loss_mean, "val_acc": train_acc_mean}
        self.log_dict(log_dict, prog_bar=True)
        self.log_dict({"step": self.current_epoch})

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)

        return [optimizer], [scheduler]

    def prepare_data(self):
        """Download images and prepare images datasets."""
        download_and_extract_archive(url=DATA_URL, download_root=self.dl_path, remove_finished=True)

    def setup(self, stage: str):
        data_path = Path(self.dl_path).joinpath("cats_and_dogs_filtered")

        # 2. Load the data + preprocessing & data augmentation
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = ImageFolder(
            root=data_path.joinpath("train"),
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        valid_dataset = ImageFolder(
            root=data_path.joinpath("validation"),
            transform=transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def __dataloader(self, train: bool):
        """Train/validation loaders."""

        _dataset = self.train_dataset if train else self.valid_dataset
        loader = DataLoader(dataset=_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=train)

        return loader

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
        parser.add_argument("--gpus", type=int, default=1, help="number of gpus to use")
        parser.add_argument(
            "--lr", "--learning-rate", default=1e-2, type=float, metavar="LR", help="initial learning rate", dest="lr"
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
            "--num-workers", default=6, type=int, metavar="W", help="number of CPU workers", dest="num_workers"
        )
        parser.add_argument(
            "--train-bn",
            default=True,
            type=bool,
            metavar="TB",
            help="Whether the BatchNorm layers should be trainable",
            dest="train_bn",
        )
        parser.add_argument(
            "--milestones", default=[5, 10], type=list, metavar="M", help="List of two epochs milestones"
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

    with TemporaryDirectory(dir=args.root_data_path) as tmp_dir:

        model = TransferLearningModel(dl_path=tmp_dir, **vars(args))

        trainer = pl.Trainer(
            weights_summary=None,
            progress_bar_refresh_rate=1,
            num_sanity_val_steps=0,
            gpus=args.gpus,
            min_epochs=args.nb_epochs,
            max_epochs=args.nb_epochs,
        )

        trainer.fit(model)


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
    return parser.parse_args()


if __name__ == "__main__":
    cli_lightning_logo()
    main(get_args())
