"""Computer vision example on Transfer Learning.

This example illustrates how to fine tune a pre-trained ResNet50 on the
'cats and dogs dataset' (~60MB, see `DATA_URL` below). For the sake of this
example, the proposed network is trained for 15 epochs. The training includes
three stages. From epoch 0 to 4, the feature extractor (by default, a ResNet50)
is frozen except for the BatchNorm layers (`train_bn = True` in `hparams`)
and lr = 1e-2. From epoch 5 to 9, the last two layer groups of the feature
extractor are unfrozen and added to the optimizer as a new parameter group
with lr = 1e-4 (while lr = 1e-3 for the first parameter group in the
optimizer). Eventually, from epoch 10, all the remaining layer groups of the
feature extractor are unfrozen and added to the optimizer as a third parameter
group. From epoch 10, the parameters of the feature extractor are trained
with lr = 1e-5 while those of the MLP (`self.fc` in `TransferLearningModel`)
are trained with lr = 1e-4. For the sake of this example, the dataset is
downloaded to a temporary folder.

See also: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import argparse
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Generator

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

import pytorch_lightning as pl
from pytorch_lightning import _logger as log

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
DATA_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'


#  --- Utility functions ---


def _make_trainable(module: torch.nn.Module) -> None:
    """Unfreeze a given module.

    Args:
        module (torch.nn.Module): The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: torch.nn.Module,
                      train_bn: Optional[bool] = True) -> None:
    """Freeze the layers of a given module.

    Args:
        module (torch.nn.Module): The module to freeze
        train_bn (bool): If True, leave the BatchNorm layers in training mode
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


def freeze(module: torch.nn.Module,
           n: Optional[int] = None,
           train_bn: Optional[bool] = True) -> None:
    """Freeze the layers up to index n (if n is not None).

    Args:
        module (torch.nn.Module): The module to freeze (at least partially)
        n (int): Max depth at which we stop freezing the layers. By default,
        train_bn (bool): If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)


def filter_params(module: torch.nn.Module,
                  train_bn: Optional[bool] = True) -> Generator:
    """Yield the trainable parameters of a given module.

    Args:
        module (torch.nn.Module): A given module
        train_bn (bool): If True, leave the BatchNorm layers in training mode

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


def _unfreeze_and_add_param_group(module, optimizer, lr=None, train_bn=True):
    """Unfreeze a module and add its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    optimizer.add_param_group(
        {'params': filter_params(module=module, train_bn=train_bn),
         'lr': params_lr / 10.,
         })


#  --- Pytorch-lightning module ---


class TransferLearningModel(pl.LightningModule):
    """Transfer Learning with pre-trained ResNet50.

    Args:
        hparams (argparse.Namespace): Model hyperparameters
        train_dataset (torch.utils.data.Dataset): training dataset
        valid_dataset (torch.utils.data.Dataset): validation dataset
    """
    def __init__(self,
                 hparams,
                 train_dataset,
                 valid_dataset):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.hparams = hparams
        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network:
        model_func = getattr(models, self.hparams.backbone)
        backbone = model_func(pretrained=True)

        _layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*_layers)
        freeze(module=self.feature_extractor, train_bn=self.hparams.train_bn)

        # 2. Classifier:
        _fc_layers = [torch.nn.Linear(2048, 256),
                      torch.nn.Linear(256, 32),
                      torch.nn.Linear(32, 1)]
        self.fc = torch.nn.Sequential(*_fc_layers)

        # 3. Loss:
        self.loss_func = F.binary_cross_entropy_with_logits

    def check_module(self, module):
        children = list(module.children())
        if not children and hasattr(module, 'training'):
            if module.training:
                print(module)
        else:
            for child in children:
                self.check_module(child)

    def forward(self, x):
        """Forward pass. Returns logits."""

        # 1. Feature extraction:
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)

        # 2. Classifier (returns logits):
        x = self.fc(x)

        return x

    def loss(self, labels, logits):
        return self.loss_func(input=logits, target=labels)

    def train(self, mode=True):
        super().train(mode=mode)

        epoch = self.current_epoch
        if epoch < self.hparams.milestones[0] and mode:
            # feature extractor is frozen (except for BatchNorm layers)
            freeze(module=self.feature_extractor,
                   train_bn=self.hparams.train_bn)

        elif self.hparams.milestones[0] <= epoch < self.hparams.milestones[1] and mode:
            # Unfreeze last two layers of the feature extractor
            freeze(module=self.feature_extractor,
                   n=-2,
                   train_bn=self.hparams.train_bn)

    def on_epoch_start(self):
        """Use `on_epoch_start` to unfreeze layers progressively."""
        optimizer = self.trainer.optimizers[0]
        if self.current_epoch == self.hparams.milestones[0]:
            _unfreeze_and_add_param_group(module=self.feature_extractor[-2:],
                                          optimizer=optimizer,
                                          train_bn=self.hparams.train_bn)

        elif self.current_epoch == self.hparams.milestones[1]:
            _unfreeze_and_add_param_group(module=self.feature_extractor[:-2],
                                          optimizer=optimizer,
                                          train_bn=self.hparams.train_bn)

    def training_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)
        y_bin = torch.ge(y_logits, 0)

        # 2. Compute loss & accuracy:
        train_loss = self.loss(y_true, y_logits)
        num_correct = torch.eq(y_bin.view(-1), y_true.view(-1)).sum().item()

        # 3. Outputs:
        tqdm_dict = {'train_loss': train_loss}
        output = OrderedDict({'loss': train_loss,
                              'num_correct': num_correct,
                              'log': tqdm_dict,
                              'progress_bar': tqdm_dict})

        return output

    def training_epoch_end(self, outputs):

        train_loss_mean = 0.
        train_acc_mean = 0.
        for output in outputs:

            train_loss = output['loss']
            train_acc = output['num_correct']
            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                train_loss = torch.mean(train_loss)
            train_loss_mean += train_loss
            train_acc_mean += train_acc

        train_loss_mean /= len(outputs)
        train_acc_mean /= (len(outputs) * self.hparams.batch_size)
        return {'log': {'train_loss': train_loss_mean,
                        'train_acc': train_acc_mean}}

    def validation_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)
        y_bin = torch.ge(y_logits, 0)

        # 2. Compute loss & accuracy:
        val_loss = self.loss(y_true, y_logits)
        num_correct = torch.eq(y_bin.view(-1), y_true.view(-1)).sum().item()

        return {'val_loss': val_loss,
                'num_correct': num_correct}

    def validation_epoch_end(self, outputs):

        val_loss_mean = 0.
        val_acc_mean = 0.
        for output in outputs:

            val_loss = output['val_loss']
            val_acc = output['num_correct']
            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss
            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= (len(outputs) * self.hparams.batch_size)
        return {'log': {'val_loss': val_loss_mean,
                        'val_acc': val_acc_mean}}

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      self.parameters()),
                               lr=self.hparams.lr)

        scheduler = MultiStepLR(optimizer,
                                milestones=self.hparams.milestones,
                                gamma=self.hparams.lr_scheduler_gamma)

        return [optimizer], [scheduler]

    def __dataloader(self, train):
        """Train/validation loaders."""

        _dataset = self.train_dataset if train else self.valid_dataset
        loader = DataLoader(dataset=_dataset,
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            shuffle=True if train else False)

        return loader

    def train_dataloader(self):
        log.info('Training data loaded.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info('Validation data loaded.')
        return self.__dataloader(train=False)

    @staticmethod
    def add_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--backbone',
                            default='resnet50',
                            type=str,
                            metavar='BK',
                            help='Name (as in ``torchvision.models``) of the feature extractor')
        parser.add_argument('--epochs',
                            default=15,
                            type=int,
                            metavar='N',
                            help='total number of epochs',
                            dest='nb_epochs')
        parser.add_argument('--batch-size',
                            default=8,
                            type=int,
                            metavar='B',
                            help='batch size',
                            dest='batch_size')
        parser.add_argument('--gpus',
                            type=int,
                            default=1,
                            help='number of gpus to use')
        parser.add_argument('--lr',
                            '--learning-rate',
                            default=1e-2,
                            type=float,
                            metavar='LR',
                            help='initial learning rate',
                            dest='lr')
        parser.add_argument('--lr-scheduler-gamma',
                            default=1e-1,
                            type=float,
                            metavar='LRG',
                            help='Factor by which the learning rate is reduced at each milestone',
                            dest='lr_scheduler_gamma')
        parser.add_argument('--num-workers',
                            default=6,
                            type=int,
                            metavar='W',
                            help='number of CPU workers',
                            dest='num_workers')
        parser.add_argument('--train-bn',
                            default=True,
                            type=bool,
                            metavar='TB',
                            help='Whether the BatchNorm layers should be trainable',
                            dest='train_bn')
        parser.add_argument('--milestones',
                            default=[5, 10],
                            type=list,
                            metavar='M',
                            help='List of two epochs milestones')
        return parser


def main(hparams):

    with TemporaryDirectory(dir=hparams.root_data_path) as tmp_dir:

        # 1. Download the images
        download_and_extract_archive(url=DATA_URL,
                                     download_root=tmp_dir,
                                     remove_finished=True)

        data_path = Path(tmp_dir).joinpath('cats_and_dogs_filtered')

        # 2. Load the data (with preprocessing & data augmentation)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = ImageFolder(root=data_path.joinpath('train'),
                                    transform=transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]))

        valid_dataset = ImageFolder(root=data_path.joinpath('validation'),
                                    transform=transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]))

        # 2. Train the proposed model (for exactly `hparams.nb_epochs` epochs)
        model = TransferLearningModel(hparams,
                                      train_dataset=train_dataset,
                                      valid_dataset=valid_dataset)

        trainer = pl.Trainer(
            weights_summary=None,
            show_progress_bar=True,
            num_sanity_val_steps=0,
            gpus=hparams.gpus,
            min_epochs=hparams.nb_epochs,
            max_epochs=hparams.nb_epochs)

        trainer.fit(model)


def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--root-data-path',
                               metavar='DIR',
                               type=str,
                               default=Path.cwd().as_posix(),
                               help='Directory where the data will be downloaded',
                               dest='root_data_path')
    parser = TransferLearningModel.add_specific_args(parent_parser)
    return parser.parse_args()


if __name__ == '__main__':

    main(get_args())
