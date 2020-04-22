"""Computer vision example on Transfer Learning.

This example illustrates how to fine tune a pre-trained ResNet50 on the
'cats and dogs dataset' (~60MB, see `DATA_URL` below). For the sake of this
example, the proposed network is trained for 15 epochs. The training includes
three stages. From epoch 0 to 4, the feature extractor (ResNet50) is frozen
except for the BatchNorm layers (`train_bn = True` in `hparams`) and lr = 1e-2.
From epoch 5 to 9, the last two layer groups of the feature extractor are
unfrozen and lr = 1e-3. From epoch 10, all the layer groups of the feature
extractor are unfrozen and lr = 1e-4.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50

import pytorch_lightning as pl
from pytorch_lightning import _logger as log

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


#  --- Utility functions ---


def _make_trainable(module):
    """Unfreeze a given module.

    Operates in-place.

    Parameters
    ----------
    module : instance of `torch.nn.Module`
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module, train_bn=True):
    """Freeze the layers of a given module.

    Operates in-place.

    Parameters
    ----------
    module : instance of `torch.nn.Module`

    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
        Otherwise, they will be set to eval mode along with the other modules.
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


def freeze(module, n=None, train_bn=True):
    """Freeze the layers up to index n.

    Operates in-place.

    Parameters
    ----------
    module : instance of `torch.nn.Module`

    n : int or None (default: None)
        By default, all the layers will be frozen. Otherwise (if not None),
        an integer must be given.

    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)


def filter_params(module, train_bn=True):
    """Yield the trainable parameters of a given module.

    Parameters
    ----------
    module : instance of `torch.nn.Module`

    train_bn : bool (default: True)

    Returns
    -------
    generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            filter_params(module=child, train_bn=train_bn)


def _unfreeze_and_add_param_group(module, optimizer, lr=None, train_bn=True):
    """Unfreeze a module and add its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    optimizer.add_param_group(
        {'params': filter_params(module=module, train_bn=train_bn),
         'lr': params_lr,
         })


#  --- Pytorch-lightning module ---


class TransferLearningModel(pl.LightningModule):
    """Transfer Learning with pre-trained ResNet50.

    Parameters
    ----------
    hparams : instance of `argparse.Namespace`
        Model hyperparameters.

    train_dataset : instance of `torch.utils.data.Dataset`
        Dataset with training images.

    valid_dataset : instance of `torch.utils.data.Dataset`
        Dataset with validation images.
    """
    def __init__(self,
                 hparams,
                 train_dataset,
                 valid_dataset):
        super(TransferLearningModel, self).__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.hparams = hparams
        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained ResNet50:
        backbone = resnet50(pretrained=True)

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
        super(TransferLearningModel, self).train(mode=mode)

        epoch = self.current_epoch
        if epoch < self.hparams.milestones[0]:
            # feature extractor is frozen (except for BatchNorm layers)
            freeze(module=self.feature_extractor,
                   train_bn=self.hparams.train_bn)

        elif self.hparams.milestones[0] <= epoch < self.hparams.milestones[1]:
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


def main(hparams, data_path):

    # 1. Load the data
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

    # 1. Instantiate model
    model = TransferLearningModel(hparams,
                                  train_dataset=train_dataset,
                                  valid_dataset=valid_dataset)

    # 2. Setup trainer (train for exactly `hparams.nb_epochs` epochs)
    trainer = pl.Trainer(
        weights_summary=None,
        show_progress_bar=True,
        num_sanity_val_steps=0,
        gpus=hparams.gpus,
        min_epochs=hparams.nb_epochs,
        max_epochs=hparams.nb_epochs)

    trainer.fit(model)


if __name__ == '__main__':

    from argparse import Namespace
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from torchvision.datasets.utils import download_and_extract_archive

    DATA_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

    with TemporaryDirectory(dir=Path.cwd().as_posix()) as tmp_dir:

        # 1. Download the data
        download_and_extract_archive(url=DATA_URL,
                                     download_root=tmp_dir,
                                     remove_finished=True)

        # 2. Define hparams
        _hparams = {'batch_size': 8,
                    'num_workers': 6,
                    'lr': 1e-2,
                    'gpus': [0],
                    'lr_scheduler_gamma': 0.1,
                    'nb_epochs': 3,
                    'train_bn': True,
                    'milestones': [1, 2]}
        hyper_parameters = Namespace(**_hparams)

        # 3. Train
        main(hyper_parameters,
             data_path=Path(tmp_dir).joinpath('cats_and_dogs_filtered'))
