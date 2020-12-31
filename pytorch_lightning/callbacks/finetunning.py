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

r"""
Finetunning Callback
^^^^^^^^^^^^^^^^^^^^

Freeze and unfreeze models for finetunning purposes

"""
from typing import Callable, Generator, Optional

import torch
from torch.nn import Module
from torch.nn.modules.container import Sequential
from torch.optim.optimizer import Optimizer

from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


def _make_trainable(module: Module) -> None:
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: Module,
                      train_bn: bool = True) -> None:
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


def freeze(module: Module,
           n: Optional[int] = None,
           train_bn: bool = True) -> None:
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


def filter_params(module: Module,
                  train_bn: bool = True) -> Generator:
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


def unfreeze_and_add_param_group(module: Module,
                                 optimizer: Optimizer,
                                 lr: Optional[float] = None,
                                 train_bn: bool = True,):
    """Unfreezes a module and adds its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    denom_lr = 10. if lr is None else 1.
    optimizer.add_param_group(
        {'params': filter_params(module=module, train_bn=train_bn),
         'lr': params_lr / denom_lr,
         })


multiplicative = lambda epoch: 2


class BaseFinetunningCallback(Callback):

    r"""
    BaseFinetunningCallback.
    Overrides the finetunning_function and add your own logic there.
    """
    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the epoch begins."""
        for opt_idx, optimizer in trainer.train_loop.prepare_optimizers():
            self.finetunning_function(pl_module, trainer.current_epoch, optimizer, opt_idx)

    def finetunning_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        raise NotImplementedError


class BackboneLambdaFinetunningCallback(BaseFinetunningCallback):

    r"""
    Finetunne a backbone model based on a learning rate user-defined scheduling.
    When the backbone learning rate reaches the current model learning rate
    and ``should_align`` is set to True, it will align with it for the rest of the training.

    Args:
        unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.
        lambda_func: Scheduling function for increasing backbone learning rate.
        verbose: verbosity mode. Default: ``False``.
        backbone_initial_ratio_lr:
            Used to scale down the backbone learning rate compared to rest of model
        backbone_initial_lr: Optional, Inital learning rate for the backbone.
            By default, we will use current_learning /  backbone_initial_ratio_lr
        should_align: Wheter to align with current learning rate when backbone learning
            reaches it.
        verbose: Display current learning rate for model and backbone
        round: Precision for displaying learning rate

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import BackboneLambdaFinetunningCallback
        >>> multiplicative = lambda epoch: 1.5
        >>> backbone_finetunning = BackboneLambdaFinetunningCallback(200, multiplicative)
        >>> trainer = Trainer(callbacks=[backbone_finetunning])
    """

    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 10,
        lambda_func: Callable = multiplicative,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        verbose: bool = False,
        round: int = 12,
    ):
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.backbone_initial_lr = backbone_initial_lr
        self.lambda_func = lambda_func
        self.backbone_initial_ratio_lr = backbone_initial_ratio_lr
        self.should_align = should_align
        self.round = round
        self.verbose = verbose

    def on_fit_start(self, trainer, pl_module):
        if hasattr(pl_module, "backbone") and \
           (isinstance(pl_module.backbone, Module) or isinstance(pl_module.backbone, Sequential)):
            return
        raise MisconfigurationException(
            "The LightningModule should have a nn.Module `backbone` attribute"
        )

    def finetunning_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        """Called when the epoch begins."""

        if epoch == 0:
            _recursive_freeze(pl_module.backbone)
            return

        elif epoch == self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]['lr']
            initial_backbone_lr = self.backbone_initial_lr if self.backbone_initial_lr is not None \
                else current_lr * self.backbone_initial_ratio_lr
            self.previous_backbone_lr = initial_backbone_lr
            unfreeze_and_add_param_group(pl_module.backbone, optimizer, initial_backbone_lr)
            if self.verbose:
                log.info(f"Current lr: {round(current_lr, self.round)}, "
                         f"Backbone lr: {round(initial_backbone_lr, self.round)}")

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]['lr']
            next_current_backbone_lr = self.lambda_func(epoch + 1) * self.previous_backbone_lr
            next_current_backbone_lr = current_lr if (self.should_align and next_current_backbone_lr > current_lr) \
                else next_current_backbone_lr
            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr
            if self.verbose:
                log.info(f"Current lr: {round(current_lr, self.round)}, "
                         f"Backbone lr: {round(next_current_backbone_lr, self.round)}")
