import numbers
import torch.nn as nn
from abc import ABC
from typing import Generator


# Copied from pl_examples (with small changes)
BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def filter_params(module: nn.Module, bn: bool = True, only_trainable=False) -> Generator:
    """Yields the trainable parameters of a given module.

    Args:
        module: A given module
        bn: If False, don't return batch norm layers

    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not isinstance(module, BN_TYPES) or bn:
            for param in module.parameters():
                if not only_trainable or param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, bn=bn, only_trainable=only_trainable):
                yield param


def unfreeze(params):
    for p in params:
        p.requires_grad = True


def freeze(params):
    for p in params:
        p.requires_grad = False


class ParametersSplitsModuleMixin(nn.Module, ABC):
    def model_splits(self):
        return self.children()

    def params_splits(self, only_trainable=False):
        """ Get parameters from model splits
        """
        for split in self.model_splits():
            params = list(filter_params(split, only_trainable=only_trainable))
            if params:
                yield params

    def trainable_params_splits(self):
        """ Get trainable parameters from model splits
            If a parameter group does not have trainable params, it does not get added
        """
        return self.params_splits(only_trainable=True)

    def freeze_to(self, n: int = None):
        """ Freezes model until certain layer
        """
        unfreeze(self.parameters())
        for params in list(self.params_splits())[:n]:
            freeze(params)

    def get_optimizer_param_groups(self, lr):
        lrs = self.get_lrs(lr)
        return [{"params": params, "lr": lr} for params, lr in zip(self.params_splits(), lrs)]

    def get_lrs(self, lr):
        n_splits = len(list(self.params_splits()))
        if isinstance(lr, numbers.Number):
            return [lr] * n_splits
        if isinstance(lr, (tuple, list)):
            assert len(lr) == len(list(self.params_splits()))
            return lr
