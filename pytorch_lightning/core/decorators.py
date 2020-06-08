from functools import wraps
from typing import Callable

import torch

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_warn


def data_loader(fn):
    """Decorator to make any fx with this use the lazy property.

    Warnings:
        This decorator deprecated in v0.7.0 and it will be removed v0.9.0.
    """
    rank_zero_warn('`data_loader` decorator deprecated in v0.7.0. Will be removed v0.9.0', DeprecationWarning)

    def inner_fx(self):
        return fn(self)
    return inner_fx


def auto_move_data(fn: Callable) -> Callable:
    """
    Decorator for :class:`~pytorch_lightning.core.lightning.LightningModule` methods for which
    input arguments should be moved automatically to the correct device.
    It as no effect if applied to a method of an object that is not an instance of
    :class:`~pytorch_lightning.core.lightning.LightningModule` and is typically applied to ``__call__``
    or ``forward``.

    Args:
        fn: A LightningModule method for which the arguments should be moved to the device
            the parameters are on.

    Example:

        .. code-block:: python

            # directly in the source code
            class LitModel(LightningModule):

                @auto_move_data
                def forward(self, x):
                    return x

            # or outside
            LitModel.forward = auto_move_data(LitModel.forward)

            model = LitModel()
            model = model.to('cuda')
            model(torch.zeros(1, 3))

            # input gets moved to device
            # tensor([[0., 0., 0.]], device='cuda:0')

    """
    @wraps(fn)
    def auto_transfer_args(self, *args, **kwargs):
        if not isinstance(self, LightningModule):
            return fn(self, *args, **kwargs)

        args = self.transfer_batch_to_device(args, self.device)
        kwargs = self.transfer_batch_to_device(kwargs, self.device)
        return fn(self, *args, **kwargs)

    return auto_transfer_args
