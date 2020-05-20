from typing import Callable

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_warn, transfer_batch_to_device


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

            model = model.cuda(0)
            model.prepare_data()
            loader = model.train_dataloader()
            for x, y in loader:
                output = model(x)
    """
    def auto_transfer_args(self, *args, **kwargs):
        if not isinstance(self, LightningModule):
            return fn(self, *args, **kwargs)

        args = transfer_batch_to_device(args, self.device)
        kwargs = transfer_batch_to_device(kwargs, self.device)
        return fn(self, *args, **kwargs)

    return auto_transfer_args
