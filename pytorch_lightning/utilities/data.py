from distutils.version import LooseVersion

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.utilities import rank_zero_warn

try:
    from torch.utils.data import IterableDataset
    ITERABLE_DATASET_EXISTS = True
except ImportError:
    ITERABLE_DATASET_EXISTS = False


def has_iterable_dataset(dataloader: DataLoader):
    return ITERABLE_DATASET_EXISTS and hasattr(dataloader, 'dataset') \
        and isinstance(dataloader.dataset, IterableDataset)


def has_len(dataloader: DataLoader) -> bool:
    """ Checks if a given Dataloader has __len__ method implemented i.e. if
    it is a finite dataloader or infinite dataloader. """

    try:
        # try getting the length
        if len(dataloader) == 0:
            raise ValueError('`Dataloader` returned 0 length.'
                             ' Please make sure that your Dataloader at least returns 1 batch')
        has_len = True
    except TypeError:
        has_len = False
    except NotImplementedError:  # e.g. raised by torchtext if a batch_size_fn is used
        has_len = False

    if has_len and has_iterable_dataset(dataloader) and LooseVersion(torch.__version__) >= LooseVersion("1.4.0"):
        rank_zero_warn(
            'Your `IterableDataset` has `__len__` defined.'
            ' In combination with multi-processing data loading (e.g. batch size > 1),'
            ' this can lead to unintended side effects since the samples will be duplicated.'
        )
    return has_len
