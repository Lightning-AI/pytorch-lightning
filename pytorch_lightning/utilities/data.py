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

from typing import Union

from torch.utils.data import DataLoader, IterableDataset

from pytorch_lightning.utilities import rank_zero_warn


def has_iterable_dataset(dataloader: DataLoader):
    return hasattr(dataloader, 'dataset') and isinstance(dataloader.dataset, IterableDataset)


def has_len(dataloader: DataLoader) -> bool:
    """ Checks if a given Dataloader has __len__ method implemented i.e. if
    it is a finite dataloader or infinite dataloader. """

    try:
        # try getting the length
        if len(dataloader) == 0:
            raise ValueError('`Dataloader` returned 0 length. Please make sure that it returns at least 1 batch')
        has_len = True
    except TypeError:
        has_len = False
    except NotImplementedError:  # e.g. raised by torchtext if a batch_size_fn is used
        has_len = False

    if has_len and has_iterable_dataset(dataloader):
        rank_zero_warn(
            'Your `IterableDataset` has `__len__` defined.'
            ' In combination with multi-processing data loading (e.g. batch size > 1),'
            ' this can lead to unintended side effects since the samples will be duplicated.'
        )
    return has_len


def get_len(dataloader: DataLoader) -> Union[int, float]:
    """ Return the length of the given DataLoader. If ``__len__`` method is not implemented, return float('inf'). """

    if has_len(dataloader):
        return len(dataloader)

    return float('inf')
