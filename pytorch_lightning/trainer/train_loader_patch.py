'''
    This patch solves two problems discussed in
    https://github.com/PyTorchLightning/pytorch-lightning/pull/1959

    The function  train_dataloader can either return a single instance of
    torch.utils.data.DataLoader or a dictionary of dataloaders.

    This patch fixes the length and iteration issus
    and make the rest of the code oblivious of the underlying data structure.

    I will keep the name of the class but a better name is probable advisable

    @christofer-f
'''

import itertools

from typing import Any, Union
from collections.abc import Iterable, Iterator, Mapping, Sequence

from torch.utils.data import DataLoader

from pytorch_lightning.utilities.data import get_len
from pytorch_lightning.utilities.apply_func import apply_to_collection


class MultiIterator(object):
    SUPPORTED_MODES = ('min_size', 'max_size_cycle')

    def __init__(self, loaders: Any, mode: str = 'min_size') -> None:
        self.loaders = loaders
        self.num_batches = self._calc_num_batches(loaders, mode)

    def _calc_num_batches(self, loaders, mode: str) -> Union[int, float]:
        all_lengths = apply_to_collection(loaders, Iterable, get_len,
                                          wrong_dtype=(Sequence, Mapping))

        if mode == 'min_size':
            compare_func = min
        elif mode == 'max_size_cycle':
            compare_func = max
        else:
            raise ValueError(f"Invalid Mode: {mode}")

        if isinstance(all_lengths, (int, float)):
            return all_lengths
        if isinstance(all_lengths, Mapping):
            return compare_func(all_lengths.values())
        elif isinstance(all_lengths, Sequence):
            return compare_func(all_lengths)

        raise TypeError(f'Got Type {type(all_lengths).__name__}, but expected one of Sequence, Mapping, int or float')

    def __len__(self) -> Union[int, float]:
        # Return type might be int or inf. Inf will cause type error when calling len()
        return self.num_batches

    def __iter__(self):
        if isinstance(self.loaders, Mapping):
            gens = {}
            for batch_idx in range(self.num_batches):
                rv = {}
                for loader_name, loader in self.loaders.items():
                    # If reaching the end of the iterator, recreate one
                    # because shuffle=True in dataloader, the iterator will have a different order
                    if batch_idx % len(loader) == 0:
                        gens[loader_name] = iter(loader)
                    rv[loader_name] = next(gens[loader_name])
                yield rv
        elif isinstance(self.loaders, Sequence):
            gens = [None] * self.num_batches
            for batch_idx in range(self.num_batches):
                rv = []
                for idx, loader in enumerate(self.loaders):
                    # If reaching the end of the iterator, recreate one
                    # because shuffle=True in dataloader, the iterator will have a different order
                    if batch_idx % len(loader) == 0:
                        gens[idx] = iter(loader)
                    rv.append(next(gens[idx]))
                yield rv

        return iter(self.loaders)
