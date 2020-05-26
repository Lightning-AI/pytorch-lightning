from typing import Optional

import torch
from pytorch_lightning.utilities.apply_func import apply_to_collection
from collections.abc import Iterable, Iterator, Mapping, Sequence


class TensorRunningAccum(object):
    """Tracks a running accumulation values (min, max, mean) without graph
    references.

    Examples:
        >>> accum = TensorRunningAccum(5)
        >>> accum.last(), accum.mean()
        (None, None)
        >>> accum.append(torch.tensor(1.5))
        >>> accum.last(), accum.mean()
        (tensor(1.5000), tensor(1.5000))
        >>> accum.append(torch.tensor(2.5))
        >>> accum.last(), accum.mean()
        (tensor(2.5000), tensor(2.))
        >>> accum.reset()
        >>> _= [accum.append(torch.tensor(i)) for i in range(13)]
        >>> accum.last(), accum.mean(), accum.min(), accum.max()
        (tensor(12.), tensor(10.), tensor(8.), tensor(12.))
    """

    def __init__(self, window_length: int):
        self.window_length = window_length
        self.memory = torch.Tensor(self.window_length)
        self.current_idx: int = 0
        self.last_idx: Optional[int] = None
        self.rotated: bool = False

    def reset(self) -> None:
        """Empty the accumulator."""
        self = TensorRunningAccum(self.window_length)

    def last(self):
        """Get the last added element."""
        if self.last_idx is not None:
            return self.memory[self.last_idx]

    def append(self, x):
        """Add an element to the accumulator."""
        # ensure same device and type
        if self.memory.device != x.device or self.memory.type() != x.type():
            x = x.to(self.memory)

        # store without grads
        with torch.no_grad():
            self.memory[self.current_idx] = x
            self.last_idx = self.current_idx

        # increase index
        self.current_idx += 1

        # reset index when hit limit of tensor
        self.current_idx = self.current_idx % self.window_length
        if self.current_idx == 0:
            self.rotated = True

    def mean(self):
        """Get mean value from stored elements."""
        return self._agg_memory('mean')

    def max(self):
        """Get maximal value from stored elements."""
        return self._agg_memory('max')

    def min(self):
        """Get minimal value from stored elements."""
        return self._agg_memory('min')

    def _agg_memory(self, how: str):
        if self.last_idx is not None:
            if self.rotated:
                return getattr(self.memory, how)()
            else:
                return getattr(self.memory[:self.current_idx], how)()


class CombinedLoaderIterator(object):
    def __init__(self, loaders: Any):
        self.loaders = loaders
        self._loader_iters = None

    @property
    def loader_iters(self):
        if self._loader_iters is None:
            self._loader_iters = self.create_loader_iters(self.loaders)

        return self._loader_iters

    def __iter__(self):
        return self

    def __next__(self):
        return self.request_next_batch(self.loader_iters)

    @staticmethod
    def request_next_batch(loader_iters: Union[Iterator, Sequence, Mapping]):
        return apply_to_collection(loader_iters, Iterator, next)

    @staticmethod
    def _calc_num_batches(loader_iters):
        all_lengths = apply_to_collection(loader_iters, Iterator, len)

        if isinstance(all_lengths, int):
            return all_lengths

        elif isinstance(all_lengths, Mapping):
            return min(all_lengths.values())

        elif isinstance(all_lengths, Sequence):
            return min(all_lengths)

        raise TypeError(f'Got Type {type(all_lengths).__name__}, but expected one of Sequence, int or Mapping')

    @staticmethod
    def create_loader_iters(loaders: Union[Any, Iterator,
                                           Sequence, Mapping]) -> Union[Any, Iterator, Sequence, Mapping]:

        # dataloaders are Iterable but not Sequences. Need this to specifically exclude sequences
        return apply_to_collection(loaders, Iterable, iter, wrong_dtype=Sequence)

    def __len__(self):
        return self._calc_num_batches(self.loader_iters)