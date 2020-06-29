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

import os
from typing import Optional

import fsspec
import torch
from pytorch_lightning.utilities.cloud_io import get_filesystem
from torch import Tensor
from pytorch_lightning.utilities.apply_func import apply_to_collection
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Union


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
                return getattr(self.memory[: self.current_idx], how)()


class Accumulator(object):
    def __init__(self):
        self.num_values = 0
        self.total = 0

    def accumulate(self, x):
        with torch.no_grad():
            self.total += x
            self.num_values += 1

    def mean(self):
        return self.total / self.num_values


class PredictionCollection(object):
    def __init__(self, global_rank: int, world_size: int):
        self.global_rank = global_rank
        self.world_size = world_size
        self.predictions = {}
        self.num_predictions = 0

    def _add_prediction(self, name, values, filename):
        if filename not in self.predictions:
            self.predictions[filename] = {name: values}
        elif name not in self.predictions[filename]:
            self.predictions[filename][name] = values
        elif isinstance(values, Tensor):
            self.predictions[filename][name] = torch.cat(
                (self.predictions[filename][name], values)
            )
        elif isinstance(values, list):
            self.predictions[filename][name].extend(values)

    def add(self, predictions):

        if predictions is None:
            return

        for filename, pred_dict in predictions.items():
            for feature_name, values in pred_dict.items():
                self._add_prediction(feature_name, values, filename)

    def to_disk(self) -> None:
        """Write predictions to file(s).
        """
        for filepath, predictions in self.predictions.items():
            fs = get_filesystem(filepath)
            # normalize local filepaths only
            if fs.protocol == "file":
                filepath = os.path.realpath(filepath)
            if self.world_size > 1:
                stem, extension = os.path.splitext(filepath)
                filepath = f"{stem}_rank_{self.global_rank}{extension}"
            dirpath = os.path.split(filepath)[0]
            fs.mkdirs(dirpath, exist_ok=True)

            # Convert any tensor values to list
            predictions = {
                k: v if not isinstance(v, Tensor) else v.tolist()
                for k, v in predictions.items()
            }

            # Check if all features for this file add up to same length
            feature_lens = {k: len(v) for k, v in predictions.items()}
            if len(set(feature_lens.values())) != 1:
                raise ValueError(
                    "Mismatching feature column lengths found in stored EvalResult predictions."
                )

            # Switch predictions so each entry has its own dict
            outputs = []
            for values in zip(*predictions.values()):
                output_element = {k: v for k, v in zip(predictions.keys(), values)}
                outputs.append(output_element)

            # Write predictions for current file to disk
            with fs.open(filepath, "wb") as fp:
                torch.save(outputs, fp)
                return getattr(self.memory[:self.current_idx], how)()


class CycleIterator(object):
    """
    Iterator for restarting a dataloader if it runs out of samples
    """
    def __init__(self, loader, length: int = None):
        """

        Args:
            loader: the loader to restart for cyclic (and optionally infinite) sampling
            length: the number of batches to sample (with restarted loaders if necessary) before raising StopIteration
                if None: infinite
        """
        if length is None:
            length = float('inf')

        self.length = length
        self.loader = loader
        self._loader_iter = None
        self.counter = 0

    def __iter__(self):
        """
        Creates the internal iterator and returns self
        Return:
            CycleIterator: self

        """
        self._loader_iter = iter(self.loader)
        return self

    def __next__(self) -> Any:
        """
        Fetches the next batch from internal dataloader and restarts
        it if necessary

        Return:
            Any: the resulting batch

        Raises:
            StopIteration: if more then :attr:`length` batches have been returned

        """
        if self.counter >= len(self):
            raise StopIteration

        try:
            return next(self._loader_iter)

        except StopIteration:
            self._loader_iter = iter(self.loader)
            return next(self._loader_iter)
        finally:
            self.counter += 1

    def __len__(self) -> int:
        return self.length


class CombinedLoaderIterator(object):
    """
    Combines different dataloaders and allows sampling in parallel
    """
    SUPPORTED_MODES = ('min_size', 'max_size_cycle')

    def __init__(self, loaders: Any, mode='min_size'):
        """

        Args:
            loaders: the loaders to sample from. Can be all kind of collection
            mode: the mode. Supported are 'min_size' which stops if the shortest loader is exhausted and
                'max_size_cycle' which stops if the longest loader is exhausted and cycles through the smaller ones.
        """
        self.loaders = loaders
        self._loader_iters = None

        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Invalid Mode: {mode}")

        self.mode = mode

        if self.mode == 'max_size_cycle':
            self._wrap_loaders_max_size_cycle()

    def _wrap_loaders_max_size_cycle(self) -> Any:
        """
        Wraps all loaders to make sure they are cycled until the longest loader is exhausted

        Return:
            Any: the wrapped loaders
        """
        all_lengths = apply_to_collection(self.loaders, Iterable, len,
                                          wrong_dtype=(Sequence, Mapping))
        if isinstance(all_lengths, int):
            length = all_lengths

        elif isinstance(all_lengths, Mapping):
            length = max(all_lengths.values())

        elif isinstance(all_lengths, Sequence):
            length = max(all_lengths)

        if isinstance(self.loaders, Mapping):
            self.loaders = type(self.loaders)({k: CycleIterator(v, length=length)
                                               for k, v in self.loaders.items()})

        elif isinstance(self.loaders, Sequence):
            self.loaders = type(self.loaders)([CycleIterator(v, length=length)
                                               for v in self.loaders])

        # dataloaders are iterable but not sequence
        elif isinstance(self.loaders, Iterable):
            self.loaders = CycleIterator(self.loaders, length=length)
        else:
            raise ValueError(f'Invalid Datatype for loaders: {type(self.loaders).__name__}')

    @property
    def loader_iters(self) -> Any:
        if self._loader_iters is None:
            self._loader_iters = self.create_loader_iters(self.loaders)

        return self._loader_iters

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        return self.request_next_batch(self.loader_iters)

    @staticmethod
    def request_next_batch(loader_iters: Union[Iterator, Sequence, Mapping]) -> Any:
        return apply_to_collection(loader_iters, Iterator, next)

    @staticmethod
    def _calc_num_batches(loaders) -> int:
        all_lengths = apply_to_collection(loaders, Iterable, len,
                                          wrong_dtype=(Sequence, Mapping))

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
        return apply_to_collection(loaders, Iterable, iter, wrong_dtype=(Sequence, Mapping))

    def __len__(self):
        return self._calc_num_batches(self.loaders)
