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
        self.memory = torch.zeros(self.window_length)
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
