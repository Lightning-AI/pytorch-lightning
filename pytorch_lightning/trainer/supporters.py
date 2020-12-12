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
import numpy as np
from pytorch_lightning.utilities.exceptions import MisconfigurationException


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
        self.memory = None
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
        if self.memory is None:
            self.memory = torch.zeros(self.window_length, *x.shape)

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


'''
This class is taken from https://github.com/pytorch/tnt/blob/master/torchnet/meter/averagevaluemeter.py#L6
'''
class AverageValueMeter:

    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value * n
        if n <= 0:
            raise ValueError("Cannot use a non-positive weight for the running stat.")
        elif self.n == 0:
            self.mean = 0.0 + value  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + n * (value - self.mean_old) / float(self.n + n)
            self.m_s += n * (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n + n - 1.0))
        self.var = self.std ** 2

        self.n += n

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class GradNormTracker:

    def __init__(self, aggregation_mode, norm_type):
        """
        Track grad norms and aggregate the values across parameters, optimizers or both.

        Args:

            aggregation_mode: How grad norms are aggregated. The supported values are `parameters`, `optimizer`,
                `optimizer+parameters`.

            norm_type: The type of the used p-norm. This class uses the value to name the output norms correctly.
        """

        self._grad_norm_dic = {}
        self.norm_type = norm_type
        self.aggregation_mode = aggregation_mode

        if aggregation_mode == 'optimizer':
            self.name_mapping = self.aggregate_over_optimizer
        elif aggregation_mode == 'optimizer+parameters':
            self.name_mapping = self.aggregate_over_optimizer_and_parameters
        elif aggregation_mode =='parameters':
            self.name_mapping = self.aggregate_over_parameters
        else:
            raise MisconfigurationException(f'Invalid value=`{self.aggration_mode}` for aggregation_mode. Supported'
                                            f' values are `parameters`, `optimizer` and `optimizer+parameters`.')

    @staticmethod
    def check_grad_norm_mode(mode):
        supported_modes = ['optimizer', 'optimizer+parameters', 'parameters']
        mode_is_valid = isinstance(mode, str) and mode in supported_modes
        return mode_is_valid

    @staticmethod
    def norm_name(norm_type):
        return f'grad_{norm_type}_norm'

    def aggregate_over_optimizer(self, name, opt_idx):
        base = self.norm_name(self.norm_type)
        if 'norm_total' in name:
            return f'opt_{opt_idx}_{base}_total'
        else:
            return f'opt_{opt_idx}_{base}'

    def aggregate_over_optimizer_and_parameters(self, name, opt_idx):
        return f'opt_{opt_idx}_{name}'

    def aggregate_over_parameters(self, name, opt_idx):
        return name

    def track_norm(self, grad_norm_dic, opt_idx):
        for name, norm in grad_norm_dic.items():
            name = self.name_mapping(name, opt_idx)
            if name not in self._grad_norm_dic:
                self._grad_norm_dic[name] = AverageValueMeter()
            self._grad_norm_dic[name].add(norm)

    def _reduce(self):
        reduced_norm = {}
        for name, norm in self._grad_norm_dic.items():
            mean, std = norm.value()
            reduced_norm[f'{name}_mean'] = mean
            reduced_norm[f'{name}_std'] = std

        # If aggregating over optimizers only keep total norm mean
        if self.aggregation_mode == 'optimizer':
            reduced_norm = {name: val for name, val in reduced_norm.items() if 'norm_total_mean' in name}
        # Remove total norm std when aggregating over parameters
        if self.aggregation_mode == 'parameters':
            total_norm_std = GradNormTracker.norm_name(self.norm_type) + '_total_std'
            del reduced_norm[total_norm_std]

        self._grad_norm_dic = reduced_norm

    def get_and_reset(self):
        self._reduce()
        grad_norm_dic = self._grad_norm_dic
        self._grad_norm_dic = {}
        return grad_norm_dic
