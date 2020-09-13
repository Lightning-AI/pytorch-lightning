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

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence
import numbers

import torch
from torch import nn
import numpy as np

from pytorch_lightning.metrics.converters import (
    at_least_1d,
    gather_all_tensors_if_available,
    convert_to_tensor,
    convert_to_numpy,
)
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin


class Metric(DeviceDtypeModuleMixin, nn.Module, ABC):
    """
    Abstract base class for metric implementation.

    Should be used to implement metrics that

        1. Return multiple Outputs
        2. Handle their own DDP sync

    Metric hooks that can be implemented are

        * input_convert: pre-forward hook that takes care of input conversion
        * output_convert: post-forward hook that takes care of output convertion
        * ddp_reduce: implementation of ddp sync + aggregation, default is ddp_sync + aggregate
        * compute: post-ddp sync for additional metric computations

    ``ddp_reduce`` by default calls the following methods, which can also be overwritten if necessary.

        * ddp_sync: implements how values should be synced across ddp-processes. Defaults to gather all.
        * aggregate: implement how values should be aggregated (defaults to mean).

    Call order

        input_convert -> forward -> output_convert -> ddp_reduce (per default being ddp_sync -> aggregate) -> compute

    """

    def __init__(self, name: str, reduce_group: Optional[Any] = None):
        """
        Args:
            name: the metric's name
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)

        """
        super().__init__()
        self.name = name
        self._dtype = torch.get_default_dtype()
        self._device = torch.device("cpu")

        self.reduce_group = reduce_group

        self._step_vals = []

        # Register hooks
        self.register_forward_pre_hook(self.input_convert)
        self.register_forward_hook(self.output_convert)
        self.register_forward_hook(self.ddp_reduce)
        self.register_forward_hook(self.compute)

    @staticmethod
    def input_convert(self, data: Any):
        """
        Implement how the inputs should be casted before calling forward

        Args:
            data: input to forward method

        Returns:
            casted data
        """
        return data

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Implements the actual metric computation.

        Returns:
            metric value or metric state

        """
        raise NotImplementedError

    @staticmethod
    def output_convert(self, data: Any, output: Any):
        """
        Implement how outputs from forward should be casted

        Args:
            data: input to forward method
            output: output from forward method

        Returns:
            casted outputs
        """
        return apply_to_collection(output, (torch.Tensor, np.ndarray), at_least_1d)

    def ddp_sync(self, tensor: Any):
        """
        Implement how the outputs from forward should be synced
        (per default just gathers all of them and adds them to self._step_vals)

        Args:
            tensor: tensor to sync

        Returns:
            synced output

        """
        gathered_tensors = apply_to_collection(tensor, torch.Tensor, gather_all_tensors_if_available, self.reduce_group)

        self._step_vals.append(gathered_tensors)

        return gathered_tensors

    @staticmethod
    def ddp_reduce(self, data: Any, output: Any):
        """
        Implement how the outputs from forward should be synced and reduced across nodes

        Args:
            data: input to forward method
            output: output from the `output_convert` hook

        Returns:
            synced output

        """
        synced = self.ddp_sync(output)
        return self.aggregate(synced)

    def aggregate(self, *tensors: torch.Tensor) -> torch.Tensor:
        """
        Implement aggregation of values on the same device

        Args:
            tensors: the values to be aggregated

        Returns:
            aggregated values

        """
        try:
            return torch.cat(tensors).mean(0)
        except (ValueError, TypeError):
            if isinstance(tensors[0], Mapping):
                return {k: torch.stack([tensor[k] for tensor in tensors]).mean(0) for k in tensors[0].keys()}
            elif isinstance(tensors[0], Sequence) and not isinstance(tensors[0], torch.Tensor):
                return tuple([torch.stack(tmp).mean(0) for tmp in zip(*tensors)])
            elif isinstance(tensors[0], torch.Tensor):
                return torch.stack(tensors).mean(0)
            else:
                raise TypeError("unknown metric value format to aggregate")

    @staticmethod
    def compute(self, data: Any, output: Any):
        """
        Implement additionally metric computations to be done after the aggregation

        Args:
            data: input to forward method
            output: output from the `aggregate` hook

        Returns:
            final metric value

        """
        return output

    @property
    def aggregated(self) -> torch.Tensor:
        aggr = self.aggregate(*self._step_vals)
        self.reset()
        return self.compute(self, None, aggr)

    def reset(self):
        self._step_vals = []


class TensorMetric(Metric):
    """
    Base class for metric implementation operating directly on tensors.
    All inputs and outputs will be casted to tensors if necessary.
    Already handles DDP sync and input/output conversions.
    """

    @staticmethod
    def input_convert(self, data: Any):
        data = apply_to_collection(
            data, (torch.Tensor, np.ndarray, numbers.Number), convert_to_tensor, self.dtype, self.device
        )
        return super(TensorMetric, self).input_convert(self, data)

    @staticmethod
    def output_convert(self, data: Any, output: Any):

        output = apply_to_collection(
            output, (torch.Tensor, np.ndarray, numbers.Number), convert_to_tensor, self.dtype, self.device
        )
        return super(TensorMetric, self).output_convert(self, data, output)


class NumpyMetric(Metric):
    """
    Base class for metric implementation operating on numpy arrays.
    All inputs will be casted to numpy if necessary and all outputs will
    be casted to tensors if necessary.
    Already handles DDP sync and input/output conversions.
    """

    @staticmethod
    def input_convert(self, data: Any):
        data = apply_to_collection(data, (torch.Tensor, np.ndarray, numbers.Number), convert_to_numpy)
        return super(NumpyMetric, self).input_convert(self, data)

    @staticmethod
    def output_convert(self, data: Any, output: Any):
        output = apply_to_collection(
            output, (torch.Tensor, np.ndarray, numbers.Number), convert_to_tensor, self.dtype, self.device
        )

        return super(NumpyMetric, self).output_convert(self, data, output)
