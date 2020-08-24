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
from typing import Any, Optional
import numbers

import torch
from torch import nn
import numpy as np

from pytorch_lightning.metrics.converters import (
    sync_ddp_if_available, gather_all_tensors_if_available,
    convert_to_tensor, convert_to_numpy)
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
        * ddp_sync: implementation of ddp sync, default is gather all
        * aggregate: implement how values should be aggregated
        * compute: post-ddp sync for additional metric computations

    Call order

        input_convert -> forward -> output_convert -> ddp_sync -> aggregate -> compute

    """

    def __init__(self, name: str):
        """
        Args:
            name: the metric's name

        """
        super().__init__()
        self.name = name
        self._dtype = torch.get_default_dtype()
        self._device = torch.device('cpu')

        # Register hooks
        self.register_forward_pre_hook(self.input_convert)
        self.register_forward_hook(self.output_convert)
        self.register_forward_hook(self.ddp_sync)
        self.register_forward_hook(self.aggregate)
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
        return output

    @staticmethod
    def ddp_sync(self, data: Any, output: Any):
        """
        Implement how the outputs from forward should be synced

        Args:

            data: input to forward method

            output: output from the `output_convert` hook

        Returns:
            synced output

        """
        return output

    @staticmethod
    def aggregate(self, data: Any, output: Any):
        """
        Implement aggregation of values on the same device

        Args:

            data: input to forward method

            output: output from the `ddp_sync` hook

        Returns:
            aggregated values

        """
        return output

    @staticmethod
    def compute(self, data: Any, output: Any):
        """
        Implement additionally metric computations to be done after the ddp sync

        Args:

            data: input to forward method

            output: output from the `aggregate` hook

        Returns:
            final metric value

        """
        return output


class TensorMetric(Metric):
    """
    Base class for metric implementation operating directly on tensors.
    All inputs and outputs will be casted to tensors if necessary.
    Already handles DDP sync and input/output conversions.
    """

    def __init__(self, name: str,
                 reduce_group: Optional[Any] = None,
                 reduce_op: Optional[Any] = None):
        """

        Args:
            name: the metric's name
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__(name)
        self.reduce_group = reduce_group
        self.reduce_op = reduce_op

    @staticmethod
    def input_convert(self, data: Any):
        return apply_to_collection(data,
                                   (torch.Tensor, np.ndarray, numbers.Number),
                                   convert_to_tensor,
                                   self.dtype, self.device)

    @staticmethod
    def output_convert(self, data: Any, output: Any):
        return apply_to_collection(output, torch.Tensor, convert_to_tensor,
                                   self.dtype, self.device)

    @staticmethod
    def ddp_sync(self, data: Any, output: Any):
        return apply_to_collection(output, torch.Tensor, sync_ddp_if_available,
                                   self.reduce_group, self.reduce_op)


class TensorCollectionMetric(Metric):
    """
    Base class for metric implementation operating directly on tensors.
    All inputs will be casted to tensors if necessary. Outputs won't be casted.
    Already handles DDP sync and input conversions.

    This class differs from :class:`TensorMetric`, as it assumes all outputs to
    be collections of tensors and does not explicitly convert them. This is
    necessary, since some collections (like for ROC, Precision-Recall Curve etc.)
    cannot be converted to tensors at the highest level.
    All numpy arrays and numbers occuring in these outputs will still be converted.

    Use this class as a baseclass, whenever you want to ensure inputs are
    tensors and outputs cannot be converted to tensors automatically

    """

    def __init__(self, name: str,
                 reduce_group: Optional[Any] = None,
                 reduce_op: Optional[Any] = None):
        """

        Args:
            name: the metric's name
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__(name)
        self.reduce_group = reduce_group
        self.reduce_op = reduce_op

    @staticmethod
    def input_convert(self, data: Any):
        return apply_to_collection(data,
                                   (torch.Tensor, np.ndarray, numbers.Number),
                                   convert_to_tensor,
                                   self.dtype, self.device)

    @staticmethod
    def output_convert(self, data: Any, output: Any):
        return apply_to_collection(output,
                                   (torch.Tensor, np.ndarray, numbers.Number),
                                   convert_to_tensor,
                                   self.dtype, self.device)

    @staticmethod
    def ddp_sync(self, data: Any, output: Any):
        return apply_to_collection(output, torch.Tensor, sync_ddp_if_available,
                                   self.reduce_group, self.reduce_op)


class NumpyMetric(Metric):
    """
    Base class for metric implementation operating on numpy arrays.
    All inputs will be casted to numpy if necessary and all outputs will
    be casted to tensors if necessary.
    Already handles DDP sync and input/output conversions.
    """

    def __init__(self, name: str,
                 reduce_group: Optional[Any] = None,
                 reduce_op: Optional[Any] = None):
        """

        Args:
            name: the metric's name
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__(name)
        self.reduce_group = reduce_group
        self.reduce_op = reduce_op

    @staticmethod
    def input_convert(self, data: Any):
        return apply_to_collection(data,
                                   (torch.Tensor, np.ndarray, numbers.Number),
                                   convert_to_numpy)

    @staticmethod
    def output_convert(self, data: Any, output: Any):
        return apply_to_collection(output,
                                   (torch.Tensor, np.ndarray, numbers.Number),
                                   convert_to_tensor,
                                   self.dtype, self.device)

    @staticmethod
    def ddp_sync(self, data: Any, output: Any):
        return apply_to_collection(output, torch.Tensor, sync_ddp_if_available,
                                   self.reduce_group, self.reduce_op)
