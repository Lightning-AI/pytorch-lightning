from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
import torch.distributed

from pytorch_lightning.metrics.converters import tensor_metric, numpy_metric
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin

__all__ = ['Metric', 'TensorMetric', 'NumpyMetric']


class Metric(DeviceDtypeModuleMixin, torch.nn.Module, ABC):
    """
    Abstract base class for metric implementation.

    Should be used to implement metrics that
    1. Return multiple Outputs
    2. Handle their own DDP sync
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

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Implements the actual metric computation.

        Returns:
            metric value

        """
        raise NotImplementedError


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
        self._orig_call = tensor_metric(group=reduce_group,
                                        reduce_op=reduce_op)(super().__call__)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        def _to_device_dtype(x: torch.Tensor) -> torch.Tensor:
            return x.to(device=self.device, dtype=self.dtype, non_blocking=True)

        return apply_to_collection(self._orig_call(*args, **kwargs), torch.Tensor,
                                   _to_device_dtype)


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
        self._orig_call = numpy_metric(group=reduce_group,
                                       reduce_op=reduce_op)(super().__call__)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        def _to_device_dtype(x: torch.Tensor) -> torch.Tensor:
            return x.to(device=self.device, dtype=self.dtype, non_blocking=True)

        return apply_to_collection(self._orig_call(*args, **kwargs), torch.Tensor,
                                   _to_device_dtype)
