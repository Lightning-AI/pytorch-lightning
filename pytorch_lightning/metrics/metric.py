from abc import ABC, abstractmethod
import numbers
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed

from pytorch_lightning.metrics.converters import (
    sync_ddp_if_available, convert_to_tensor, convert_to_numpy)
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin


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
        self.register_forward_pre_hook(self.input_convert)
        self.register_forward_hook(self.output_convert)
        self.register_forward_hook(self.ddp_sync)
        self.register_forward_hook(self.compute)
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Implements the actual metric computation. 

        Returns:
            metric value or metric state

        """
        raise NotImplementedError
        
    def compute(self, module, input, output) -> torch.Tensor:
        """
        Output contains the 
        """
        return output
    
    def ddp_sync(self, module, input, output):
        return output
    
    def input_convert(self, module, input):
        return input
    
    def output_convert(self, module, input, output):
        return output
        
class TensorMetric(Metric):
    """
    Base class for metric implementation operating directly on tensors.
    All inputs and outputs will be casted to tensors if necessary.
    Already handles DDP sync and input/output conversions.
    """

    def __init__(self, name: str,
                 reduce_group: Optional[Any] = None,
                 reduce_op: Optional[Any] = None,
                 ddp_normalize: bool = False):
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
        self.ddp_normalize = ddp_normalize
    
    def input_convert(self, module, input):
        return apply_to_collection(input, 
                                   (torch.Tensor, np.ndarray, numbers.Number), 
                                   convert_to_tensor, self.dtype, self.device)

    def output_convert(self, module, input, output):
        return apply_to_collection(output, 
                                   (torch.Tensor, np.ndarray, numbers.Number), 
                                   convert_to_tensor, self.dtype, self.device)
    
    def ddp_sync(self, module, input, output):
        return apply_to_collection(output, torch.Tensor, sync_ddp_if_available,
                                   self.reduce_group, self.reduce_op, self.ddp_normalize)


class NumpyMetric(Metric):
    """
    Base class for metric implementation operating on numpy arrays.
    All inputs will be casted to numpy if necessary and all outputs will
    be casted to tensors if necessary.
    Already handles DDP sync and input/output conversions.
    """

    def __init__(self, name: str,
                 reduce_group: Optional[Any] = None,
                 reduce_op: Optional[Any] = None,
                 ddp_normalize: bool = False):
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
        self.ddp_normalize = ddp_normalize

    def input_convert(self, module, input):
        return apply_to_collection(input, 
                                   (torch.Tensor, np.ndarray, numbers.Number), 
                                   convert_to_numpy)

    def output_convert(self, module, input, output):
        return apply_to_collection(output, 
                                   (torch.Tensor, np.ndarray, numbers.Number), 
                                   convert_to_tensor, self.dtype, self.device)

    
    def ddp_sync(self, module, input, output):
        return apply_to_collection(output, torch.Tensor, sync_ddp_if_available,
                                   self.reduce_group, self.reduce_op, self.ddp_normalize)
    
    
    