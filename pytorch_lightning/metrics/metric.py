from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.distributed

from pytorch_lightning.metrics.utils import tensor_metric, numpy_metric
from pytorch_lightning.utilities.apply_to_collection import apply_to_collection

__all__ = ['AbstractMetric', 'TensorMetric', 'NumpyMetric']


class AbstractMetric(torch.nn.Module, ABC):
    def __init__(self, name: str):
        """
        Abstract Base Class for metric implementation.
        Should be used to implement metrics that
        1.) Return multiple Outputs
        2.) Handle their own DDP sync

        Args:
            name: the metric's name

        """
        super().__init__()
        self.name = name
        self._dtype = torch.get_default_dtype()
        self._device = torch.device('cpu')

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: Any):
        # Necessary to avoid infinite recursion
        raise RuntimeError('Cannot set the dtype explicitly. Please use metric.to(new_dtype).')

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, new_device):
        # Necessary to avoid infinite recursion
        raise RuntimeError('Cannot set the device explicitly. Please use metric.to(new_device).')

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Implements the actual metric computation.

        Returns:
            metric value

        """
        raise NotImplementedError

    def to(self, *args, **kwargs):
        """Moves and/or casts the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)

        .. function:: to(dtype, non_blocking=False)

        .. function:: to(tensor, non_blocking=False)

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device: the desired device of the parameters
                and buffers in this module
            dtype: the desired floating point type of
                the floating point parameters and buffers in this module
            tensor: Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module

        Returns:
            Module: self

        Example::

            >>> linear = nn.Linear(2, 2)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]])
            >>> linear.to(torch.double)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]], dtype=torch.float64)
            >>> gpu1 = torch.device("cuda:1")
            >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
            >>> cpu = torch.device("cpu")
            >>> linear.to(cpu)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16)

        """
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self._device = device

        if dtype is not None:
            self._dtype = dtype

        return super().to(*args, **kwargs)

    def cuda(self, device=None):
        """Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module:
        """

        self._device = torch.device('cuda', index=device)
        return super().cuda(device=device)

    def cpu(self):
        """Moves all model parameters and buffers to the CPU.

        Returns:
            Module: self
        """
        self._device = torch.device('cpu')
        return super().cpu()

    def type(self, dst_type):
        """Casts all parameters and buffers to :attr:`dst_type`.

        Arguments:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        self._dtype = dst_type
        return super().type(dst_type=dst_type)

    def float(self):
        """Casts all floating point parameters and buffers to float datatype.

        Returns:
            Module: self
        """
        self._dtype = torch.float
        return super().float()

    def double(self):
        """Casts all floating point parameters and buffers to ``double`` datatype.

        Returns:
            Module: self
        """
        self._dtype = torch.double
        return super().double()

    def half(self):
        """Casts all floating point parameters and buffers to ``half`` datatype.

        Returns:
            Module: self
        """
        self._dtype = torch.half
        return super().half()


class TensorMetric(AbstractMetric):
    def __init__(self, name: str,
                 reduce_group: Optional[Any] = torch.distributed.group.WORLD,
                 reduce_op: Optional[Any] = torch.distributed.ReduceOp.SUM):
        """
        Base class for metric implementation operating directly on tensors.
        All inputs and outputs will be casted to tensors if necessary.
        Already handles DDP sync and input/output conversions

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
        return apply_to_collection(self._orig_call(*args, **kwargs), torch.Tensor,
                                   lambda x: x.to(device=self.device, dtype=self.dtype))


class NumpyMetric(AbstractMetric):
    def __init__(self, name: str,
                 reduce_group: Optional[Any] = torch.distributed.group.WORLD,
                 reduce_op: Optional[Any] = torch.distributed.ReduceOp.SUM):
        """
        Base class for metric implementation operating on numpy arrays.
        All inputs will be casted to numpy if necessary and all outputs will
        be casted to tensors if necessary.
        Already handles DDP sync and input/output conversions

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
        return apply_to_collection(self._orig_call(*args, **kwargs), torch.Tensor,
                                   lambda x: x.to(device=self.device, dtype=self.dtype))
