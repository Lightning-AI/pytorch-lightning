from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
import torch.distributed

from pytorch_lightning.metrics.converters import tensor_metric, numpy_metric
from pytorch_lightning.utilities.apply_func import apply_to_collection

__all__ = ['Metric', 'TensorMetric', 'NumpyMetric']


class Metric(torch.nn.Module, ABC):
    """
    Abstract Base Class for metric implementation.

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

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: Union[str, torch.dtype]):
        # necessary to avoid infinite recursion
        raise RuntimeError('Cannot set the dtype explicitly. Please use metric.to(new_dtype).')

    @property
    def device(self) -> Union[str, torch.device]:
        return self._device

    @device.setter
    def device(self, new_device: Union[str, torch.device]):
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

    def to(self, *args, **kwargs) -> torch.nn.Module:
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

        Note:
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
            >>> class ExampleMetric(Metric):
            ...     def __init__(self, weight: torch.Tensor):
            ...         super().__init__('example')
            ...         self.register_buffer('weight', weight)
            ...     def forward(self, pred, target) -> torch.Tensor:
            ...         return (pred - target) * self.weight
            >>> _ = torch.manual_seed(0)
            >>> metric = ExampleMetric(torch.rand(3, 4))
            >>> metric.weight
            tensor([[0.4963, 0.7682, 0.0885, 0.1320],
                    [0.3074, 0.6341, 0.4901, 0.8964],
                    [0.4556, 0.6323, 0.3489, 0.4017]])
            >>> metric.to(torch.double) #doctest: +ELLIPSIS
            ExampleMetric()
            >>> metric.weight
            tensor([[...]], dtype=torch.float64)
            >>> cpu = torch.device('cpu')
            >>> metric.to(cpu, dtype=torch.half, non_blocking=True)
            ExampleMetric()
            >>> metric.weight #doctest: +ELLIPSIS
            tensor([[...]], dtype=torch.float16)
            >>> metric.to(cpu)
            ExampleMetric()
            >>> metric.weight #doctest: +ELLIPSIS
            tensor([[...]], dtype=torch.float16)


        """
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self._device = device

        if dtype is not None:
            self._dtype = dtype

        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[int] = None) -> torch.nn.Module:
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

    def cpu(self) -> torch.nn.Module:
        """Moves all model parameters and buffers to the CPU.

        Returns:
            Module: self
        """
        self._device = torch.device('cpu')
        return super().cpu()

    def type(self, dst_type: Union[str, torch.dtype]) -> torch.nn.Module:
        """Casts all parameters and buffers to :attr:`dst_type`.

        Arguments:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        self._dtype = dst_type
        return super().type(dst_type=dst_type)

    def float(self) -> torch.nn.Module:
        """Casts all floating point parameters and buffers to float datatype.

        Returns:
            Module: self
        """
        self._dtype = torch.float
        return super().float()

    def double(self) -> torch.nn.Module:
        """Casts all floating point parameters and buffers to ``double`` datatype.

        Returns:
            Module: self
        """
        self._dtype = torch.double
        return super().double()

    def half(self) -> torch.nn.Module:
        """Casts all floating point parameters and buffers to ``half`` datatype.

        Returns:
            Module: self
        """
        self._dtype = torch.half
        return super().half()


class TensorMetric(Metric):
    """
    Base class for metric implementation operating directly on tensors.
    All inputs and outputs will be casted to tensors if necessary.
    Already handles DDP sync and input/output conversions.
    """
    def __init__(self, name: str,
                 reduce_group: Optional[Any] = torch.distributed.group.WORLD,
                 reduce_op: Optional[Any] = torch.distributed.ReduceOp.SUM):
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
            return x.to(device=self.device, dtype=self.dtype)

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
                 reduce_group: Optional[Any] = torch.distributed.group.WORLD,
                 reduce_op: Optional[Any] = torch.distributed.ReduceOp.SUM):
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
            return x.to(device=self.device, dtype=self.dtype)

        return apply_to_collection(self._orig_call(*args, **kwargs), torch.Tensor,
                                   _to_device_dtype)
