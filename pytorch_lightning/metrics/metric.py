import numbers
from abc import ABC, abstractmethod
from collections import Mapping, Sequence
from functools import partial
from typing import Union, Any, Optional

import torch
import torch.distributed
from torch.utils.data._utils.collate import np_str_obj_array_pattern

__all__ = ['BaseMetric']


class BaseMetric(torch.nn.Module, ABC):
    def __init__(self, name: str,
                 reduce_group: Optional[Any] = torch.distributed.group.WORLD,
                 reduce_op: Optional[Any] = torch.distributed.ReduceOp.SUM):
        """
        Abstract Base Class for metric implementation.

        Automatically handles the computation
        Args:
            name: the metric's name
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__()
        self.name = name
        self.reduce_op = reduce_op
        self.reduce_group = reduce_group
        self._dtype = torch.get_default_dtype()
        self._device = torch.device('cpu')

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Implements the actual metric computation.

        Returns:
            metric value

        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return _sync_collections(super().__call__(*args, **kwargs),
                                 group=self.reduce_group,
                                 reduce_op=self.reduce_op)

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


def _sync_ddp(result: Union[torch.Tensor, numbers.Number],
              group: Any = torch.distributed.group.WORLD,
              reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM) -> torch.Tensor:
    """
    Function to reduce the tensors from several ddp processes to one master process

    Args:
        result: the value to sync and reduce (typically tensor or number)
        group: the process group to gather results from. Defaults to all processes (world)
        reduce_op: the reduction operation. Defaults to sum

    Returns:
        reduced value

    """

    # convert to tensor if necessary
    if not isinstance(result, torch.Tensor):
        result = torch.tensor(result)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # sync all processes before reduction
        torch.distributed.barrier(group=group)
        torch.distributed.all_reduce(result, op=reduce_op, group=group,
                                     async_op=False)

    return result


def _sync_collections(result: Union[torch.Tensor, numbers.Number,
                                    Mapping, Sequence],
                      group: Any = torch.distributed.group.WORLD,
                      reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM
                      ) -> Union[torch.Tensor, numbers.Number,
                                 Mapping, Sequence]:
    """
    Recursively applies sync_ddp to collections

    Args:
        result: Tensor or Number or Mapping or Sequence holding the values to be reduced
        group: the process group to gather results from. Defaults to all processes (world)
        reduce_op: the reduction operation. Defaults to sum

    Returns:
        the reduced collection

    """
    # function adapted from torch.utils.data._utils.collate
    elem_type = type(result)

    func = partial(_sync_collections, group=group, reduce_op=reduce_op)

    # convert numpy to tensor if possible
    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array not of string classes and object
        if elem_type.__name__ != 'ndarray' \
                or np_str_obj_array_pattern.search(result.dtype.str) is None:
            result = torch.as_tensor(result)

    if isinstance(result, (torch.Tensor, numbers.Number)):
        return _sync_ddp(result, group=group, reduce_op=reduce_op)

    elif isinstance(result, Mapping):
        return elem_type({key: func(result[key]) for key in result})
    elif isinstance(result, tuple) and hasattr(result, '_fields'):  # namedtuple
        return elem_type(*(func(r) for r in result))
    elif isinstance(result, Sequence) and not isinstance(result, str):
        return elem_type([func(r) for r in result])
    # not possible to reduce this type
    else:
        return result
