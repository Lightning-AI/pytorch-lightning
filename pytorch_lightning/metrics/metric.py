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

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Implements the actual metric computation.

        Returns:
            metric value

        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return sync_collections(super().__call__(*args, **kwargs),
                                group=self.reduce_group,
                                reduce_op=self.reduce_op)


def sync_ddp(result: Union[torch.Tensor, numbers.Number],
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


def sync_collections(result: Union[torch.Tensor, numbers.Number,
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

    func = partial(sync_collections, group=group, reduce_op=reduce_op)

    # convert numpy to tensor if possible
    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array not of string classes and object
        if elem_type.__name__ != 'ndarray' \
                or np_str_obj_array_pattern.search(result.dtype.str) is None:
            result = torch.as_tensor(result)

    if isinstance(result, (torch.Tensor, numbers.Number)):
        return sync_ddp(result, group=group, reduce_op=reduce_op)

    elif isinstance(result, Mapping):
        return elem_type({key: func(result[key]) for key in result})
    elif isinstance(result, tuple) and hasattr(result, '_fields'):  # namedtuple
        return elem_type(*(func(r) for r in result))
    elif isinstance(result, Sequence) and not isinstance(result, str):
        return elem_type([func(r) for r in result])
    else:
        return func(result)
