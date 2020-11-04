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
import warnings
from functools import wraps

import torch
from pytorch_lightning import _logger as log
from typing import Union, Optional, Any

if torch.distributed.is_available():
    from torch.distributed import ReduceOp
else:
    class ReduceOp:
        SUM = None


def rank_zero_only(fn):

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, 'rank', int(os.environ.get('LOCAL_RANK', 0)))


def _warn(*args, **kwargs):
    warnings.warn(*args, **kwargs)


def _info(*args, **kwargs):
    log.info(*args, **kwargs)


def _debug(*args, **kwargs):
    log.debug(*args, **kwargs)


rank_zero_debug = rank_zero_only(_debug)
rank_zero_info = rank_zero_only(_info)
rank_zero_warn = rank_zero_only(_warn)


def find_free_network_port() -> int:
    """
    Finds a free port on localhost.
    It is useful in single-node training when we don't want to connect to a real master node but
    have to set the `MASTER_PORT` environment variable.
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def gather_all_tensors(result: Union[torch.Tensor], group: Optional[Any] = None):
    """
    Function to gather all tensors from several ddp processes onto a list that
    is broadcasted to all processes

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    if group is None:
        group = torch.distributed.group.WORLD

    world_size = torch.distributed.get_world_size(group)

    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]

    # sync and broadcast all
    torch.distributed.barrier(group=group)
    torch.distributed.all_gather(gathered_result, result, group)

    return gathered_result


def sync_ddp_if_available(
    result: Union[torch.Tensor], group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
) -> torch.Tensor:
    """
    Function to reduce a tensor across worker processes during distributed training
    Args:
        result: the value to sync and reduce (typically tensor or number)
        group: the process group to gather results from. Defaults to all processes (world)
        reduce_op: the reduction operation. Defaults to sum.
            Can also be a string of 'avg', 'mean' to calculate the mean during reduction.
    Return:
        reduced value
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return sync_ddp(result, group=group, reduce_op=reduce_op)
    return result


def sync_ddp(
    result: Union[torch.Tensor], group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
) -> torch.Tensor:
    """
    Function to reduce the tensors from several ddp processes to one master process

    Args:
        result: the value to sync and reduce (typically tensor or number)
        group: the process group to gather results from. Defaults to all processes (world)
        reduce_op: the reduction operation. Defaults to sum.
            Can also be a string of 'avg', 'mean' to calculate the mean during reduction.

    Return:
        reduced value
    """
    divide_by_world_size = False

    if group is None:
        group = torch.distributed.group.WORLD

    if reduce_op is None:
        reduce_op = torch.distributed.ReduceOp.SUM
    elif isinstance(reduce_op, str) and reduce_op in ("avg", "mean"):
        reduce_op = torch.distributed.ReduceOp.SUM
        divide_by_world_size = True

    # sync all processes before reduction
    torch.distributed.barrier(group=group)
    torch.distributed.all_reduce(result, op=reduce_op, group=group, async_op=False)

    if divide_by_world_size:
        result = result / torch.distributed.get_world_size(group)

    return result
