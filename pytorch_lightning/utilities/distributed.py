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
import inspect
import logging
import os
import warnings
from functools import wraps
from typing import Any, Optional, Union

import torch
from torch.utils.data import DataLoader, Sampler, BatchSampler

from pytorch_lightning.utilities.exceptions import MisconfigurationException

log = logging.getLogger(__name__)

if torch.distributed.is_available():
    from torch.distributed import group, ReduceOp

else:

    class ReduceOp:
        SUM = None

    class group:
        WORLD = None


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

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)

    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]

    # sync and broadcast all
    torch.distributed.barrier(group=group)
    torch.distributed.all_gather(gathered_result, result, group)

    return gathered_result


def sync_ddp_if_available(
    result: Union[torch.Tensor],
    group: Optional[Any] = None,
    reduce_op: Optional[Union[ReduceOp, str]] = None
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
    result: Union[torch.Tensor],
    group: Optional[Any] = None,
    reduce_op: Optional[Union[ReduceOp, str]] = None
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

    op = reduce_op if isinstance(reduce_op, ReduceOp) else ReduceOp.SUM

    if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
        divide_by_world_size = True

    # sync all processes before reduction
    torch.distributed.barrier(group=group)
    torch.distributed.all_reduce(result, op=op, group=group, async_op=False)

    if divide_by_world_size:
        result = result / torch.distributed.get_world_size(group)

    return result


class AllGatherGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, group=group.WORLD):
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, *grad_output):
        grad_output = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()]


def all_gather_ddp_if_available(
    tensor: Union[torch.Tensor], group: Optional[Any] = None, sync_grads: bool = False
) -> torch.Tensor:
    """
    Function to gather a tensor from several distributed processes

    Args:
        tensor: tensor of shape (batch, ...)
        group: the process group to gather results from. Defaults to all processes (world)
        sync_grads: flag that allows users to synchronize gradients for all_gather op

    Return:
        A tensor of shape (world_size, batch, ...)
    """
    group = group if group is not None else torch.distributed.group.WORLD
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if sync_grads:
            return AllGatherGrad.apply(tensor, group)
        else:
            with torch.no_grad():
                return AllGatherGrad.apply(tensor, group)
    return tensor


def replace_sampler(dataloader: DataLoader, sampler: Sampler):
    skip_keys = ('sampler', 'batch_sampler', 'dataset_kind')
    skip_signature_keys = ('args', 'kwargs', 'self')

    attrs = {k: v for k, v in vars(dataloader).items() if not k.startswith("_")}

    params = set(inspect.signature(dataloader.__init__).parameters)
    contains_dataset = True

    if type(dataloader) is not DataLoader:
        contains_dataset = "dataset" in params
        params.update(inspect.signature(DataLoader.__init__).parameters)

    dl_args = {name: attrs[name] for name in params if name in attrs and name not in skip_keys}
    dl_args = _resolve_batch_sampler(dl_args, dataloader, sampler)

    multiprocessing_context = dataloader.multiprocessing_context
    dl_args['multiprocessing_context'] = multiprocessing_context

    missing_kwargs = params.difference(skip_signature_keys).difference(dl_args)
    if missing_kwargs:
        """
        Example:
        class CustomDataLoader(DataLoader):
            def __init__(self, num_features, dataset, *args, **kwargs):
                self.num_features = num_features
                super().__init__(dataset, *args, **kwargs)
        """
        dataloader_cls_name = dataloader.__class__.__name__
        raise MisconfigurationException(
            f"Trying to inject DistributedSampler within {dataloader_cls_name} class."
            "This would fail as your DataLoader doesn't expose all its __init__ parameters as attributes. "
            f"Missing attributes are {missing_kwargs}. "
            f"HINT: If you wrote the {dataloader_cls_name} class, add the `__init__` arguments as attributes or ",
            "manually add DistributedSampler as "
            f"{dataloader_cls_name}(dataset, ..., sampler=DistributedSampler(dataset, ...)).",
        )

    if not contains_dataset:
        dl_args.pop('dataset')

    dataloader = type(dataloader)(**dl_args)
    dataloader.multiprocessing_context = multiprocessing_context
    return dataloader


def _resolve_batch_sampler(dl_args, dataloader, sampler):
    batch_sampler = getattr(dataloader, "batch_sampler")
    if batch_sampler is not None and type(batch_sampler) is not BatchSampler:
        batch_sampler = type(batch_sampler)(
            sampler,
            batch_size=batch_sampler.batch_size,
            drop_last=batch_sampler.drop_last,
        )
        dl_args['batch_sampler'] = batch_sampler
        dl_args['batch_size'] = 1
        dl_args['shuffle'] = False
        dl_args['sampler'] = None
        dl_args['drop_last'] = False
    else:
        dl_args['sampler'] = sampler
        dl_args['shuffle'] = False
        dl_args['batch_sampler'] = None
    return dl_args