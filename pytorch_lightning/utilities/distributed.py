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

import logging
import os
from functools import wraps
from platform import python_version
from typing import Any, Optional, Union

import torch
from torch.nn.parallel.distributed import DistributedDataParallel

from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_8, _TORCH_GREATER_EQUAL_1_9, _TPU_AVAILABLE

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm

if torch.distributed.is_available():
    from torch.distributed import group, ReduceOp

else:

    class ReduceOp:
        SUM = None

    class group:
        WORLD = None


log = logging.getLogger(__name__)


def rank_zero_only(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


# TODO: this should be part of the cluster environment
def _get_rank() -> int:
    rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, "rank", _get_rank())


def rank_zero_warn(*args, stacklevel: int = 5, **kwargs):
    from pytorch_lightning.utilities.warnings import rank_zero_deprecation, rank_zero_warn

    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.rank_zero_warn` has been moved to"
        " `pytorch_lightning.utilities.rank_zero_warn` in v1.3.7 and will be removed in v1.6"
    )
    return rank_zero_warn(*args, stacklevel=stacklevel, **kwargs)


def rank_zero_deprecation(*args, stacklevel: int = 5, **kwargs):
    from pytorch_lightning.utilities.warnings import rank_zero_deprecation

    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.rank_zero_deprecation` has been moved to"
        " `pytorch_lightning.utilities.rank_zero_deprecation` in v1.3.7 and will be removed in v1.6"
    )
    return rank_zero_deprecation(*args, stacklevel=stacklevel, **kwargs)


def _info(*args, stacklevel: int = 2, **kwargs):
    if python_version() >= "3.8.0":
        kwargs["stacklevel"] = stacklevel
    log.info(*args, **kwargs)


def _debug(*args, stacklevel: int = 2, **kwargs):
    if python_version() >= "3.8.0":
        kwargs["stacklevel"] = stacklevel
    log.debug(*args, **kwargs)


@rank_zero_only
def rank_zero_debug(*args, stacklevel: int = 4, **kwargs):
    _debug(*args, stacklevel=stacklevel, **kwargs)


@rank_zero_only
def rank_zero_info(*args, stacklevel: int = 4, **kwargs):
    _info(*args, stacklevel=stacklevel, **kwargs)


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


def distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized() or tpu_distributed()


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
    if distributed_available():
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

    if isinstance(reduce_op, str):
        if reduce_op.lower() in ("avg", "mean"):
            op = ReduceOp.SUM
            divide_by_world_size = True
        else:
            op = getattr(ReduceOp, reduce_op.upper())
    else:
        op = reduce_op

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

        return grad_output[torch.distributed.get_rank()], None


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
    if distributed_available():
        if sync_grads:
            return AllGatherGrad.apply(tensor, group)
        with torch.no_grad():
            return AllGatherGrad.apply(tensor, group)
    return tensor


def register_ddp_comm_hook(
    model: DistributedDataParallel,
    ddp_comm_state: Optional[object] = None,
    ddp_comm_hook: Optional[callable] = None,
    ddp_comm_wrapper: Optional[callable] = None,
) -> None:
    """
    Function to register communication hook for DDP model
    https://pytorch.org/docs/master/ddp_comm_hooks.html

    Args:
        model:
            DDP model
        ddp_comm_state:
            state is passed to the hook and can be used to maintain
            and update any state information that users would like to
            maintain as part of the training process. Examples: error
            feedback in gradient compression, peers to communicate with
            next in GossipGrad etc.
        ddp_comm_hook:
            hook(state: object, bucket: dist._GradBucket) -> torch.futures.Future

            This callable function is called once the bucket is ready. The
            hook can perform whatever processing is needed and return
            a Future indicating completion of any async work (ex: allreduce).
            If the hook doesn't perform any communication, it can also
            just return a completed Future. The Future should hold the
            new value of grad bucket's tensors. Once a bucket is ready,
            c10d reducer would call this hook and use the tensors returned
            by the Future and copy grads to individual parameters.
        ddp_comm_wrapper:
            communication hook wraper to support a communication hook such
            as FP16 compression as wrapper, which could be combined with
            ddp_comm_hook

    .. warning ::
        DDP communication hook needs pytorch version at least 1.8.0

    .. warning ::
        DDP communication wrapper needs pytorch version at least 1.9.0

    Example:

        from torch.distributed.algorithms.ddp_comm_hooks import (
            default_hooks as default,
            powerSGD_hook as powerSGD,
        )

        # fp16_compress_hook for compress gradients
        register_ddp_comm_hook(
            model=ddp_model,
            ddp_comm_hook=default.fp16_compress_hook,
        )

        # powerSGD_hook
        register_ddp_comm_hook(
            model=ddp_model,
            ddp_comm_state=powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=5000,
            ),
            ddp_comm_hook=powerSGD.powerSGD_hook,
        )

        # fp16_compress_wrapper combined with other communication hook
        register_ddp_comm_hook(
            model=ddp_model,
            ddp_comm_state=powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=5000,
            ),
            ddp_comm_hook=powerSGD.powerSGD_hook,
            ddp_comm_wrapper=default.fp16_compress_wrapper,
        )
    """
    from pytorch_lightning.utilities import rank_zero_warn

    if not _TORCH_GREATER_EQUAL_1_8:
        rank_zero_warn("Not registering DDP comm hook. To use communication hooks, please use pytorch>=1.8.0.")
        return
    if ddp_comm_hook is None:
        return
    if ddp_comm_wrapper is not None:
        if not _TORCH_GREATER_EQUAL_1_9:
            rank_zero_warn("Not applying DDP comm wrapper. To use communication wrapper, please use pytorch>=1.9.0.")
        else:
            rank_zero_info(
                f"DDP comm wrapper is provided, apply {ddp_comm_wrapper.__qualname__}({ddp_comm_hook.__qualname__})."
            )
            ddp_comm_hook = ddp_comm_wrapper(ddp_comm_hook)

    rank_zero_debug(f"Registering DDP comm hook: {ddp_comm_hook.__qualname__}.")
    model.register_comm_hook(state=ddp_comm_state, hook=ddp_comm_hook)


def tpu_distributed() -> bool:
    return _TPU_AVAILABLE and xm.xrt_world_size() > 1
