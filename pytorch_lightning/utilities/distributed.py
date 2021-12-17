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
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel

import pytorch_lightning as pl
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_8, _TORCH_GREATER_EQUAL_1_9, _TPU_AVAILABLE

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm

if torch.distributed.is_available():
    from torch.distributed import group, ReduceOp

else:

    class ReduceOp:  # type: ignore # (see https://github.com/python/mypy/issues/1153)
        SUM = None

    class group:  # type: ignore
        WORLD = None


log = logging.getLogger(__name__)


def rank_zero_only(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
        return None

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


def rank_zero_warn(*args: Any, stacklevel: int = 5, **kwargs: Any) -> None:
    from pytorch_lightning.utilities.warnings import rank_zero_deprecation, rank_zero_warn

    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.rank_zero_warn` has been moved to"
        " `pytorch_lightning.utilities.rank_zero_warn` in v1.3.7 and will be removed in v1.6"
    )
    return rank_zero_warn(*args, stacklevel=stacklevel, **kwargs)


def rank_zero_deprecation(*args: Any, stacklevel: int = 5, **kwargs: Any) -> None:
    from pytorch_lightning.utilities.warnings import rank_zero_deprecation

    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.rank_zero_deprecation` has been moved to"
        " `pytorch_lightning.utilities.rank_zero_deprecation` in v1.3.7 and will be removed in v1.6"
    )
    return rank_zero_deprecation(*args, stacklevel=stacklevel, **kwargs)


def _info(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
    if python_version() >= "3.8.0":
        kwargs["stacklevel"] = stacklevel
    log.info(*args, **kwargs)


def _debug(*args: Any, stacklevel: int = 2, **kwargs: Any) -> None:
    if python_version() >= "3.8.0":
        kwargs["stacklevel"] = stacklevel
    log.debug(*args, **kwargs)


@rank_zero_only
def rank_zero_debug(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
    _debug(*args, stacklevel=stacklevel, **kwargs)


@rank_zero_only
def rank_zero_info(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
    _info(*args, stacklevel=stacklevel, **kwargs)


def gather_all_tensors(result: torch.Tensor, group: Optional[Any] = None) -> List[torch.Tensor]:
    """Function to gather all tensors from several ddp processes onto a list that is broadcasted to all processes.

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
    result: torch.Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
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
    result: torch.Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
) -> torch.Tensor:
    """Function to reduce the tensors from several ddp processes to one master process.

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
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Optional["torch.distributed.ProcessGroup"] = group.WORLD,
    ) -> torch.Tensor:
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None


def all_gather_ddp_if_available(
    tensor: torch.Tensor, group: Optional["torch.distributed.ProcessGroup"] = None, sync_grads: bool = False
) -> torch.Tensor:
    """Function to gather a tensor from several distributed processes.

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
    ddp_comm_hook: Optional[Callable] = None,
    ddp_comm_wrapper: Optional[Callable] = None,
) -> None:
    """Function to register communication hook for DDP model https://pytorch.org/docs/master/ddp_comm_hooks.html.

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
        Post-localSGD hook needs pytorch version at least 1.9.0

    Example:

        from torch.distributed.algorithms.ddp_comm_hooks import (
            default_hooks as default,
            powerSGD_hook as powerSGD,
            post_localSGD_hook as post_localSGD,
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

        # post_localSGD_hook
        subgroup, _ = torch.distributed.new_subgroups()
        register_comm_hook(
            model=ddp_model,
            state=post_localSGD.PostLocalSGDState(
                process_group=None,
                subgroup=subgroup,
                start_localSGD_iter=1_000,
            ),
            ddp_comm_hook=post_localSGD.post_localSGD_hook,
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
    # inform mypy that ddp_comm_hook is callable
    ddp_comm_hook: Callable = ddp_comm_hook

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


def init_dist_connection(
    cluster_environment: "pl.plugins.environments.ClusterEnvironment",
    torch_distributed_backend: str,
    global_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """Utility function to initialize distributed connection by setting env variables and initiliazing the
    distributed process group.

    Args:
        cluster_environment: ``ClusterEnvironment`` instance
        torch_distributed_backend: backend to use (includes `nccl` and `gloo`)
        global_rank: rank of the current process
        world_size: number of processes in the group
        kwargs: kwargs for ``init_process_group``
    """
    global_rank = global_rank if global_rank is not None else cluster_environment.global_rank()
    world_size = world_size if world_size is not None else cluster_environment.world_size()
    os.environ["MASTER_ADDR"] = cluster_environment.master_address()
    os.environ["MASTER_PORT"] = str(cluster_environment.master_port())
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        log.info(f"initializing distributed: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
        torch.distributed.init_process_group(
            torch_distributed_backend, rank=global_rank, world_size=world_size, **kwargs
        )

        # on rank=0 let everyone know training is starting
        rank_zero_info(
            f"{'-' * 100}\n"
            f"distributed_backend={torch_distributed_backend}\n"
            f"All distributed processes registered. Starting with {world_size} processes\n"
            f"{'-' * 100}\n"
        )


class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input: torch.Tensor) -> None:
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the subclass.
        # Here, we are bypassing some tensor sanity checks and trusting that the user
        # provides the right input dimensions at inference.
        return


def _revert_sync_batchnorm(module: Module) -> Module:
    # Code adapted from https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547
    # Original author: Kapil Yedidi (@kapily)
    converted_module = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        # Unfortunately, SyncBatchNorm does not store the original class - if it did
        # we could return the one that was originally created.
        converted_module = _BatchNormXd(
            module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats
        )
        if module.affine:
            with torch.no_grad():
                converted_module.weight = module.weight
                converted_module.bias = module.bias
        converted_module.running_mean = module.running_mean
        converted_module.running_var = module.running_var
        converted_module.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            converted_module.qconfig = module.qconfig
    for name, child in module.named_children():
        converted_module.add_module(name, _revert_sync_batchnorm(child))
    del module
    return converted_module
