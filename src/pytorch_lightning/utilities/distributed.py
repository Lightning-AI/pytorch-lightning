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
"""Utilities that can be used with distributed training."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel

import pytorch_lightning as pl
from pytorch_lightning.utilities.imports import _HPU_AVAILABLE, _TPU_AVAILABLE
from pytorch_lightning.utilities.rank_zero import rank_zero_debug as new_rank_zero_debug
from pytorch_lightning.utilities.rank_zero import rank_zero_only  # noqa: F401
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation
from pytorch_lightning.utilities.rank_zero import rank_zero_info as new_rank_zero_info

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


def gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> List[Tensor]:
    """Function to gather all tensors from several ddp processes onto a list that is broadcasted to all processes.

    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

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
    torch.distributed.barrier(group=group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result


def _simple_gather_all_tensors(result: Tensor, group: Any, world_size: int) -> List[Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result, group)
    return gathered_result


def distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized() or tpu_distributed()


def sync_ddp_if_available(
    result: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
) -> Tensor:
    """Function to reduce a tensor across worker processes during distributed training.

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


def sync_ddp(result: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None) -> Tensor:
    """Function to reduce the tensors from several ddp processes to one main process.

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

    op: Optional[ReduceOp]
    if isinstance(reduce_op, str):
        if reduce_op.lower() in ("avg", "mean"):
            op = ReduceOp.SUM
            divide_by_world_size = True
        else:
            op = getattr(ReduceOp, reduce_op.upper())
    else:
        op = reduce_op

    # WA for HPU. HPU doesn't support Long types, forcefully set it to float
    if _HPU_AVAILABLE:
        is_hpu_backend = os.environ.get("HCCL_DISTRIBUTED_BACKEND") == "1"
        if is_hpu_backend:
            if (result.type() == "torch.LongTensor") or (result.type() == "torch.hpu.LongTensor"):
                new_rank_zero_info("Long tensor unsupported on HPU, casting to float")
                result = result.float()

    # sync all processes before reduction
    torch.distributed.barrier(group=group)
    torch.distributed.all_reduce(result, op=op, group=group, async_op=False)

    if divide_by_world_size:
        result = result / torch.distributed.get_world_size(group)

    return result


class AllGatherGrad(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        tensor: Tensor,
        group: Optional["torch.distributed.ProcessGroup"] = group.WORLD,
    ) -> Tensor:
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[Tensor, None]:
        grad_output = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None


def all_gather_ddp_if_available(
    tensor: Tensor, group: Optional["torch.distributed.ProcessGroup"] = None, sync_grads: bool = False
) -> Tensor:
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
            communication hook wrapper to support a communication hook such
            as FP16 compression as wrapper, which could be combined with
            ddp_comm_hook

    Examples:

        >>> from torch.distributed.algorithms.ddp_comm_hooks import ( # doctest: +SKIP
        ...     default_hooks as default,
        ...     powerSGD_hook as powerSGD,
        ...     post_localSGD_hook as post_localSGD,
        ... )
        >>>
        >>> # fp16_compress_hook for compress gradients
        >>> ddp_model = ...
        >>> register_ddp_comm_hook( # doctest: +SKIP
        ...     model=ddp_model,
        ...     ddp_comm_hook=default.fp16_compress_hook,
        ... )
        >>>
        >>> # powerSGD_hook
        >>> ddp_model = ...
        >>> register_ddp_comm_hook( # doctest: +SKIP
        ...     model=ddp_model,
        ...     ddp_comm_state=powerSGD.PowerSGDState(
        ...         process_group=None,
        ...         matrix_approximation_rank=1,
        ...         start_powerSGD_iter=5000,
        ...     ),
        ...     ddp_comm_hook=powerSGD.powerSGD_hook,
        ... )
        >>>
        >>> # post_localSGD_hook
        >>> subgroup, _ = torch.distributed.new_subgroups() # doctest: +SKIP
        >>> ddp_model = ...
        >>> register_ddp_comm_hook( # doctest: +SKIP
        ...     model=ddp_model,
        ...     state=post_localSGD.PostLocalSGDState(
        ...         process_group=None,
        ...         subgroup=subgroup,
        ...         start_localSGD_iter=1_000,
        ...     ),
        ...     ddp_comm_hook=post_localSGD.post_localSGD_hook,
        ... )
        >>>
        >>> # fp16_compress_wrapper combined with other communication hook
        >>> ddp_model = ...
        >>> register_ddp_comm_hook( # doctest: +SKIP
        ...     model=ddp_model,
        ...     ddp_comm_state=powerSGD.PowerSGDState(
        ...         process_group=None,
        ...         matrix_approximation_rank=1,
        ...         start_powerSGD_iter=5000,
        ...     ),
        ...     ddp_comm_hook=powerSGD.powerSGD_hook,
        ...     ddp_comm_wrapper=default.fp16_compress_wrapper,
        ... )
    """
    if ddp_comm_hook is None:
        return
    # inform mypy that ddp_comm_hook is callable
    ddp_comm_hook: Callable = ddp_comm_hook

    if ddp_comm_wrapper is not None:
        new_rank_zero_info(
            f"DDP comm wrapper is provided, apply {ddp_comm_wrapper.__qualname__}({ddp_comm_hook.__qualname__})."
        )
        ddp_comm_hook = ddp_comm_wrapper(ddp_comm_hook)

    new_rank_zero_debug(f"Registering DDP comm hook: {ddp_comm_hook.__qualname__}.")
    model.register_comm_hook(state=ddp_comm_state, hook=ddp_comm_hook)  # type: ignore[operator]


def tpu_distributed() -> bool:
    return _TPU_AVAILABLE and xm.xrt_world_size() > 1


def get_default_process_group_backend_for_device(device: torch.device) -> str:
    return "nccl" if device.type == "cuda" else "gloo"


def _get_process_group_backend_from_env() -> Optional[str]:
    torch_backend = os.getenv("PL_TORCH_DISTRIBUTED_BACKEND")
    if torch_backend is not None:
        rank_zero_deprecation(
            "Environment variable `PL_TORCH_DISTRIBUTED_BACKEND`"
            " was deprecated in v1.6 and will be removed in v1.8."
            " Specify `process_group_backend` directly on the strategy constructor."
        )
    return torch_backend


def init_dist_connection(
    cluster_environment: "pl.plugins.environments.ClusterEnvironment",
    torch_distributed_backend: str,
    global_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """Utility function to initialize distributed connection by setting env variables and initializing the
    distributed process group.

    Args:
        cluster_environment: ``ClusterEnvironment`` instance
        torch_distributed_backend: backend to use (includes `nccl` and `gloo`)
        global_rank: rank of the current process
        world_size: number of processes in the group
        kwargs: kwargs for ``init_process_group``

    Raises:
        RuntimeError:
            If ``torch.distributed`` is not available
    """
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available. Cannot initialize distributed process group")
    if torch.distributed.is_initialized():
        log.debug("torch.distributed is already initialized. Exiting early")
        return
    global_rank = global_rank if global_rank is not None else cluster_environment.global_rank()
    world_size = world_size if world_size is not None else cluster_environment.world_size()
    os.environ["MASTER_ADDR"] = cluster_environment.main_address
    os.environ["MASTER_PORT"] = str(cluster_environment.main_port)
    log.info(f"Initializing distributed: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
    torch.distributed.init_process_group(torch_distributed_backend, rank=global_rank, world_size=world_size, **kwargs)

    # on rank=0 let everyone know training is starting
    new_rank_zero_info(
        f"{'-' * 100}\n"
        f"distributed_backend={torch_distributed_backend}\n"
        f"All distributed processes registered. Starting with {world_size} processes\n"
        f"{'-' * 100}\n"
    )


def _broadcast_object_list(obj: Any, rank: int) -> Any:
    objects = [obj if torch.distributed.get_rank() == rank else None]
    torch.distributed.broadcast_object_list(objects, src=rank)
    return objects[0]


# TODO: Refactor with the Strategy Collectives once finalized.
def _collect_states_on_rank_zero(state: Dict[str, Any]) -> Dict[int, Any]:
    """This distributed utility collects dictionary state across all processes.

    Args:
        state: Dictionary containing the state of the current process

    Returns:
        states: On global rank 0, a dictionary where the primary keys are
            the process rank and the values their associated states. Otherwise, returns None.
    """
    if not distributed_available():
        return {0: state}
    return {rank: _broadcast_object_list(state, rank) for rank in range(torch.distributed.get_world_size())}


def rank_zero_info(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "pytorch_lightning.utilities.distributed.rank_zero_info has been deprecated in v1.6"
        " and will be removed in v1.8."
        " Use the equivalent function from the pytorch_lightning.utilities.rank_zero module instead."
    )
    return new_rank_zero_info(*args, **kwargs)


def rank_zero_debug(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "pytorch_lightning.utilities.distributed.rank_zero_debug has been deprecated in v1.6"
        " and will be removed in v1.8."
        " Use the equivalent function from the pytorch_lightning.utilities.rank_zero module instead."
    )
    return new_rank_zero_debug(*args, **kwargs)
