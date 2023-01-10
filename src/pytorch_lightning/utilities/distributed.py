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

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel

from lightning_fabric.utilities.distributed import _all_gather_ddp_if_available as new_all_gather_ddp_if_available
from lightning_fabric.utilities.distributed import _distributed_available as new_distributed_available
from lightning_fabric.utilities.distributed import _gather_all_tensors as new_gather_all_tensors
from lightning_fabric.utilities.distributed import (
    _get_default_process_group_backend_for_device as new_get_default_process_group_backend_for_device,
)
from lightning_fabric.utilities.distributed import _init_dist_connection as new_init_dist_connection
from lightning_fabric.utilities.distributed import _sync_ddp as new_sync_ddp
from lightning_fabric.utilities.distributed import _sync_ddp_if_available as new_sync_ddp_if_available
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_deprecation, rank_zero_info


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
        rank_zero_info(
            f"DDP comm wrapper is provided, apply {ddp_comm_wrapper.__qualname__}({ddp_comm_hook.__qualname__})."
        )
        ddp_comm_hook = ddp_comm_wrapper(ddp_comm_hook)

    rank_zero_debug(f"Registering DDP comm hook: {ddp_comm_hook.__qualname__}.")
    model.register_comm_hook(state=ddp_comm_state, hook=ddp_comm_hook)  # type: ignore[operator]


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
    if not new_distributed_available():
        return {0: state}
    return {rank: _broadcast_object_list(state, rank) for rank in range(torch.distributed.get_world_size())}


def all_gather_ddp_if_available(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.all_gather_ddp_if_available` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_all_gather_ddp_if_available(*args, **kwargs)


def distributed_available() -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.distributed_available` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_distributed_available()


def gather_all_tensors(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.gather_all_tensors` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_gather_all_tensors(*args, **kwargs)


class AllGatherGrad(torch.autograd.Function):
    """Gathers tensors from the whole group and stacks them.

    This implementation is copied from PyTorch.

    .. deprecated:: v1.8.0
        This function has been deprecated in v1.8.0 in favor of :func:`torch.distributed.nn.functional.all_gather` and
        will be removed in v2.0.0.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        tensor: Tensor,
        group: Optional["torch.distributed.ProcessGroup"] = None,
    ) -> Tensor:
        rank_zero_deprecation(
            "`AllGatherGrad` has been deprecated in v1.8.0 and will be removed in v2.0.0."
            " Use `torch.distributed.nn.functional.all_gather` instead.",
            stacklevel=6,
        )
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


def get_default_process_group_backend_for_device(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.get_default_process_group_backend_for_device` has been deprecated"
        " in v1.8.0 and will be removed in v2.0.0. This function is internal but you can copy over its implementation."
        " `lightning_fabric.utilities.distributed.get_default_process_group_backend_for_device` instead."
    )
    return new_get_default_process_group_backend_for_device(*args, **kwargs)


def init_dist_connection(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.init_dist_connection` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_init_dist_connection(*args, **kwargs)


def sync_ddp(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.sync_ddp` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_sync_ddp(*args, **kwargs)


def sync_ddp_if_available(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.sync_ddp_if_available` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    return new_sync_ddp_if_available(*args, **kwargs)


def tpu_distributed() -> bool:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.tpu_distributed` has been deprecated in v1.8.0 and will"
        " be removed in v2.0.0. This function is internal but you can copy over its implementation."
    )
    from lightning_fabric.accelerators.tpu import _tpu_distributed

    return _tpu_distributed()


def rank_zero_only(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.distributed.rank_zero_only` has been deprecated in v1.8.1 and will"
        " be removed in v2.0.0. You can import it from `pytorch_lightning.utilities` instead."
    )
    from pytorch_lightning.utilities.rank_zero import rank_zero_only as new_rank_zero_only

    return new_rank_zero_only(*args, **kwargs)
